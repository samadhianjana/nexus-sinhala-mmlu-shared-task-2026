#!/usr/bin/env bash
# Queue of experiments B, A, C (one GPU: run back to back). Every result is scored like the zero-shot baseline
# (English prompt, logit scoring, 4 cyclic option orders). Resumable: finished evaluations are skipped.
#   B  LoRA fine-tuning on SinhalaMMLU dev, 5-fold cross-validation (each fold scored on its held-out 370 questions)
#   A  LoRA fine-tuning on Global-MMLU si, 1 epoch
#   C  continued-pretraining pilot, 30M tokens, checkpoints at 15M and 30M scored zero-shot
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
mkdir -p results/logs
ts() { date '+%d %b %H:%M'; }
evaluate() {  # model run_name data...
  local m=$1 name=$2; shift 2
  [ -f "results/$name/metrics.json" ] && { echo "skip eval $name"; return; }
  $PY -m eval.run_eval --model "$m" --data "$@" --method logit --n-perms 4 --run-name "$name" \
      > "results/logs/$name.log" 2>&1 && grep -E "acc=" "results/logs/$name.log" | sed 's/.*\(exp_\)/\1/' \
    || { echo "FAILED eval $name"; tail -3 "results/logs/$name.log"; }
}
train() {  # log_name args...
  local name=$1; shift
  $PY -m train.train "$@" 2>&1 | grep -vE "[Ww]arn|W10[0-9]{2}|Writing model" > "results/logs/train_$name.log"
  grep -E "^(sft|cpt):|^done|Error|error" "results/logs/train_$name.log" | tail -3
}

echo "=== B: dev 5-fold LoRA ($(ts))"
for f in 0 1 2 3 4; do
  name=exp_devcv_lora_f$f
  [ -f "results/$name/metrics.json" ] && { echo "skip fold $f"; continue; }
  train devcv_lora_f$f --config configs/train/sft_lora_devcv.yaml \
      --data data/processed/smmlu_cv5_f${f}_train.jsonl --out models/devcv_lora_f$f
  evaluate "$(ls -d models/devcv_lora_f$f/step-* | tail -1)" $name data/processed/smmlu_cv5_f${f}_test.jsonl
  rm -rf models/devcv_lora_f$f   # 15 GB each; the fold result is what we need
done
$PY -m eval.combine_cv --runs "results/exp_devcv_lora_f*" --name exp_devcv_lora_all

echo "=== A: Global-MMLU LoRA 1 epoch ($(ts))"
if [ ! -f results/exp_sft_gmmlu_lora_logit/metrics.json ]; then
  train sft_gmmlu_lora --config configs/train/sft_lora.yaml --out models/sft_gmmlu_lora
  evaluate "$(ls -d models/sft_gmmlu_lora/step-* | tail -1)" exp_sft_gmmlu_lora_logit sinhalammlu_dev gmmlu_si_val
fi

echo "=== C: pretraining pilot 30M tokens ($(ts))"
if [ ! -d models/cpt_pilot ] || [ -z "$(ls -d models/cpt_pilot/step-* 2>/dev/null)" ]; then
  train cpt_pilot --config configs/train/cpt_pilot.yaml
fi
for ck in $(ls -d models/cpt_pilot/step-* | sort -t- -k2 -n); do
  evaluate "$ck" "exp_cpt_pilot_$(basename "$ck")_logit" sinhalammlu_dev gmmlu_si_val
done

$PY -m eval.summarize --runs results/base_gemma-4-E4B-it_logit results/exp_devcv_lora_all \
    results/exp_sft_gmmlu_lora_logit "results/exp_cpt_pilot_step-*_logit" --out reports/experiments_abc.md > /dev/null
echo "summary -> reports/experiments_abc.md"
echo "=== done ($(ts))"
