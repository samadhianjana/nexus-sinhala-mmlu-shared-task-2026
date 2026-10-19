#!/usr/bin/env bash
# Step 4: fine-tune text-only Gemma 4 E4B on Global-MMLU si, then score every epoch checkpoint exactly like the
# zero-shot baseline (English prompt, logit scoring, 4 cyclic option orders) on SinhalaMMLU dev + Global-MMLU val.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
CFG=configs/train/sft_gmmlu.yaml
OUT=models/sft_gmmlu_v1
mkdir -p results/logs

echo "=== train ($(date '+%d %b %H:%M'))"
$PY -m train.train --config $CFG 2>&1 | grep -vE "[Ww]arn|W10[0-9]{2}|Writing model" | tee results/logs/train_sft_gmmlu_v1.log

for ck in $(ls -d $OUT/step-* | sort -t- -k2 -n); do
  name="exp_sft_gmmlu_v1_$(basename "$ck")_logit"
  echo "=== eval $ck ($(date '+%H:%M'))"
  $PY -m eval.run_eval --model "$ck" --data sinhalammlu_dev gmmlu_si_val --method logit --n-perms 4 \
      --run-name "$name" > "results/logs/$name.log" 2>&1 && grep -E "acc=" "results/logs/$name.log" \
    || { echo "FAILED eval $ck"; tail -3 "results/logs/$name.log"; }
done

$PY -m eval.summarize --runs results/base_gemma-4-E4B-it_logit "results/exp_sft_gmmlu_v1_step-*_logit" \
    --out reports/sft_gmmlu_v1.md > /dev/null && echo "summary -> reports/sft_gmmlu_v1.md"
echo "=== done ($(date '+%d %b %H:%M'))"
