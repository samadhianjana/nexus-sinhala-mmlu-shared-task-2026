#!/usr/bin/env bash
# Day 1 (18 Oct) queue. GPU jobs run one after another; teacher downloads run in parallel with v2 training.
#   1. v2 = LoRA on Global-MMLU si + all SinhalaMMLU dev (1 epoch), merged
#   2. v2 test predictions (4-order vote) + format check; v2 sanity score on Global-MMLU validation
#   3. Teacher candidates scored zero-shot on dev + Global-MMLU validation (logit) and dev (generated answers)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
TEACHERS=(google/gemma-4-31B-it google/gemma-4-26B-A4B-it Qwen/Qwen3.5-27B)
mkdir -p results/logs results/submissions
ts() { date '+%d %b %H:%M'; }

# Teacher downloads run separately (scripts/runs/download_teachers.sh, tmux session "downloads").
# Wait until no other GPU training (the WavLM ta-100h run of Model-Training-Pipeline) is running.
until ! pgrep -f "Model-Training-Pipeline/.venv/bin/python" > /dev/null; do
  echo "waiting for the GPU: another project's training is still running ($(ts))"; sleep 300
done
echo "GPU free ($(ts))"

# --- 1. v2 training ---
echo "=== v2 training ($(ts))"
if [ -z "$(ls -d models/sft_v2_gmmlu_dev/step-* 2>/dev/null)" ]; then
  $PY -m train.train --config configs/train/sft_lora.yaml \
      --data data/sft/gmmlu_si_train.jsonl data/processed/sinhalammlu_dev.jsonl --out models/sft_v2_gmmlu_dev 2>&1 \
    | grep -vE "[Ww]arn|W10[0-9]{2}|Writing model" > results/logs/train_sft_v2_gmmlu_dev.log
  grep -E "^sft:|^done|Error" results/logs/train_sft_v2_gmmlu_dev.log | tail -3
fi
V2=$(ls -d models/sft_v2_gmmlu_dev/step-* | tail -1)
sed -i "s|^model: models/sft_v2_gmmlu_dev/step-[0-9]*|model: $V2|" configs/submission_v2.yaml

# --- 2. v2 test predictions + checks ---
echo "=== v2 test predictions ($(ts))"
OUT=results/submissions/sub3_v2_lora_gmmlu_dev_vote4.zip
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_v2.yaml --input data/test \
    --output "$OUT" > results/logs/predict_v2.log 2>&1
$PY -m inference.check_predictions --predictions "$OUT" --gold data/test 2>&1 | tail -2
[ -f results/exp_sft_v2_gmmlu_dev_gmmluval/metrics.json ] || $PY -m eval.run_eval --model "$V2" --data gmmlu_si_val \
    --n-perms 4 --run-name exp_sft_v2_gmmlu_dev_gmmluval > results/logs/exp_sft_v2_gmmlu_dev_gmmluval.log 2>&1
grep -E "acc=" results/logs/exp_sft_v2_gmmlu_dev_gmmluval.log | sed 's/.*\(exp_\)/\1/'

# --- 3. teacher evaluations ---
for m in "${TEACHERS[@]}"; do
  n=$(basename "$m")
  until [ -f "results/logs/.downloaded_$n" ] || ! tmux has-session -t downloads 2>/dev/null; do sleep 60; done
  [ -f "results/logs/.downloaded_$n" ] || echo "download of $m failed or not finished"
  [ -f "results/logs/.downloaded_$n" ] || continue
  echo "=== teacher $m ($(ts))"
  [ -f "results/teacher_${n}_logit/metrics.json" ] || $PY -m eval.run_eval --model "$m" --data sinhalammlu_dev gmmlu_si_val \
      --method logit --n-perms 1 --token-budget 16384 --run-name "teacher_${n}_logit" > "results/logs/teacher_${n}_logit.log" 2>&1
  grep -E "acc=" "results/logs/teacher_${n}_logit.log" | sed 's/.*\(teacher_\)/\1/' || tail -3 "results/logs/teacher_${n}_logit.log"
  [ -f "results/teacher_${n}_gen/metrics.json" ] || $PY -m eval.run_eval --model "$m" --data sinhalammlu_dev \
      --method generate --max-new-tokens 48 --token-budget 16384 --run-name "teacher_${n}_gen" > "results/logs/teacher_${n}_gen.log" 2>&1
  grep -E "acc=" "results/logs/teacher_${n}_gen.log" | sed 's/.*\(teacher_\)/\1/' || tail -3 "results/logs/teacher_${n}_gen.log"
done
$PY -m eval.summarize --runs results/base_gemma-4-E4B-it_logit "results/teacher_*" --out reports/teachers.md > /dev/null
echo "summary -> reports/teachers.md"
echo "=== done ($(ts))"
