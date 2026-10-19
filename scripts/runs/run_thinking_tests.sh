#!/usr/bin/env bash
# 19 Oct, step 1: does thinking mode help? Model A (current submission model) and the base model, think-then-answer,
# on Global-MMLU academic subjects (STEM + Business, proxy for A/L "hard") and a 400-question dev sample.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
A=models/sft_gmmlu_lora/step-357
run() {  # name model data...
  local name=$1 m=$2; shift 2
  [ -f "results/$name/metrics.json" ] && { echo "skip $name"; return; }
  $PY -m eval.run_eval --model "$m" --data "$@" --method generate --thinking --max-new-tokens 2048 \
      --run-name "$name" > "results/logs/$name.log" 2>&1 && grep -E "acc=" "results/logs/$name.log" | sed 's/.*\(think_\)/\1/' \
    || { echo "FAILED $name"; tail -3 "results/logs/$name.log"; }
}
echo "=== thinking tests ($(date '+%H:%M'))"
run think_A_academic "$A" gmmlu_si_val_academic
run think_A_dev400 "$A" smmlu_dev_sample400
run think_base_academic models/gemma-4-E4B-it-text gmmlu_si_val_academic
echo "=== done ($(date '+%H:%M'))"
