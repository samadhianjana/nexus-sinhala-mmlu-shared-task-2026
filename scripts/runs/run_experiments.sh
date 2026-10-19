#!/usr/bin/env bash
# Prompting / scoring experiments on the top baseline models, compared against their zero-shot English logit run.
#   logit_si     Sinhala instruction, 4 cyclic option orders
#   logit_5shot  fixed 5 examples from the dev pool (same for every question), original option order
#   generate     greedy generation + answer parsing, original option order
#   cloze        length-normalised option likelihood (options not shown)
# Usage: scripts/run_experiments.sh [model ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
DATA="sinhalammlu_dev gmmlu_si_val"
if [ $# -gt 0 ]; then MODELS=("$@"); else MODELS=(google/gemma-4-E4B-it google/gemma-3n-E4B-it Qwen/Qwen3.5-4B); fi

declare -A EXP=(
  [logit_si]="--method logit --lang si --n-perms 4"
  [logit_5shot]="--method logit --fewshot-file sinhalammlu_dev_5shot --n-shot 5 --n-perms 1"
  [generate]="--method generate --n-perms 1 --max-new-tokens 16"
  [cloze]="--method cloze"
)
ORDER=(logit_si logit_5shot generate cloze)

mkdir -p results/logs
for m in "${MODELS[@]}"; do
  for e in "${ORDER[@]}"; do
    name="exp_$(basename "$m")_${e}"
    if [ -f "results/$name/metrics.json" ]; then echo "skip $name (done)"; continue; fi
    echo "=== $name ($(date +%H:%M))"
    $PY -m eval.run_eval --model "$m" --data $DATA ${EXP[$e]} --run-name "$name" > "results/logs/$name.log" 2>&1 \
      && grep -E "acc=|scoring" "results/logs/$name.log" \
      || { echo "FAILED $name (see results/logs/$name.log)"; tail -5 "results/logs/$name.log"; }
  done
done
runs=()
for m in "${MODELS[@]}"; do runs+=("results/base_$(basename "$m")_logit" "results/exp_$(basename "$m")_*"); done
$PY -m eval.summarize --runs "${runs[@]}" --out reports/experiments_prompting.md > /dev/null
echo "summary -> reports/experiments_prompting.md"
