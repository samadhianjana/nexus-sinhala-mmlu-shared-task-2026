#!/usr/bin/env bash
# Gemma 4 E4B: compile with activation recomputation (fixed) and/or 8-bit optimizer states, for memory headroom.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
G=models/gemma-4-E4B-it-text
RUNS=(
  "--ckpt-every 2 --optim adamw"
  "--ckpt-every 0 --optim adamw8bit"
  "--ckpt-every 2 --optim adamw8bit"
)
for r in "${RUNS[@]}"; do
  tag="gemma4E4B_freezeembed_liger_compile_$(echo "$r" | tr -d '-' | tr ' ' '_')"
  echo "=== $tag ($(date +%H:%M))"
  $PY -m train.bench_throughput --model $G --mode freeze-embed --loss liger --compile $r --tag memory \
      > "results/logs/bench_$tag.log" 2>&1 \
    && tail -1 "results/logs/bench_$tag.log" || { echo "FAILED"; grep -vE "^\s*$|^W1011" "results/logs/bench_$tag.log" | tail -3; }
  echo "recompile-limit warnings: $(grep -c 'hit config.recompile_limit' "results/logs/bench_$tag.log")"
done
echo "=== done ($(date +%H:%M))"
