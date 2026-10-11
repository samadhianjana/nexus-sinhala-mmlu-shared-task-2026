#!/usr/bin/env bash
# Rerun of the Gemma 4 E4B settings affected by the compile recompile-limit and OOM-retry bugs (10 Oct grid).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
G=models/gemma-4-E4B-it-text
for ck in 1 2 0; do
  tag="gemma4E4Bittext_freezeembed_liger_ckpt${ck}_compile_fixed"
  echo "=== $tag ($(date +%H:%M))"
  $PY -m train.bench_throughput --model $G --mode freeze-embed --loss liger --ckpt-every $ck --compile \
      --tag compile_fixed > "results/logs/bench_$tag.log" 2>&1 \
    && tail -1 "results/logs/bench_$tag.log" || { echo "FAILED"; grep -vE "^\s*$" "results/logs/bench_$tag.log" | tail -3; }
  echo "recompilation warnings: $(grep -c 'recompile' "results/logs/bench_$tag.log")"
done
echo "=== done ($(date +%H:%M))"
