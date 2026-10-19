#!/usr/bin/env bash
# Throughput with the speed options of train/efficiency.py (results appended to reports/throughput.jsonl).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
G=models/gemma-4-E4B-it-text
Q=Qwen/Qwen3.5-4B
RUNS=(
  "$G freeze-embed --loss chunked --ckpt-every 1"
  "$G freeze-embed --loss liger --ckpt-every 1"
  "$G freeze-embed --loss liger --ckpt-every 2"
  "$G freeze-embed --loss liger --ckpt-every 0"
  "$G freeze-embed --loss liger --ckpt-every 1 --compile"
  "$G freeze-embed --loss liger --ckpt-every 2 --compile"
  "$G lora --loss liger --ckpt-every 1 --compile"
  "$Q full --loss hf --ckpt-every 1"
  "$Q full --loss liger --ckpt-every 1"
  "$Q full --loss liger --ckpt-every 1 --compile"
  "$Q lora --loss liger --ckpt-every 1 --compile"
)
mkdir -p results/logs
for r in "${RUNS[@]}"; do
  set -- $r
  m=$1; mode=$2; shift 2
  tag=$(echo "$(basename "$m")_${mode}_$*" | tr ' ' '_' | tr -d '-')
  echo "=== $tag ($(date +%H:%M))"
  $PY -m train.bench_throughput --model "$m" --mode "$mode" "$@" --tag speedups > "results/logs/bench_$tag.log" 2>&1 \
    && tail -1 "results/logs/bench_$tag.log" || { echo "FAILED"; grep -vE "^\s*$" "results/logs/bench_$tag.log" | tail -3; }
done
echo "=== done ($(date +%H:%M))"
