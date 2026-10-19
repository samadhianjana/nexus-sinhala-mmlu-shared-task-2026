#!/usr/bin/env bash
# Phase 1 checks (PLAN.md):
#   1. text-only Gemma 4 E4B: extract, evaluate, compare with the multimodal baseline
#   2. offline submission pipeline on the dev files, validated + scored like the leaderboard
#   3. training throughput benchmarks
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
mkdir -p results/logs results/dev_submission
step() { echo; echo "=== $* ($(date +%H:%M))"; }

step "1a. extract text-only Gemma 4 E4B"
[ -f models/gemma-4-E4B-it-text/SOURCE.json ] || $PY scripts/make_text_only.py google/gemma-4-E4B-it 2>&1 | grep -v -i warn | tail -3

step "1b. evaluate text-only model (original option order)"
if [ ! -f results/exp_gemma-4-E4B-it-text_logit/metrics.json ]; then
  $PY -m eval.run_eval --model models/gemma-4-E4B-it-text --data sinhalammlu_dev gmmlu_si_val --n-perms 1 \
      --run-name exp_gemma-4-E4B-it-text_logit > results/logs/exp_gemma-4-E4B-it-text_logit.log 2>&1
fi
grep -E "acc=" results/logs/exp_gemma-4-E4B-it-text_logit.log
$PY - <<'EOF'
from eval.common import read_jsonl
base = {r["uid"]: r for r in read_jsonl("results/base_gemma-4-E4B-it_logit/predictions.jsonl") if r["perm_k"] == 0}
text = {r["uid"]: r for r in read_jsonl("results/exp_gemma-4-E4B-it-text_logit/predictions.jsonl")}
common = base.keys() & text.keys()
same = sum(base[u]["pred_orig"] == text[u]["pred_orig"] for u in common)
dp = max(max(abs(a - b) for a, b in zip(base[u]["probs_orig"], text[u]["probs_orig"])) for u in common)
print(f"text-only vs multimodal: same prediction on {same}/{len(common)} questions; max probability difference {dp:.4f}")
EOF

step "2. offline submission pipeline on dev files"
HF_HUB_OFFLINE=1 $PY -m inference.predict --input data/raw/sinhalammlu/TEST/easy \
    --output results/dev_submission/predictions.zip 2>&1 | grep -vE "it/s\]|s/it\]|[Ww]arn" | tail -3
$PY -m inference.check_predictions --predictions results/dev_submission/predictions.zip --gold data/raw/sinhalammlu/TEST/easy

step "3. training throughput"
for spec in "models/gemma-4-E4B-it-text lora" "models/gemma-4-E4B-it-text freeze-embed" "Qwen/Qwen3.5-4B full" "Qwen/Qwen3.5-4B lora"; do
  set -- $spec
  log="results/logs/bench_$(basename "$1")_$2.log"
  $PY -m train.bench_throughput --model "$1" --mode "$2" > "$log" 2>&1 && tail -1 "$log" || { echo "FAILED $spec"; tail -4 "$log"; }
done
step "done"
