#!/usr/bin/env bash
# Cross-model pooling (27-28 Oct), fully unattended. Medium only at test time.
#   1. validation runs on FRESH stand-in questions   2. pre-registered rule (eval/pool_decide.py, gain >= +1.0)
#   3. if GO: base and RFT2 test runs with saved ballots (Medium)   4. pool, merge into B, format check
# Environment variables: B_ZIP (candidate B predictions), MAX_BATCH (default 64), GPU_MEM_GB (default 100)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
B_ZIP=${B_ZIP:-results/submissions/final_candidate_base_think_gated_sc.zip}
MAX_BATCH=${MAX_BATCH:-64}
ADAPTER=models/rft_v2/step-141/adapter
export SMMLU_GPU_MEM_GB=${GPU_MEM_GB:-100}
ts() { date '+%d %b %H:%M'; }
mkdir -p results/logs results/submissions

echo "=== 0. checks ($(ts))"
for f in "$ADAPTER/adapter_model.safetensors" models/gemma-4-E4B-it-text/model.safetensors "$B_ZIP" data/test; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
$PY scripts/prepare_subsets.py
D=gmmlu_si_val_nonacademic_fresh300

echo "=== 1. validation on $D ($(ts))"
run() {  # run_name extra-args...
  local name=$1; shift
  [ -f "results/$name/metrics.json" ] || $PY -m eval.run_eval --model models/gemma-4-E4B-it-text --data $D \
      --max-batch "$MAX_BATCH" --run-name "$name" "$@" > "results/logs/$name.log" 2>&1
  echo "  $name: $(grep -oE "acc=[0-9.]+" "results/logs/$name.log" | tail -1) ($(ts))"
}
THINK="--method generate --thinking --max-new-tokens 3072"
SAMPLE="--temperature 0.8 --perm-mode identity --n-perms 3"
run pool_fast_base_fresh300 --method logit --n-perms 4
run pool_think_base_fresh300 $THINK
run pool_sc3_base_fresh300 $THINK $SAMPLE
run pool_think_rft2_fresh300 $THINK --adapter "$ADAPTER"
run pool_sc3_rft2_fresh300 $THINK $SAMPLE --adapter "$ADAPTER"
$PY -m eval.pool_decide --fast pool_fast_base_fresh300 --base-greedy pool_think_base_fresh300 --base-samples pool_sc3_base_fresh300 \
    --rft-greedy pool_think_rft2_fresh300 --rft-samples pool_sc3_rft2_fresh300 | tee results/logs/pool_decision.log | tail -20
grep -q "DECISION: GO" results/logs/pool_decision.log || { echo "NO-GO: candidate B stays final ($(ts))"; exit 0; }

echo "=== 2. medium test questions ($(ts))"
for part in base rft2; do
  OUT=results/submissions/pool_${part}_medium.zip
  sed -e "s|^max_batch: .*|max_batch: $MAX_BATCH|" -e "s|^gpu_mem_gb: .*|gpu_mem_gb: ${GPU_MEM_GB:-100}|" \
      configs/submission_pool_${part}_medium.yaml > configs/submission_pool_${part}_medium.resolved.yaml
  [ -f "$OUT.ballots.jsonl" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_pool_${part}_medium.resolved.yaml \
      --input data/test --levels medium --output "$OUT" > results/logs/predict_pool_${part}_medium.log 2>&1
  grep -E "questions|thinking|self-consistency|wrote|Error|Traceback" results/logs/predict_pool_${part}_medium.log
  echo "  $part done ($(ts))"
done

echo "=== 3. pool, merge and check ($(ts))"
FINAL=results/submissions/final_candidate_B_plus_pool_medium.zip
$PY scripts/pool_ballots.py --base "$B_ZIP" --out "$FINAL" \
    --ballots results/submissions/pool_base_medium.zip.ballots.jsonl results/submissions/pool_rft2_medium.zip.ballots.jsonl
$PY -m inference.check_predictions --predictions "$FINAL" --gold data/test 2>&1 | tail -1
echo "=== done ($(ts))"
