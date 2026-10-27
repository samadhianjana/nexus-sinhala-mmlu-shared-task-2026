#!/usr/bin/env bash
# Option 2 (27 Oct), fully unattended; meant for a second GPU machine.
# Reasoning-preserving fine-tuning round 2, MEDIUM questions only at test time.
#   0. checks   1. traces on Global-MMLU + synthetic questions (time-boxed)   2. LoRA on dev traces + new traces
#   3. validation on held-out questions + pre-registered rule (eval/decide_rft.py)   4. if GO: medium test run + merge into B
# Environment variables: B_ZIP (candidate B predictions), TRACE_MINUTES (default 240), MAX_BATCH (default 64; lower on a small GPU),
#                        GPU_MEM_GB (default 100; set to ~85% of the GPU's memory)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
B_ZIP=${B_ZIP:-results/submissions/final_candidate_base_think_gated_sc.zip}
TRACE_MINUTES=${TRACE_MINUTES:-240}
MAX_BATCH=${MAX_BATCH:-64}
export SMMLU_GPU_MEM_GB=${GPU_MEM_GB:-100}
ts() { date '+%d %b %H:%M'; }
mkdir -p results/logs data/synth results/submissions

echo "=== 0. checks ($(ts))"
for f in data/processed/smmlu_dev_rest1450.jsonl data/processed/smmlu_dev_sample400.jsonl data/processed/gmmlu_si_val_nonacademic300.jsonl \
         data/sft/gmmlu_si_train.jsonl data/synth/rft_traces.jsonl results/think_base_smmlu_dev_sample400/predictions.jsonl \
         results/think_base_gmmlu_si_val_nonacademic300/predictions.jsonl models/gemma-4-E4B-it-text/model.safetensors "$B_ZIP" data/test; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
$PY scripts/build_rft2_questions.py

echo "=== 1. traces ($(ts))"
$PY scripts/gen_traces.py --data data/synth/rft2_questions.jsonl --out data/synth/rft2_traces --samples 1 \
    --max-minutes "$TRACE_MINUTES" --max-batch "$MAX_BATCH" 2>&1 | grep -E "questions|samples,|time box|done|Error|Traceback" | tee -a results/logs/rft2_traces.log | tail -4
N=$(wc -l < data/synth/rft2_traces.jsonl 2>/dev/null || echo 0)
echo "new kept traces: $N (plus $(wc -l < data/synth/rft_traces.jsonl) dev traces)"
[ "$N" -ge 300 ] || { echo "too few new traces: stopping, candidate B stays final"; exit 1; }

echo "=== 2. training ($(ts))"
if [ -z "$(ls -d models/rft_v2/step-* 2>/dev/null)" ]; then
  $PY -m train.train --config configs/train/rft_trace2.yaml 2>&1 | grep -vE "[Ww]arn|W10[0-9]{2}|Writing model" > results/logs/train_rft_v2.log
  grep -E "trace examples|^trace:|^done|Error|Traceback" results/logs/train_rft_v2.log | tail -4
fi
CK=$(ls -d models/rft_v2/step-* 2>/dev/null | tail -1)
[ -n "$CK" ] && [ -f "$CK/adapter/adapter_model.safetensors" ] || { echo "training failed: candidate B stays final"; exit 1; }

echo "=== 3. validation ($(ts))"
val() {  # run_name data
  [ -f "results/$1/metrics.json" ] || $PY -m eval.run_eval --model models/gemma-4-E4B-it-text --adapter "$CK/adapter" --data $2 \
      --method generate --thinking --max-new-tokens 3072 --max-batch "$MAX_BATCH" --run-name $1 > results/logs/$1.log 2>&1
  grep -E "acc=" results/logs/$1.log | sed 's/.*\(think_\)/\1/'
}
val think_rft2_dev400 smmlu_dev_sample400
val think_rft2_nonacademic300 gmmlu_si_val_nonacademic300
$PY -m eval.decide_rft --out rft2_decision.json --pairs think_base_smmlu_dev_sample400:think_rft2_dev400:sinhalammlu \
    think_base_gmmlu_si_val_nonacademic300:think_rft2_nonacademic300:gmmlu_si | tee results/logs/rft2_decision.log | tail -22
grep -q "DECISION: GO" results/logs/rft2_decision.log || { echo "NO-GO: candidate B stays final ($(ts))"; exit 0; }

echo "=== 4. medium test questions ($(ts))"
sed -e "s|models/rft_v2/step-ADAPTER/adapter|$CK/adapter|" -e "s|^max_batch: .*|max_batch: $MAX_BATCH|" -e "s|^gpu_mem_gb: .*|gpu_mem_gb: ${GPU_MEM_GB:-100}|" \
    configs/submission_rft2_medium.yaml > configs/submission_rft2_medium.resolved.yaml
OUT=results/submissions/rft2_medium.zip
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_rft2_medium.resolved.yaml \
    --input data/test --levels medium --output "$OUT" > results/logs/predict_rft2_medium.log 2>&1
grep -E "questions|thinking|self-consistency|wrote|Error|Traceback" results/logs/predict_rft2_medium.log
$PY scripts/merge_predictions.py --base "$B_ZIP" --new "$OUT" --out results/submissions/final_candidate_B_plus_rft2_medium.zip
$PY -m inference.check_predictions --predictions results/submissions/final_candidate_B_plus_rft2_medium.zip --gold data/test 2>&1 | tail -1
echo "=== done ($(ts))"
