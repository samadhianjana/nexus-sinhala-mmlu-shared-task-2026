#!/usr/bin/env bash
# 26-27 Oct, fully unattended: reasoning-preserving fine-tuning (rejection-sampled own traces) and, only if it passes the
# pre-registered validation rule, the final test predictions.
#   1. traces: base model, thinking + sampling on 1,450 dev questions (the 400-question dev sample is held out), 2 samples,
#      keep the correct ones (time-boxed to 4 h)
#   2. LoRA r=16 on the traces (loss on the trace only)
#   3. validation: greedy thinking with the adapter on the held-out dev 400 and the Global-MMLU non-academic 300, paired
#      against the base model's greedy thinking; eval/decide_rft.py applies the fixed rule (GO / NO-GO)
#   4. if GO: test predictions = candidate C (configs/submission_rft_think_gated_sc.yaml), format check, comparison with B
# Every stage is resumable (finished outputs are skipped). On NO-GO or any failure, candidate B stays final.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ts() { date '+%d %b %H:%M'; }
mkdir -p results/logs data/synth

echo "=== 1. traces ($(ts))"
$PY scripts/gen_traces.py --data smmlu_dev_rest1450 --out data/synth/rft_traces --samples 2 --max-minutes 240 \
    2>&1 | grep -E "questions|samples,|time box|done|Error|Traceback" | tee -a results/logs/rft_traces.log | tail -4
N=$(wc -l < data/synth/rft_traces.jsonl 2>/dev/null || echo 0)
echo "kept traces: $N"
[ "$N" -ge 300 ] || { echo "too few traces ($N): stopping, candidate B stays final"; exit 1; }

echo "=== 2. training ($(ts))"
if [ -z "$(ls -d models/rft_v1/step-* 2>/dev/null)" ]; then
  $PY -m train.train --config configs/train/rft_trace.yaml 2>&1 | grep -vE "[Ww]arn|W10[0-9]{2}|Writing model" > results/logs/train_rft_v1.log
  grep -E "trace examples|^trace:|^done|Error|Traceback" results/logs/train_rft_v1.log | tail -4
fi
CK=$(ls -d models/rft_v1/step-* 2>/dev/null | tail -1)
[ -n "$CK" ] && [ -f "$CK/adapter/adapter_model.safetensors" ] || { echo "training failed: candidate B stays final"; exit 1; }
echo "adapter: $CK/adapter"

echo "=== 3. validation on held-out questions ($(ts))"
val() {  # run_name data
  [ -f "results/$1/metrics.json" ] || $PY -m eval.run_eval --model models/gemma-4-E4B-it-text --adapter "$CK/adapter" --data $2 \
      --method generate --thinking --max-new-tokens 3072 --run-name $1 > results/logs/$1.log 2>&1
  grep -E "acc=" results/logs/$1.log | sed 's/.*\(think_\)/\1/'
}
val think_rft_dev400 smmlu_dev_sample400
val think_rft_nonacademic300 gmmlu_si_val_nonacademic300
$PY -m eval.decide_rft --pairs think_base_smmlu_dev_sample400:think_rft_dev400:sinhalammlu \
    think_base_gmmlu_si_val_nonacademic300:think_rft_nonacademic300:gmmlu_si | tee results/logs/rft_decision.log | tail -22
grep -q "DECISION: GO" results/logs/rft_decision.log || { echo "NO-GO: candidate B stays final ($(ts))"; exit 0; }

echo "=== 4. candidate C test predictions ($(ts))"
sed "s|models/rft_v1/step-ADAPTER/adapter|$CK/adapter|" configs/submission_rft_think_gated_sc.yaml > configs/submission_rft_think_gated_sc.resolved.yaml
OUT=results/submissions/final_candidate_rft_think_gated_sc.zip
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_rft_think_gated_sc.resolved.yaml \
    --input data/test --output "$OUT" > results/logs/predict_final_candidate_rft.log 2>&1
grep -E "questions|thinking|self-consistency|wrote|Error|Traceback" results/logs/predict_final_candidate_rft.log
$PY -m inference.check_predictions --predictions "$OUT" --gold data/test 2>&1 | tail -1
$PY - <<'PYEOF'
import json, zipfile
def load(p): return {(r["level"], r["subject"], r["q_no"]): r["answer"] for r in map(json.loads, zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines())}
a = load("results/submissions/final_candidate_base_think_gated_sc.zip")
b = load("results/submissions/final_candidate_rft_think_gated_sc.zip")
for lvl in ("easy", "medium", "hard"):
    ks = [k for k in a if k[0] == lvl]
    print(f"  {lvl}: answers changed vs candidate B: {sum(a[k] != b[k] for k in ks)}/{len(ks)}")
PYEOF
echo "=== done ($(ts))"
