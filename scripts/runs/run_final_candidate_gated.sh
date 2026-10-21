#!/usr/bin/env bash
# 21-22 Oct: final candidate B = base Gemma 4 E4B text-only (7.463B), thinking for every question, agreement-gated
# self-consistency on all levels (configs/submission_base_think_gated_sc.yaml). Starts after scripts/runs/run_final_candidate.sh
# (tmux session "final") has finished, so the two runs never share the GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
CFG=configs/submission_base_think_gated_sc.yaml
OUT=results/submissions/final_candidate_base_think_gated_sc.zip
until ! tmux has-session -t final 2>/dev/null || grep -q "=== done" results/logs/run_final_candidate.log 2>/dev/null; do sleep 120; done
echo "=== smoke test of the gate on 4 dev questions ($(date '+%d %b %H:%M'))"
sed 's/^thinking_levels: .*/thinking_levels: [easy]/; s/^sc_levels: .*/sc_levels: [easy]/' $CFG > results/logs/_smoke_gated.yaml
HF_HUB_OFFLINE=1 $PY -m inference.predict --config results/logs/_smoke_gated.yaml --input data/raw/sinhalammlu/TEST/easy \
    --output results/logs/_smoke_gated.zip --limit 4 2>&1 | grep -E "thinking|self-consistency|wrote|Error|Traceback" \
  || { echo "SMOKE TEST FAILED - not starting the full run"; exit 1; }
echo "=== final candidate B predictions ($(date '+%d %b %H:%M'))"
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config $CFG --input data/test --output "$OUT" \
    > results/logs/predict_final_candidate_gated.log 2>&1
grep -E "questions|thinking|self-consistency|wrote|Error|Traceback" results/logs/predict_final_candidate_gated.log
$PY -m inference.check_predictions --predictions "$OUT" --gold data/test 2>&1 | tail -1
$PY - <<'PYEOF'
import json, zipfile
def load(p): return {(r["level"], r["subject"], r["q_no"]): r["answer"] for r in map(json.loads, zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines())}
a = load("results/submissions/final_candidate_base_think_sc_hard.zip")
b = load("results/submissions/final_candidate_base_think_gated_sc.zip")
for lvl in ("easy", "medium", "hard"):
    ks = [k for k in a if k[0] == lvl]
    print(f"  {lvl}: answers changed vs candidate A (self-consistency on hard only): {sum(a[k] != b[k] for k in ks)}/{len(ks)}")
PYEOF
echo "=== done ($(date '+%d %b %H:%M'))"
