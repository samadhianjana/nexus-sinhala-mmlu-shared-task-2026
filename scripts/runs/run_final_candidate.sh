#!/usr/bin/env bash
# 21 Oct: final candidate = base Gemma 4 E4B text-only (7.463B, no adapter), thinking for every question,
# self-consistency (greedy + 3 samples) for hard. Config: configs/submission_base_think_sc_hard.yaml
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=results/submissions/final_candidate_base_think_sc_hard.zip
echo "=== final candidate predictions ($(date '+%d %b %H:%M'))"
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_base_think_sc_hard.yaml --input data/test \
    --output "$OUT" > results/logs/predict_final_candidate.log 2>&1
grep -E "questions|thinking|self-consistency|wrote|Error|Traceback" results/logs/predict_final_candidate.log
$PY -m inference.check_predictions --predictions "$OUT" --gold data/test 2>&1 | tail -1
$PY - <<'PYEOF'
import json, zipfile
def load(p): return {(r["level"], r["subject"], r["q_no"]): r["answer"] for r in map(json.loads, zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines())}
a, b = load("results/submissions/sub6_base_adapterA_think_medhard.zip"), load("results/submissions/final_candidate_base_think_sc_hard.zip")
for lvl in ("easy", "medium", "hard"):
    ks = [k for k in a if k[0] == lvl]
    print(f"  {lvl}: answers changed vs the previous submission: {sum(a[k] != b[k] for k in ks)}/{len(ks)}")
PYEOF
echo "=== done ($(date '+%d %b %H:%M'))"
