#!/usr/bin/env bash
# 20 Oct: submission 6 = base Gemma 4 E4B + adapter A; base-model thinking (adapter off) for medium and hard,
# adapter A fast 4-order vote for easy.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=results/submissions/sub6_base_adapterA_think_medhard.zip
echo "=== submission 6 predictions ($(date '+%d %b %H:%M'))"
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_sub6_think_medhard.yaml --input data/test \
    --output "$OUT" 2>&1 | grep -E "questions|thinking|wrote|Error" > results/logs/predict_sub6.log
cat results/logs/predict_sub6.log
$PY -m inference.check_predictions --predictions "$OUT" --gold data/test 2>&1 | tail -1
$PY - <<'PYEOF'
import json, zipfile
def load(p): return {(r["level"], r["subject"], r["q_no"]): r["answer"] for r in map(json.loads, zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines())}
a, b = load("results/submissions/sub5_base_adapterA_vote4_think_hard.zip"), load("results/submissions/sub6_base_adapterA_think_medhard.zip")
for lvl in ("easy", "medium", "hard"):
    ks = [k for k in a if k[0] == lvl]
    print(f"  {lvl}: answers changed vs submission 5: {sum(a[k] != b[k] for k in ks)}/{len(ks)}")
PYEOF
echo "=== done ($(date '+%d %b %H:%M'))"
