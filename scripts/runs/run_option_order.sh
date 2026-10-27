#!/usr/bin/env bash
# 27 Oct, unattended: option-order self-consistency for Medium.
#   1. validate on the Global-MMLU non-academic stand-in (300 q, held out): thinking under 3 re-ordered option lists, voted with
#      the existing greedy + 3 sampled paths (eval/perm_sc.py; GO if +1.0 over the current vote)
#   2. if GO: run the medium test questions only (configs/submission_medium_option_order.yaml, --levels medium)
#   3. merge into candidate B and run the format check
# On NO-GO or failure candidate B stays final.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ts() { date '+%d %b %H:%M'; }
mkdir -p results/logs

echo "=== 1. validation ($(ts))"
[ -f results/permthink_nonacademic300/metrics.json ] || $PY -m eval.run_eval --model models/gemma-4-E4B-it-text \
    --data gmmlu_si_val_nonacademic300 --method generate --thinking --max-new-tokens 3072 --n-perms 4 --first-perm 1 \
    --perm-mode cyclic --run-name permthink_nonacademic300 > results/logs/permthink_nonacademic300.log 2>&1
$PY -m eval.perm_sc --greedy results/think_base_gmmlu_si_val_nonacademic300 --samples results/sc3_base_nonacademic \
    --perms results/permthink_nonacademic300 | tee results/logs/perm_decision.log | tail -16
grep -q "DECISION: GO" results/logs/perm_decision.log || { echo "NO-GO: candidate B stays final ($(ts))"; exit 0; }

echo "=== 2. medium test questions ($(ts))"
OUT=results/submissions/medium_option_order.zip
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_medium_option_order.yaml \
    --input data/test --levels medium --output "$OUT" > results/logs/predict_medium_option_order.log 2>&1
grep -E "questions|option-order|wrote|Error|Traceback" results/logs/predict_medium_option_order.log

echo "=== 3. merge and check ($(ts))"
$PY scripts/merge_predictions.py --base results/submissions/final_candidate_base_think_gated_sc.zip --new "$OUT" \
    --out results/submissions/final_candidate_B_plus_option_order_medium.zip
$PY -m inference.check_predictions --predictions results/submissions/final_candidate_B_plus_option_order_medium.zip --gold data/test 2>&1 | tail -1
echo "=== done ($(ts))"
