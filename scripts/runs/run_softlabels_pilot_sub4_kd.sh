#!/usr/bin/env bash
# 19 Oct, steps 2-4 (start after scripts/runs/run_thinking_tests.sh has finished; tmux session/log names day2a):
#   2. Gemma 4 31B soft labels for the 11,447 Global-MMLU training questions
#   3. synthetic pilot: ~450 MCQs from 150 Wikipedia passages, round-trip filtered, review file + overlap checks
#   3b. test predictions: model A + thinking on hard (configs/submission_A_think_hard.yaml)
#   4. student LoRA with distillation (0.5 CE + 0.5 KL to teacher), scored on dev + Global-MMLU val, compared with A
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
T=google/gemma-4-31B-it
ts() { date '+%d %b %H:%M'; }
until ! tmux has-session -t day2a 2>/dev/null || grep -q "=== done" results/logs/run_day2a.log 2>/dev/null; do sleep 60; done
echo "start ($(ts))"

echo "=== 2. teacher soft labels on Global-MMLU train ($(ts))"
[ -f results/teacher_31b_gmmlu_train/metrics.json ] || $PY -m eval.run_eval --model $T --data data/sft/gmmlu_si_train.jsonl \
    --method logit --n-perms 1 --token-budget 16384 --run-name teacher_31b_gmmlu_train > results/logs/teacher_31b_gmmlu_train.log 2>&1
grep -E "acc=" results/logs/teacher_31b_gmmlu_train.log | sed 's/.*\(teacher_\)/\1/'
$PY scripts/add_teacher_labels.py --data data/sft/gmmlu_si_train.jsonl --run results/teacher_31b_gmmlu_train \
    --out data/sft/gmmlu_si_train_t31b.jsonl

echo "=== 3. synthetic pilot ($(ts))"
[ -f data/synth/pilot.jsonl ] || $PY scripts/generate_mcq.py --passages data/synth/passages_pilot.jsonl --out data/synth/pilot \
    --teacher $T --per-passage 3 2>&1 | grep -vE "[Ww]arn|W10|it/s\]|s/it\]" > results/logs/synth_pilot.log
tail -25 results/logs/synth_pilot.log
for b in sinhalammlu_dev gmmlu_si_val; do
  echo "-- overlap pilot vs $b (character / meaning)"
  $PY scripts/overlap.py --a data/synth/pilot.jsonl --b $b --show 0 | sed -n '6,8p'
  $PY scripts/semantic_overlap.py --a data/synth/pilot.jsonl --b $b --show 0 2>/dev/null | grep -E "100th|>= 0.9"
done

echo "=== 3b. test predictions: A + thinking on hard ($(ts))"
OUT=results/submissions/sub4_A_vote4_think_hard.zip
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_A_think_hard.yaml --input data/test \
    --output "$OUT" 2>&1 | grep -E "questions|thinking|wrote|Error" > results/logs/predict_sub4.log
cat results/logs/predict_sub4.log
$PY -m inference.check_predictions --predictions "$OUT" --gold data/test 2>&1 | tail -1
$PY - <<'PYEOF'
import json, zipfile
def load(p): return {(r["level"], r["subject"], r["q_no"]): r["answer"] for r in map(json.loads, zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines())}
a, b = load("results/submissions/sub2_lora_gmmlu_vote4.zip"), load("results/submissions/sub4_A_vote4_think_hard.zip")
for lvl in ("easy", "medium", "hard"):
    ks = [k for k in a if k[0] == lvl]
    print(f"  {lvl}: answers changed vs submission 2 (A, fast): {sum(a[k] != b[k] for k in ks)}/{len(ks)}")
PYEOF

echo "=== 4. distillation LoRA ($(ts))"
if [ -z "$(ls -d models/sft_kd_t31b/step-* 2>/dev/null)" ]; then
  $PY -m train.train --config configs/train/sft_lora_kd.yaml 2>&1 | grep -vE "[Ww]arn|W10[0-9]{2}|Writing model" > results/logs/train_sft_kd_t31b.log
  grep -E "^sft:|^done|Error" results/logs/train_sft_kd_t31b.log | tail -3
fi
KD=$(ls -d models/sft_kd_t31b/step-* | tail -1)
[ -f results/exp_sft_kd_t31b_logit/metrics.json ] || $PY -m eval.run_eval --model "$KD" --data sinhalammlu_dev gmmlu_si_val \
    --method logit --n-perms 4 --run-name exp_sft_kd_t31b_logit > results/logs/exp_sft_kd_t31b_logit.log 2>&1
grep -E "acc=" results/logs/exp_sft_kd_t31b_logit.log | sed 's/.*\(exp_\)/\1/'
echo "-- paired vs model A (dev):"
$PY -m eval.compare --base results/exp_sft_gmmlu_lora_logit --runs results/exp_sft_kd_t31b_logit
echo "=== done ($(ts))"
