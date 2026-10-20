#!/usr/bin/env bash
# 19-20 Oct queue:
#   1. submission 5: base Gemma 4 E4B + adapter A (extracted by scripts/extract_lora.py; verified 97.8% same answers
#      as merged A on 400 dev questions); hard -> thinking without adapter, easy/medium -> fast 4-order vote with adapter
#   2. base-model thinking on easy-style (400 dev) and non-academic (300 Global-MMLU humanities/social) questions,
#      compared with model A fast on the same questions -> decides whether medium/easy also get thinking
#   3. distillation LoRA rerun (interrupted by the 19 Oct crash) + dev/val scores vs model A
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ts() { date '+%d %b %H:%M'; }
mkdir -p results/logs results/submissions
[ -f models/adapter_A/adapter_model.safetensors ] || $PY scripts/extract_lora.py --base models/gemma-4-E4B-it-text \
    --merged models/sft_gmmlu_lora/step-357 --rank 16 --out models/adapter_A

echo "=== 1. submission 5 predictions ($(ts))"
OUT=results/submissions/sub5_base_adapterA_vote4_think_hard.zip
[ -f "$OUT" ] || HF_HUB_OFFLINE=1 $PY -m inference.predict --config configs/submission_sub5_route.yaml --input data/test \
    --output "$OUT" 2>&1 | grep -E "questions|thinking|wrote|Error" > results/logs/predict_sub5.log
cat results/logs/predict_sub5.log
$PY -m inference.check_predictions --predictions "$OUT" --gold data/test 2>&1 | tail -1
$PY - <<'PYEOF'
import json, zipfile
def load(p): return {(r["level"], r["subject"], r["q_no"]): r["answer"] for r in map(json.loads, zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines())}
a, b = load("results/submissions/sub4_A_vote4_think_hard.zip"), load("results/submissions/sub5_base_adapterA_vote4_think_hard.zip")
for lvl in ("easy", "medium", "hard"):
    ks = [k for k in a if k[0] == lvl]
    print(f"  {lvl}: answers changed vs submission 4: {sum(a[k] != b[k] for k in ks)}/{len(ks)}")
PYEOF

echo "=== 2. base-model thinking on easy-style and non-academic questions ($(ts))"
for d in smmlu_dev_sample400 gmmlu_si_val_nonacademic300; do
  n=think_base_${d}
  [ -f "results/$n/metrics.json" ] || $PY -m eval.run_eval --model models/gemma-4-E4B-it-text --data $d --method generate \
      --thinking --max-new-tokens 3072 --run-name $n > results/logs/$n.log 2>&1
  grep -E "acc=" results/logs/$n.log | sed 's/.*\(think_\)/\1/'
done
echo "-- paired vs model A fast (vote over 4 orders is the 'vote' line):"
$PY -m eval.compare --base results/exp_sft_gmmlu_lora_logit --runs results/think_base_smmlu_dev_sample400 --dataset sinhalammlu
$PY -m eval.compare --base results/exp_sft_gmmlu_lora_logit --runs results/think_base_gmmlu_si_val_nonacademic300 --dataset gmmlu_si

echo "=== 3. distillation LoRA rerun ($(ts))"
if [ -z "$(ls -d models/sft_kd_t31b/step-* 2>/dev/null)" ]; then
  rm -f models/sft_kd_t31b/log.jsonl
  $PY -m train.train --config configs/train/sft_lora_kd.yaml 2>&1 | grep -vE "[Ww]arn|W10[0-9]{2}|Writing model" > results/logs/train_sft_kd_t31b.log
  grep -E "^sft:|^done|Error" results/logs/train_sft_kd_t31b.log | tail -3
fi
KD=$(ls -d models/sft_kd_t31b/step-* | tail -1)
[ -f results/exp_sft_kd_t31b_logit/metrics.json ] || $PY -m eval.run_eval --model "$KD" --data sinhalammlu_dev gmmlu_si_val \
    --method logit --n-perms 4 --run-name exp_sft_kd_t31b_logit > results/logs/exp_sft_kd_t31b_logit.log 2>&1
grep -E "acc=" results/logs/exp_sft_kd_t31b_logit.log | sed 's/.*\(exp_\)/\1/'
echo "-- distillation vs model A (dev):"
$PY -m eval.compare --base results/exp_sft_gmmlu_lora_logit --runs results/exp_sft_kd_t31b_logit
echo "=== done ($(ts))"
