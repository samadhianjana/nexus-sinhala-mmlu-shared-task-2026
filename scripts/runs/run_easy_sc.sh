#!/usr/bin/env bash
# 20 Oct:
#   1. Easy decision: base-model thinking on the remaining 1,450 dev questions; combined with the earlier 400 into the
#      full dev set and compared with model A fast (original order and 4-order vote)
#   2. Self-consistency test: 3 sampled thinking paths (T=0.8, top-p 0.95, top-k 64) on the academic (551) and
#      non-academic (300) Global-MMLU proxies; vote with the existing greedy runs, compared with greedy alone
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
B=models/gemma-4-E4B-it-text
ts() { date '+%d %b %H:%M'; }
run() {  # name data extra-args...
  local n=$1 d=$2; shift 2
  [ -f "results/$n/metrics.json" ] || $PY -m eval.run_eval --model $B --data $d --method generate --thinking \
      --max-new-tokens 3072 --run-name $n "$@" > results/logs/$n.log 2>&1
  grep -E "acc=" results/logs/$n.log | sed 's/.*\(think_\|sc3_\)/\1/' || tail -3 results/logs/$n.log
}
echo "=== 1. easy: base-model thinking on the rest of dev ($(ts))"
run think_base_smmlu_dev_rest1450 smmlu_dev_rest1450
$PY -m eval.combine_cv --runs results/think_base_smmlu_dev_sample400 results/think_base_smmlu_dev_rest1450 --name think_base_dev_all
echo "-- full dev, base thinking vs model A fast (orig) and A 4-order vote (vote line):"
$PY -m eval.compare --base results/exp_sft_gmmlu_lora_logit --runs results/think_base_dev_all

echo "=== 2. self-consistency on the proxies ($(ts))"
run sc3_base_academic gmmlu_si_val_academic --temperature 0.8 --perm-mode identity --n-perms 3
run sc3_base_nonacademic gmmlu_si_val_nonacademic300 --temperature 0.8 --perm-mode identity --n-perms 3
$PY -m eval.self_consistency --greedy results/think_base_academic --samples results/sc3_base_academic
$PY -m eval.self_consistency --greedy results/think_base_gmmlu_si_val_nonacademic300 --samples results/sc3_base_nonacademic
echo "=== done ($(ts))"
