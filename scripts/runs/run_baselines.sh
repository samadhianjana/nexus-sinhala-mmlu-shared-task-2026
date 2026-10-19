#!/usr/bin/env bash
# Zero-shot baselines: same prompt for every model, letter-logit scoring, 4 cyclic option orders.
# Usage: scripts/run_baselines.sh [model ...]   (default: every `baseline` entry in configs/models.yaml)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
METHOD=${METHOD:-logit}
NPERMS=${NPERMS:-4}
DATA=${DATA:-"sinhalammlu_dev gmmlu_si_val belebele_si"}

if [ $# -gt 0 ]; then
  MODELS=("$@")
else
  mapfile -t MODELS < <($PY -c "import yaml; [print(c['id']) for c in yaml.safe_load(open('configs/models.yaml'))['candidates'] if c.get('baseline', True)]")
fi

mkdir -p results/logs
for m in "${MODELS[@]}"; do
  name="base_$(basename "$m")_${METHOD}"
  if [ -f "results/$name/metrics.json" ]; then echo "skip $name (done)"; continue; fi
  echo "=== $m -> $name ($(date +%H:%M))"
  $PY -m eval.run_eval --model "$m" --data $DATA --method "$METHOD" --n-perms "$NPERMS" \
      --run-name "$name" > "results/logs/$name.log" 2>&1 \
    && grep -E "acc=|scoring" "results/logs/$name.log" \
    || { echo "FAILED $m (see results/logs/$name.log)"; tail -5 "results/logs/$name.log"; }
done
$PY -m eval.summarize --runs "results/base_*_${METHOD}" --out "reports/baselines_zeroshot_${METHOD}.md" > /dev/null
echo "summary -> reports/baselines_zeroshot_${METHOD}.md"
