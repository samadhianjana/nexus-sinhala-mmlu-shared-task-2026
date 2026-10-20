#!/usr/bin/env bash
# Reproduce a submission end to end, offline, with one command.
#
#   ./run_submission.sh [config] [test_dir] [output_zip]
#
# Defaults: the current final configuration, the shared-task test set in data/test (download it with the
# organisers' instructions), output in results/submissions/final.zip. Requires the environment from README.md and the
# model weights referenced by the config under models/. No network access is used (HF_HUB_OFFLINE=1).
set -euo pipefail
cd "$(dirname "$0")"
CONFIG=${1:-configs/submission_sub6_think_medhard.yaml}   # final configuration (updated at the config freeze)
INPUT=${2:-data/test}
OUTPUT=${3:-results/submissions/final.zip}

echo "config: $CONFIG"
grep -E "^(model|adapter|n_perms|thinking_levels|think_max_new_tokens):" "$CONFIG" | sed 's/^/  /'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m inference.predict --config "$CONFIG" --input "$INPUT" --output "$OUTPUT"
.venv/bin/python -m inference.check_predictions --predictions "$OUTPUT" --gold "$INPUT" | tail -1
echo "submission file: $OUTPUT (upload to Codabench)"
