#!/usr/bin/env bash
# Download the teacher candidates (network/disk only, no GPU). Marks each finished model in results/logs/.
set -uo pipefail
cd "$(dirname "$0")/../.."
for m in google/gemma-4-31B-it google/gemma-4-26B-A4B-it Qwen/Qwen3.5-27B; do
  n=$(basename "$m"); [ -f "results/logs/.downloaded_$n" ] && { echo "have $m"; continue; }
  echo "download $m ($(date '+%H:%M'))"
  .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from eval.common import load_env; load_env()
from huggingface_hub import HfApi, snapshot_download
m = sys.argv[1]; sha = HfApi().model_info(m).sha
snapshot_download(m, revision=sha, allow_patterns=['*.json', '*.safetensors', '*.jinja', 'tokenizer*', '*.model', '*.txt'])
print('ok', m, sha)" "$m" 2>&1 | grep -E "^ok|Error" && touch "results/logs/.downloaded_$n"
done
echo "downloads done ($(date '+%H:%M'))"
