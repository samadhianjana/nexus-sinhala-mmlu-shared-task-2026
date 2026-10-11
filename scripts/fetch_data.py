"""Download evaluation data into data/raw/ (git-ignored).

  SinhalaMMLU (public Easy tier)  - dev set, CC BY-NC-ND 4.0, gated: accept terms on HF first
  Global-MMLU `si`                - harder internal validation (translated MMLU), Apache-2.0
  Belebele `sin_Sinh`             - reading-comprehension sanity check, EVALUATION ONLY

Usage: python scripts/fetch_data.py [--only sinhalammlu,global_mmlu,belebele]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, load_env  # noqa: E402

RAW = ROOT / "data" / "raw"

SOURCES = {
    "sinhalammlu": dict(repo_id="naist-nlp/SinhalaMMLU", allow_patterns=["README.md", "TEST/**/*.json"]),
    "global_mmlu": dict(repo_id="CohereLabs/Global-MMLU", allow_patterns=["si/*.parquet"]),
    "belebele": dict(repo_id="facebook/belebele", allow_patterns=["data/sin_Sinh.jsonl"]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SOURCES))
    args = ap.parse_args()
    load_env()
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    for name in args.only.split(","):
        src = SOURCES[name]
        sha = api.dataset_info(src["repo_id"]).sha
        path = snapshot_download(
            repo_id=src["repo_id"], repo_type="dataset", revision=sha,
            allow_patterns=src["allow_patterns"], local_dir=RAW / name,
        )
        (RAW / name / "REVISION").write_text(f"{src['repo_id']}@{sha}\n")
        print(f"{name}: {src['repo_id']}@{sha[:10]} -> {path}")


if __name__ == "__main__":
    main()
