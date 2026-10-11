"""Extract the text-only language model from a multimodal checkpoint (vision/audio towers dropped).

The shared task counts every component that is loaded, so the submission loads this checkpoint instead of the
multimodal one. Weights are copied unchanged; the script verifies the copy is exact.

Usage: python scripts/make_text_only.py google/gemma-4-E4B-it [--out models/gemma-4-E4B-it-text]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, load_env  # noqa: E402

# multimodal class -> (text-only class, attribute path of the language model inside it)
TEXT_ONLY = {
    "Gemma4ForConditionalGeneration": ("Gemma4ForCausalLM", "model.language_model"),
    "Gemma3ForConditionalGeneration": ("Gemma3ForCausalLM", "model.language_model"),
    "Gemma3nForConditionalGeneration": ("Gemma3nForCausalLM", "model.language_model"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out")
    args = ap.parse_args()
    load_env()
    import torch
    import transformers
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    sha = HfApi().model_info(args.repo).sha
    out = Path(args.out or ROOT / "models" / f"{args.repo.split('/')[-1]}-text")
    cfg = AutoConfig.from_pretrained(args.repo, revision=sha)
    full = AutoModelForImageTextToText.from_pretrained(args.repo, revision=sha, dtype=torch.bfloat16, device_map="cpu")
    text_cls_name, lm_path = TEXT_ONLY[type(full).__name__]
    lm = full
    for attr in lm_path.split("."):
        lm = getattr(lm, attr)

    text = getattr(transformers, text_cls_name)(cfg.text_config).to(torch.bfloat16)
    text.model.load_state_dict(lm.state_dict(), strict=True)
    text.tie_weights()
    if not getattr(cfg.text_config, "tie_word_embeddings", True) and hasattr(full, "lm_head"):
        text.lm_head.load_state_dict(full.lm_head.state_dict())
    text.generation_config = full.generation_config

    # Exactness check: every text tensor equals its source
    src = lm.state_dict()
    for k, v in text.model.state_dict().items():
        assert torch.equal(v, src[k]), f"mismatch in {k}"
    n_params = sum(p.numel() for p in text.parameters())

    out.mkdir(parents=True, exist_ok=True)
    text.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.repo, revision=sha).save_pretrained(out)
    snap = Path(snapshot_download(args.repo, revision=sha, allow_patterns=["chat_template.jinja", "chat_template.json"]))
    for f in snap.glob("chat_template.*"):
        shutil.copy(f, out / f.name)
    (out / "SOURCE.json").write_text(json.dumps({
        "source_repo": args.repo, "source_commit": sha, "class": text_cls_name,
        "total_params": n_params, "dropped": [n for n, _ in full.model.named_children() if n != "language_model"],
    }, indent=2))
    print(f"{args.repo}@{sha[:10]} -> {out}  ({text_cls_name}, {n_params / 1e9:.3f}B parameters)")


if __name__ == "__main__":
    main()
