"""Exact parameter accounting for candidate models -> MODEL_MANIFEST.md + reports/param_counts.json.

Two independent counts per model:
  stored        sum of every tensor in the repo's safetensors headers (read via HTTP range requests,
                no weight download), split into text / vision / audio / MTP heads by tensor name.
                Tied input/output embeddings are counted once.
  instantiated  parameters of the module transformers builds (meta device, no weights) with
                AutoModelForCausalLM, falling back to AutoModelForImageTextToText.

LoRA/adapter repos are counted as base model + adapter tensors (shared-task rule).
Usage: python scripts/count_params.py [--models id1 id2 ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, load_env  # noqa: E402

LIMIT = 8_000_000_000
CUTOFF = "2026-10-01"
MANIFEST = ROOT / "MODEL_MANIFEST.md"
BEGIN, END = "<!-- AUTO:PARAMS:BEGIN -->", "<!-- AUTO:PARAMS:END -->"

PATTERNS = [
    ("vision", re.compile(r"vision|visual|image|siglip|patch_embed|multi_modal_projector|mm_projector|embed_vision")),
    ("audio", re.compile(r"audio|embed_audio|speech")),
    ("mtp", re.compile(r"(^|\.)mtp[._]")),
]


def tensor_kind(name: str) -> str:
    for kind, pat in PATTERNS:
        if pat.search(name):
            return kind
    return "text"


def _tied(cfg: dict) -> bool:
    for c in (cfg, cfg.get("text_config") or {}):
        if "tie_word_embeddings" in c:
            return bool(c["tie_word_embeddings"])
    return False


def stored_counts(api, repo: str, sha: str) -> dict:
    from huggingface_hub import hf_hub_download

    meta = api.get_safetensors_metadata(repo, revision=sha)
    counts = {"text": 0, "vision": 0, "audio": 0, "mtp": 0}
    names = {}
    for fm in meta.files_metadata.values():
        for name, t in fm.tensors.items():
            n = 1
            for d in t.shape:
                n *= d
            names[name] = n
            counts[tensor_kind(name)] += n
    cfg = json.load(open(hf_hub_download(repo, "config.json", revision=sha)))
    tied = _tied(cfg)
    lm_head = [k for k in names if k.endswith("lm_head.weight")]
    deduped = 0
    if tied and lm_head:  # stored twice but shared at runtime
        deduped = sum(names[k] for k in lm_head)
        counts["text"] -= deduped
    emb = sum(v for k, v in names.items() if re.search(r"embed_tokens|wte|lm_head|embeddings\.word", k)
              and tensor_kind(k) == "text") - deduped
    return {**counts, "embeddings": emb, "tied_embeddings": tied, "total": sum(counts.values()),
            "model_type": cfg.get("model_type"), "config": cfg}


def instantiated_count(repo: str, sha: str) -> tuple[int | None, str]:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        cfg = AutoConfig.from_pretrained(repo, revision=sha)
    except Exception as e:  # noqa: BLE001
        return None, f"config: {type(e).__name__}"
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            import transformers

            cls = getattr(transformers, cls_name)
            with init_empty_weights():
                m = cls.from_config(cfg)
            m.tie_weights()  # meta init leaves tied embeddings as separate tensors
            return sum(p.numel() for p in m.parameters()), type(m).__name__
        except Exception as e:  # noqa: BLE001
            err = f"{cls_name}: {type(e).__name__}: {str(e)[:80]}"
    return None, err


def count_repo(api, repo: str) -> dict:
    info = api.model_info(repo)
    sha = info.sha
    row = {"id": repo, "sha": sha, "created": str(info.created_at)[:10],
           "last_modified": str(info.last_modified)[:10],
           "license": (info.card_data or {}).get("license") if info.card_data else None,
           "gated": info.gated}
    files = {s.rfilename for s in info.siblings}
    if "adapter_config.json" in files:
        from huggingface_hub import hf_hub_download

        acfg = json.load(open(hf_hub_download(repo, "adapter_config.json", revision=sha)))
        base = acfg["base_model_name_or_path"]
        fm = api.parse_safetensors_file_metadata(repo, "adapter_model.safetensors", revision=sha)
        adapter = sum(_numel(t.shape) for t in fm.tensors.values())
        b = count_repo(api, base)
        row.update(kind="adapter", base=base, adapter_params=adapter,
                   modules_to_save=acfg.get("modules_to_save"),
                   text=b.get("text"), vision=b.get("vision"), audio=b.get("audio"), mtp=b.get("mtp"),
                   embeddings=b.get("embeddings"), tied_embeddings=b.get("tied_embeddings"),
                   total=(b.get("total") or 0) + adapter, text_loaded=(b.get("text_loaded") or 0) + adapter,
                   instantiated=None, inst_class=f"base {b.get('total')} + adapter")
        return row
    s = stored_counts(api, repo, sha)
    s.pop("config")
    inst, cls = instantiated_count(repo, sha)
    row.update(kind="model", **s, text_loaded=s["text"], instantiated=inst, inst_class=cls)
    return row


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def verdict(r: dict) -> str:
    if r.get("error"):
        return "ERROR"
    flags = []
    if r["created"] >= CUTOFF:
        flags.append("RELEASED TOO LATE")
    if r["total"] <= LIMIT:
        flags.append("OK")
    elif r["text_loaded"] <= LIMIT:
        flags.append("OK only text-only (strip vision/audio/MTP)")
    else:
        flags.append("OVER 8B")
    return "; ".join(flags)


def b(n) -> str:
    return "-" if n in (None, 0) else f"{n / 1e9:.3f}B"


def write_manifest(rows: list[dict]) -> None:
    lines = [BEGIN, "",
             "| Model | Verdict | Total (all stored) | Text model | Vision | Audio | MTP | Embeddings | Tied | "
             "Instantiated (class) | Repo created | License | Commit |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            lines.append(f"| `{r['id']}` | ERROR: {r['error']} |" + " |" * 11)
            continue
        inst = f"{b(r['instantiated'])} ({r['inst_class']})" if r.get("instantiated") else r.get("inst_class", "-")
        name = f"`{r['id']}`" + (f"<br>adapter on `{r['base']}` (+{b(r['adapter_params'])})" if r["kind"] == "adapter" else "")
        lines.append(f"| {name} | {verdict(r)} | {b(r['total'])} | {b(r['text_loaded'])} | {b(r['vision'])} | "
                     f"{b(r['audio'])} | {b(r['mtp'])} | {b(r['embeddings'])} | {'yes' if r['tied_embeddings'] else 'no'} | "
                     f"{inst} | {r['created']} | {r['license'] or '?'} | `{r['sha'][:10]}` |")
    lines += ["", END]
    text = MANIFEST.read_text()
    start, stop = text.index(BEGIN), text.index(END) + len(END)
    MANIFEST.write_text(text[:start] + "\n".join(lines) + text[stop:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    load_env()
    import yaml
    from huggingface_hub import HfApi

    repos = args.models or [c["id"] for c in yaml.safe_load(open(ROOT / "configs" / "models.yaml"))["candidates"]]
    api = HfApi()
    rows = []
    for repo in repos:
        try:
            r = count_repo(api, repo)
        except Exception as e:  # noqa: BLE001
            r = {"id": repo, "error": f"{type(e).__name__}: {str(e)[:100]}"}
        rows.append(r)
        print(f"{repo:42s} {verdict(r):45s} total={b(r.get('total'))} text={b(r.get('text_loaded'))} "
              f"inst={b(r.get('instantiated'))}")
    out = ROOT / "reports" / "param_counts.json"
    out.parent.mkdir(exist_ok=True)
    if args.models and out.exists():  # partial recount: update those rows, keep the rest
        fresh = {r["id"]: r for r in rows}
        old = json.loads(out.read_text())
        rows = [fresh.pop(r["id"], r) for r in old] + list(fresh.values())
    out.write_text(json.dumps(rows, indent=2, default=str))
    write_manifest(rows)


if __name__ == "__main__":
    main()
