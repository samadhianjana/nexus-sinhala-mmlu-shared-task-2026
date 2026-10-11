"""Offline submission pipeline: question file(s) in, Codabench predictions.zip out. No human in the loop.

  python -m inference.predict --input <test dir | .json | .jsonl> --output predictions.zip

Runs with HF_HUB_OFFLINE=1 (no network); every setting comes from configs/submission.yaml.
Accepts the SinhalaMMLU JSON layout (list of {q_no, subject, category, question, choices, metadata.difficulty})
and JSONL; q_no / subject / category / level are echoed exactly as given. Writes <output>.run.json next to the
zip with the model commit, loaded parameter count, input hash and library versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEVELS = ("easy", "medium", "hard")


def read_records(path: Path) -> list[tuple[Path, dict]]:
    files = sorted(p for p in path.rglob("*") if p.suffix in (".json", ".jsonl")) if path.is_dir() else [path]
    out = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        rows = [json.loads(l) for l in text.splitlines() if l.strip()] if f.suffix == ".jsonl" else json.loads(text)
        out += [(f, r) for r in (rows if isinstance(rows, list) else [rows])]
    if not out:
        raise SystemExit(f"no questions found under {path}")
    return out


def to_item(f: Path, r: dict) -> dict:
    options = r.get("choices", r.get("options"))
    level = r.get("level") or (r.get("metadata") or {}).get("difficulty") \
        or next((p for p in f.parts[::-1] if p.lower() in LEVELS), None)
    missing = [k for k, v in {"q_no": r.get("q_no"), "subject": r.get("subject"), "category": r.get("category"),
                              "question": r.get("question"), "choices": options, "level": level}.items() if v is None]
    if missing:
        raise SystemExit(f"{f}: record {r.get('q_no')} lacks {missing}; keys present: {sorted(r)}")
    return {"uid": f"{level}/{r['subject']}/{r['q_no']}", "dataset": "test", "q_no": r["q_no"],
            "subject": r["subject"].strip(), "subject_raw": r["subject"], "category": r["category"], "level": level,
            "question": r["question"].strip(), "options": [str(o).strip() for o in options]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "submission.yaml")
    ap.add_argument("--allow-online", action="store_true", help="debugging only; the submission runs offline")
    args = ap.parse_args()
    if not args.allow_online:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import yaml

    from eval.common import cap_gpu_memory, env_fingerprint, set_seed
    from eval.metrics import aggregate_items
    from eval.prompts import PromptBuilder
    from eval.run_eval import score_items
    from eval.scorers import Scorer, load_model, loaded_param_stats

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["seed"])
    cap_gpu_memory(cfg.get("gpu_mem_gb"))

    raw = read_records(args.input)
    items = {}
    for f, r in raw:
        it = to_item(f, r)
        if it["uid"] in items:
            raise SystemExit(f"duplicate question id {it['uid']}")
        items[it["uid"]] = it
    print(f"{len(items)} questions from {args.input}")

    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    model_path = str(root / cfg["model"]) if (root / cfg["model"]).exists() else cfg["model"]
    model, tok = load_model(model_path, cfg.get("revision"), cfg.get("dtype", "bfloat16"))
    pb = PromptBuilder(tok, lang=cfg["lang"], chat=cfg["chat"])
    scorer = Scorer(model, tok, pb.add_special_tokens, cfg.get("token_budget", 32768), cfg.get("max_batch", 64))
    records, _ = score_items(items, pb, scorer, cfg["method"], cfg["n_perms"], cfg["perm_mode"], cfg["seed"])
    agg = aggregate_items(items, records)

    rows = []
    for uid, it in items.items():
        n = len(it["options"])
        ans = agg[uid]["vote"]
        if ans is None or not 1 <= ans <= n:
            raise SystemExit(f"no valid prediction for {uid}")
        rows.append({"q_no": it["q_no"], "subject": it["subject_raw"], "category": it["category"],
                     "level": it["level"], "answer": str(ans)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    digest = hashlib.sha256()
    for f in sorted({str(f) for f, _ in raw}):
        digest.update(Path(f).read_bytes())
    run = {"config": cfg, "n_questions": len(rows), "seconds": round(time.time() - t0, 1),
           "input_sha256": digest.hexdigest(), "params": loaded_param_stats(model), "env": env_fingerprint()}
    Path(str(args.output) + ".run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"wrote {len(rows)} predictions -> {args.output} "
          f"({run['params']['loaded_total'] / 1e9:.3f}B parameters loaded, {run['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
