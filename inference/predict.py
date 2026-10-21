"""Offline submission pipeline: question file(s) in, Codabench predictions.zip out. No human in the loop.

  python -m inference.predict --input <test dir | .json | .jsonl> --output predictions.zip

Runs with HF_HUB_OFFLINE=1 (no network); every setting comes from configs/submission.yaml.
Optional self-consistency (config `sc_levels`, `sc_samples`, `sc_gate`): greedy + K sampled reasoning paths, majority
vote; with `sc_gate: disagree` only for questions where the fast answer disagrees with the greedy thinking answer.
Optional LoRA adapter (config `adapter`) on top of `model`. Optional per-question thinking (config `thinking_levels`,
e.g. [hard]): questions whose own `level` field is listed are answered with the model's thinking mode (with
`thinking_without_adapter: true`, the adapter is switched off for them) (greedy, original option order); if no answer can be read from the output,
the fast answer is used. All other questions use the fast logit answer (voted over `n_perms` option orders).

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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEVELS = ("easy", "medium", "hard")


def read_records(path: Path) -> list[tuple[Path, dict]]:
    files = sorted(p for p in path.rglob("*") if p.suffix in (".json", ".jsonl")
                   and not any(part.startswith(".") for part in p.relative_to(path).parts)) if path.is_dir() else [path]
    # (hidden dirs skipped: huggingface_hub keeps download metadata as JSON under .cache/)
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
    ap.add_argument("--limit", type=int, help="first N questions only (smoke tests; never for a submission)")
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
    if args.limit:
        items = dict(list(items.items())[: args.limit])
    print(f"{len(items)} questions from {args.input}")

    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    model_path = str(root / cfg["model"]) if (root / cfg["model"]).exists() else cfg["model"]
    adapter = cfg.get("adapter")
    adapter = str(root / adapter) if adapter and (root / adapter).exists() else adapter
    model, tok = load_model(model_path, cfg.get("revision"), cfg.get("dtype", "bfloat16"), adapter)
    pb = PromptBuilder(tok, lang=cfg["lang"], chat=cfg["chat"])
    scorer = Scorer(model, tok, pb.add_special_tokens, cfg.get("token_budget", 32768), cfg.get("max_batch", 64))
    records, _ = score_items(items, pb, scorer, cfg["method"], cfg["n_perms"], cfg["perm_mode"], cfg["seed"])
    agg = aggregate_items(items, records)

    # Thinking mode for selected levels (decided per question from its own level field)
    think_levels = set(cfg.get("thinking_levels") or [])
    think_uids = [u for u, it in items.items() if it["level"] in think_levels]
    think_pred, fallbacks, sc_changed, sc_count = {}, 0, 0, 0
    if think_uids:
        tpb = PromptBuilder(tok, lang=cfg["lang"], chat=cfg["chat"], chat_kwargs={"enable_thinking": True})
        tscorer = Scorer(model, tok, tpb.add_special_tokens, cfg.get("token_budget", 32768), cfg.get("max_batch", 64),
                         max_new_tokens=cfg.get("think_max_new_tokens", 2048))

        def think(uids: list[str]) -> list[dict]:
            prompts = [tpb.mcq(items[u], items[u]["options"]) for u in uids]
            n_opts = [len(items[u]["options"]) for u in uids]
            if cfg.get("thinking_without_adapter") and hasattr(model, "disable_adapter"):
                with model.disable_adapter():  # base model reasons; the adapter (trained to answer at once) does not
                    return tscorer.generate(prompts, n_opts)
            return tscorer.generate(prompts, n_opts)

        for u, r in zip(think_uids, think(think_uids)):
            if r["pred"]:
                think_pred[u] = r["pred"]
            else:
                fallbacks += 1
        print(f"thinking: {len(think_uids)} questions ({sorted(think_levels)}), {fallbacks} fell back to the fast answer")

        # Self-consistency for selected levels: greedy answer + K sampled reasoning paths, majority vote,
        # ties go to the greedy answer. Fixed seed: reproducible for the same input, code and hardware.
        sc_levels, k = set(cfg.get("sc_levels") or []), int(cfg.get("sc_samples", 0))
        sc_uids = [u for u in think_uids if items[u]["level"] in sc_levels]
        if cfg.get("sc_gate") == "disagree":  # only where the free one-step answer disagrees with greedy thinking
            sc_uids = [u for u in sc_uids if think_pred.get(u) != agg[u]["pred"]]
        sc_count = len(sc_uids) if k > 0 else 0
        if sc_uids and k > 0:
            import torch

            torch.manual_seed(cfg["seed"])
            tscorer.sampling = {"do_sample": True, "temperature": cfg.get("sc_temperature", 0.8),
                                "top_p": cfg.get("sc_top_p", 0.95), "top_k": cfg.get("sc_top_k", 64)}
            ballots = {u: [think_pred.get(u)] for u in sc_uids}
            for _ in range(k):
                for u, r in zip(sc_uids, think(sc_uids)):
                    ballots[u].append(r["pred"])
            tscorer.sampling = {}
            for u, b in ballots.items():
                counts = Counter(x for x in b if x)
                if not counts:
                    continue
                top = max(counts.values())
                winners = [a for a, c in counts.items() if c == top]
                choice = b[0] if b[0] in winners else min(winners)
                sc_changed += choice != think_pred.get(u)
                think_pred[u] = choice
            print(f"self-consistency: {len(sc_uids)} questions ({sorted(sc_levels)}, gate={cfg.get('sc_gate')}), greedy + {k} samples, "
                  f"{sc_changed} answers changed by the vote")

    rows = []
    for uid, it in items.items():
        n = len(it["options"])
        ans = think_pred.get(uid, agg[uid]["vote"])
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
           "thinking": {"levels": sorted(think_levels), "questions": len(think_uids), "fallbacks": fallbacks},
           "self_consistency": {"levels": sorted(set(cfg.get("sc_levels") or [])), "samples": int(cfg.get("sc_samples", 0)),
                                "gate": cfg.get("sc_gate"), "questions": sc_count, "answers_changed": sc_changed},
           "input_sha256": digest.hexdigest(), "params": loaded_param_stats(model), "env": env_fingerprint()}
    Path(str(args.output) + ".run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"wrote {len(rows)} predictions -> {args.output} "
          f"({run['params']['loaded_total'] / 1e9:.3f}B parameters loaded, {run['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
