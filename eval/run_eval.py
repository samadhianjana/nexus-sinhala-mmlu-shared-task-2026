"""Evaluate one model on one or more processed MCQ files.

Example:
  python -m eval.run_eval --model Qwen/Qwen3-4B-Instruct-2507 --data sinhalammlu_dev gmmlu_si_val \
      --method logit --n-perms 4 --run-name qwen3-4b-inst_logit

Outputs results/<run-name>/{config.json, predictions.jsonl, metrics.json, report.md}
and, with --write-submission, a Codabench-format predictions.zip per dataset.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import zipfile
from pathlib import Path

import yaml

from .common import ROOT, cap_gpu_memory, env_fingerprint, load_env, read_jsonl, set_seed, write_jsonl
from .metrics import compute_metrics, markdown_report
from .prompts import PromptBuilder, user_text

PROCESSED = ROOT / "data" / "processed"
BASE_NAME = re.compile(r"(-base|-pt)$|^google/gemma-4-e[24]b$", re.I)


def make_perm(n: int, k: int, mode: str, seed: int, uid: str) -> list[int]:
    """perm[j] = original 0-based index shown at display position j. k=0 is the original order."""
    if k == 0:
        return list(range(n))
    if mode == "cyclic" and k < n:
        return [(j + k) % n for j in range(n)]
    rng = random.Random(f"{seed}|{uid}|{k}")
    p = list(range(n))
    rng.shuffle(p)
    return p


def resolve_data(name: str) -> Path:
    p = Path(name)
    return p if p.suffix == ".jsonl" else PROCESSED / f"{name}.jsonl"


def score_items(items: dict[str, dict], pb: PromptBuilder, scorer, method: str, n_perms: int,
                perm_mode: str, seed: int) -> tuple[list[dict], str | None]:
    """One record per (question, option order). Shared by evaluation and the submission pipeline.
    Items without a gold answer (test data) get gold/correct = None."""
    n_perms = 1 if method == "cloze" else n_perms
    jobs = [(uid, k, make_perm(len(it["options"]), k, perm_mode, seed, uid))
            for uid, it in items.items() for k in range(n_perms)]
    first_prompt = None
    if method == "cloze":
        res = scorer.cloze([pb.cloze_context(items[u]) for u, _, _ in jobs], [items[u]["options"] for u, _, _ in jobs])
    else:
        prompts = [pb.mcq(items[u], [items[u]["options"][j] for j in perm]) for u, _, perm in jobs]
        fn = scorer.logit if method == "logit" else scorer.generate
        res = fn(prompts, [len(items[u]["options"]) for u, _, _ in jobs])
        first_prompt = prompts[0] if prompts else None

    records = []
    for (uid, k, perm), r in zip(jobs, res):
        gold = items[uid].get("answer")
        probs_orig = [0.0] * len(perm)
        for j, p in enumerate(r["probs"]):
            probs_orig[perm[j]] = p
        pred_orig = perm[r["pred"] - 1] + 1 if r["pred"] else None
        rec = {"uid": uid, "perm_k": k, "perm": perm, "method": method,
               "pred_display": r["pred"], "gold_display": perm.index(gold - 1) + 1 if gold else None,
               "pred_orig": pred_orig, "gold": gold, "correct": pred_orig == gold if gold else None,
               "probs_orig": probs_orig}
        for extra in ("label_mass", "text", "new_tokens"):
            if extra in r:
                rec[extra] = r[extra]
        records.append(rec)
    return records, first_prompt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML with any of the options below (CLI wins)")
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--data", nargs="+", default=["sinhalammlu_dev"])
    ap.add_argument("--method", choices=["logit", "generate", "cloze"], default="logit")
    ap.add_argument("--lang", choices=["en", "si"], default="en", help="instruction language")
    ap.add_argument("--chat", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--answer-prefix", default="")
    ap.add_argument("--thinking", action="store_true", help="enable the chat template's thinking mode (use with --method generate)")
    ap.add_argument("--n-perms", type=int, default=1, help="option orders per question (1 = original only)")
    ap.add_argument("--perm-mode", choices=["cyclic", "random"], default="cyclic")
    ap.add_argument("--fewshot-file", help="processed JSONL; the same examples precede every question")
    ap.add_argument("--n-shot", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--token-budget", type=int, default=32768, help="padded tokens per forward batch")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-mem-gb", type=float, help="PyTorch memory cap (default 100, env SMMLU_GPU_MEM_GB)")
    ap.add_argument("--limit", type=int, help="first N questions per dataset (smoke tests)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--run-name")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--write-submission", action="store_true")
    args, _ = ap.parse_known_args()
    if args.config:
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in yaml.safe_load(open(args.config)).items()})
    args = ap.parse_args()
    if not args.model:
        ap.error("--model is required")
    return args


def main() -> None:
    args = parse_args()
    load_env()
    set_seed(args.seed)
    cap_gpu_memory(args.gpu_mem_gb)
    from .scorers import Scorer, load_model, loaded_param_stats

    run_name = args.run_name or f"{args.model.split('/')[-1]}_{args.method}_{args.lang}_{args.n_shot}shot"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    fewshot = read_jsonl(resolve_data(args.fewshot_file))[: args.n_shot] if args.n_shot else []
    shot_uids = {e["uid"] for e in fewshot}
    items: dict[str, dict] = {}
    for d in args.data:
        rows = [r for r in read_jsonl(resolve_data(d)) if r["uid"] not in shot_uids]
        items.update({r["uid"]: r for r in (rows[: args.limit] if args.limit else rows)})

    t0 = time.time()
    model, tok = load_model(args.model, args.revision, args.dtype)
    chat = {"on": True, "off": False}.get(args.chat, tok.chat_template is not None and not BASE_NAME.search(args.model))
    pb = PromptBuilder(tok, lang=args.lang, chat=chat, fewshot=fewshot, answer_prefix=args.answer_prefix,
                       chat_kwargs={"enable_thinking": args.thinking})
    scorer = Scorer(model, tok, pb.add_special_tokens, args.token_budget, args.max_batch, args.max_new_tokens)
    load_s = time.time() - t0

    t0 = time.time()
    records, first_prompt = score_items(items, pb, scorer, args.method, args.n_perms, args.perm_mode, args.seed)
    score_s = time.time() - t0
    if first_prompt:
        (out / "example_prompt.txt").write_text(first_prompt, encoding="utf-8")

    # Tokens per character of the question block: tokenizer efficiency on Sinhala
    sample = list(items.values())[:500]
    texts = [user_text(it, it["options"], args.lang) for it in sample]
    n_tok = sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])

    metrics = {"by_dataset": metrics_by_dataset(items, records)}
    metrics["run"] = {
        "run_name": run_name, "args": vars(args), "chat": chat, "n_questions": len(items),
        "n_forward_items": len(records), "load_seconds": round(load_s, 1), "score_seconds": round(score_s, 1),
        "tokens_per_char": n_tok / sum(len(t) for t in texts),
        "params": loaded_param_stats(model), "env": env_fingerprint(),
    }
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    write_jsonl(out / "predictions.jsonl", records)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    report = "\n".join(markdown_report(f"{run_name} / {d}", m) for d, m in metrics["by_dataset"].items())
    (out / "report.md").write_text(report, encoding="utf-8")

    if args.write_submission:
        write_submissions(out, items, records)

    for d, m in metrics["by_dataset"].items():
        o = m["overall"]
        print(f"{run_name} | {d}: acc={o['acc']:.4f} vote={o['acc_vote']:.4f} "
              f"perm={o['acc_perm_mean']:.4f}±{o['acc_perm_std']:.4f} n={o['n']}")
    print(f"load {load_s:.0f}s, scoring {score_s:.0f}s -> {out}")


def metrics_by_dataset(items: dict, records: list[dict]) -> dict:
    out = {}
    for d in sorted({it["dataset"] for it in items.values()}):
        sub_items = {u: it for u, it in items.items() if it["dataset"] == d}
        out[d] = compute_metrics(sub_items, [r for r in records if r["uid"] in sub_items])
    return out


def write_submissions(out: Path, items: dict, records: list[dict]) -> None:
    """Codabench format: predictions.zip containing predictions.jsonl, answer = permutation-vote prediction."""
    from .metrics import aggregate_items

    agg = aggregate_items(items, records)
    for d in sorted({it["dataset"] for it in items.values()}):
        rows = [{"q_no": it["q_no"], "subject": it.get("subject_raw", it["subject"]), "category": it["category"],
                 "level": it["level"], "answer": str(agg[u]["vote"] or 1)}
                for u, it in items.items() if it["dataset"] == d]
        path = out / f"submission_{d}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("predictions.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
