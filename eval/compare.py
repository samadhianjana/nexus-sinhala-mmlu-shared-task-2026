"""Paired comparison of runs on the same questions (bootstrap 95% CI of the accuracy difference).

  python -m eval.compare --base results/exp_sft_gmmlu_lora_logit --runs results/exp_x results/exp_y [--dataset sinhalammlu]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import read_jsonl
from .metrics import aggregate_items


def correctness(run: Path, dataset: str) -> tuple[dict, dict]:
    recs = read_jsonl(run / "predictions.jsonl")
    items = {r["uid"]: {"options": r["perm"], "answer": r["gold"]} for r in recs}
    agg = aggregate_items(items, recs)
    keep = {u for u in agg if u.startswith({"sinhalammlu": "smmlu/", "gmmlu_si": "gmmlu_si/"}.get(dataset, ""))}
    return ({u: agg[u]["pred"] == agg[u]["gold"] for u in keep}, {u: agg[u]["vote"] == agg[u]["gold"] for u in keep})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--dataset", default="sinhalammlu", choices=["sinhalammlu", "gmmlu_si"])
    args = ap.parse_args()
    base = correctness(args.base, args.dataset)
    rng = np.random.default_rng(0)
    for run in args.runs:
        c = correctness(run, args.dataset)
        for k, label in ((0, "orig"), (1, "vote")):
            u = sorted(set(base[k]) & set(c[k]))
            if not u:
                continue
            d = np.array([c[k][x] - base[k][x] for x in u], dtype=float)
            lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [2.5, 97.5])
            print(f"{run.name:40s} {label}: {100 * np.mean([c[k][x] for x in u]):5.1f} vs {100 * np.mean([base[k][x] for x in u]):5.1f}"
                  f"  diff {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]  n={len(u)}")


if __name__ == "__main__":
    main()
