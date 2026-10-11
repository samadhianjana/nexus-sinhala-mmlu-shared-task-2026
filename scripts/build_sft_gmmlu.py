"""Fine-tuning MCQs from Global-MMLU `si` (Apache-2.0) -> data/sft/gmmlu_si_train.jsonl.

  pool   Global-MMLU si test split minus the 2,000 validation ids, plus its dev split
  drop   items with char-5-gram TF-IDF cosine >= --val-threshold to any Global-MMLU validation item
         (MMLU repeats questions across subjects: 33 validation items had a >= 0.9 copy in the pool),
         >= --dev-threshold to any SinhalaMMLU dev item, and near-copies within the pool (>= 0.95, first kept)
  shuffle options to a deterministic, balanced gold position, except items whose options refer to other options
         ("all of the above", "both", option letters/numbers), which keep their order

Output records use the processed MCQ schema (answer = 1-based gold index). Aggregate counts go to
reports/data_stats_sft_gmmlu.json; nothing from the dev/validation sets is written to the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.common import ROOT, read_jsonl, write_jsonl  # noqa: E402
from overlap import item_text, ngram_matrix, tfidf  # noqa: E402
from prepare_data import _gmmlu_rows  # noqa: E402

# Options that refer to other options must keep their order.
REFERS_TO_OTHERS = re.compile(
    r"ඉහත|සියල්ල|කිසිවක්\s*නැත|දෙකම|සියල්ලම|all of the above|none of the above|both|neither|"
    r"\b[A-D]\s*(and|හා|සහ|හෝ|or)\s*[A-D]\b|\b[1-4]\s*(and|හා|සහ|හෝ|or)\s*[1-4]\b", re.I)


def max_sim(a, b, chunk: int = 2000, earlier_only: bool = False) -> np.ndarray:
    """For each row of a: max cosine to rows of b. earlier_only (a is b): only rows before it, so the first
    occurrence of a duplicate group is kept and later copies are flagged."""
    best = np.zeros(a.shape[0])
    bt = b.T.tocsc()
    for i in range(0, a.shape[0], chunk):
        s = (a[i:i + chunk] @ bt).toarray()
        if earlier_only:
            rows = np.arange(i, i + s.shape[0])
            s[np.arange(s.shape[1])[None, :] >= rows[:, None]] = 0
        best[i:i + chunk] = s.max(1)
    return best


def balanced_permutation(r: dict, n_items_seen: Counter) -> dict:
    """Move the gold answer to the least-used position so far (ties broken by a stable hash)."""
    n = len(r["options"])
    if any(REFERS_TO_OTHERS.search(o) for o in r["options"]):
        return r | {"meta": r["meta"] | {"shuffled": False}}
    h = int(hashlib.sha1(r["uid"].encode()).hexdigest(), 16)
    target = min(range(n), key=lambda p: (n_items_seen[p], (p + h) % n))
    gold = r["answer"] - 1
    others = [j for j in range(n) if j != gold]
    rot = h % len(others)
    others = others[rot:] + others[:rot]
    order = others[:target] + [gold] + others[target:]
    n_items_seen[target] += 1
    return r | {"options": [r["options"][j] for j in order], "answer": target + 1,
                "meta": r["meta"] | {"shuffled": True}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-threshold", type=float, default=0.6)
    ap.add_argument("--dev-threshold", type=float, default=0.5)
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    ap.add_argument("--out", default=str(ROOT / "data" / "sft" / "gmmlu_si_train.jsonl"))
    args = ap.parse_args()
    import pandas as pd

    raw = ROOT / "data" / "raw" / "global_mmlu" / "si"
    val_ids = set((ROOT / "configs" / "splits" / "gmmlu_si_val_ids.txt").read_text().split())
    pool = [r for r in _gmmlu_rows(pd.read_parquet(raw / "test-00000-of-00001.parquet"), "test") if r["uid"] not in val_ids]
    pool += _gmmlu_rows(pd.read_parquet(raw / "dev-00000-of-00001.parquet"), "dev")
    val = read_jsonl(ROOT / "data" / "processed" / "gmmlu_si_val.jsonl")
    dev = read_jsonl(ROOT / "data" / "processed" / "sinhalammlu_dev.jsonl")

    tp = [item_text(r, True) for r in pool]
    mp, mv = tfidf(ngram_matrix(tp, 5), ngram_matrix([item_text(r, True) for r in val], 5))
    sim_val = max_sim(mp, mv)
    mp2, md = tfidf(ngram_matrix(tp, 5), ngram_matrix([item_text(r, True) for r in dev], 5))
    sim_dev = max_sim(mp2, md)
    sim_self = max_sim(mp, mp, earlier_only=True)

    keep, reasons = [], Counter()
    for r, sv, sd, ss in zip(pool, sim_val, sim_dev, sim_self):
        if sv >= args.val_threshold:
            reasons["near_validation"] += 1
        elif sd >= args.dev_threshold:
            reasons["near_sinhalammlu_dev"] += 1
        elif ss >= args.dup_threshold:
            reasons["duplicate_within_pool"] += 1
        else:
            keep.append(r)

    seen = Counter()
    out = [balanced_permutation(r, seen) for r in sorted(keep, key=lambda r: r["uid"])]
    out = [r | {"dataset": "gmmlu_si_train"} for r in out]
    write_jsonl(args.out, out)

    stats = {
        "pool": len(pool), "kept": len(out), "dropped": dict(reasons),
        "thresholds": {"validation": args.val_threshold, "sinhalammlu_dev": args.dev_threshold,
                       "within_pool": args.dup_threshold},
        "max_similarity_kept": {"to_validation": float(max((sv for sv in sim_val if sv < args.val_threshold), default=0)),
                                "to_sinhalammlu_dev": float(sim_dev.max())},
        "answer_positions": dict(sorted(Counter(r["answer"] for r in out).items())),
        "shuffled": sum(r["meta"]["shuffled"] for r in out),
        "by_category": dict(Counter(r["category"] for r in out).most_common()),
    }
    (ROOT / "reports" / "data_stats_sft_gmmlu.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
