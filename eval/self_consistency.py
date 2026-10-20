"""Self-consistency: majority vote over a greedy thinking run and sampled thinking runs of the same questions.

  python -m eval.self_consistency --greedy results/think_base_academic --samples results/sc3_base_academic

Votes = greedy answer + every sampled answer (unparsed samples abstain); ties go to the greedy answer. Prints accuracy of
greedy alone, samples alone (each), and the vote, with a paired bootstrap CI of vote minus greedy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", required=True, type=Path)
    ap.add_argument("--samples", required=True, type=Path)
    args = ap.parse_args()
    greedy = {r["uid"]: r for r in read_jsonl(args.greedy / "predictions.jsonl") if r["perm_k"] == 0}
    samples = defaultdict(list)
    for r in read_jsonl(args.samples / "predictions.jsonl"):
        samples[r["uid"]].append(r["pred_orig"])
    uids = sorted(set(greedy) & set(samples))
    g = np.array([greedy[u]["pred_orig"] == greedy[u]["gold"] for u in uids], dtype=float)
    vote = []
    for u in uids:
        ballots = [greedy[u]["pred_orig"]] + samples[u]
        c = Counter(b for b in ballots if b is not None)
        if not c:
            vote.append(False)
            continue
        top = max(c.values())
        winners = [a for a, n in c.items() if n == top]
        choice = greedy[u]["pred_orig"] if greedy[u]["pred_orig"] in winners else min(winners)
        vote.append(choice == greedy[u]["gold"])
    v = np.array(vote, dtype=float)
    k = max(len(samples[u]) for u in uids)
    per_sample = [np.mean([samples[u][i] == greedy[u]["gold"] for u in uids if len(samples[u]) > i]) for i in range(k)]
    d = v - g
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [2.5, 97.5])
    print(f"{args.samples.name}: n={len(uids)}  greedy {100 * g.mean():.1f}  samples {' / '.join(f'{100 * x:.1f}' for x in per_sample)}"
          f"  vote(greedy+{k}) {100 * v.mean():.1f}  diff {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")


if __name__ == "__main__":
    main()
