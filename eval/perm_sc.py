"""Option-order self-consistency validation (decision for the Medium run).

Prior answer = the current pipeline's answer on the stand-in questions: majority vote of greedy thinking + 3 sampled paths
(ties -> greedy), the analogue of candidate B's final answer. Candidate method: add the answers from K re-orderings of the
options (thinking each time, mapped back to the original numbering) and vote again (ties -> prior).

GO if the candidate beats the prior by at least +1.0 point on the stand-in. Writes results/perm_decision.json.

  python -m eval.perm_sc --greedy results/think_base_gmmlu_si_val_nonacademic300 --samples results/sc3_base_nonacademic \
      --perms results/permthink_nonacademic300
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, read_jsonl

MIN_GAIN = 1.0


def vote(ballots: list, first_wins: bool = True):
    c = Counter(b for b in ballots if b)
    if not c:
        return None
    top = max(c.values())
    w = [a for a, n in c.items() if n == top]
    return ballots[0] if ballots[0] in w else min(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", required=True, type=Path)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--perms", required=True, type=Path)
    args = ap.parse_args()
    g = {r["uid"]: r for r in read_jsonl(args.greedy / "predictions.jsonl") if r["perm_k"] == 0}
    s, p = defaultdict(list), defaultdict(list)
    for r in read_jsonl(args.samples / "predictions.jsonl"):
        s[r["uid"]].append(r["pred_orig"])
    for r in read_jsonl(args.perms / "predictions.jsonl"):
        if r["perm_k"] >= 1:
            p[r["uid"]].append(r["pred_orig"])
    uids = sorted(set(g) & set(s) & set(p))
    gold = {u: g[u]["gold"] for u in uids}
    prior = {u: vote([g[u]["pred_orig"]] + s[u]) for u in uids}
    cand = {u: vote([prior[u]] + p[u]) for u in uids}
    perms_only = {u: vote([g[u]["pred_orig"]] + p[u]) for u in uids}
    acc = lambda m: np.array([m[u] == gold[u] for u in uids], dtype=float)  # noqa: E731
    a_g, a_prior, a_cand, a_perm = acc({u: g[u]["pred_orig"] for u in uids}), acc(prior), acc(cand), acc(perms_only)
    d = a_cand - a_prior
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [2.5, 97.5])
    singles = [100 * float(np.mean([p[u][i] == gold[u] for u in uids])) for i in range(min(len(p[u]) for u in uids))]
    out = {"n": len(uids), "greedy": 100 * a_g.mean(), "prior_vote_greedy_plus_samples": 100 * a_prior.mean(),
           "single_reordered_runs": singles, "greedy_plus_reorders_vote": 100 * a_perm.mean(),
           "candidate_prior_plus_reorders": 100 * a_cand.mean(), "gain_vs_prior": 100 * float(d.mean()),
           "ci95": [100 * float(lo), 100 * float(hi)], "rule": f"candidate - prior >= {MIN_GAIN}",
           "go": bool(100 * d.mean() >= MIN_GAIN)}
    (ROOT / "results" / "perm_decision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("DECISION:", "GO" if out["go"] else "NO-GO")


if __name__ == "__main__":
    main()
