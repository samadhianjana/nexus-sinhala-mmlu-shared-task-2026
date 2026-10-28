"""Go / no-go for cross-model pooling, by a rule fixed (27 Oct, before any of these runs existed) on FRESH stand-in questions
(gmmlu_si_val_nonacademic_fresh300: never used for any earlier decision, training or trace generation).

Baseline = candidate B's method: greedy thinking of the base model; where the fast one-step answer (base model, 4-order logit
vote) disagrees with it, vote over greedy + 3 sampled paths (ties -> greedy).
Candidate = vote over base greedy + 3 base samples + RFT2 greedy + 3 RFT2 samples, on every question (ties -> base greedy).

GO if candidate - baseline >= +1.0 point. Writes results/pool_decision.json.

  python -m eval.pool_decide --fast pool_fast_base_fresh300 --base-greedy pool_think_base_fresh300 \
      --base-samples pool_sc3_base_fresh300 --rft-greedy pool_think_rft2_fresh300 --rft-samples pool_sc3_rft2_fresh300
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .common import ROOT, read_jsonl
from .perm_sc import vote

MIN_GAIN = 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("fast", "base-greedy", "base-samples", "rft-greedy", "rft-samples"):
        ap.add_argument(f"--{a}", required=True)
    args = ap.parse_args()

    def runs(name: str) -> tuple[dict, dict]:
        preds, probs = defaultdict(list), defaultdict(lambda: np.zeros(4))
        for r in read_jsonl(ROOT / "results" / name / "predictions.jsonl"):
            preds[r["uid"]].append(r["pred_orig"])
            probs[r["uid"]][: len(r["probs_orig"])] += r["probs_orig"]
        return preds, probs

    _, fast = runs(args.fast)
    bg, _ = runs(args.base_greedy)
    bs, _ = runs(args.base_samples)
    rg, _ = runs(args.rft_greedy)
    rs, _ = runs(args.rft_samples)
    gold = {r["uid"]: r["gold"] for r in read_jsonl(ROOT / "results" / args.base_greedy / "predictions.jsonl")}
    uids = sorted(set(bg) & set(bs) & set(rg) & set(rs) & set(fast))
    gated = {u: int(fast[u].argmax()) + 1 != bg[u][0] for u in uids}
    baseline = {u: vote([bg[u][0]] + (bs[u] if gated[u] else [])) for u in uids}
    cand = {u: vote([bg[u][0]] + bs[u] + [rg[u][0]] + rs[u]) for u in uids}
    acc = lambda m: np.array([m[u] == gold[u] for u in uids], dtype=float)  # noqa: E731
    d = acc(cand) - acc(baseline)
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [2.5, 97.5])
    out = {"n": len(uids), "gated_fraction": float(np.mean(list(gated.values()))),
           "base_greedy": 100 * float(acc({u: bg[u][0] for u in uids}).mean()),
           "rft2_greedy": 100 * float(acc({u: rg[u][0] for u in uids}).mean()),
           "base_vote_all4": 100 * float(acc({u: vote([bg[u][0]] + bs[u]) for u in uids}).mean()),
           "rft2_vote_all4": 100 * float(acc({u: vote([rg[u][0]] + rs[u]) for u in uids}).mean()),
           "baseline_B": 100 * float(acc(baseline).mean()), "candidate_pooled8": 100 * float(acc(cand).mean()),
           "gain": 100 * float(d.mean()), "ci95": [100 * float(lo), 100 * float(hi)],
           "rule": f"candidate - baseline >= {MIN_GAIN}", "go": bool(100 * d.mean() >= MIN_GAIN)}
    (ROOT / "results" / "pool_decision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("DECISION:", "GO" if out["go"] else "NO-GO")


if __name__ == "__main__":
    main()
