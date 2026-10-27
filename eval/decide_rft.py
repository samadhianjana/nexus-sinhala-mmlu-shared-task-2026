"""Go / no-go for the reasoning-trace model, decided by a rule fixed in advance (before any result is seen).

Compares greedy thinking of the trained model (with its adapter) with greedy thinking of the base model on the SAME held-out
questions: the 400-question SinhalaMMLU dev sample (never used for traces or training) and the 300-question Global-MMLU
non-academic proxy (never used). Paired differences are pooled over both sets.

GO if pooled difference >= +1.5 points AND neither set is worse than -3.0 points. Writes results/rft_decision.json.

  python -m eval.decide_rft --pairs think_base_smmlu_dev_sample400:think_rft_dev400:sinhalammlu \
      think_base_gmmlu_si_val_nonacademic300:think_rft_nonacademic300:gmmlu_si
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import ROOT
from .compare import correctness

MIN_POOLED, MIN_EACH = 1.5, -3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="base_run:new_run:dataset")
    ap.add_argument("--out", default="rft_decision.json", help="file name under results/")
    args = ap.parse_args()
    diffs, per = [], {}
    for pair in args.pairs:
        b, n, ds = pair.split(":")
        base, new = correctness(ROOT / "results" / b, ds)[0], correctness(ROOT / "results" / n, ds)[0]
        uids = sorted(set(base) & set(new))
        d = np.array([new[u] - base[u] for u in uids], dtype=float)
        per[n] = {"n": len(uids), "base": 100 * float(np.mean([base[u] for u in uids])),
                  "new": 100 * float(np.mean([new[u] for u in uids])), "diff": 100 * float(d.mean())}
        diffs.append(d)
    pooled = np.concatenate(diffs)
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([pooled[rng.integers(0, len(pooled), len(pooled))].mean() for _ in range(5000)], [2.5, 97.5])
    go = bool(100 * pooled.mean() >= MIN_POOLED and all(v["diff"] >= MIN_EACH for v in per.values()))
    out = {"rule": f"pooled diff >= {MIN_POOLED} and each set >= {MIN_EACH}", "per_set": per, "n": len(pooled),
           "pooled_diff": 100 * float(pooled.mean()), "ci95": [100 * float(lo), 100 * float(hi)], "go": go}
    (ROOT / "results" / args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("DECISION:", "GO" if go else "NO-GO")


if __name__ == "__main__":
    main()
