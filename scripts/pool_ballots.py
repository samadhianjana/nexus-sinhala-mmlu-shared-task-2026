"""Cross-model pooling: majority vote over the thinking answers of several runs (each <predictions.zip>.ballots.jsonl,
written by inference/predict.py), merged into a base predictions.zip for the questions the ballot files cover.

Ballots are concatenated in the order given; unparsed answers abstain; ties go to the first ballot (the first run's greedy
answer) if it is among the winners, else to the lowest label (the same rule as inference/predict.py).

  python scripts/pool_ballots.py --base results/submissions/final_candidate_base_think_gated_sc.zip \
      --ballots results/submissions/pool_base_medium.zip.ballots.jsonl results/submissions/pool_rft2_medium.zip.ballots.jsonl \
      --out results/submissions/final_candidate_B_plus_pool_medium.zip
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


def vote(b: list):
    c = Counter(x for x in b if x)
    if not c:
        return None
    top = max(c.values())
    w = [a for a, n in c.items() if n == top]
    return b[0] if b[0] in w else min(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--ballots", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    runs = []
    for f in args.ballots:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        runs.append({(r["level"], r["subject"], r["q_no"]): r["ballots"] for r in rows})
    keys = set(runs[0])
    assert all(set(r) == keys for r in runs), "ballot files cover different questions"
    pooled = {k: vote([b for r in runs for b in r[k]]) for k in keys}
    with zipfile.ZipFile(args.base) as z:
        base = [json.loads(l) for l in z.read("predictions.jsonl").decode("utf-8").splitlines() if l.strip()]
    changed = matched = 0
    for r in base:
        k = (r["level"], r["subject"], r["q_no"])
        if k in pooled and pooled[k]:
            matched += 1
            changed += str(pooled[k]) != r["answer"]
            r["answer"] = str(pooled[k])
    assert matched == sum(1 for v in pooled.values() if v), "some pooled questions are missing from the base file"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in base))
    n_b = Counter(len([b for r in runs for b in r[k]]) for k in keys)
    print(f"{len(keys)} questions pooled over {len(runs)} runs (ballots per question: {dict(n_b)}); "
          f"{changed} answers differ from {args.base.name} -> {args.out}")


if __name__ == "__main__":
    main()
