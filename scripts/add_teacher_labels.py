"""Attach a teacher's option probabilities (from an eval.run_eval logit run) to an MCQ training file.

  python scripts/add_teacher_labels.py --data data/sft/gmmlu_si_train.jsonl --run results/teacher_31b_gmmlu_train \
      --out data/sft/gmmlu_si_train_t31b.jsonl

teacher_probs are in the file's stored option order (probs_orig of permutation 0). Reports teacher accuracy on the
file's keys and how often teacher and key disagree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    preds = {r["uid"]: r["probs_orig"] for r in read_jsonl(Path(args.run) / "predictions.jsonl") if r["perm_k"] == 0}
    rows, agree, missing = [], 0, 0
    for r in read_jsonl(args.data):
        p = preds.get(r["uid"])
        if p is None:
            missing += 1
            rows.append(r)
            continue
        agree += max(range(len(p)), key=p.__getitem__) + 1 == r["answer"]
        rows.append(r | {"teacher_probs": [round(x, 5) for x in p]})
    write_jsonl(args.out, rows)
    n = len(rows) - missing
    print(f"{n} items labelled ({missing} without teacher probs); teacher agrees with the key on {agree / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
