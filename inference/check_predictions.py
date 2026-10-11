"""Score a Codabench-format predictions.zip against gold answers (dev only; test gold is hidden).

  python -m inference.check_predictions --predictions dev_predictions.zip --gold data/raw/sinhalammlu/TEST/easy

Also validates the file the way the leaderboard will need it: one row per question, required fields present,
answers are digits within the option range. Reports accuracy per level and their macro average (task metric).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.predict import read_records, to_item  # noqa: E402

FIELDS = ("q_no", "subject", "category", "level", "answer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    args = ap.parse_args()

    with zipfile.ZipFile(args.predictions) as z:
        assert z.namelist() == ["predictions.jsonl"], f"zip must contain only predictions.jsonl, has {z.namelist()}"
        preds = [json.loads(l) for l in z.read("predictions.jsonl").decode("utf-8").splitlines() if l.strip()]

    gold = {}
    for f, r in read_records(args.gold):
        it = to_item(f, r)
        gold[(it["level"], it["subject_raw"], it["q_no"])] = (r.get("answer"), len(it["options"]))

    errors, seen = [], set()
    correct = defaultdict(lambda: [0, 0])
    for p in preds:
        if any(k not in p for k in FIELDS):
            errors.append(f"missing fields in {p}")
            continue
        key = (p["level"], p["subject"], p["q_no"])
        if key in seen:
            errors.append(f"duplicate {key}")
        seen.add(key)
        if key not in gold:
            errors.append(f"unknown question {key}")
            continue
        ans, n = gold[key]
        if not (isinstance(p["answer"], str) and p["answer"].isdigit() and 1 <= int(p["answer"]) <= n):
            errors.append(f"bad answer {p['answer']!r} for {key}")
            continue
        if isinstance(ans, int) and 1 <= ans <= n:  # skip items with an invalid gold label
            correct[p["level"]][0] += int(p["answer"]) == ans
            correct[p["level"]][1] += 1
    missing = set(gold) - seen
    if missing:
        errors.append(f"{len(missing)} questions without a prediction, e.g. {sorted(missing)[:3]}")

    for lvl, (k, n) in sorted(correct.items()):
        print(f"{lvl}: {k}/{n} = {100 * k / n:.2f}%")
    if correct:
        print(f"macro average over levels: {100 * sum(k / n for k, n in correct.values()) / len(correct):.2f}%")
    print(f"{len(preds)} rows, {len(errors)} format errors")
    for e in errors[:20]:
        print("  ", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
