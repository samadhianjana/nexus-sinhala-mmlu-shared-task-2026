"""Replace the answers of some levels in a base predictions.zip with those of another predictions.zip (e.g. a run made with
`inference/predict.py --levels medium`). Rows are matched on (level, subject, q_no); everything else is kept as in the base.

  python scripts/merge_predictions.py --base results/submissions/B.zip --new results/submissions/medium.zip \
      --out results/submissions/merged.zip
"""
from __future__ import annotations

import argparse
import json
import zipfile


def load(p: str) -> list[dict]:
    with zipfile.ZipFile(p) as z:
        return [json.loads(l) for l in z.read("predictions.jsonl").decode("utf-8").splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, new = load(args.base), load(args.new)
    key = lambda r: (r["level"], r["subject"], r["q_no"])  # noqa: E731
    new_by = {key(r): r for r in new}
    levels = sorted({r["level"] for r in new})
    out, replaced, changed = [], 0, 0
    for r in base:
        if r["level"] in levels and key(r) in new_by:
            replaced += 1
            changed += new_by[key(r)]["answer"] != r["answer"]
            r = new_by[key(r)]
        out.append(r)
    assert replaced == len(new), f"{len(new) - replaced} new rows had no match in the base file"
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"levels {levels}: {replaced} rows replaced, {changed} answers differ from the base file -> {args.out}")


if __name__ == "__main__":
    main()
