"""Recompute metrics.json / report.md of finished runs from their saved predictions (no model needed).
Use after changing data grouping or metrics code.  Usage: python -m eval.recompute results/base_*
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import read_jsonl
from .metrics import markdown_report
from .run_eval import metrics_by_dataset, resolve_data


def main() -> None:
    for d in map(Path, sys.argv[1:]):
        if not (d / "metrics.json").exists():
            continue
        old = json.loads((d / "metrics.json").read_text())
        records = read_jsonl(d / "predictions.jsonl")
        uids = {r["uid"] for r in records}
        items = {r["uid"]: r for name in old["run"]["args"]["data"] for r in read_jsonl(resolve_data(name)) if r["uid"] in uids}
        missing = uids - items.keys()
        if missing:
            print(f"{d.name}: {len(missing)} predicted uids not in current data, skipped")
            continue
        new = {"by_dataset": metrics_by_dataset(items, records), "run": old["run"]}
        (d / "metrics.json").write_text(json.dumps(new, indent=2, ensure_ascii=False))
        (d / "report.md").write_text("\n".join(markdown_report(f"{d.name} / {k}", m) for k, m in new["by_dataset"].items()))
        print(f"{d.name}: recomputed")


if __name__ == "__main__":
    main()
