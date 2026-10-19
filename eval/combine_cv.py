"""Combine k-fold runs (each scored only on its held-out fold) into one run over the full dev set.

  python -m eval.combine_cv --runs results/exp_devcv_lora_f* --name exp_devcv_lora_all

Writes results/<name>/{metrics.json, report.md, predictions.jsonl} in the usual layout, so eval.summarize can put it
next to the zero-shot baseline. Fails if a question is missing or scored twice.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .common import ROOT, read_jsonl, write_jsonl
from .metrics import markdown_report
from .run_eval import metrics_by_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/processed/sinhalammlu_dev.jsonl")
    args = ap.parse_args()
    dirs = sorted(Path(d) for pat in args.runs for d in glob.glob(pat))
    records, first = [], None
    for d in dirs:
        records += read_jsonl(d / "predictions.jsonl")
        first = first or json.loads((d / "metrics.json").read_text())
    items = {r["uid"]: r for r in read_jsonl(ROOT / args.data)}
    scored = [r["uid"] for r in records if r["perm_k"] == 0]
    assert len(scored) == len(set(scored)), "a question was scored by more than one fold"
    assert set(scored) == set(items), f"{len(set(items) - set(scored))} dev questions were not scored"
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"by_dataset": metrics_by_dataset(items, records),
               "run": first["run"] | {"run_name": args.name, "combined_from": [d.name for d in dirs]}}
    write_jsonl(out / "predictions.jsonl", records)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out / "report.md").write_text("\n".join(markdown_report(f"{args.name} / {k}", m)
                                             for k, m in metrics["by_dataset"].items()))
    o = metrics["by_dataset"]["sinhalammlu"]["overall"]
    print(f"{args.name}: dev acc={o['acc']:.4f} vote={o['acc_vote']:.4f} over {o['n']} questions from {len(dirs)} folds")


if __name__ == "__main__":
    main()
