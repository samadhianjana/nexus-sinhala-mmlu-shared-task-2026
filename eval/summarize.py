"""Combine results/*/metrics.json into comparison tables (reports/<name>.md).

Usage: python -m eval.summarize --runs results/base_* --out reports/baselines_zeroshot.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load(run_dirs: list[str]) -> dict[str, dict]:
    runs = {}
    for d in sorted(run_dirs):
        f = Path(d) / "metrics.json"
        if f.exists():
            runs[Path(d).name] = json.loads(f.read_text())
    return runs


def fmt(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = load([d for pat in args.runs for d in glob.glob(pat)])
    names = list(runs)
    short = {n: n.removeprefix("base_").removeprefix("exp_") for n in names}

    lines = ["# Baseline comparison", "",
             "Accuracy in %. `orig` = original option order; `vote` = mean probabilities over cyclic option "
             "permutations; `±` = std of accuracy across permutations (position sensitivity). Random = 25.", ""]

    # Overview
    lines += ["## Overview", "", "| model | params loaded | tok/char | SinhalaMMLU dev orig | vote | ± | macro-subj | "
              "Global-MMLU si | Belebele si | label mass | consistency |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        r, bd = runs[n]["run"], runs[n]["by_dataset"]
        s = bd.get("sinhalammlu", {}).get("overall", {})
        g = bd.get("gmmlu_si", {}).get("overall", {})
        b = bd.get("belebele_si", {}).get("overall", {})
        lines.append(f"| {short[n]} | {r['params']['loaded_total'] / 1e9:.2f}B | {r['tokens_per_char']:.2f} | "
                     f"{fmt(s.get('acc'))} | {fmt(s.get('acc_vote'))} | {fmt(s.get('acc_perm_std'))} | "
                     f"{fmt(s.get('acc_macro_subject'))} | {fmt(g.get('acc'))} | {fmt(b.get('acc'))} | "
                     f"{s.get('label_mass_mean', float('nan')):.2f} | {s.get('consistency', float('nan')):.2f} |")

    def table(dataset: str, key: str, title: str) -> None:
        groups = sorted({g for n in names for g in runs[n]["by_dataset"].get(dataset, {}).get("groups", {}).get(key, {})})
        if not groups:
            return
        lines.extend(["", f"## {title}", "", "| " + key + " | n | " + " | ".join(short[n] for n in names) + " |",
                      "|---|---:|" + "---:|" * len(names)])
        for g in groups:
            cells, cnt = [], "-"
            for n in names:
                blk = runs[n]["by_dataset"].get(dataset, {}).get("groups", {}).get(key, {}).get(g)
                cells.append(fmt(blk["acc"]) if blk else "-")
                cnt = blk["n"] if blk else cnt
            lines.append(f"| {g} | {cnt} | " + " | ".join(cells) + " |")

    table("sinhalammlu", "subject", "SinhalaMMLU dev: per subject")
    table("sinhalammlu", "category", "SinhalaMMLU dev: per domain")
    table("sinhalammlu", "grade", "SinhalaMMLU dev: per grade")
    table("gmmlu_si", "category", "Global-MMLU si: per category")
    table("gmmlu_si", "cultural_sensitivity", "Global-MMLU si: culturally sensitive (CS) / agnostic (CA) / unlabelled (-)")

    lines += ["", "## Position bias on SinhalaMMLU dev (share of predictions per displayed position)", "",
              "| model | 1 | 2 | 3 | 4 |", "|---|---:|---:|---:|---:|"]
    for n in names:
        pb = runs[n]["by_dataset"].get("sinhalammlu", {}).get("position_bias")
        if pb:
            lines.append(f"| {short[n]} | " + " | ".join(fmt(pb["pred_dist"].get(str(p))) for p in range(1, 5)) + " |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
