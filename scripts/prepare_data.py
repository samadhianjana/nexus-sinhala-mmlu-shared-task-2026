"""Normalise raw eval data into one MCQ schema under data/processed/.

Schema (one JSON object per line):
  uid, dataset, q_no, subject, category, level, question, options[list[str]], answer (1-based int), meta{}

Outputs:
  sinhalammlu_dev.jsonl   public Easy tier (dev)
  gmmlu_si_val.jsonl      fixed held-out sample of Global-MMLU si test (harder internal validation);
                          ids in configs/splits/ so they are excluded from any later training mix
  sinhalammlu_dev_5shot.jsonl  fixed 5 few-shot examples drawn from the dev pool (ids frozen in configs/splits)
  gmmlu_si_dev.jsonl      Global-MMLU si dev split (5/subject), candidate pool for fixed few-shot examples
  belebele_si.jsonl       Belebele sin_Sinh, EVALUATION ONLY
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, read_jsonl, write_jsonl  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SPLITS = ROOT / "configs" / "splits"
LETTERS = "ABCD"


def prep_sinhalammlu() -> list[dict]:
    rows, dropped = [], []
    for f in sorted((RAW / "sinhalammlu" / "TEST").rglob("*.json")):
        recs = json.loads(f.read_text(encoding="utf-8"))
        # A few records spell their subject differently from the rest of the file
        # ("Eastern_music" vs "Eastern music", trailing spaces): group by the file's majority spelling,
        # keep the original in subject_raw because Codabench predictions must echo it.
        canonical = Counter(r["subject"].strip() for r in recs).most_common(1)[0][0]
        for r in recs:
            level = r["metadata"].get("difficulty", f.parent.name)
            uid = f"smmlu/{level}/{r['subject']}/{r['q_no']}"
            if not (isinstance(r["answer"], int) and 1 <= r["answer"] <= len(r["choices"])):
                dropped.append((uid, r["answer"]))
                continue
            rows.append(dict(
                uid=uid, dataset="sinhalammlu", q_no=r["q_no"], subject=canonical, subject_raw=r["subject"],
                category=r["category"], level=level, question=r["question"].strip(),
                options=[str(c).strip() for c in r["choices"]], answer=r["answer"],
                meta={k: r["metadata"].get(k) for k in ("grade", "year", "type", "subject_original")},
            ))
    for uid, a in dropped:
        print(f"  dropped {uid}: invalid answer {a!r}")
    return rows


def select_fewshot(rows: list[dict], k: int, seed: int) -> list[dict]:
    """Fixed few-shot set from the dev pool: one question per domain, gold answers spread over 1..4.
    Chosen at random (not by inspection) and frozen in configs/splits; excluded from evaluation by the harness."""
    ids_file = SPLITS / f"sinhalammlu_dev_{k}shot_ids.txt"
    by_uid = {r["uid"]: r for r in rows}
    if ids_file.exists():
        return [by_uid[u] for u in ids_file.read_text().split()]
    rng = random.Random(seed)
    pool = sorted(rows, key=lambda r: r["uid"])
    rng.shuffle(pool)
    picked, answers = [], []
    for cat in sorted({r["category"] for r in rows}):
        r = next(r for r in pool if r["category"] == cat and r["answer"] not in answers)
        picked.append(r)
        answers.append(r["answer"])
    for r in pool:
        if len(picked) >= k:
            break
        if r not in picked and (len(set(answers)) == 4 or r["answer"] not in answers):
            picked.append(r)
            answers.append(r["answer"])
    rng.shuffle(picked)
    ids_file.write_text("\n".join(r["uid"] for r in picked) + "\n")
    return picked


def _gmmlu_rows(df, split: str) -> list[dict]:
    rows = []
    for r in df.itertuples():
        rows.append(dict(
            uid=f"gmmlu_si/{split}/{r.sample_id}", dataset="gmmlu_si", q_no=r.sample_id,
            subject=r.subject, category=r.subject_category, level="gmmlu",
            question=r.question.strip(),
            options=[str(getattr(r, f"option_{c}")).strip() for c in "abcd"],
            answer=LETTERS.index(r.answer) + 1,
            meta={"cultural_sensitivity": r.cultural_sensitivity_label},
        ))
    return rows


def prep_gmmlu(val_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    import pandas as pd

    d = RAW / "global_mmlu" / "si"
    test = _gmmlu_rows(pd.read_parquet(d / "test-00000-of-00001.parquet"), "test")
    dev = _gmmlu_rows(pd.read_parquet(d / "dev-00000-of-00001.parquet"), "dev")

    ids_file = SPLITS / "gmmlu_si_val_ids.txt"
    if ids_file.exists():  # split is frozen once created
        keep = set(ids_file.read_text().split())
    else:  # proportional stratified sample by subject
        by_subj = defaultdict(list)
        for r in test:
            by_subj[r["subject"]].append(r["uid"])
        rng = random.Random(seed)
        keep = set()
        for subj in sorted(by_subj):
            uids = sorted(by_subj[subj])
            k = max(1, round(val_size * len(uids) / len(test)))
            keep.update(rng.sample(uids, min(k, len(uids))))
        SPLITS.mkdir(parents=True, exist_ok=True)
        ids_file.write_text("\n".join(sorted(keep)) + "\n")
    return [r for r in test if r["uid"] in keep], dev


def prep_belebele() -> list[dict]:
    rows = []
    for r in read_jsonl(RAW / "belebele" / "data" / "sin_Sinh.jsonl"):
        rows.append(dict(
            uid=f"belebele_si/{r['link']}#{r['question_number']}", dataset="belebele_si",
            q_no=r["question_number"], subject="reading_comprehension", category="reading",
            level="belebele", question=f"{r['flores_passage'].strip()}\n\n{r['question'].strip()}",
            options=[r[f"mc_answer{i}"].strip() for i in range(1, 5)],
            answer=int(r["correct_answer_num"]), meta={},
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmmlu-val-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    outputs = {"sinhalammlu_dev": prep_sinhalammlu()}
    outputs["sinhalammlu_dev_5shot"] = select_fewshot(outputs["sinhalammlu_dev"], 5, args.seed)
    outputs["gmmlu_si_val"], outputs["gmmlu_si_dev"] = prep_gmmlu(args.gmmlu_val_size, args.seed)
    outputs["belebele_si"] = prep_belebele()
    for name, rows in outputs.items():
        write_jsonl(OUT / f"{name}.jsonl", rows)
        ans = Counter(r["answer"] for r in rows)
        print(f"{name:18s} n={len(rows):5d} answers={dict(sorted(ans.items()))}")


if __name__ == "__main__":
    main()
