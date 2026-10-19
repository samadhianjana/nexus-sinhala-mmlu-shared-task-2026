"""5-fold split of SinhalaMMLU dev, stratified by subject (fixed seed) -> data/processed/smmlu_cv5_f{k}_{train,test}.jsonl.

Each question is in exactly one test fold, so a model trained on the other four folds is always scored on questions
it never saw. Fold membership (uids only) is frozen in configs/splits/sinhalammlu_dev_cv5.json.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, read_jsonl, write_jsonl  # noqa: E402

K, SEED = 5, 1234


def main() -> None:
    rows = read_jsonl(ROOT / "data" / "processed" / "sinhalammlu_dev.jsonl")
    split_file = ROOT / "configs" / "splits" / "sinhalammlu_dev_cv5.json"
    if split_file.exists():
        folds = json.loads(split_file.read_text())
    else:
        by_subject = defaultdict(list)
        for r in rows:
            by_subject[r["subject"]].append(r["uid"])
        rng, folds, k = random.Random(SEED), [[] for _ in range(K)], 0
        for subject in sorted(by_subject):
            uids = sorted(by_subject[subject])
            rng.shuffle(uids)
            for u in uids:
                folds[k % K].append(u)
                k += 1
        split_file.write_text(json.dumps(folds, indent=1))
    for f, test_uids in enumerate(folds):
        test = set(test_uids)
        write_jsonl(ROOT / "data" / "processed" / f"smmlu_cv5_f{f}_test.jsonl", [r for r in rows if r["uid"] in test])
        write_jsonl(ROOT / "data" / "processed" / f"smmlu_cv5_f{f}_train.jsonl",
                    [r | {"dataset": "sinhalammlu_train"} for r in rows if r["uid"] not in test])
        print(f"fold {f}: train {len(rows) - len(test)}, test {len(test)}")


if __name__ == "__main__":
    main()
