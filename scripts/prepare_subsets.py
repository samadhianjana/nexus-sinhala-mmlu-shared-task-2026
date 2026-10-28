"""Recreate the fixed evaluation subsets used by the reasoning-trace experiments (deterministic, seed 1234):
  smmlu_dev_sample400            400 random SinhalaMMLU dev questions (HELD OUT from trace training)
  smmlu_dev_rest1450             the other 1,450 dev questions (trace source)
  gmmlu_si_val_nonacademic300    300 random non-academic (Humanities / Social Sciences) Global-MMLU val questions
  gmmlu_si_val_nonacademic_fresh300  300 other random non-academic val questions (cross-model pooling decision only)
  gmmlu_si_val_academic          Global-MMLU val STEM + Business questions
Run after scripts/fetch_data.py and scripts/prepare_data.py.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, read_jsonl, write_jsonl  # noqa: E402

P = ROOT / "data" / "processed"
dev = read_jsonl(P / "sinhalammlu_dev.jsonl")
sample = random.Random(1234).sample(dev, 400)
keep = {r["uid"] for r in sample}
write_jsonl(P / "smmlu_dev_sample400.jsonl", sample)
write_jsonl(P / "smmlu_dev_rest1450.jsonl", [r for r in dev if r["uid"] not in keep])
val = read_jsonl(P / "gmmlu_si_val.jsonl")
non = [r for r in val if r["category"] in ("Humanities", "Social Sciences")]
na300 = random.Random(1234).sample(non, 300)
write_jsonl(P / "gmmlu_si_val_nonacademic300.jsonl", na300)
# Fresh stand-in (27 Oct), never looked at before the cross-model pooling rule was fixed: 300 other non-academic val questions
used = {r["uid"] for r in na300}
write_jsonl(P / "gmmlu_si_val_nonacademic_fresh300.jsonl", random.Random(2026).sample([r for r in non if r["uid"] not in used], 300))
write_jsonl(P / "gmmlu_si_val_academic.jsonl", [r for r in val if r["category"] in ("STEM", "Business")])
print("dev", len(dev), "| sample400", len(sample), "| rest", len(dev) - len(sample), "| nonacademic300 300 | academic",
      sum(r["category"] in ("STEM", "Business") for r in val))
