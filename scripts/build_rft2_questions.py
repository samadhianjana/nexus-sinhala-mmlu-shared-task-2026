"""Question pool for the second round of reasoning-trace training: Global-MMLU `si` training questions (Humanities, Social
Sciences, Other, Business: closest to Medium's culture/civics/social subjects) plus the synthetic Wikipedia-based MCQs
(data/synth/pilot.jsonl) if present. Output: data/synth/rft2_questions.jsonl. The dev held-out questions are never included.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, read_jsonl, write_jsonl  # noqa: E402

N_GMMLU = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
g = [r for r in read_jsonl(ROOT / "data" / "sft" / "gmmlu_si_train.jsonl")
     if r["category"] in ("Humanities", "Social Sciences", "Other", "Business")]
g = random.Random(1234).sample(g, min(N_GMMLU, len(g)))
syn_path = ROOT / "data" / "synth" / "pilot.jsonl"
syn = [{k: v for k, v in r.items() if k != "teacher_probs"} for r in read_jsonl(syn_path)] if syn_path.exists() else []
write_jsonl(ROOT / "data" / "synth" / "rft2_questions.jsonl", g + syn)
print(f"{len(g)} Global-MMLU + {len(syn)} synthetic questions -> data/synth/rft2_questions.jsonl")
