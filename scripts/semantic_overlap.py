"""Meaning-based overlap check between two MCQ sets (catches the same question in different wording, which the
character n-gram check in scripts/overlap.py misses).

  python scripts/semantic_overlap.py --a sinhalammlu_dev --b data/sft/gmmlu_si_train.jsonl [--show 15]

Embeds "question + options" with intfloat/multilingual-e5-large (covers Sinhala), cosine similarity, and reports for
each A item its nearest B item. A reference scale is printed from pairs that are known copies (Global-MMLU questions
repeated across MMLU subjects) and from random unrelated pairs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import cap_gpu_memory, load_env, read_jsonl  # noqa: E402
from eval.run_eval import resolve_data  # noqa: E402

MODEL = "intfloat/multilingual-e5-large"


def text(r: dict) -> str:
    return "query: " + r["question"] + " " + " | ".join(r["options"])


def embed(texts: list[str], tok, model, bs: int = 64) -> np.ndarray:
    import torch

    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()
    load_env()
    cap_gpu_memory(20)
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    A, B = read_jsonl(resolve_data(args.a)), read_jsonl(resolve_data(args.b))
    ea, eb = embed([text(r) for r in A], tok, model), embed([text(r) for r in B], tok, model)
    sims = ea @ eb.T
    best, idx = sims.max(1), sims.argmax(1)

    # Reference scale: known copies (validation items with a character-level copy in B) and random pairs
    rng = np.random.default_rng(0)
    rand = sims[rng.integers(0, len(A), 5000), rng.integers(0, len(B), 5000)]
    print(f"A={args.a} ({len(A)})  B={args.b} ({len(B)})  model={MODEL}")
    print(f"reference: random unrelated pairs  median {np.median(rand):.3f}, 99th pct {np.percentile(rand, 99):.3f}")
    for q in (50, 90, 99, 100):
        print(f"nearest-neighbour similarity, {q:>3}th percentile: {np.percentile(best, q):.3f}")
    for t in (0.85, 0.90, 0.95):
        print(f"A items with a B neighbour >= {t}: {(best >= t).sum()}")
    for i in np.argsort(-best)[: args.show]:
        print(f"\nsim={best[i]:.3f}\n  A {A[i]['uid']}: {A[i]['question'][:150]}\n     options: {' | '.join(A[i]['options'])[:150]}"
              f"\n  B {B[idx[i]]['uid']}: {B[idx[i]]['question'][:150]}\n     options: {' | '.join(B[idx[i]]['options'])[:150]}")


if __name__ == "__main__":
    main()
