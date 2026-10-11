"""Near-duplicate check between two MCQ (or text) sets: TF-IDF over character n-grams, cosine similarity.

Catches exact copies, reorderings and light edits; translation of the same question into different Sinhala
wording is only partly caught. Used for contamination checks of every training set against dev/validation.

  python scripts/overlap.py --a gmmlu_si_train --b sinhalammlu_dev [--threshold 0.8] [--show 5]

--a / --b are processed dataset names (data/processed/<name>.jsonl) or JSONL paths with "question" (+ "options")
or "text" fields. Prints the similarity distribution of each A item's nearest B item and the pairs above threshold.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import read_jsonl  # noqa: E402
from eval.run_eval import resolve_data  # noqa: E402


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[\s‌‍]+", " ", text)  # whitespace, ZWNJ/ZWJ (common in Sinhala)
    # keep the whole Sinhala block: vowel signs and virama are combining marks, which \w does not match
    return re.sub(r"[^\w \u0D80-\u0DFF]", "", text).strip()


def item_text(r: dict, with_options: bool) -> str:
    if "text" in r:
        return r["text"]
    return r["question"] + (" " + " ".join(r["options"]) if with_options else "")


def ngram_matrix(texts: list[str], n: int, dim: int = 2**21) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        t = f" {norm(t)} "
        grams = {}
        for j in range(len(t) - n + 1):
            h = hash(t[j:j + n]) % dim
            grams[h] = grams.get(h, 0) + 1
        rows += [i] * len(grams)
        cols += list(grams)
        vals += list(grams.values())
    m = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(texts), dim))
    return m


def tfidf(a: sp.csr_matrix, b: sp.csr_matrix) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    df = np.asarray(((a > 0).sum(0) + (b > 0).sum(0))).ravel()
    idf = np.log((a.shape[0] + b.shape[0] + 1) / (df + 1)) + 1
    out = []
    for m in (a, b):
        m = m.multiply(idf).tocsr()
        m.data = np.log1p(m.data)
        norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel()) + 1e-9
        out.append(sp.diags(1 / norms) @ m)
    return out[0], out[1]


def nearest(a: sp.csr_matrix, b: sp.csr_matrix, chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    best, idx = np.zeros(a.shape[0]), np.zeros(a.shape[0], dtype=int)
    bt = b.T.tocsc()
    for i in range(0, a.shape[0], chunk):
        s = (a[i:i + chunk] @ bt).toarray()
        idx[i:i + chunk] = s.argmax(1)
        best[i:i + chunk] = s.max(1)
    return best, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=5, help="character n-gram size")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--no-options", action="store_true", help="compare question stems only")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    A, B = read_jsonl(resolve_data(args.a)), read_jsonl(resolve_data(args.b))
    ta = [item_text(r, not args.no_options) for r in A]
    tb = [item_text(r, not args.no_options) for r in B]
    ma, mb = tfidf(ngram_matrix(ta, args.n), ngram_matrix(tb, args.n))
    best, idx = nearest(ma, mb)

    print(f"A={args.a} ({len(A)})  B={args.b} ({len(B)})  char {args.n}-grams, "
          f"{'stems only' if args.no_options else 'question + options'}")
    for q in (50, 90, 99, 99.9, 100):
        print(f"  nearest-neighbour similarity, {q:>5}th percentile: {np.percentile(best, q):.3f}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        print(f"  A items with a B neighbour >= {t}: {(best >= t).sum()}")
    order = np.argsort(-best)
    for i in order[: args.show]:
        print(f"\n  sim={best[i]:.3f}\n    A {A[i].get('uid', i)}: {ta[i][:160]}\n    B {B[idx[i]].get('uid', idx[i])}: {tb[idx[i]][:160]}")


if __name__ == "__main__":
    main()
