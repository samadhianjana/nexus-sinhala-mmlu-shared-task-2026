"""Choose the continued-pretraining mix from data/corpus/clean -> data/corpus/<name>/ (+ reports/<name>_stats.json).

Default mix of ~300M estimated Gemma tokens:
  wikipedia   all of it (encyclopedic, highest priority)
  fineweb2    the rest of the Sinhala budget: 70% highest Sri Lanka/curriculum keyword score (sl_score), 30% random
  nsina       20% of the budget, random news articles
  english     10% of the budget, FineWeb-Edu sample (keeps general/English reasoning), not deduplicated against Sinhala
Cross-source near-duplicates (MinHash on word 5-gram shingles, Jaccard >= 0.8) are removed in priority order
wikipedia > fineweb2 > nsina. Writes 30 random Sinhala documents to sample_for_review.md for a human check.

Usage: python scripts/select_cpt_subset.py [--target-tokens 300e6] [--name cpt_v1]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cpt_corpus import norm  # noqa: E402
from eval.common import ROOT, load_env  # noqa: E402

TOK_PER_CHAR = {"si": 0.40, "en": 0.25}  # Gemma tokenizer, measured on Sinhala MCQs / typical English
NUM_PERM = 64
ENGLISH = ("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet")


def minhash(text: str):
    from datasketch import MinHash

    w = norm(text).split()
    m = MinHash(num_perm=NUM_PERM, seed=1)
    m.update_batch([" ".join(w[i:i + 5]).encode() for i in range(max(len(w) - 4, 1))])
    return m


def load_source(clean: Path, name: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted(clean.glob(f"{name}-*.parquet"))], ignore_index=True)


def take_chars(df: pd.DataFrame, chars: float) -> pd.DataFrame:
    return df[df["n_chars"].cumsum() <= chars]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=float, default=300e6)
    ap.add_argument("--news-share", type=float, default=0.20)
    ap.add_argument("--english-share", type=float, default=0.10)
    ap.add_argument("--name", default="cpt_v1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    load_env()
    from datasketch import MinHashLSH
    from huggingface_hub import HfApi, hf_hub_download

    clean, out = ROOT / "data" / "corpus" / "clean", ROOT / "data" / "corpus" / args.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    si_chars = args.target_tokens * (1 - args.english_share) / TOK_PER_CHAR["si"]
    margin = 1.15  # select extra, then trim after deduplication

    wiki = load_source(clean, "wikipedia")
    fw = load_source(clean, "fineweb2")
    news = load_source(clean, "nsina")
    news_chars = args.target_tokens * args.news_share / TOK_PER_CHAR["si"]
    fw_chars = si_chars - wiki["n_chars"].sum() - news_chars
    fw_top = take_chars(fw.sort_values("sl_score", ascending=False), 0.7 * fw_chars * margin)
    fw_rest = fw.drop(fw_top.index).sample(frac=1, random_state=args.seed)
    fw_sel = pd.concat([fw_top.assign(pick="top_sl_score"),
                        take_chars(fw_rest, 0.3 * fw_chars * margin).assign(pick="random")])
    news_sel = take_chars(news.sample(frac=1, random_state=args.seed), news_chars * margin).assign(pick="random")
    cand = pd.concat([wiki.assign(pick="all"), fw_sel, news_sel], ignore_index=True)
    print(f"candidates: {len(cand):,} docs, {cand['n_chars'].sum() / 1e6:.0f}M chars", flush=True)

    with Pool(args.workers) as pool:
        sigs = pool.map(minhash, cand["text"].tolist(), chunksize=256)
    lsh = MinHashLSH(threshold=0.8, num_perm=NUM_PERM)
    keep = np.zeros(len(cand), dtype=bool)
    for i, m in enumerate(sigs):  # cand is ordered wikipedia, fineweb2, nsina = priority order
        if not lsh.query(m):
            lsh.insert(str(i), m)
            keep[i] = True
    removed = cand[~keep].groupby("source").size().to_dict()
    cand = cand[keep]
    parts = {
        "wikipedia": cand[cand.source == "wikipedia"],
        "fineweb2": pd.concat([take_chars(cand[(cand.source == "fineweb2") & (cand.pick == p)], s * fw_chars)
                               for p, s in (("top_sl_score", 0.7), ("random", 0.3))]),
        "nsina": take_chars(cand[cand.source == "nsina"], news_chars),
    }

    # English: FineWeb-Edu sample
    sha = HfApi().dataset_info(ENGLISH[0]).sha
    path = hf_hub_download(ENGLISH[0], ENGLISH[1], repo_type="dataset", revision=sha,
                           local_dir=ROOT / "data" / "raw" / "corpus" / "fineweb-edu")
    en = pd.read_parquet(path, columns=["id", "url", "text"]).sample(frac=1, random_state=args.seed)
    en["n_chars"] = en["text"].str.len()
    en = take_chars(en, args.target_tokens * args.english_share / TOK_PER_CHAR["en"])
    parts["english"] = en.assign(source="fineweb_edu", sl_score=0.0, pick="random")

    stats = {"target_tokens": args.target_tokens, "near_duplicates_removed": removed,
             "english_revision": f"{ENGLISH[0]}@{sha}", "parts": {}}
    for name, df in parts.items():
        lang = "en" if name == "english" else "si"
        df[["id", "source", "url", "text", "n_chars", "sl_score", "pick"]].to_parquet(out / f"{name}.parquet", index=False)
        stats["parts"][name] = {"docs": len(df), "chars": int(df["n_chars"].sum()),
                                "est_gemma_tokens": int(df["n_chars"].sum() * TOK_PER_CHAR[lang]),
                                "picks": df["pick"].value_counts().to_dict()}
    stats["total_est_gemma_tokens"] = sum(p["est_gemma_tokens"] for p in stats["parts"].values())
    (ROOT / "reports" / f"{args.name}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # Human review sample (stays under data/, git-ignored)
    r = random.Random(args.seed)
    lines = [f"# {args.name}: 30 random Sinhala documents for review", "",
             "Check: is it real Sinhala prose? boilerplate/menus? machine-translated junk? exam material?", ""]
    for name, k in (("wikipedia", 8), ("fineweb2", 14), ("nsina", 8)):
        df = parts[name]
        for i in r.sample(range(len(df)), k):
            d = df.iloc[i]
            lines += [f"## {name} / {d['pick']} / sl_score {d['sl_score']} / {d['url']}", "", d["text"][:1200], ""]
    (out / "sample_for_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"review sample -> {out / 'sample_for_review.md'}")


if __name__ == "__main__":
    main()
