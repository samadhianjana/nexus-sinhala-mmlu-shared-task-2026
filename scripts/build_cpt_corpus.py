"""Sinhala continued-pretraining corpus -> data/corpus/clean/*.parquet (+ aggregate stats in reports/).

Sources (downloaded to data/raw/corpus by this script, pinned revisions in DATA_LOG.md):
  fineweb2   HuggingFaceFW/fineweb-2 sin_Sinh (ODC-By; already MinHash-deduplicated upstream)
  wikipedia  wikimedia/wikipedia 20231101.si (CC BY-SA 3.0)
  nsina      sinhala-nlp/NSINA news (CC BY-SA 4.0)

Per document, in order (first failing rule is recorded):
  1. blocked_domain   exam / past-paper / curriculum sites (SinhalaMMLU's sources and anything like them)
  2. exam_like        question-paper text: "ප්‍රශ්න පත්‍ර", "වාර විභාග", "past paper" ... or many numbered option lines
  3. too_short        < 200 characters
  4. not_sinhala      < 50% of letters in the Sinhala block
  5. repetitive       < 30% unique lines (boilerplate, menus)
  6. duplicate        same normalised text already kept (exact, across sources; FineWeb-2 is near-deduped upstream)
  7. contaminated     shares a word 8-gram with any SinhalaMMLU dev or Global-MMLU validation question
Kept documents get sl_score = Sri Lanka / curriculum-topic keyword hits per 1k characters (used to prioritise).

Stages are resumable: finished shards are skipped on rerun.
Usage: python scripts/build_cpt_corpus.py [--limit N] [--workers 12] [--out data/corpus]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, load_env, read_jsonl  # noqa: E402

SOURCES = {
    "fineweb2": ("HuggingFaceFW/fineweb-2", "data/sin_Sinh/train/000_00000.parquet"),
    "wikipedia": ("wikimedia/wikipedia", "20231101.si/train-00000-of-00001.parquet"),
    "nsina": ("sinhala-nlp/NSINA", "NSINA.json"),
}
BLOCKED_DOMAIN = re.compile(
    r"pastpaper|past-paper|past_paper|thaksalawa|doenets|exam|paperbank|papersbank|model-?paper|term-?test|"
    r"nie\.lk|edupub|e-?learning|tutor|guru\.lk|alevel|a-level|olevel|o-level|scholarship", re.I)
EXAM_PHRASES = re.compile(
    r"ප්‍රශ්න\s*පත්‍ර|වාර\s*විභාග|පසුගිය\s*විභාග|ආදර්ශ\s*ප්‍රශ්න|පෙරහුරු\s*ප්‍රශ්න|"
    r"past\s*papers?|model\s*papers?|term\s*test|marking\s*scheme|ලකුණු\s*දීමේ\s*පටිපාටිය", re.I)
OPTION_LINE = re.compile(r"^\s*(\(?[1-5][\).]|\(?[iv]{1,3}\)|[අආඇඉ]\))\s*\S", re.M)
SL_TOPICS = re.compile(
    r"ශ්‍රී\s*ලංකා|ලංකාව|සිංහල|බුදු|බෞද්ධ|ධර්ම|විහාර|රජතුමා|රාජධානි|අනුරාධපුර|පොළොන්නරු|මහනුවර|ඉතිහාස|"
    r"පුරාවිද්‍යා|සාහිත්‍ය|කාව්‍ය|නර්තන|නැටුම|සංගීත|රාග|චිත්‍ර|කලාව|නාට්‍ය|කතෝලික|ක්‍රිස්තියානි|ඉස්ලාම්|"
    r"භූගෝල|පුරවැසි|ආණ්ඩුක්‍රම|සෞඛ්‍ය|විද්‍යාව")
SINHALA = re.compile(r"[඀-෿]")
LETTER = re.compile(r"[^\W\d_]")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[‌‍]", "", text)
    # keep the whole Sinhala block: vowel signs and virama are combining marks, which \w does not match
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\u0D80-\u0DFF]", " ", text)).strip()


def word_grams(text: str, n: int = 8):
    w = norm(text).split()
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


_CONTAM: set[int] = set()


def _init(contam: set[int]) -> None:
    global _CONTAM
    _CONTAM = contam


def check(doc: dict) -> tuple[str, dict]:
    """Return (reason or 'kept', doc with stats). Duplicate check happens in the parent (needs global state)."""
    text, url = doc["text"], doc.get("url") or ""
    if url and BLOCKED_DOMAIN.search(url.split("/")[2] if "://" in url else url):
        return "blocked_domain", doc
    if EXAM_PHRASES.search(text[:5000]) or len(OPTION_LINE.findall(text)) >= 8:
        return "exam_like", doc
    if len(text) < 200:
        return "too_short", doc
    letters = LETTER.findall(text)
    if not letters or len(SINHALA.findall(text)) / len(letters) < 0.5:
        return "not_sinhala", doc
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.3:
        return "repetitive", doc
    if _CONTAM and word_grams(text) & _CONTAM:
        return "contaminated", doc
    doc["n_chars"] = len(text)
    doc["sl_score"] = round(1000 * len(SL_TOPICS.findall(text)) / len(text), 3)
    doc["hash"] = hashlib.sha1(norm(text).encode()).hexdigest()
    return "kept", doc


def iter_source(name: str, path: Path, limit: int | None, batch: int = 20000):
    if name == "nsina":
        rows = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for i, r in enumerate(rows[:limit] if limit else rows):
            text = "\n".join(x for x in (r.get("Headline"), r.get("News Content")) if x)
            out.append({"id": f"nsina/{i}", "source": name, "url": r.get("Source") or "", "text": text})
            if len(out) == batch:
                yield out
                out = []
        if out:
            yield out
        return
    import pyarrow.parquet as pq

    pf, n = pq.ParquetFile(path), 0
    for rb in pf.iter_batches(batch_size=batch, columns=["id", "url", "text"] + (["title"] if name == "wikipedia" else [])):
        out = []
        for r in rb.to_pylist():
            text = f"{r['title']}\n{r['text']}" if name == "wikipedia" else r["text"]
            out.append({"id": f"{name}/{r['id']}", "source": name, "url": r.get("url") or "", "text": text})
        if limit:
            out = out[: max(limit - n, 0)]
        n += len(out)
        if out:
            yield out
        if limit and n >= limit:
            return


def contamination_grams() -> set[int]:
    grams = set()
    for f in ("sinhalammlu_dev", "gmmlu_si_val"):
        for r in read_jsonl(ROOT / "data" / "processed" / f"{f}.jsonl"):
            grams |= word_grams(r["question"])
            grams |= word_grams(r["question"] + " " + r["options"][r["answer"] - 1])
    return grams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="documents per source (smoke test)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "data" / "corpus"))
    args = ap.parse_args()
    load_env()
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(args.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revisions = {}
    for name, (repo, f) in SOURCES.items():
        sha = api.dataset_info(repo).sha
        revisions[name] = f"{repo}@{sha}"
        hf_hub_download(repo, f, repo_type="dataset", revision=sha,
                        local_dir=ROOT / "data" / "raw" / "corpus" / repo.split("/")[1])
    contam = contamination_grams()
    print(f"contamination 8-grams: {len(contam):,}", flush=True)

    seen_hashes: set[str] = set()
    stats = {"revisions": revisions, "sources": {}}
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(contam,)) as pool:
        for name, (repo, f) in SOURCES.items():
            path = ROOT / "data" / "raw" / "corpus" / repo.split("/")[1] / f
            reasons, kept_chars, sl = Counter(), 0, []
            for shard, docs in enumerate(iter_source(name, path, args.limit)):
                shard_file = out / "clean" / f"{name}-{shard:04d}.parquet"
                kept = []
                for reason, doc in pool.imap(check, docs, chunksize=64):
                    if reason == "kept":
                        if doc["hash"] in seen_hashes:
                            reason = "duplicate"
                        else:
                            seen_hashes.add(doc["hash"])
                            kept.append(doc)
                    reasons[reason] += 1
                if not shard_file.exists() and kept:
                    pd.DataFrame(kept).drop(columns=["hash"]).to_parquet(shard_file, index=False)
                kept_chars += sum(d["n_chars"] for d in kept)
                sl += [d["sl_score"] for d in kept]
                print(f"{name} shard {shard}: {sum(reasons.values()):,} docs seen, {reasons['kept']:,} kept "
                      f"({time.time() - t0:.0f}s)", flush=True)
            stats["sources"][name] = {
                "docs_seen": sum(reasons.values()), "reasons": dict(reasons), "kept_chars": kept_chars,
                "est_gemma_tokens": int(kept_chars * 0.40),  # 0.40 Gemma tokens / Sinhala char (baseline runs)
                "sl_score_quantiles": {q: float(pd.Series(sl).quantile(q / 100)) for q in (50, 90, 99)} if sl else {},
            }
    stats["total_est_gemma_tokens"] = sum(s["est_gemma_tokens"] for s in stats["sources"].values())
    stats["seconds"] = round(time.time() - t0)
    name = "corpus_stats_smoke.json" if args.limit else "corpus_stats.json"
    (ROOT / "reports" / name).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
