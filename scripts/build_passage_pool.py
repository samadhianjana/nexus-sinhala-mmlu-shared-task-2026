"""Passage pool for synthetic MCQ generation: Sinhala passages assigned to curriculum subjects.

Subjects are the published SinhalaMMLU subject list (Pramodya et al. 2025, Table 2), never test questions. Passages come
from the cleaned pretraining mix (data/corpus/cpt_v1: Wikipedia, FineWeb-2, NSINA; exam sites already removed).
Documents are split into 400-1200 character chunks; a chunk is assigned to the subject whose keyword list it matches
most (at least 2 distinct keywords). Wikipedia chunks are preferred.

  python scripts/build_passage_pool.py --per-subject 6 --out data/synth/passages_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, write_jsonl  # noqa: E402

# subject -> (level description used in the generation prompt, Sinhala keywords)
SUBJECTS = {
    "History": ("grades 6-11", ["ඉතිහාස", "රජතුමා", "රාජධානි", "අනුරාධපුර", "පොළොන්නරු", "මහාවංශ", "වංශකථා", "ශිලා ලිපි", "යටත් විජිත", "පෘතුගීසි", "ලන්දේසි", "නිදහස් අරගල"]),
    "Buddhism": ("grades 6-11", ["බුදුරජාණන්", "ධර්මය", "භික්ෂු", "සංඝ", "සූත්‍ර", "ත්‍රිපිටක", "නිර්වාණ", "පන්සිල්", "භාවනා", "ජාතක"]),
    "Buddhist Civilization": ("grades 12-13 (A/L)", ["බෞද්ධ ශිෂ්ටාචාර", "ස්තූප", "දාගැබ", "චෛත්‍ය", "ආරාම", "ධාතු", "අභයගිරි", "මහාවිහාර", "ජේතවන"]),
    "Catholicism": ("grades 6-11", ["කතෝලික", "පාප්වහන්සේ", "සාන්තුවර", "දිව්‍ය පූජා", "කන්‍යා මරියා", "රදගුරු"]),
    "Christianity": ("grades 6-11", ["ක්‍රිස්තියානි", "යේසුස්", "බයිබලය", "ශුද්ධ", "ගිවිසුම", "දේවස්ථාන", "ගෝලයන්"]),
    "Islam": ("grades 6-11", ["ඉස්ලාම්", "අල්ලාහ්", "නබි", "කුර්ආන්", "මුස්ලිම්", "රාමසාන්", "හජ්", "මස්ජිද්"]),
    "Dancing": ("grades 6-13", ["නර්තන", "නැටුම", "උඩරට", "පහතරට", "සබරගමු", "වන්නම", "ගැටබෙර", "වෙස්", "තාලම්"]),
    "Eastern Music": ("grades 6-11", ["සංගීත", "රාග", "තාල", "ස්වර", "වාදන", "සිතාර්", "තබ්ලා", "ගායන", "හින්දුස්ථානි"]),
    "Drama and Theatre": ("grades 6-11", ["නාට්‍ය", "රංගන", "නාඩගම්", "කෝලම්", "සොකරි", "වේදිකා", "සරච්චන්ද්‍ර", "නළු"]),
    "Arts": ("grades 6-8", ["චිත්‍ර", "කලාව", "මූර්ති", "බිතු සිතුවම්", "සීගිරි", "වර්ණ", "කලාකරු"]),
    "Sinhala Language and Literature": ("grades 6-13", ["කාව්‍ය", "සාහිත්‍ය", "නවකතා", "කවි", "ව්‍යාකරණ", "සන්දේශ", "කෙටිකතා", "ලේඛක"]),
    "Citizenship Education": ("grades 6-11", ["ආණ්ඩුක්‍රම", "පාර්ලිමේන්තු", "ප්‍රජාතන්ත්‍ර", "මූලික අයිතිවාසිකම්", "ඡන්ද", "පළාත් සභා", "පුරවැසි"]),
    "Political Science": ("grades 12-13 (A/L)", ["දේශපාලන", "රාජ්‍යය", "ව්‍යවස්ථාව", "පරමාධිපත්‍ය", "ඒකීය", "ෆෙඩරල්", "විධායක"]),
    "Geography": ("grades 6-13", ["භූගෝල", "දේශගුණ", "මෝසම්", "ගංගා", "කඳු", "වර්ෂාපතන", "සාගර", "පාංශු", "ජනගහන"]),
    "Health and Physical Education": ("grades 6-11", ["සෞඛ්‍ය", "පෝෂණ", "රෝග", "ව්‍යායාම", "ක්‍රීඩා", "විටමින්", "ප්‍රතිශක්ති"]),
    "Science": ("grades 6-11", ["විද්‍යාව", "පදාර්ථ", "ශාක", "සත්ත්ව", "විද්‍යුත්", "චුම්බක", "ආලෝකය", "පරිසර"]),
    "Physics": ("grades 12-13 (A/L)", ["භෞතික", "බලය", "ප්‍රවේග", "ත්වරණ", "තරංග", "විද්‍යුත් ධාරා", "ගුරුත්වාකර්ෂණ", "ශක්තිය"]),
    "Chemistry": ("grades 12-13 (A/L)", ["රසායන", "මූලද්‍රව්‍ය", "සංයෝග", "අණු", "පරමාණු", "අම්ල", "භෂ්ම", "ප්‍රතික්‍රියා"]),
    "Biology": ("grades 12-13 (A/L)", ["ජීව විද්‍යා", "සෛල", "ජාන", "ප්‍රෝටීන", "එන්සයිම", "පරිණාම", "ප්‍රභාසංස්ලේෂණ", "ජීවීන්"]),
    "Economics": ("grades 12-13 (A/L)", ["ආර්ථික", "ඉල්ලුම", "සැපයුම", "උද්ධමන", "දළ දේශීය", "බදු", "වෙළඳපොළ", "අපනයන"]),
    "Business and Accounting Studies": ("grades 12-13 (A/L)", ["ව්‍යාපාර", "ගිණුම්", "ලාභ", "වත්කම්", "බැංකු", "කොටස්", "සමාගම", "ව්‍යවසාය"]),
    "Agriculture and Food Technology": ("grades 9-13", ["කෘෂිකර්ම", "වගාව", "පොහොර", "බෝග", "කුඹුරු", "වී ", "පළිබෝධ", "ජලසම්පාදන"]),
    "Home Economics": ("grades 9-13", ["ආහාර", "පෝෂණ", "ඇඳුම්", "නිවාස", "ගෘහ", "පවුල", "ආහාර පිසීම"]),
    "Communication and Media Studies": ("grades 9-13", ["මාධ්‍ය", "ජනමාධ්‍ය", "පුවත්පත", "රූපවාහිනී", "ගුවන්විදුලි", "සන්නිවේදන"]),
    "Design and Construction Technology": ("grades 9-11", ["ඉදිකිරීම්", "ගොඩනැගිලි", "කොන්ක්‍රීට්", "සැලසුම්", "ඉංජිනේරු", "ද්‍රව්‍ය"]),
}


def chunks(text: str, lo: int = 400, hi: int = 1200):
    cur = ""
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 1 <= hi:
            cur = f"{cur}\n{para}" if cur else para
        else:
            if len(cur) >= lo:
                yield cur
            cur = para[:hi]
    if len(cur) >= lo:
        yield cur


def assign(chunk: str) -> tuple[str | None, int]:
    best, hits = None, 0
    for subj, (_, kws) in SUBJECTS.items():
        h = sum(kw in chunk for kw in kws)
        if h > hits:
            best, hits = subj, h
    return (best, hits) if hits >= 2 else (None, hits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth" / "passages_pilot.jsonl"))
    ap.add_argument("--exclude", help="earlier pool file: its passages are not reused")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    import pandas as pd

    rng = random.Random(args.seed)
    used = set()
    if args.exclude and Path(args.exclude).exists():
        used = {json.loads(l)["pid"] for l in open(args.exclude, encoding="utf-8")}
    by_subject: dict[str, list[dict]] = {s: [] for s in SUBJECTS}
    for source in ("wikipedia", "fineweb2", "nsina"):
        df = pd.read_parquet(ROOT / "data" / "corpus" / "cpt_v1" / f"{source}.parquet", columns=["id", "text"])
        docs = df.sample(frac=1, random_state=args.seed).itertuples()
        for d in docs:
            for k, c in enumerate(chunks(d.text)):
                pid = f"{d.id}#{k}"
                if pid in used:
                    continue
                subj, hits = assign(c)
                if subj and len(by_subject[subj]) < 50 * args.per_subject:
                    by_subject[subj].append({"pid": pid, "source": source, "subject": subj,
                                             "level": SUBJECTS[subj][0], "keyword_hits": hits, "text": c})
            if all(len(v) >= 50 * args.per_subject for v in by_subject.values()):
                break
    out = []
    for subj, cands in by_subject.items():
        cands.sort(key=lambda r: (r["source"] != "wikipedia", -r["keyword_hits"], rng.random()))
        out += cands[: args.per_subject]
    write_jsonl(args.out, out)
    print(f"{len(out)} passages -> {args.out}")
    print(dict(Counter(r["subject"] for r in out)))
    print("by source:", dict(Counter(r["source"] for r in out)))


if __name__ == "__main__":
    main()
