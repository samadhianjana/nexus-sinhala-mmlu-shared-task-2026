"""Synthetic Sinhala MCQs from passages with an open teacher model (data generation only, never used at inference).

For each passage the teacher writes k exam-style Sinhala MCQs (JSON). Every question is then
  1. validated: 4 distinct non-empty options, key 1-4, mostly Sinhala, does not refer to "the passage";
  2. round-trip checked: the teacher re-answers it WITH the passage (logit scoring) and must agree with its own key;
  3. soft-labelled: the teacher's closed-book option probabilities (same prompt as the student sees) are stored.
Kept questions go to <out>.jsonl in the processed MCQ schema (+ teacher_probs, passage id, subject); rejected ones to
<out>.rejected.jsonl with the reason; 50 random kept questions to <out>.review.md for a native-speaker check.

  python scripts/generate_mcq.py --passages data/synth/passages_pilot.jsonl --out data/synth/pilot \
      --teacher google/gemma-4-31B-it --per-passage 3
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, cap_gpu_memory, load_env, read_jsonl, set_seed, write_jsonl  # noqa: E402

PROMPT = """You are writing exam questions for Sri Lankan school students.
Subject: {subject}. Level: {level}.

Read the Sinhala passage below. Write {k} multiple-choice questions IN SINHALA that test important facts or concepts
stated in the passage, in the style of Sri Lankan government exam papers (term tests, O/L, A/L).

Rules:
- Each question must make sense on its own, without the passage. Never write "according to the passage" or refer to
  "the passage", "the text" or "the article".
- Exactly 4 options and exactly one correct answer, which must be supported by the passage.
- Wrong options must be plausible, on the same topic, and similar in length to the correct one.
- Vary the format: direct questions, sentence completion ("...... වේ."), and when possible one question asking which
  statement is NOT correct.
- Use standard Sinhala school-curriculum terminology. No English words unless the term is normally written in English.

Return only a JSON list, nothing else:
[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": <number 1-4>}}]

Passage:
{passage}"""

REFERS_TO_PASSAGE = re.compile(r"ඡේද|ඡේදය|පාඨය|ලිපිය|ඉහත සඳහන්|ඉහත කොටස|passage|text|article", re.I)
SINHALA = re.compile(r"[඀-෿]")
LETTER = re.compile(r"[^\W\d_]")


def parse(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def validate(q: dict) -> str | None:
    opts = q.get("options")
    if not isinstance(q.get("question"), str) or not isinstance(opts, list) or len(opts) != 4:
        return "format"
    opts = [str(o).strip() for o in opts]
    if any(not o for o in opts) or len(set(opts)) < 4:
        return "options"
    try:
        a = int(q.get("answer"))
    except (TypeError, ValueError):
        return "answer"
    if not 1 <= a <= 4:
        return "answer"
    text = q["question"] + " " + " ".join(opts)
    if len(q["question"].strip()) < 10:
        return "too_short"
    if REFERS_TO_PASSAGE.search(q["question"]):
        return "refers_to_passage"
    letters = LETTER.findall(text)
    if not letters or len(SINHALA.findall(text)) / len(letters) < 0.5:
        return "not_sinhala"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--teacher", default="google/gemma-4-31B-it")
    ap.add_argument("--per-passage", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    load_env()
    set_seed(args.seed)
    cap_gpu_memory()
    from eval.prompts import PromptBuilder, user_text
    from eval.scorers import Scorer, load_model

    passages = read_jsonl(args.passages)
    model, tok = load_model(args.teacher)
    t0 = time.time()

    # 1. generation (greedy, thinking off)
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(
        subject=p["subject"], level=p["level"], k=args.per_passage, passage=p["text"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in passages]
    gen = Scorer(model, tok, add_special_tokens=False, token_budget=10**9, max_batch=args.gen_batch,
                 max_new_tokens=args.max_new_tokens)
    raw = gen.generate(prompts, [4] * len(prompts))
    print(f"generated for {len(passages)} passages in {(time.time() - t0) / 60:.1f} min", flush=True)

    # 2. validation
    items, rejected = [], []
    for p, r in zip(passages, raw):
        qs = parse(r["text"])
        if not qs:
            rejected.append({"pid": p["pid"], "reason": "unparseable", "text": r["text"][:500]})
        for j, q in enumerate(qs[: args.per_passage]):
            why = validate(q)
            if why:
                rejected.append({"pid": p["pid"], "reason": why, "q": q})
                continue
            items.append({"uid": f"synth/{p['pid']}/{j}", "dataset": "synthetic", "q_no": j, "subject": p["subject"],
                          "category": "synthetic", "level": p["level"], "question": q["question"].strip(),
                          "options": [str(o).strip() for o in q["options"]], "answer": int(q["answer"]),
                          "meta": {"pid": p["pid"], "source": p["source"], "teacher": args.teacher}})

    # 3. round trip (open book) and 4. soft labels (closed book), both logit scoring with the teacher
    by_pid = {p["pid"]: p for p in passages}
    pb = PromptBuilder(tok, lang="en", chat=True)
    scorer = Scorer(model, tok, add_special_tokens=False, token_budget=16384, max_batch=32)
    open_prompts = [tok.apply_chat_template([{"role": "user", "content":
                    f"{by_pid[it['meta']['pid']]['text']}\n\n{user_text(it, it['options'], 'en')}"}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False) for it in items]
    ob = scorer.logit(open_prompts, [4] * len(items))
    cb = scorer.logit([pb.mcq(it, it["options"]) for it in items], [4] * len(items))
    kept = []
    for it, o, c in zip(items, ob, cb):
        it["meta"] |= {"open_book_pred": o["pred"], "open_book_conf": round(max(o["probs"]), 4),
                       "closed_book_pred": c["pred"]}
        if o["pred"] != it["answer"]:
            rejected.append({"pid": it["meta"]["pid"], "reason": "round_trip_disagrees", "q": it})
            continue
        kept.append(it | {"teacher_probs": [round(x, 5) for x in c["probs"]]})

    write_jsonl(f"{args.out}.jsonl", kept)
    write_jsonl(f"{args.out}.rejected.jsonl", rejected)
    stats = {"passages": len(passages), "generated_valid": len(items), "kept": len(kept),
             "rejected": dict(Counter(r["reason"] for r in rejected)),
             "answer_positions": dict(Counter(it["answer"] for it in kept)),
             "teacher_closed_book_agrees_with_key": round(sum(it["meta"]["closed_book_pred"] == it["answer"] for it in kept) / max(len(kept), 1), 3),
             "minutes": round((time.time() - t0) / 60, 1)}
    Path(f"{args.out}.stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    rng = random.Random(args.seed)
    lines = [f"# Review: {min(50, len(kept))} random synthetic MCQs ({args.teacher})", "",
             "For each question mark: OK / wrong answer / unnatural Sinhala / ambiguous / not exam-like.", ""]
    for n, it in enumerate(rng.sample(kept, min(50, len(kept))), 1):
        p = by_pid[it["meta"]["pid"]]
        lines += [f"## {n}. {it['subject']} ({it['level']})", "", f"**Q:** {it['question']}", ""]
        lines += [f"{k}. {o}{'  ✅' if k == it['answer'] else ''}" for k, o in enumerate(it["options"], 1)]
        lines += ["", f"<details><summary>passage</summary>\n\n{p['text'][:800]}\n\n</details>", ""]
    Path(f"{args.out}.review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"review file -> {args.out}.review.md")


if __name__ == "__main__":
    main()
