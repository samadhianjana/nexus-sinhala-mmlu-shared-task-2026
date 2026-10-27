"""Rejection-sampled reasoning traces from the base model itself (STaR-style), for SinhalaMMLU dev questions.

For every question the base model answers with thinking mode and temperature sampling; traces whose final answer equals
the gold key are kept. Output: <out>.all.jsonl (every sample) and <out>.jsonl (correct ones, one record per kept sample:
the question item + "completion" = raw model output including the thought channel and the end-of-turn token).

Time-boxed and resumable: samples are generated in rounds (all questions once, then again), results are appended after
every chunk, and the run stops when --max-minutes is reached.

  python scripts/gen_traces.py --data smmlu_dev_rest1450 --out data/synth/rft_traces --samples 2 --max-minutes 240
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import cap_gpu_memory, load_env, read_jsonl, set_seed  # noqa: E402
from eval.run_eval import resolve_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="models/gemma-4-E4B-it-text")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-batch", type=int, default=64, help="sequences per generate call (lower it on a smaller GPU)")
    ap.add_argument("--max-minutes", type=float, default=240)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    load_env()
    set_seed(args.seed)
    cap_gpu_memory()
    from eval.prompts import PromptBuilder
    from eval.scorers import Scorer, load_model

    items = read_jsonl(resolve_data(args.data))[: args.limit]
    all_path, ok_path = Path(f"{args.out}.all.jsonl"), Path(f"{args.out}.jsonl")
    all_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["uid"], r["k"]) for r in read_jsonl(all_path)} if all_path.exists() else set()
    jobs = [(it, k) for k in range(args.samples) for it in items if (it["uid"], k) not in done]
    print(f"{len(items)} questions x {args.samples} samples; {len(done)} already done, {len(jobs)} to go", flush=True)

    model, tok = load_model(args.model)
    pb = PromptBuilder(tok, lang="en", chat=True, chat_kwargs={"enable_thinking": True})
    scorer = Scorer(model, tok, pb.add_special_tokens, token_budget=32768, max_batch=args.max_batch, max_new_tokens=args.max_new_tokens)
    scorer.keep_raw = True
    scorer.sampling = {"do_sample": True, "temperature": args.temperature, "top_p": 0.95, "top_k": 64}

    t0 = time.time()
    n_ok = n_all = 0
    for i in range(0, len(jobs), args.chunk):
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"time box of {args.max_minutes} min reached after {n_all} samples", flush=True)
            break
        chunk = jobs[i:i + args.chunk]
        res = scorer.generate([pb.mcq(it, it["options"]) for it, _ in chunk], [len(it["options"]) for it, _ in chunk])
        with open(all_path, "a", encoding="utf-8") as fa, open(ok_path, "a", encoding="utf-8") as fo:
            for (it, k), r in zip(chunk, res):
                rec = {"uid": it["uid"], "k": k, "pred": r["pred"], "gold": it["answer"], "new_tokens": r["new_tokens"],
                       "correct": r["pred"] == it["answer"]}
                fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_all += 1
                if rec["correct"] and r["raw"].endswith("<turn|>") and "<|channel>" in r["raw"]:
                    fo.write(json.dumps(it | {"completion": r["raw"], "k": k}, ensure_ascii=False) + "\n")
                    n_ok += 1
        print(f"{n_all}/{len(jobs)} samples, {n_ok} kept ({100 * n_ok / n_all:.0f}%), {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"done: {n_all} samples, {n_ok} kept traces -> {ok_path}", flush=True)


if __name__ == "__main__":
    main()
