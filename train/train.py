"""Continued pretraining (mode: cpt) and MCQ fine-tuning (mode: sft) of the submission model.

  python -m train.train --config configs/train/sft_gmmlu.yaml [--max-steps N] [--out models/x]

cpt  documents from <data_dir>/*.parquet are shuffled, tokenised once (<bos> text <eos>) into a uint32 token file and
     cut into fixed seq_len sequences (fixed shapes keep torch.compile to one graph).
sft  each MCQ is rendered with the *inference* prompt (eval/prompts.py: same template, chat wrapping) and the loss is
     the cross-entropy of the answer digit at the last prompt position, which is exactly what logit scoring reads.
     Options can be re-shuffled every epoch (except options that refer to other options) to avoid position priors.

Speed/memory options come from train/efficiency.py (measured in reports/throughput.jsonl). The optimizer is torchao
AdamW8bit with bf16 stochastic rounding (weights stay bf16; small updates are not rounded away).
Checkpoints are complete, loadable model directories (weights, tokenizer, chat template, SOURCE.json, TRAINING.json).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, cap_gpu_memory, env_fingerprint, load_env, read_jsonl, set_seed  # noqa: E402


# ---------------------------------------------------------------- data
def prepare_cpt_tokens(cfg: dict, tok) -> np.memmap:
    import pandas as pd

    data_dir = ROOT / cfg["data_dir"]
    suffix = f"_limit{cfg['limit_docs']}" if cfg.get("limit_docs") else ""
    path = data_dir / f"tokens_{Path(cfg['model']).name}{suffix}.u32"
    if not path.exists():
        docs = pd.concat([pd.read_parquet(f, columns=["text"]) for f in sorted(data_dir.glob("*.parquet"))],
                         ignore_index=True)["text"].tolist()
        random.Random(cfg["seed"]).shuffle(docs)
        docs = docs[: cfg["limit_docs"]] if cfg.get("limit_docs") else docs
        tmp = path.with_suffix(".tmp")
        n = 0
        with open(tmp, "wb") as f:
            for i in range(0, len(docs), 2000):
                for ids in tok(docs[i:i + 2000], add_special_tokens=False)["input_ids"]:
                    arr = np.array([tok.bos_token_id] + ids + [tok.eos_token_id], dtype=np.uint32)
                    f.write(arr.tobytes())
                    n += len(arr)
                print(f"tokenised {min(i + 2000, len(docs)):,}/{len(docs):,} docs, {n / 1e6:.1f}M tokens", flush=True)
        tmp.rename(path)
    return np.memmap(path, dtype=np.uint32, mode="r")


def cpt_batches(tokens: np.memmap, seq_len: int, micro_batch: int):
    import torch

    n_seq = len(tokens) // seq_len
    i = 0
    while True:  # cycles if the run asks for more tokens than the file holds (epoch > 1)
        rows = [(i + k) % n_seq for k in range(micro_batch)]
        i += micro_batch
        x = np.stack([tokens[r * seq_len:(r + 1) * seq_len] for r in rows]).astype(np.int64)
        yield {"input_ids": torch.from_numpy(x)}


def sft_examples(cfg: dict, tok, epoch: int) -> list[dict]:
    from eval.prompts import PromptBuilder
    from scripts.build_sft_gmmlu import REFERS_TO_OTHERS

    pb = PromptBuilder(tok, lang=cfg.get("lang", "en"), chat=cfg.get("chat", True))
    rng = random.Random(f"{cfg['seed']}|{epoch}")
    label_ids = {}
    out = []
    for f in cfg["data"]:
        for r in read_jsonl(ROOT / f):
            opts, ans = list(r["options"]), r["answer"]
            tprobs = r.get("teacher_probs")  # teacher's option distribution, in the stored option order
            order = list(range(len(opts)))
            if cfg.get("permute_options", True) and not any(REFERS_TO_OTHERS.search(o) for o in opts):
                rng.shuffle(order)
            opts, ans = [opts[j] for j in order], order.index(ans - 1) + 1
            tprobs = [tprobs[j] for j in order] if tprobs else None
            ids = tok(pb.mcq(r, opts), add_special_tokens=pb.add_special_tokens)["input_ids"]
            n = len(opts)
            if n not in label_ids:
                label_ids[n] = [tok.encode(str(k), add_special_tokens=False) for k in range(1, n + 1)]
                assert all(len(t) == 1 for t in label_ids[n]), "answer labels must be single tokens"
            out.append({"ids": ids, "target": label_ids[n][ans - 1][0], "labels": [t[0] for t in label_ids[n]],
                        "tprobs": tprobs})
    rng.shuffle(out)
    return out


def sft_steps(examples: list[dict], batch_size: int, micro_tokens: int, pad_id: int):
    """Yield optimizer steps; each is a list of right-padded micro-batches (token budget), batch_size examples."""
    import torch

    for s in range(0, len(examples) - batch_size + 1, batch_size):
        chunk = sorted(examples[s:s + batch_size], key=lambda e: len(e["ids"]))
        micros, cur = [], []
        for e in chunk:
            if cur and max(len(x["ids"]) for x in cur + [e]) * (len(cur) + 1) > micro_tokens:
                micros.append(cur)
                cur = []
            cur.append(e)
        micros.append(cur)
        batches = []
        for m in micros:
            L = max(len(e["ids"]) for e in m)
            ids = torch.full((len(m), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(m), L), dtype=torch.long)
            for i, e in enumerate(m):
                ids[i, : len(e["ids"])] = torch.tensor(e["ids"])
                mask[i, : len(e["ids"])] = 1
            b = {"input_ids": ids, "attention_mask": mask,
                 "last": torch.tensor([len(e["ids"]) - 1 for e in m]),
                 "target": torch.tensor([e["target"] for e in m])}
            if all(e["tprobs"] is not None for e in m) and len({len(e["labels"]) for e in m}) == 1:
                b["labels"] = torch.tensor([e["labels"] for e in m])
                b["tprobs"] = torch.tensor([e["tprobs"] for e in m], dtype=torch.float32)
            batches.append(b)
        yield batches


def sft_loss(model, batch, batch_size: int, kd_alpha: float = 0.0):
    """Cross-entropy of the answer digit; with kd_alpha > 0 and teacher probabilities present, mixed with the KL
    divergence from the teacher's distribution over the option labels (knowledge distillation)."""
    import torch
    import torch.nn.functional as F

    from train.efficiency import lm_parts

    body, head, softcap = lm_parts(model)
    h = body(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    h = h[torch.arange(h.shape[0], device=h.device), batch["last"]]
    logits = head(h).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    ce = F.cross_entropy(logits, batch["target"], reduction="none")
    if kd_alpha > 0 and "tprobs" in batch:
        student = torch.log_softmax(logits.gather(1, batch["labels"]), dim=-1)
        teacher = batch["tprobs"].clamp_min(1e-8)
        teacher = teacher / teacher.sum(-1, keepdim=True)
        kl = (teacher * (teacher.log() - student)).sum(-1)
        return ((1 - kd_alpha) * ce + kd_alpha * kl).sum() / batch_size
    return ce.sum() / batch_size


# ---------------------------------------------------------------- model / saving
def load(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from train.efficiency import configure

    path = str(ROOT / cfg["model"]) if (ROOT / cfg["model"]).exists() else cfg["model"]
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    if cfg.get("lora"):  # adapters on all linear layers; merged into the base weights when saved
        from peft import LoraConfig, get_peft_model

        lc = cfg["lora"]
        model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc.get("alpha", 2 * lc["r"]),
                                                 target_modules=lc.get("targets", "all-linear"),
                                                 lora_dropout=lc.get("dropout", 0.0), task_type="CAUSAL_LM"))
    elif cfg.get("freeze_embeddings", True):
        for name, p in model.named_parameters():
            if "embed_tokens" in name:  # incl. Gemma per-layer embedding tables; lm_head is tied to embed_tokens
                p.requires_grad_(False)
    configure(model, cfg.get("ckpt_every", 1), cfg.get("compile", False))
    if cfg.get("lora") and cfg.get("ckpt_every", 1) > 0:
        model.enable_input_require_grads()
    model.train()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok, Path(path)


def save(model, tok, src: Path, out: Path, info: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "merge_and_unload"):  # LoRA: save plain merged weights (parameter count = base model)
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    model.config.use_cache = False
    tok.save_pretrained(out)
    for f in src.glob("chat_template.*"):
        shutil.copy(f, out / f.name)
    source = json.loads((src / "SOURCE.json").read_text()) if (src / "SOURCE.json").exists() else {}
    source["model_id"] = f"{info['run']}/{out.name}@{source.get('source_commit', 'unknown')[:10]}"
    source.setdefault("derived_from", []).append(str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src))
    (out / "SOURCE.json").write_text(json.dumps(source, indent=2))
    (out / "TRAINING.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"saved {out}", flush=True)


def lr_at(step: int, total: int, cfg: dict) -> float:
    warm, peak, floor = cfg["warmup_steps"], cfg["lr"], cfg["lr"] * cfg.get("min_lr_ratio", 0.1)
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(total - warm, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(t, 1.0)))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="override out_dir")
    ap.add_argument("--max-steps", type=int, help="stop early (smoke tests)")
    ap.add_argument("--limit-docs", type=int, help="cpt: tokenise only N documents (smoke tests)")
    ap.add_argument("--data", nargs="+", help="sft: override the training files")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg["out_dir"] = args.out or cfg["out_dir"]
    cfg["limit_docs"] = args.limit_docs or cfg.get("limit_docs")
    if args.data:
        cfg["data"] = args.data
    load_env()
    set_seed(cfg["seed"])
    cap_gpu_memory(cfg.get("gpu_mem_gb"))
    import torch
    from torchao.optim import AdamW8bit

    from train.efficiency import lm_loss

    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    model, tok, src = load(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW8bit(trainable, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg.get("weight_decay", 0.0),
                    bf16_stochastic_round=True)

    if cfg["mode"] == "cpt":
        tokens = prepare_cpt_tokens(cfg, tok)
        per_step = cfg["seq_len"] * cfg["micro_batch"] * cfg["grad_accum"]
        total = int(min(cfg["max_tokens"], len(tokens) * cfg.get("epochs", 1)) // per_step)
        it = cpt_batches(tokens, cfg["seq_len"], cfg["micro_batch"])
        steps = ([next(it) for _ in range(cfg["grad_accum"])] for _ in range(total))
        save_every = max(1, int(cfg.get("save_every_tokens", 1e18) // per_step))
    else:
        n_ex = len(sft_examples(cfg, tok, 0))
        per_epoch = n_ex // cfg["batch_size"]
        total = per_epoch * cfg["epochs"]
        steps = (s for e in range(cfg["epochs"])
                 for s in sft_steps(sft_examples(cfg, tok, e), cfg["batch_size"], cfg["micro_tokens"],
                                    tok.pad_token_id))
        save_every = per_epoch  # end of every epoch
    if args.max_steps:
        total = min(total, args.max_steps)
    if cfg.get("lora"):
        save_every = total
    print(f"{cfg['mode']}: {total} optimizer steps, "
          f"{sum(p.numel() for p in trainable) / 1e9:.2f}B trainable params", flush=True)

    info = {"run": Path(cfg["out_dir"]).name, "config": cfg, "env": env_fingerprint(), "tokens_seen": 0}
    log = open(out / "log.jsonl", "a")
    t0, tokens_seen = time.time(), 0
    for step, micro_batches in enumerate(steps):
        if step >= total:
            break
        for g in opt.param_groups:  # torchao keeps lr as a tensor: update in place
            lr = lr_at(step, total, cfg)
            g["lr"].fill_(lr) if torch.is_tensor(g["lr"]) else g.__setitem__("lr", lr)
        loss_sum = 0.0
        for b in micro_batches:
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            if cfg["mode"] == "cpt":
                loss = lm_loss(model, b["input_ids"], b["input_ids"], cfg.get("loss", "liger")) / len(micro_batches)
                tokens_seen += b["input_ids"].numel()
            else:
                loss = sft_loss(model, b, cfg["batch_size"], cfg.get("kd_alpha", 0.0))
                tokens_seen += int(b["attention_mask"].sum())
            loss.backward()
            loss_sum += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.get("max_grad_norm", 1.0)).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % cfg.get("log_every", 10) == 0 or step == total - 1:
            el = time.time() - t0
            rec = {"step": step, "loss": round(loss_sum, 4), "lr": float(opt.param_groups[0]["lr"]), "grad_norm": round(gnorm, 3),
                   "tokens": tokens_seen, "tok_per_s": round(tokens_seen / el, 1),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1), "elapsed_min": round(el / 60, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if (step + 1) % save_every == 0 or step == total - 1:
            info.update(step=step + 1, total_steps=total, tokens_seen=tokens_seen)
            ckpt = out / f"step-{step + 1}"
            save(model, tok, src, ckpt, info)
            keep = cfg.get("keep_checkpoints", 3)
            old = sorted(out.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))[:-keep]
            for p in old:
                shutil.rmtree(p)
    print(f"done: {tokens_seen / 1e6:.1f}M tokens in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
