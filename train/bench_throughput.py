"""Training throughput benchmark: tokens/s and peak memory for a model + training mode on this machine.

  python -m train.bench_throughput --model Qwen/Qwen3.5-4B --mode full
  python -m train.bench_throughput --model models/gemma-4-E4B-it-text --mode freeze-embed
  python -m train.bench_throughput --model models/gemma-4-E4B-it-text --mode lora

Speed options (train/efficiency.py): --loss hf|chunked|liger, --ckpt-every N (1 = all layers, 0 = none), --compile.

Modes:  full          every parameter trained
        freeze-embed  embedding tables frozen (for Gemma E-models: the 3.5B per-layer embeddings), rest trained
        lora          LoRA r=16 on all linear layers, base frozen

Random token ids are used: step time does not depend on the text. The micro-batch is searched upward until the
memory cap is hit, then timed. AdamW states are kept in the parameter dtype (bf16); the result also reports the
extra memory fp32 optimizer states would need. Appends one JSON line per run to reports/throughput.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, cap_gpu_memory, env_fingerprint, load_env, set_seed  # noqa: E402


def build(model_name: str, mode: str, lora_r: int, ckpt_every: int, compile_layers: bool):
    import torch
    from transformers import AutoModelForCausalLM

    from eval.scorers import resolve_revision

    revision, commit = resolve_revision(model_name, None)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.config.use_cache = False
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules="all-linear",
                                                 lora_dropout=0.0, task_type="CAUSAL_LM"))
    elif mode == "freeze-embed":
        for name, p in model.named_parameters():
            if "embed_tokens" in name:  # includes Gemma's per-layer embedding tables; lm_head is tied to it
                p.requires_grad_(False)
    from train.efficiency import configure

    configure(model, ckpt_every, compile_layers)
    if mode == "lora":
        model.enable_input_require_grads()
    model.train()
    return model, commit


def run_steps(model, opt, vocab: int, mb: int, seq: int, steps: int, loss_kind: str) -> float:
    import torch

    from train.efficiency import lm_loss

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        ids = torch.randint(100, vocab - 100, (mb, seq), device="cuda")
        loss = lm_loss(model, ids, ids, loss_kind)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["full", "freeze-embed", "lora"], required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-micro-batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--gpu-mem-gb", type=float)
    ap.add_argument("--loss", choices=["hf", "chunked", "liger"], default="hf")
    ap.add_argument("--ckpt-every", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    load_env()
    set_seed(1234)
    cap_gpu_memory(args.gpu_mem_gb)
    import torch

    model, commit = build(args.model, args.mode, args.lora_r, args.ckpt_every, args.compile)
    total = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    if args.optim == "adamw8bit":  # 8-bit optimizer states (torchao): ~4x less optimizer memory than fp32
        from torchao.optim import AdamW8bit

        opt = AdamW8bit(trainable, lr=1e-5)
    else:
        opt = torch.optim.AdamW(trainable, lr=1e-5, fused=True)
    vocab = model.get_input_embeddings().num_embeddings

    best = None
    mb = 1
    while mb <= args.max_micro_batch:
        try:
            run_steps(model, opt, vocab, mb, args.seq_len, 2, args.loss)  # warm-up / fit check (and compile)
            best = mb
            mb *= 2
        except torch.OutOfMemoryError:
            break
        finally:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if best is None:
        raise SystemExit("micro-batch 1 does not fit under the memory cap")

    # The failed probe can leave memory fragmented, so the largest fitting size may not fit again: back off.
    while True:
        torch.cuda.reset_peak_memory_stats()
        try:
            secs = run_steps(model, opt, vocab, best, args.seq_len, args.steps, args.loss)
            break
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if best == 1:
                raise
            best //= 2
    tok_s = best * args.seq_len * args.steps / secs
    res = {
        "model": args.model, "commit": commit, "mode": args.mode, "loss": args.loss, "ckpt_every": args.ckpt_every,
        "compile": args.compile, "optim": args.optim, "tag": args.tag, "seq_len": args.seq_len, "micro_batch": best,
        "total_params": total, "trainable_params": n_train,
        "tokens_per_s": round(tok_s, 1), "step_seconds": round(secs / args.steps, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
        "extra_gb_for_fp32_adam": round(n_train * 4 / 1024**3, 1),  # bf16 -> fp32 for both moments
        "days_per_1B_tokens": round(1e9 / tok_s / 86400, 2),
        "env": env_fingerprint(),
    }
    out = ROOT / "reports" / "throughput.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"{args.model} [{args.mode} loss={args.loss} ckpt_every={args.ckpt_every} compile={args.compile} optim={args.optim}] "
          f"seq={args.seq_len} mb={best}: {tok_s:,.0f} tok/s, "
          f"peak {res['peak_mem_gb']} GB, trainable {n_train / 1e9:.2f}B / {total / 1e9:.2f}B, "
          f"{res['days_per_1B_tokens']} days per 1B tokens")


if __name__ == "__main__":
    main()
