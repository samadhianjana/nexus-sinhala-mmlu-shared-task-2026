"""Recover a LoRA adapter from a merged LoRA model: for every weight that differs from the base model,
delta = merged - base (fp32) is factorised by SVD and truncated to rank r (LoRA training with rank r makes the delta
rank <= r, up to bf16 rounding). Saved in PEFT format with lora_alpha = r (scaling 1), so base + adapter = merged.

  python scripts/extract_lora.py --base models/gemma-4-E4B-it-text --merged models/sft_gmmlu_lora/step-357 \
      --rank 16 --out models/adapter_A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_dir(d: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        out.update(load_file(f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--merged", required=True, type=Path)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    base, merged = load_dir(args.base), load_dir(args.merged)
    assert base.keys() == merged.keys(), "base and merged checkpoints have different tensors"

    tensors, targets, n_params, worst = {}, [], 0, 0.0
    for k in sorted(base):
        if torch.equal(base[k], merged[k]):
            continue
        if base[k].dim() != 2 or not k.endswith(".weight"):
            raise SystemExit(f"non-linear tensor changed ({k}); this is not a pure LoRA merge")
        delta = merged[k].cuda().float() - base[k].cuda().float()
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        r = args.rank
        B = (U[:, :r] * S[:r]).contiguous()          # out x r
        A = Vh[:r].contiguous()                       # r x in
        err = ((delta - B @ A).norm() / delta.norm().clamp_min(1e-12)).item()
        worst = max(worst, err)
        module = k[: -len(".weight")]
        tensors[f"base_model.model.{module}.lora_A.weight"] = A.to(torch.bfloat16).cpu()
        tensors[f"base_model.model.{module}.lora_B.weight"] = B.to(torch.bfloat16).cpu()
        targets.append(module)
        n_params += A.numel() + B.numel()

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out / "adapter_model.safetensors")
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": args.rank, "lora_alpha": args.rank,
           "lora_dropout": 0.0, "bias": "none", "target_modules": targets, "fan_in_fan_out": False,
           "use_rslora": False, "modules_to_save": None, "base_model_name_or_path": str(args.base),
           "inference_mode": True}
    (args.out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    (args.out / "SOURCE.json").write_text(json.dumps({"extracted_from": str(args.merged), "base": str(args.base),
                                                      "rank": args.rank, "modules": len(targets),
                                                      "adapter_params": n_params,
                                                      "max_relative_reconstruction_error": worst}, indent=2))
    print(f"{len(targets)} modules, {n_params / 1e6:.1f}M adapter parameters, "
          f"max relative reconstruction error {worst:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
