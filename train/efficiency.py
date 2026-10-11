"""Training speed/memory options shared by the benchmark and the training scripts.

  loss="hf"       transformers' built-in loss: materialises the full [tokens x vocab] logits (Gemma: 262k vocab)
  loss="chunked"  plain-PyTorch chunked cross-entropy; each chunk's logits are recomputed in backward
  loss="liger"    Liger fused linear + cross-entropy Triton kernel (never materialises the logits)
  ckpt_every=N    recompute activations of every Nth decoder layer only (1 = all layers, 0 = none)
  compile=True    torch.compile each decoder layer (regional compilation)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def lm_parts(model):
    """(decoder body, lm_head, final logit softcap) for plain and PEFT-wrapped causal LMs."""
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    return lm.model, lm.lm_head, getattr(lm.config, "final_logit_softcapping", None)


def _ce_chunk(h, w, y, softcap):
    logits = (h @ w.t()).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(logits, y, ignore_index=-100, reduction="sum")


def chunked_ce(h, w, y, softcap, chunk: int = 4096):
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(_ce_chunk, h[i:i + chunk], w, y[i:i + chunk], softcap, use_reentrant=False)
    return total / (y != -100).sum().clamp(min=1)


_liger = {}


def lm_loss(model, input_ids, labels, loss: str = "hf", attention_mask=None):
    """Next-token cross-entropy, mean over non-ignored target tokens. All three variants give the same value."""
    if loss == "hf":
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    body, head, softcap = lm_parts(model)
    h = body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = h[:, :-1].reshape(-1, h.shape[-1])
    y = labels[:, 1:].reshape(-1)
    if loss == "chunked":
        return chunked_ce(h, head.weight, y, softcap)
    if loss == "liger":
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        fn = _liger.setdefault(softcap, LigerFusedLinearCrossEntropyLoss(softcap=softcap))
        return fn(head.weight, h, y)
    raise ValueError(loss)


def configure(model, ckpt_every: int = 1, compile_layers: bool = False) -> None:
    body = lm_parts(model)[0]
    if ckpt_every > 0:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for i, layer in enumerate(body.layers):
            layer.gradient_checkpointing = i % ckpt_every == 0
    else:
        model.gradient_checkpointing_disable()
    if compile_layers:
        # Layers differ (sliding vs global attention, linear-attention blocks), so dynamo compiles one graph per
        # layer variant; past the default limit of 8 it silently falls back to eager. The limit is passed per call:
        # the global config value was not honoured in the 11 Oct rerun. dynamic=False keeps one static graph per
        # shape, so activation recomputation cannot pick a different (dynamic) graph than the forward pass used
        # (pytorch/pytorch#166926).
        limit = max(64, 4 * len(body.layers))
        for layer in body.layers:
            layer.compile(dynamic=False, recompile_limit=limit)
