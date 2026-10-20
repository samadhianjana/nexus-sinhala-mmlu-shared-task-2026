"""Model loading and the three scoring methods.

  logit     next-token log-probs restricted to the option labels "1".."N" (argmax)
  generate  greedy decoding, first standalone label digit in the output is the prediction
  cloze     length-normalised log-likelihood of each option text given the question
"""
from __future__ import annotations

import math
import re

import torch
from tqdm.auto import tqdm


def resolve_revision(name: str, revision: str | None) -> tuple[str | None, str | None]:
    """(revision to load, commit to record). Local dirs record their SOURCE.json; offline runs need a full sha."""
    import json
    import os
    from pathlib import Path

    if Path(name).is_dir():
        src = Path(name) / "SOURCE.json"
        if not src.exists():
            return None, "local"
        meta = json.loads(src.read_text())
        return None, meta.get("model_id") or meta["source_commit"] + "+text-only"
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision, revision
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise ValueError(f"offline run needs a full 40-character commit sha for {name}")
    from huggingface_hub import HfApi

    sha = HfApi().model_info(name, revision=revision).sha  # pin the exact commit
    return sha, sha


def load_model(name: str, revision: str | None = None, dtype: str = "bfloat16", adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision, commit = resolve_revision(name, revision)
    kw = dict(dtype=getattr(torch, dtype), device_map="cuda", revision=revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except ValueError:  # some multimodal checkpoints only register under image-text-to-text
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    if adapter:  # LoRA adapter kept separate (can be switched off per question with model.disable_adapter())
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        commit = f"{commit}+adapter:{adapter}"
    model.eval()
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.hub_commit = commit
    return model, tok


def loaded_param_stats(model) -> dict:
    """Parameters actually resident at inference (tied weights counted once)."""
    by_module: dict[str, int] = {}
    for name, p in model.named_parameters():  # named_parameters() already de-duplicates tied tensors
        top = ".".join(name.split(".")[:2])
        by_module[top] = by_module.get(top, 0) + p.numel()
    non_text = {k: v for k, v in by_module.items() if re.search(r"vision|visual|audio|image|mm_proj", k)}
    total = sum(by_module.values())
    return {"loaded_total": total, "loaded_non_text": sum(non_text.values()),
            "non_text_modules": non_text, "commit": getattr(model.config, "hub_commit", None)}


def label_token_ids(tok, n: int) -> list[list[int]]:
    """Single-token encodings of each label ("1" and " 1" variants)."""
    out = []
    for i in range(1, n + 1):
        ids = set()
        for v in (str(i), f" {i}"):
            enc = tok.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            raise ValueError(f"label {i} has no single-token encoding")
        out.append(sorted(ids))
    flat = [t for ids in out for t in ids]
    assert len(flat) == len(set(flat)), "label token ids overlap"
    return out


def _batches(lengths: list[int], token_budget: int, max_batch: int):
    """Length-sorted batches bounded by padded-token budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batch, longest = [], 0
    for i in order:
        new_longest = max(longest, lengths[i])
        if batch and (new_longest * (len(batch) + 1) > token_budget or len(batch) >= max_batch):
            yield batch
            batch, new_longest = [], lengths[i]
        batch.append(i)
        longest = new_longest
    if batch:
        yield batch


def _forward_last(model, enc, keep: int):
    try:
        return model(**enc, logits_to_keep=keep).logits
    except TypeError:
        return model(**enc).logits[:, -keep:]


class Scorer:
    def __init__(self, model, tok, add_special_tokens: bool, token_budget: int = 32768,
                 max_batch: int = 64, max_new_tokens: int = 16):
        self.model, self.tok = model, tok
        self.add_special = add_special_tokens
        self.token_budget, self.max_batch = token_budget, max_batch
        self.max_new_tokens = max_new_tokens
        self.sampling: dict = {}  # e.g. {"do_sample": True, "temperature": 0.8, "top_p": 0.95, "top_k": 64}

    def _encode(self, texts: list[str]):
        return self.tok(texts, return_tensors="pt", padding=True,
                        add_special_tokens=self.add_special).to(self.model.device)

    def _lengths(self, texts: list[str]) -> list[int]:
        return [len(x) for x in self.tok(texts, add_special_tokens=self.add_special)["input_ids"]]

    @torch.no_grad()
    def logit(self, prompts: list[str], n_options: list[int]) -> list[dict]:
        cache: dict[int, list[list[int]]] = {}
        results: list[dict | None] = [None] * len(prompts)
        lengths = self._lengths(prompts)
        for idx in tqdm(list(_batches(lengths, self.token_budget, self.max_batch)), desc="logit", leave=False):
            enc = self._encode([prompts[i] for i in idx])
            logp = torch.log_softmax(_forward_last(self.model, enc, 1)[:, -1].float(), dim=-1)
            for row, i in enumerate(idx):
                n = n_options[i]
                labels = cache.setdefault(n, label_token_ids(self.tok, n))
                scores = torch.stack([torch.logsumexp(logp[row, ids], 0) for ids in labels])
                probs = torch.softmax(scores, 0)
                results[i] = {"probs": probs.tolist(), "pred": int(probs.argmax()) + 1,
                              "label_mass": float(scores.exp().sum())}
        return results

    @torch.no_grad()
    def generate(self, prompts: list[str], n_options: list[int]) -> list[dict]:
        results: list[dict | None] = [None] * len(prompts)
        lengths = self._lengths(prompts)
        budget = max(self.token_budget // 2, 1)
        for idx in tqdm(list(_batches(lengths, budget, self.max_batch)), desc="generate", leave=False):
            enc = self._encode([prompts[i] for i in idx])
            gen_kw = self.sampling or {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                      **gen_kw)
            new = out[:, enc["input_ids"].shape[1]:]
            raw = self.tok.batch_decode(new, skip_special_tokens=False)
            texts = self.tok.batch_decode(new, skip_special_tokens=True)
            for row, i in enumerate(idx):
                n = n_options[i]
                answer_text = texts[row]
                if "<|channel>" in raw[row]:  # thinking mode: the answer is what follows the closed thought channel
                    answer_text = raw[row].split("<channel|>")[-1] if "<channel|>" in raw[row] else ""
                    answer_text = re.sub(r"<[^>]*\|>|<\|[^>]*>", " ", answer_text)
                m = re.search(rf"(?<![0-9])([1-{n}])(?![0-9])", answer_text)
                pred = int(m.group(1)) if m else None
                probs = [1.0 if pred == k else 0.0 for k in range(1, n + 1)]
                n_new = int((new[row] != self.tok.pad_token_id).sum())
                results[i] = {"probs": probs, "pred": pred, "text": texts[row][-2000:], "new_tokens": n_new}
        return results

    @torch.no_grad()
    def cloze(self, contexts: list[str], options: list[list[str]]) -> list[dict]:
        """Left-padded so every continuation ends at the last position; keeps only needed logits."""
        pairs = []  # (item index, ctx ids, cont ids, n chars)
        for i, (ctx, opts) in enumerate(zip(contexts, options)):
            ctx_ids = self.tok.encode(ctx, add_special_tokens=True)
            for opt in opts:
                cont = self.tok.encode(" " + opt, add_special_tokens=False)
                pairs.append((i, ctx_ids, cont, len(opt)))
        lengths = [len(c) + len(o) for _, c, o, _ in pairs]
        scores = [0.0] * len(pairs)
        pad = self.tok.pad_token_id
        for idx in tqdm(list(_batches(lengths, self.token_budget, self.max_batch)), desc="cloze", leave=False):
            L = max(lengths[j] for j in idx)
            keep = max(len(pairs[j][2]) for j in idx) + 1
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for row, j in enumerate(idx):
                seq = pairs[j][1] + pairs[j][2]
                ids[row, L - len(seq):] = torch.tensor(seq)
                mask[row, L - len(seq):] = 1
            enc = {"input_ids": ids.to(self.model.device), "attention_mask": mask.to(self.model.device)}
            logp = torch.log_softmax(_forward_last(self.model, enc, keep).float(), dim=-1)
            for row, j in enumerate(idx):
                cont = pairs[j][2]
                k = len(cont)
                # logits at kept positions [-k-1 .. -2] predict tokens at [-k .. -1]
                lp = logp[row, keep - k - 1: keep - 1].gather(-1, torch.tensor(cont, device=logp.device)[:, None])
                scores[j] = float(lp.sum()) / max(pairs[j][3], 1)
        results: list[dict] = []
        pos = 0
        for opts in options:
            s = torch.tensor(scores[pos: pos + len(opts)])
            pos += len(opts)
            probs = torch.softmax(s, 0)
            results.append({"probs": probs.tolist(), "pred": int(probs.argmax()) + 1})
        return results


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
