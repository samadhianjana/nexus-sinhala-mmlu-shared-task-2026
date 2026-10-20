# Future work: from shared-task system to research paper

_Written 19 Oct 2026, before the end of the test phase. Results referenced here are from `reports/`, `PLAN.md` and
`scripts/runs/README.md`. Start this work after the shared task closes (27 Oct)._

## 1. What we can claim today

| Finding | Evidence so far | Novelty |
|---|---|---|
| **Reasoning helps the academic tiers, not the culturally grounded ones.** Thinking mode helps A/L-level academic questions a lot, but not easy culture/humanities questions | Global-MMLU `si` academic proxy (551 q): base model thinking 79.5 vs fast 60.1 (+19.4, paired CI [+14.9, +24.0]). Real test: Hard 58.97 → 62.68 even with the weaker thinking of model A. Dev (easy, 400 q): no gain | Medium: directly answers the SinhalaMMLU paper's question of *why* models fail (Hard = reasoning gap, Humanities = knowledge gap) |
| **Answer-only fine-tuning silently removes the reasoning benefit**; switching the LoRA adapter off per question by difficulty recovers it within the 8B budget | Model A reasons on 26% of academic questions vs 100% for the base model; 68.4 vs 79.5 with thinking on. Adapter extracted from the merged weights reproduces A on 97.8% of answers | Low–medium: the collapse itself is already reported (see §4); the setting and the parameter-budget fix are new |
| **Negative results** with paired statistics | Prompt language, fixed few-shot and output format: ±0.4. Full fine-tuning: −2.9 (forgetting). 30M tokens of raw Sinhala continued pretraining: no gain on cultural subjects | Low–medium, useful for low-resource practitioners |
| **Leakage analysis** of Global-MMLU vs SinhalaMMLU | Character n-grams: max 0.45; meaning-based (multilingual-e5): max 0.93, none at copy level (≥ 0.95) | Low, a short section |
| **Teacher-student gap and distillation** | Gemma 4 31B 73.6 vs Gemma 4 E4B 55.2 on dev; distillation and passage-based synthetic MCQs pending | Depends on pending results |

**Positioning.** The contribution is not "we discovered reasoning collapse". It is: *in a low-resource, culturally
grounded exam benchmark, reasoning helps the academic tiers and not the cultural ones; fine-tuning removes that benefit;
and a parameter-budget-friendly fix (difficulty-routed adapter switching) recovers it.* The paper's strength therefore
rests on **generalisation** and on **comparison with existing fixes**.

## 2. Work items

| # | Item | Strengthens the paper | Cost | Notes |
|---|---|---|---|---|
| 1 | **Second model and second benchmark** | **High, essential beyond a system paper** | 4–6 GPU-days | Benchmarks built from local exams *with school levels*, so the knowledge-vs-reasoning-by-level claim can be tested: e.g. IndoMMLU or ArabicMMLU. Second model with a thinking mode: Qwen3.5-4B or Qwen3-4B |
| 2 | **Mitigations compared** | **High** | 3–5 GPU-days | Must include known fixes, not only ours: KL anchoring during fine-tuning, weight interpolation (as in ThinkSwitch), training with reasoning traces from the 31B teacher, vs. merged LoRA vs. adapter switching |
| 3 | **Label-free router** | **High** (reviewers will ask) | 1–2 GPU-days | Our router uses the test file's `level` field. Add: (a) confidence-based (think only when the fast answer's top probability < τ), (b) agreement-based (as in DART). Compare with level routing |
| 4 | **Accuracy vs. compute** | Medium | Mostly analysis | Thinking costs ~600 tokens per question. Plot reasoning tokens spent vs accuracy gained, per level and per router |
| 5 | **Error analysis by a native speaker** | Medium, cheap | ~1 day of team time | Which question types thinking fixes or breaks: calculations, negation, sub-option ("A and C") questions, chronology |
| 6 | **Several seeds** per LoRA configuration | Medium (required for rigour) | ~1 GPU-day per extra seed set | Report mean ± std; keep paired bootstrap CIs |
| 7 | **Reasoning-suppression curve** | Low–medium | ~1 GPU-day | Share of questions the model reasons on, and accuracy, against LoRA training steps and rank. Mostly confirms a known effect in a new setting; good figure, not the core claim |
| 8 | **Related work and positioning** | Required | Writing | Against the papers in §4 and the SinhalaMMLU paper's own findings |
| 9 | **Release** | Required | Small | Code, configs, adapter weights. Never redistribute test questions (dataset is CC BY-NC-ND) |

**Priority after 27 Oct:** 3 (label-free router) and 1 (second benchmark) first. Those two decide whether this is a
Sinhala-specific report or a general finding. Then 2, 5, 6.

## 3. Venues

| Target | Needs | Time after 27 Oct |
|---|---|---|
| **ICTer 2026 shared-task system paper** (required) | Current results + item 5 | ~1 week of writing |
| **Low-resource language workshop** (e.g. LoResLM, SIGUL, regional NLP workshops) | + items 2, 3, 6 | ~2–3 weeks |
| **ACL/EMNLP Findings** | + items 1, 4 (and everything above) | ~4–5 weeks; plausible but competitive |

## 4. Related work to position against

- Pramodya et al., *SinhalaMMLU*, EMNLP 2025 — https://aclanthology.org/2025.emnlp-main.1673/
- *On the Impact of Fine-Tuning on Chain-of-Thought Reasoning*, NAACL 2025 (fine-tuning degrades CoT, more in small models) — https://aclanthology.org/2025.naacl-long.584.pdf
- Crusoe, *Preserving the trace: a guide to fine-tuning chain-of-thought models* (answer-only fine-tuning drives Qwen3-4B-Thinking's reasoning rate to zero; cites a "Reasoning-Trace Collapse" paper, still to be located) — https://www.crusoe.ai/resources/blog/preserving-the-trace-a-guide-to-fine-tuning-chain-of-thought-models
- *ThinkSwitch: Context Distillation with LoRA and Weight Interpolation* — https://arxiv.org/html/2606.01080
- *Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation* — https://arxiv.org/pdf/2608.20256
- *DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets* — https://arxiv.org/pdf/2606.23181
- *Thinkless: LLM Learns When to Think* — https://fugumt.com/fugumt/paper_check/2505.13379v2_enmode
- *AdaCoT: Cross-Lingual Factual Reasoning* (multilingual, multiple-choice) — https://arxiv.org/html/2501.16154v1
- *Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It* — https://hyper.ai/en/papers/2609.36585
- Cheng et al., *Adapting LLMs via Reading Comprehension*, ICLR 2024 (raw CPT vs reading-comprehension format) — https://iclr.cc/virtual/2024/poster/17426
- Sutanto et al., *LLM Distillation for Few-Shot MCQA*, Findings of EMNLP 2025 — https://aclanthology.org/2025.findings-emnlp.452/

## 5. Practical notes

- **Disclosure:** state in the paper that Claude Code (an AI assistant) was used for code and analysis assistance,
  following the venue's AI-use policy. The authors verify and write all claims.
- **Data:** report scores only; never publish SinhalaMMLU test questions. Dev examples in the paper must respect the
  dataset licence.
- **Reproducibility record:** `scripts/runs/README.md` maps every run to its result; `DATA_LOG.md` lists all data;
  `eval/compare.py` gives the paired statistics.
