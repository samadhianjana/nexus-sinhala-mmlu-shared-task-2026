# Plan: SinhalaMMLU shared task

_Last updated: Sat 10 Oct 2026. Test phase: **12 Oct to 27 Oct 23:59**. Limit: 3 Codabench submissions per day._

Score = mean of Easy, Medium and Hard accuracy. Our dev set is Easy only (grades 6-8), so dev accuracy will
overestimate the test score. Decide using **both** SinhalaMMLU dev and the Global-MMLU `si` validation split.

## Where we are

| Done | Result |
|---|---|
| Environment (venv, PyTorch 2.14 / CUDA 13 on GB10), repo layout | `requirements.lock.txt` pinned |
| Evaluation harness (logit / generate / cloze, option shuffling, per-subject/domain/grade tables) | `eval/` |
| Parameter accounting for 19 candidates | `MODEL_MANIFEST.md` |
| Zero-shot baselines, 10 models | `reports/baselines_zeroshot_logit.md` |

Zero-shot leaders (dev / Global-MMLU si): **Gemma 4 E4B 55.2 / 52.0**, Gemma 3n E4B 50.3 / 46.2,
Qwen3.5-4B 40.6 (47.6 with option voting) / 44.2. Llama 3.2 3B, SmolLM3, Granite 4.2 3B and Phi-4-mini are
near random: dropped. Weakest subjects: dancing, history, Eastern music, Sinhala language & literature.

**Prompting experiments (done, `reports/experiments_prompting.md`):** for Gemma 4 E4B and Gemma 3n E4B, the
Sinhala instruction, fixed 5-shot and generated answers are all within ±0.4 points of the English zero-shot logit
baseline, i.e. no gain. Cloze scoring is near random: dropped. Qwen3.5-4B gains from 5-shot (+6.0 on dev) and from
option voting (+7.0) because both reduce its position bias, but it still trails Gemma 4 by about 8 points.
**Decision:** Gemma 4 E4B, English zero-shot prompt, logit scoring. Further gains must come from training.

## Timeline

| Dates | Phase | Exit criterion |
|---|---|---|
| 10-11 Oct | 1. Choose inference setup + measure training speed | Prompt/scoring config fixed; tokens/s known |
| 11-13 Oct | 2. Submission pipeline + first Codabench submission | One valid submission on the leaderboard |
| 11-15 Oct | 3. Build training data | Cleaned, deduplicated, logged MCQ and corpus sets |
| 12-17 Oct | 4. Continued pretraining (only if speed allows) | Checkpoint that beats the base on dev and val |
| 16-20 Oct | 5. Fine-tuning in test format | Fine-tuned model clearly above zero-shot |
| 20-23 Oct | 6. Inference improvements | Final configuration frozen |
| 23-26 Oct | 7. Package, reproduce, final submission | Final run on leaderboard by 26 Oct |
| 27 Oct | Buffer | Nothing new, fixes only |

## 1. Choose inference setup and measure speed (10-11 Oct)

- [x] Read the prompting experiments: English instruction, zero-shot, logit scoring for Gemma 4 E4B.
- [x] Option-order voting: not worth it for Gemma 4 (-0.7 dev, +0.4 val); revisit after fine-tuning.
- [ ] Cap PyTorch memory in every script (e.g. 100 GB of the 128 GB unified memory). On 10 Oct at 11:15 the
      Qwen3.5 cloze run exhausted system memory and the kernel OOM killer took down the desktop session and tmux.
      On the Spark, GPU and CPU share memory, so a GPU over-allocation kills other processes instead of raising
      a CUDA out-of-memory error.
- [ ] Install `flash-linear-attention` if Qwen3.5 stays in play (its linear-attention layers run on a slow
      fallback; Qwen3.5 runs were about 2× slower than Gemma 4).
- [ ] **Text-only Gemma 4 E4B:** load without the vision and audio towers (7.52B instead of 7.996B), confirm the
      accuracy is unchanged. This frees about 0.48B for LoRA or other components.
- [x] **Training throughput benchmark** (done 11 Oct, `reports/throughput.jsonl`). Best: Gemma 4 E4B text-only,
      per-layer embeddings frozen (3.97B trained), Liger fused loss, per-layer `torch.compile(dynamic=False)`,
      activations recomputed every 2nd layer, torchao `AdamW8bit`: **1,263 tokens/s ≈ 109M tokens/day**.
      Qwen3.5-4B full + compile: 1,584 tokens/s, but its tokenizer is 33% less efficient on Sinhala, so both
      cover ~250M Sinhala characters/day; Gemma stays (15 points stronger zero-shot). LoRA is slower than
      training the frozen-embedding model. Continued-pretraining budget: ~100M tokens per GPU-day.
      Originally planned settings:
      - Qwen3.5-4B full fine-tuning, BF16
      - Gemma 4 E4B, transformer layers trained, its 3.5B per-layer embedding tables frozen
      - Gemma 4 E4B LoRA
      Then set the continued-pretraining budget: tokens/s × seconds available.
- [ ] First git commit (data, `.env` and results are git-ignored).

## 2. Submission pipeline and first submission (11-13 Oct)

- [ ] `inference/predict.py`: test file in, `predictions.zip` out. One command, offline
      (`HF_HUB_OFFLINE=1`), pinned model commit, fixed seed. Echo `q_no`, `subject`, `category`, `level` exactly
      as given in the test file.
- [ ] Check the real test-file format when it is released on 12 Oct and adapt the loader.
- [ ] Submit the best zero-shot configuration once, to confirm the format and get a first test-score reference.
- [ ] Rule for the rest of the task: submit only at milestones. Never choose between variants by leaderboard
      score; choose on dev + validation.

## 3. Training data (11-15 Oct)

All sources go into `DATA_LOG.md` (URL, revision, license, size, filters).

**MCQ data for fine-tuning** (formatted exactly like the test: Sinhala question, 4 numbered options, answer digit):
- [ ] Global-MMLU `si` test split **minus** the 2,000 validation ids, plus its dev split.
- [ ] Translate English MCQ sets (ARC, OpenBookQA, SciQ, RACE, MMLU auxiliary train) into Sinhala with Gemma 4 E4B
      or NLLB-200. Filter by back-translation agreement and answer-consistency checks.
- [ ] Rebalance gold answer positions (the dev set has few "4" answers; avoid teaching a position prior).
- **Decided 11 Oct: no closed-API synthetic data** (no budget for paid APIs).
- [ ] Optional, open model only: Gemma 4 E4B writes Sinhala MCQs from Sinhala Wikipedia articles on the weakest
      subjects (Sri Lankan history, Buddhism, dance, Eastern music, literature). Free apart from GPU time; filter by
      answer agreement and drop questions that overlap dev/validation.

**Sinhala corpus for continued pretraining:**
- [ ] FineWeb-2 `sin_Sinh` (main), CulturaX `si`, MADLAD-400 `si`, Sinhala Wikipedia, NSINA news.
      Check each license.
- [ ] Remove pastpapers.wiki, e-thaksalawa.moe.gov.lk and any other exam/past-paper site (the sources of
      SinhalaMMLU questions).
- [ ] Deduplicate across sources. Run an n-gram overlap filter against dev and validation questions.
- [ ] Upweight encyclopedic and Sri Lanka content (history, religion, arts, literature, civics).
- [ ] Mix in 10-20% English (FineWeb-Edu) to preserve reasoning.

## 4. Continued pretraining (12-17 Oct, only if throughput allows)

- [ ] Go/no-go from the phase 1 benchmark. Skip if under about 300M tokens fit in the time available.
- [ ] Candidate setups: Qwen3.5-4B full training, or Gemma 4 E4B with frozen per-layer embeddings.
- [ ] Evaluate a checkpoint every ~50-100M tokens on dev + validation. Stop if accuracy stops rising.

## 5. Fine-tuning in test format (16-20 Oct)

- [ ] SFT on the phase 3 MCQ data: LoRA on Gemma 4 E4B (merge before submission) and/or full fine-tuning on
      Qwen3.5-4B.
- [ ] Short answer-only targets (the digit), the same prompt as inference.
- [ ] If any SinhalaMMLU dev questions are used for training, keep a clean held-out slice (or use k-fold) for
      evaluation.
- [ ] Optional: preference optimisation against confident wrong answers, only if SFT gains are small.

## 6. Inference improvements (20-23 Oct)

- [ ] Option-order voting (2-4 orders) on the final model.
- [ ] Reasoning before answering only for STEM; compare reasoning in English vs Sinhala.
- [ ] Fixed few-shot examples, if phase 1 showed a gain.
- [ ] Freeze the configuration by 23 Oct.

## 7. Package and reproduce (23-26 Oct)

- [ ] Dockerfile on a pinned NGC PyTorch image (needs Docker access), plus `requirements.lock.txt`.
- [ ] Fresh clone → one command → identical `predictions.zip`, with the network disabled.
- [ ] Merged final weights in a private Hugging Face repo shared with the organisers; hash in the manifest.
- [ ] `MODEL_MANIFEST.md` final row: every component loaded and its exact total parameter count.
- [ ] `DATA_LOG.md` complete, including any closed-API generation.
- [ ] System description report draft, citing Pramodya et al., EMNLP 2025.
- [ ] Final submission by 26 Oct.

## Open items for the team

| Item | Why it matters | Owner |
|---|---|---|
| Email organisers: exact 8B threshold, vocabulary trimming, per-subject few-shot, government textbooks | Decides model choice and data sources | Team |
| Register on Codabench (corresponding member) | Needed before 12 Oct | Team |
| Docker group access (`sudo usermod -aG docker anas22`) | Needed for NGC containers (FP8, packaging) | Team |
| Optional: Hugging Face access to Llama 3 / 3.1 | Only to count SinLlama and Llama 3.1 8B (both likely over 8B) | Team |

## Risks

- **Gemma 4 E4B is at the limit** (7.996B as stored). Any addition must be merged or offset by dropping the
  vision/audio towers.
- **Test Hard tier will be much lower than dev.** Accuracy already falls from 60% (grade 6) to 53% (grades 7-8).
- **Reproduction on different hardware** (organisers are likely on x86): keep greedy decoding and logit scoring,
  and test the container on x86 if possible.
- **Time:** continued pretraining is the item most likely to overrun. It runs after a working submission exists,
  so it can be dropped without losing the result.
