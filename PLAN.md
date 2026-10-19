# Plan: SinhalaMMLU shared task, 18 to 27 October 2026

_Updated Sun 18 Oct (start slipped from 14 Oct; schedule compressed). Test phase closes **27 Oct 23:59**. Final submission: **Tue 27 Oct after 15:00**. No submissions possible on 25–26 Oct (exams). Limit: 3 Codabench submissions per day. Hard constraint:
≤ 8B parameters at inference. Every choice below is made on SinhalaMMLU dev (and Global-MMLU `si` validation), never
on test questions._

## 1. Where we stand

| System | Dev (easy) | Test (macro avg of easy/medium/hard) |
|---|---:|---:|
| Zero-shot Gemma 4 E4B, text-only (7.463B) | 55.2 | **51.59** |
| LoRA r=16 on Global-MMLU `si` (1 epoch, merged) + 4-order voting | 57.2 | **54.59** (easy 55.30, medium 49.51, hard 58.97). 1st on the leaderboard on 18 Oct |

The dev gain (+2.7, paired 95% CI [+0.9, +4.4]) carried over to the test set (+3.0), so **dev is a usable guide**
even though it only contains easy questions.

What the experiments showed (details in `reports/`):

| Tried | Result on dev | Lesson |
|---|---|---|
| Prompt variants (Sinhala instruction, 5-shot, generation) | ±0.4 | Gemma 4 E4B is not format-limited. Prompting is done |
| Full fine-tuning (3.97B params, lr 1e-5) on Global-MMLU | **−2.9** | Overwrites existing knowledge (forgetting) |
| LoRA on Global-MMLU, or on dev (5-fold CV) | +1.0 (+2.7 voted) | Gentle training helps; translated and in-domain MCQs help about equally |
| Continued pretraining on raw Sinhala web/news/Wikipedia, 30M tokens | −1.2 (+0.9 voted, n.s.) | Raw text does not move the weak, culturally grounded subjects |

The remaining weakness is **knowledge**, not format: dance 34%, history 39%, Eastern music 45%, Sinhala literature 47%,
Buddhism 52%. The test also contains medium (grades 9–11, O/L) and hard (grades 12–13, A/L) subjects that dev does not
cover at all (paper, Table 2: physics, chemistry, biology, economics, business studies, agriculture, political science,
home economics, media studies, Buddhist civilization, design and construction technology).

## 2. What the research says

1. **Raw continued pretraining gives knowledge but damages question answering; reformatting the same text as
   reading-comprehension tasks fixes this.** Cheng et al., ICLR 2024 ("Adapting LLMs via Reading Comprehension")
   converted raw domain corpora into texts followed by question/answer tasks, and a 7B model matched much larger
   domain models. This matches our pilot exactly.
2. **Facts become retrievable only if they appear in varied forms** (paraphrases, questions). Allen-Zhu & Li,
   ICML 2024 ("Physics of Language Models 3.1") recommend rewriting pretraining data into such forms with auxiliary
   models.
3. **Distillation for multiple-choice QA works:** an LLM generates MCQs, scores the options, and a small student is
   trained on those scores. Sutanto et al., Findings of EMNLP 2025, report MMLU 28.9 → 39.3. Round-trip filtering
   (re-answer the generated question, keep only agreeing items) is the standard quality filter (Samuel et al. 2023).
4. **Machine-translated Sinhala is noticeably unnatural.** SinhalaMMLU paper, Table 8: native questions rated 97.3
   for naturalness, Global-MMLU `si` 71.1, with wrong STEM terminology. Training data written from **native** Sinhala
   text is preferable to more translation.
5. **Few-shot prompting gives little or nothing for instruction-tuned models** (paper, Table 4; our experiments
   agree). Questions with negation and with sub-options (~10% of the data) are hardest for every model (Table 5).
6. **Larger open models are much stronger multilingually and may be used to generate training data** (the rules allow
   synthetic data from open models). Gemma 4 model card, MMMLU: 31B 88.4%, 26B-A4B 86.3%, E4B 76.6%.
7. **Weight averaging of fine-tuned runs** ("model soups", Wortsman et al., ICML 2022) often improves accuracy at no
   inference cost. That is allowed here because the result is still one 7.46B model.

## 2b. Update 19 Oct: where the gap is

Test (macro avg): v2 = 54.85 (easy 54.83 / medium 50.77 / hard 58.97), equal to model A within noise. A new leader
scores 57.78 with **hard 70.10** (+11 over us, about 5 standard errors) while easy/medium are equal to or below ours.
Hard = A/L (grades 12–13: physics, chemistry, biology, economics, business, agriculture), where multi-step reasoning
matters. Our system answers from one next-token probability with no reasoning.

**Plan change (19 Oct):** add a reasoning track for hard questions, move teacher soft labels earlier, shrink the
synthetic track. Order of GPU jobs on 19 Oct: (1) thinking-mode test on Global-MMLU academic subjects (proxy for A/L)
and a dev sample; (2) Gemma 4 31B soft labels for the 11,447 Global-MMLU training questions; (3) synthetic pilot of
500 MCQs; (4) overnight: student LoRA trained toward the teacher's soft labels, compared with model A on dev.
If thinking helps on the academic proxy without hurting dev, inference uses it for questions labelled `hard` (each
question decided from its own fields only). Teacher (Gate 1, 19 Oct): Gemma 4 31B-it, dev 73.6 vs student 55.2.

## 3. Strategy

**Main bet: turn native Sinhala text into curriculum-style MCQs with a larger open teacher model, then train the 8B
student on them with LoRA (hard labels plus the teacher's option probabilities).** This targets the knowledge gap
(points 1–3), keeps native terminology (point 4) and uses the training recipe that already works (gentle LoRA).

Supporting work: train the final model on everything (Global-MMLU + synthetic + all of dev), voting over option
orders, and averaging two runs if time allows.

**Dropped:** more prompt engineering, full fine-tuning, raw continued pretraining, closed-API data (no budget),
ensembles of several models (would exceed 8B).

### Pipeline

1. **Teacher selection (Gate 1).** Score Gemma 4 31B-it, Gemma 4 26B-A4B-it and Qwen3.5-27B on dev and Global-MMLU
   validation, the same way as everything else (closed-book, logit scoring). Data generation only; never used at
   inference.
2. **Passage pool.** From the cleaned corpus (Sinhala Wikipedia first, then FineWeb-2 documents with high
   Sri Lanka-topic scores, news for civics), pick passages per curriculum subject using keyword lists built from
   the paper's published subject list (Table 2), not from test questions. Exam/past-paper sites are already
   removed.
3. **Generation.** The teacher writes 2–4 Sinhala MCQs per passage in Sri Lankan exam style (4 options, one
   correct; include negation and sub-option formats, point 5).
4. **Filtering.**
   - (a) round trip: the teacher re-answers each question *with* the passage and must agree with the key;
   - (b) the teacher's closed-book option probabilities are stored as soft labels;
   - (c) near-duplicate removal, and no item with similarity ≥ 0.5 to dev or validation (`scripts/overlap.py`);
   - (d) balanced answer positions;
   - (e) a native-speaker review of 50 random items before scaling up.
5. **Student training.** LoRA on Gemma 4 E4B text-only. Loss = cross-entropy on the key + KL divergence to the
   teacher's option distribution. Compare: Global-MMLU only (= model A), +synthetic, +soft labels, rank 16 vs 64.
6. **Final model.** Best recipe + all of dev, merged, with 4- or 8-order voting.

### Decision gates

| Gate | When | Question | If yes | If no |
|---|---|---|---|---|
| 1 | Sun 18 Oct | Is the best teacher ≥ 5 points above Gemma 4 E4B on dev? | Soft labels from the teacher | Use the teacher only to write passage-grounded questions (answers come from the text, so closed-book strength matters less) |
| 2 | Mon 19 Oct | Native-speaker review: ≥ 80% of 50 sample MCQs well-formed, correct, natural Sinhala? | Scale generation up | Fix the prompt/filters and review again the same day |
| 3 | Thu 22 Oct | Does +synthetic beat model A on dev (paired CI above 0)? | Synthetic data in the final recipe | Final = Global-MMLU + dev LoRA (v2), effort shifts to inference and packaging |
| 4 | Sat 24 Oct, noon | Config freeze | Only packaging and reproducibility after this | — |

## 4. Schedule (one GPU: GPU jobs run one after another, CPU work in parallel)

Everything that needs a decision or a submission happens by **Sat 24 Oct**. On 25–26 Oct nothing needs you;
on 27 Oct you only upload a file that was already built and validated on 24 Oct.

| Date | GPU (unattended, in tmux) | CPU / people | Exit criterion |
|---|---|---|---|
| **Sun 18 Oct** | Teacher evaluations: Gemma 4 31B-it, Gemma 4 26B-A4B-it, Qwen3.5-27B on dev + Global-MMLU val (~2 h incl. downloads). **Overnight:** train **v2** = LoRA on Global-MMLU + all dev (~3.5 h), then its test predictions | Passage pool + subject keywords (from the paper's Table 2); generation and filter scripts | **Gate 1**; v2 ready |
| **Mon 19 Oct** | Pilot generation: 500 MCQs from the chosen teacher (~1 h) + round-trip filter | **Submit v2** (vote 4). **Review 50 pilot MCQs (team)** | **Gate 2** |
| **Tue 20 Oct** | Large generation, ~25k MCQs (~5–8 h, overnight 20→21) | Overlap/dedup checks vs dev + validation (character and meaning-based); DATA_LOG entries | Raw synthetic set |
| **Wed 21 Oct** | Filtering + teacher soft labels (~3 h); then student run (a) Global-MMLU + synthetic, hard labels (~4 h, overnight 21→22) | Stats: subjects covered, answer balance, rejection rates | Clean synthetic set; run (a) done |
| **Thu 22 Oct** | Student run (b) = (a) + soft labels (~4 h); compare (a), (b), model A on dev (paired bootstrap). **Overnight:** final training = best recipe + all dev | Packaging starts: one-command run script, README, Dockerfile/venv instructions | **Gate 3**; final model training |
| **Fri 23 Oct** | Final model: test predictions with 4- and 8-order voting (choice of 4 vs 8 made on dev with the non-dev model from 22 Oct) | **Submit final candidate.** MODEL_MANIFEST final row; DATA_LOG complete; draft system report (cite Pramodya et al. 2025) | Final candidate on the leaderboard |
| **Sat 24 Oct** | **Reproduction test:** fresh clone → one command → identical `predictions.zip`, network off. Build `results/submissions/FINAL.zip` | **Config freeze (Gate 4, noon).** Merged weights to a private HF repo (write token needed). Submit only if 23 Oct had a problem | Frozen, reproducible package; FINAL.zip validated |
| Sun 25 – Mon 26 Oct | Nothing scheduled | **Exams: no action needed from you** | — |
| **Tue 27 Oct, after 15:00** | — | **Upload `FINAL.zip`** (already validated) and make sure it is the run selected for the final ranking. Deadline 23:59 | Final submission on the leaderboard |

**Fallback at every point:** v2 (Global-MMLU + dev LoRA + voting) is ready on 19 Oct; model A (current leader, 54.59)
exists already. If the synthetic track fails Gate 2 or 3, v2 or A becomes the final system and the remaining days go
to packaging.

**Dropped to fit the compressed schedule:** thinking-mode experiment for STEM, LoRA rank 64, seed averaging (soup).
They return only if Gate 3 is passed early.

## 5. Evaluation and submission rules

- **Promotion rule:** a change is kept only if it beats the current best on dev with a paired bootstrap 95% CI above
  0, or is neutral on dev and clearly better on Global-MMLU validation **and** reduces position bias.
- Models trained on dev are compared via the 5-fold protocol (`configs/splits/sinhalammlu_dev_cv5.json`).
- **Submissions:** at most one per milestone (planned: 19 Oct v2, 23 Oct final candidate, 27 Oct final; 24 Oct only
  as a repair). We never choose between variants by trying them on the leaderboard.
- Test questions are used only by `inference/predict.py`. Synthetic data is never checked against or filtered by
  test questions.
- Predictions are made one question at a time: nothing is estimated from the test set as a whole. Any calibration
  (e.g. PriDe's position prior) is estimated on dev only, never on test inputs.
- Every teacher-model use for data generation is logged in `DATA_LOG.md`; the system report also discloses that
  Claude Code (a closed model) was used as a coding assistant only, never for data generation or at inference.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Generated Sinhala is unnatural or wrong | Round-trip filter + native-speaker review (Gate 2); passage-grounded answers |
| Teacher (31B) generation too slow on the Spark | Use 26B-A4B (4B active, much faster decoding); batch generation; or fewer, better items |
| Memory: teacher 31B in bf16 ≈ 62 GB | Within the 100 GB cap; batch size tuned; no other GPU job runs alongside |
| Synthetic data hurts (style mismatch) | Gate 3; v2 fallback |
| Reproduction on different hardware changes some answers | Logit scoring (deterministic), fixed order of batches, pinned commits; document expected ±0.3-point noise |
| Rules ambiguity (8B threshold, textbooks) | Model is 7.463B; no textbooks or past papers used; organiser email still recommended |

## 7. Open items for the team

| Item | Needed by |
|---|---|
| Native-speaker review of 50 pilot MCQs (+ the 30 corpus samples in `data/corpus/cpt_v1/sample_for_review.md`) | Mon 19 Oct |
| Hugging Face **write** token, scoped to one private model repo, for the final weights | Sat 24 Oct |
| Check the deadline's time zone on Codabench and when the code package + system report are due | Sat 24 Oct |
| Confirm nobody on the team has access to the full SinhalaMMLU dataset (participation rule) | Any time |
| Organiser email: exact 8B threshold, government textbooks, Global-MMLU use (optional confirmation) | Any time |

## Sources

- Pramodya et al., *SinhalaMMLU*, EMNLP 2025 — https://aclanthology.org/2025.emnlp-main.1673/
- Cheng, Huang & Wei, *Adapting Large Language Models via Reading Comprehension*, ICLR 2024 — https://iclr.cc/virtual/2024/poster/17426
- Allen-Zhu & Li, *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction*, ICML 2024 — https://arxiv.org/abs/2309.14316
- Sutanto et al., *LLM Distillation for Efficient Few-Shot Multiple Choice Question Answering*, Findings of EMNLP 2025 — https://aclanthology.org/2025.findings-emnlp.452/
- Samuel et al., *Can LLMs Augment Low-Resource Reading Comprehension Datasets?* (round-trip filtering) — https://arxiv.org/pdf/2309.12426
- Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors*, ICLR 2024 — https://arxiv.org/abs/2309.03882
- Wortsman et al., *Model soups*, ICML 2022 — https://proceedings.mlr.press/v162/wortsman22a.html
- Gemma 4 model card — https://huggingface.co/google/gemma-4-31B-it
- Toukmaji & Flanigan, adaptation methods for low-resource languages, GEM 2025 (forgetting under gradient-based adaptation) — via https://aclanthology.org/people/christopher-toukmaji/unverified/
