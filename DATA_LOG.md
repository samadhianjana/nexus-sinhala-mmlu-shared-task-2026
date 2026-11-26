# Data log

A record of every dataset used in this project, how it was used, and how model-generated data was produced.
Task data is never committed; `scripts/fetch_data.py` re-downloads it at the pinned revision.

**The submitted system uses no training data.** It is the base model, prompted zero-shot, so the test questions are the
only data it sees. Everything below the evaluation table was used in development experiments that did not enter the
final system.

## Evaluation data

| Dataset | Revision | License | Use | Notes |
|---|---|---|---|---|
| SinhalaMMLU public Easy tier (`naist-nlp/SinhalaMMLU`, TEST/easy) | `7dd30681d9` | CC BY-NC-ND 4.0 | Development evaluation. In development experiments only, also used as training data (5-fold cross-validated LoRA, and questions for round-1 reasoning traces) | 1,851 questions, 14 subjects, grades 6-8, 4 options; 1,850 used because `Geography/64` was dropped (answer label `24`). Cite Pramodya et al., EMNLP 2025. |
| Global-MMLU `si` (`CohereLabs/Global-MMLU`) | `0e619dbeb3` | Apache-2.0 | Harder internal validation: fixed 2,000-question stratified sample of `test` (ids in `configs/splits/gmmlu_si_val_ids.txt`); the `dev` split was added to the training pool | Validation ids are excluded from every training mix. |
| Shared-task test set (`SinhalaMMLU-SharedTask/Sinhala-MMLU-SharedTask-Test-Dataset`) | `56fb2da50c` | Set by the organisers | **Prediction only.** Never used for training, tuning, prompt design or model selection; no answers included | 2,238 questions (easy 642, medium 1,111, hard 485), 4 options, same JSON layout as dev. Stored in `data/test/` (git-ignored). |
| Belebele `sin_Sinh` (`facebook/belebele`) | `7899cdfa4e` | CC BY-SA 4.0 | **Evaluation only**, never training | 900 reading-comprehension questions. |

## Training data (development experiments only)

| Dataset | Revision | License | Use | Processing |
|---|---|---|---|---|
| Global-MMLU `si` test (minus the 2,000 validation ids) + dev split | `0e619dbeb3` | Apache-2.0 | Fine-tuning MCQs: `data/sft/gmmlu_si_train.jsonl` via `scripts/build_sft_gmmlu.py` | 12,327 pool → 11,447 kept. Dropped 819 with char-5-gram TF-IDF cosine ≥ 0.6 to any validation item (MMLU repeats questions across subjects) and 61 within-pool copies (≥ 0.95). Max similarity of kept items to SinhalaMMLU dev: 0.42. Options shuffled to balanced gold positions except options that refer to other options. Stats: `reports/data_stats_sft_gmmlu.json` |
| FineWeb-2 `sin_Sinh` (`HuggingFaceFW/fineweb-2`) | `af9c13333e` | ODC-By 1.0 | Continued pre-training; passages for synthetic MCQs | `scripts/build_cpt_corpus.py` (filters below). Already MinHash-deduplicated upstream |
| Sinhala Wikipedia (`wikimedia/wikipedia`, `20231101.si`) | `b04c8d1ceb` | CC BY-SA 3.0 / GFDL | Continued pre-training; passages for synthetic MCQs | Same filters |
| NSINA news (`sinhala-nlp/NSINA`) | `a986af81ad` | CC BY-SA 4.0 | Continued pre-training; passages for synthetic MCQs | Same filters |
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample/10BT`) | `87f09149ef` | ODC-By 1.0 | English share (10%) of the continued pre-training mix | `scripts/select_cpt_subset.py`; not deduplicated against the Sinhala sources |

Corpus filters (`scripts/build_cpt_corpus.py`), first failing rule recorded per document; counts in
`reports/corpus_stats.json`: blocked exam / past-paper / curriculum domains; exam-like text (question-paper phrases or
≥ 8 numbered option lines); < 200 characters; < 50% Sinhala letters; < 30% unique lines; exact duplicate across
sources (normalised text); any word 8-gram shared with a SinhalaMMLU dev or Global-MMLU validation question.
Not used: CulturaX `si` (gated, access not requested) and MADLAD-400 `si` (path not found).

## Model-generated data (development experiments only; not released)

All of it was produced locally with open-weight models.

| Data | Generator | Source material | Size | Checks |
|---|---|---|---|---|
| Synthetic Sinhala MCQs (pilot) | `google/gemma-4-31B-it`, `scripts/generate_mcq.py` | Sinhala passages assigned to curriculum subjects by keyword lists built from the published subject list (`scripts/build_passage_pool.py`; Wikipedia preferred, then FineWeb-2 and NSINA) | 441 kept | Schema checks (4 distinct options, key 1-4, mostly Sinhala, no reference to "the passage"); round trip (the generator re-answers each question with the passage and must agree with its own key); 50 random items exported for native-speaker review |
| Teacher soft labels | `google/gemma-4-31B-it`, `scripts/add_teacher_labels.py` | The 11,447 Global-MMLU training questions | Option probabilities per question | Used for the distillation experiment |
| Own reasoning traces, round 1 | Gemma 4 E4B-it in thinking mode, `scripts/gen_traces.py` | SinhalaMMLU dev questions outside a held-out 400 | 1,577 kept of 2,560 samples | Only traces whose final answer matches the key are kept |
| Own reasoning traces, round 2 | Gemma 4 E4B-it in thinking mode, `scripts/gen_traces.py` | Global-MMLU `si` Humanities / Social Sciences / Other / Business training questions and the synthetic MCQs | 683 kept of 1,024 samples | Same |

The traces and part of the synthetic data derive from CC BY-NC-ND dev questions, so none of this data is redistributed.

### Contamination controls (apply to every training and pre-training source)

- SinhalaMMLU dev questions come from **pastpapers.wiki** (93%) and **e-thaksalawa.moe.gov.lk** (5%), i.e. Sri Lankan
  term-test and exam papers. Documents from these domains, and from other past-paper and exam sites, were removed from
  the web corpora (FineWeb-2, NSINA, Wikipedia) before training.
- Every training corpus was filtered by n-gram overlap against SinhalaMMLU dev and the Global-MMLU validation split; the
  Global-MMLU pool was also checked against dev with multilingual-e5-large embeddings (`scripts/semantic_overlap.py`).
- No scraped Sri Lankan O/L or A/L past papers and no government textbooks were used.

## Closed-API data generation

None. No closed or paid API was used to generate data at any stage; all generated data came from the locally run
open-weight models listed above. Claude Code, an AI coding assistant, was used only for writing and running code and for
analysis, never to produce training data, labels or answers.
