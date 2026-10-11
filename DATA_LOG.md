# Data log

Every dataset touched by this project, how it is used, and every closed-API data-generation call (required
disclosure). Task data is never committed; `scripts/fetch_data.py` re-downloads it at the pinned revision.

## Evaluation data

| Dataset | Revision | License | Use | Notes |
|---|---|---|---|---|
| SinhalaMMLU public Easy tier (`naist-nlp/SinhalaMMLU`, TEST/easy) | `7dd30681d9` | CC BY-NC-ND 4.0 | Dev evaluation | 1,851 questions, 14 subjects, grades 6-8, 4 options. `Geography/64` dropped (answer label `24`). Cite Pramodya et al., EMNLP 2025. |
| Global-MMLU `si` (`CohereLabs/Global-MMLU`) | `0e619dbeb3` | Apache-2.0 | Harder internal validation: fixed 2,000-question stratified sample of `test` (ids in `configs/splits/gmmlu_si_val_ids.txt`); `dev` split = few-shot pool | Validation ids must be excluded from any training mix. |
| Belebele `sin_Sinh` (`facebook/belebele`) | `7899cdfa4e` | CC BY-SA 4.0 | **Evaluation only**, never training | 900 reading-comprehension questions. |

## Training data

| Dataset | Revision | License | Use | Processing |
|---|---|---|---|---|
| Global-MMLU `si` test (minus the 2,000 validation ids) + dev split | `0e619dbeb3` | Apache-2.0 | Fine-tuning MCQs: `data/sft/gmmlu_si_train.jsonl` via `scripts/build_sft_gmmlu.py` | 12,327 pool → 11,447 kept. Dropped 819 with char-5-gram TF-IDF cosine ≥ 0.6 to any validation item (MMLU repeats questions across subjects) and 61 within-pool copies (≥ 0.95). Max similarity of kept items to SinhalaMMLU dev: 0.42. Options shuffled to balanced gold positions except options that refer to other options. Stats: `reports/data_stats_sft_gmmlu.json` |
| FineWeb-2 `sin_Sinh` (`HuggingFaceFW/fineweb-2`) | `af9c13333e` | ODC-By 1.0 | Continued pretraining | `scripts/build_cpt_corpus.py` (filters below). Already MinHash-deduplicated upstream |
| Sinhala Wikipedia (`wikimedia/wikipedia`, `20231101.si`) | `b04c8d1ceb` | CC BY-SA 3.0 / GFDL | Continued pretraining | Same filters |
| NSINA news (`sinhala-nlp/NSINA`) | `a986af81ad` | CC BY-SA 4.0 | Continued pretraining | Same filters |

Corpus filters (`scripts/build_cpt_corpus.py`), first failing rule recorded per document; counts in
`reports/corpus_stats.json`: blocked exam / past-paper / curriculum domains; exam-like text (question-paper phrases or
≥ 8 numbered option lines); < 200 characters; < 50% Sinhala letters; < 30% unique lines; exact duplicate across
sources (normalised text); any word 8-gram shared with a SinhalaMMLU dev or Global-MMLU validation question.
Not yet available: CulturaX `si` (gated, access not requested), MADLAD-400 `si` (path not found).

### Contamination controls (apply to every training/pretraining source)

- SinhalaMMLU dev questions come from **pastpapers.wiki** (93%) and **e-thaksalawa.moe.gov.lk** (5%), i.e. Sri Lankan
  term-test and exam papers. Drop all documents from these domains, and from any other past-paper / exam site, from
  web corpora (FineWeb-2, CulturaX, MADLAD-400, CC-100, OSCAR) before training.
- N-gram overlap filter of every training corpus against SinhalaMMLU dev and the Global-MMLU validation split.
- No scraped Sri Lankan O/L or A/L past papers. Ask the organisers before using government textbooks.

## Closed-API data generation (disclosure)

_None yet._ For each use: date, provider and model, purpose, prompt template location, number of items generated,
filtering applied, and where the output is stored.
