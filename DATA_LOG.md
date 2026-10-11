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

_None yet._ For each source record: name, URL, revision/date, license, size used, filtering and dedup applied.

### Contamination controls (apply to every training/pretraining source)

- SinhalaMMLU dev questions come from **pastpapers.wiki** (93%) and **e-thaksalawa.moe.gov.lk** (5%), i.e. Sri Lankan
  term-test and exam papers. Drop all documents from these domains, and from any other past-paper / exam site, from
  web corpora (FineWeb-2, CulturaX, MADLAD-400, CC-100, OSCAR) before training.
- N-gram overlap filter of every training corpus against SinhalaMMLU dev and the Global-MMLU validation split.
- No scraped Sri Lankan O/L or A/L past papers. Ask the organisers before using government textbooks.

## Closed-API data generation (disclosure)

_None yet._ For each use: date, provider and model, purpose, prompt template location, number of items generated,
filtering applied, and where the output is stored.
