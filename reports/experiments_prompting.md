# Baseline comparison

Accuracy in %. `orig` = original option order; `vote` = mean probabilities over cyclic option permutations; `±` = std of accuracy across permutations (position sensitivity). Random = 25.

## Overview

| model | params loaded | tok/char | SinhalaMMLU dev orig | vote | ± | macro-subj | Global-MMLU si | Belebele si | label mass | consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B_logit | 4.21B | 0.53 | 40.6 | 47.6 | 1.5 | 40.5 | 44.2 | 65.7 | 1.00 | 0.19 |
| gemma-3n-E4B-it_logit | 7.82B | 0.40 | 50.3 | 51.7 | 1.5 | 50.5 | 46.2 | 73.8 | 1.00 | 0.39 |
| gemma-4-E4B-it_logit | 7.94B | 0.40 | 55.2 | 54.5 | 1.1 | 55.4 | 52.0 | 79.8 | 0.99 | 0.49 |
| Qwen3.5-4B_generate | 4.21B | 0.53 | 40.6 | 40.6 | 0.0 | 40.4 | 44.2 | - | nan | 1.00 |
| Qwen3.5-4B_logit_5shot | 4.21B | 0.53 | 46.6 | 46.6 | 0.0 | 46.3 | 45.4 | - | 1.00 | 1.00 |
| Qwen3.5-4B_logit_si | 4.21B | 0.79 | 43.8 | 45.5 | 1.6 | 43.7 | 42.8 | - | 0.94 | 0.19 |
| gemma-3n-E4B-it_cloze | 7.82B | 0.40 | 32.1 | 32.1 | 0.0 | 32.2 | 30.6 | - | nan | 1.00 |
| gemma-3n-E4B-it_generate | 7.82B | 0.40 | 50.3 | 50.3 | 0.0 | 50.4 | 46.2 | - | nan | 1.00 |
| gemma-3n-E4B-it_logit_5shot | 7.82B | 0.40 | 50.1 | 50.1 | 0.0 | 50.2 | 46.0 | - | 1.00 | 1.00 |
| gemma-3n-E4B-it_logit_si | 7.82B | 0.53 | 50.3 | 51.9 | 1.1 | 50.5 | 46.2 | - | 0.99 | 0.37 |
| gemma-4-E4B-it_cloze | 7.94B | 0.40 | 26.5 | 26.5 | 0.0 | 26.7 | 24.9 | - | nan | 1.00 |
| gemma-4-E4B-it_generate | 7.94B | 0.40 | 55.1 | 55.1 | 0.0 | 55.2 | 52.3 | - | nan | 1.00 |
| gemma-4-E4B-it_logit_5shot | 7.94B | 0.40 | 55.1 | 55.1 | 0.0 | 55.2 | 51.7 | - | 1.00 | 1.00 |
| gemma-4-E4B-it_logit_si | 7.94B | 0.53 | 55.5 | 55.4 | 1.4 | 55.6 | 52.1 | - | 0.96 | 0.46 |

## SinhalaMMLU dev: per subject

| subject | n | Qwen3.5-4B_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | Qwen3.5-4B_generate | Qwen3.5-4B_logit_5shot | Qwen3.5-4B_logit_si | gemma-3n-E4B-it_cloze | gemma-3n-E4B-it_generate | gemma-3n-E4B-it_logit_5shot | gemma-3n-E4B-it_logit_si | gemma-4-E4B-it_cloze | gemma-4-E4B-it_generate | gemma-4-E4B-it_logit_5shot | gemma-4-E4B-it_logit_si |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arts | 100 | 33.0 | 49.0 | 56.0 | 31.0 | 36.0 | 43.0 | 36.0 | 48.0 | 48.0 | 50.0 | 26.0 | 55.0 | 55.0 | 58.0 |
| Buddhism | 162 | 29.6 | 46.3 | 51.9 | 31.5 | 40.6 | 34.6 | 29.0 | 46.3 | 50.6 | 44.4 | 28.4 | 51.9 | 53.8 | 51.9 |
| Catholicism | 132 | 31.8 | 47.0 | 50.0 | 31.8 | 38.6 | 38.6 | 31.8 | 48.5 | 43.2 | 47.7 | 23.5 | 48.5 | 47.7 | 51.5 |
| Christianity | 130 | 47.7 | 46.9 | 62.3 | 46.2 | 41.5 | 46.2 | 34.6 | 46.2 | 50.0 | 50.8 | 29.2 | 63.1 | 62.3 | 60.8 |
| Civics | 123 | 61.8 | 66.7 | 74.8 | 61.8 | 71.5 | 60.2 | 32.5 | 67.5 | 65.9 | 67.5 | 27.6 | 74.8 | 74.8 | 72.4 |
| Eastern music | 143 | 26.6 | 49.7 | 45.5 | 26.6 | 43.4 | 38.5 | 30.1 | 49.0 | 42.7 | 41.3 | 25.9 | 46.9 | 46.2 | 47.6 |
| Geography | 116 | 37.1 | 55.2 | 64.7 | 37.1 | 44.8 | 39.7 | 30.2 | 55.2 | 50.9 | 54.3 | 31.0 | 63.8 | 66.4 | 67.2 |
| Health and Physical Science | 130 | 53.1 | 61.5 | 69.2 | 53.1 | 55.8 | 59.2 | 40.8 | 61.5 | 60.5 | 66.2 | 32.3 | 69.2 | 69.0 | 67.7 |
| History | 151 | 35.1 | 34.4 | 39.1 | 34.4 | 33.1 | 34.4 | 31.8 | 34.4 | 39.7 | 39.1 | 26.5 | 39.1 | 36.4 | 39.7 |
| Islam | 108 | 41.7 | 52.8 | 59.3 | 41.7 | 43.5 | 38.9 | 37.0 | 52.8 | 61.1 | 54.6 | 25.9 | 59.3 | 53.7 | 56.5 |
| Sinhala language and literature | 155 | 37.4 | 43.2 | 46.5 | 38.7 | 40.9 | 35.5 | 33.5 | 43.2 | 42.9 | 43.9 | 27.1 | 45.2 | 50.6 | 47.1 |
| dancing | 108 | 29.6 | 40.7 | 34.3 | 29.6 | 34.3 | 32.4 | 25.0 | 39.8 | 36.1 | 38.9 | 27.8 | 33.3 | 37.0 | 30.6 |
| drama and Theatre | 128 | 44.5 | 53.1 | 55.5 | 44.5 | 53.9 | 46.9 | 27.3 | 51.6 | 54.7 | 49.2 | 22.7 | 56.2 | 53.9 | 59.4 |
| science | 164 | 58.5 | 60.4 | 66.5 | 57.9 | 69.9 | 64.0 | 31.1 | 61.6 | 57.1 | 59.1 | 19.5 | 67.1 | 65.6 | 67.7 |

## SinhalaMMLU dev: per domain

| category | n | Qwen3.5-4B_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | Qwen3.5-4B_generate | Qwen3.5-4B_logit_5shot | Qwen3.5-4B_logit_si | gemma-3n-E4B-it_cloze | gemma-3n-E4B-it_generate | gemma-3n-E4B-it_logit_5shot | gemma-3n-E4B-it_logit_si | gemma-4-E4B-it_cloze | gemma-4-E4B-it_generate | gemma-4-E4B-it_logit_5shot | gemma-4-E4B-it_logit_si |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| humanities | 1162 | 35.3 | 46.4 | 50.2 | 35.1 | 40.6 | 39.1 | 31.2 | 46.0 | 47.2 | 45.9 | 26.2 | 50.2 | 49.4 | 50.5 |
| language | 155 | 37.4 | 43.2 | 46.5 | 38.7 | 40.9 | 35.5 | 33.5 | 43.2 | 42.9 | 43.9 | 27.1 | 45.2 | 50.6 | 47.1 |
| social_science | 369 | 50.9 | 61.2 | 69.6 | 50.9 | 57.6 | 53.4 | 34.7 | 61.5 | 59.2 | 62.9 | 30.4 | 69.4 | 70.1 | 69.1 |
| stem | 164 | 58.5 | 60.4 | 66.5 | 57.9 | 69.9 | 64.0 | 31.1 | 61.6 | 57.1 | 59.1 | 19.5 | 67.1 | 65.6 | 67.7 |

## SinhalaMMLU dev: per grade

| grade | n | Qwen3.5-4B_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | Qwen3.5-4B_generate | Qwen3.5-4B_logit_5shot | Qwen3.5-4B_logit_si | gemma-3n-E4B-it_cloze | gemma-3n-E4B-it_generate | gemma-3n-E4B-it_logit_5shot | gemma-3n-E4B-it_logit_si | gemma-4-E4B-it_cloze | gemma-4-E4B-it_generate | gemma-4-E4B-it_logit_5shot | gemma-4-E4B-it_logit_si |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 584 | 44.3 | 52.9 | 59.9 | 44.3 | 48.4 | 46.9 | 36.5 | 52.1 | 53.5 | 52.2 | 26.0 | 59.9 | 60.7 | 61.6 |
| 7 | 634 | 39.4 | 49.7 | 53.0 | 39.6 | 47.2 | 42.6 | 29.2 | 50.2 | 48.8 | 50.2 | 25.4 | 52.8 | 51.3 | 52.8 |
| 8 | 632 | 38.4 | 48.6 | 53.0 | 38.1 | 44.4 | 42.2 | 31.0 | 48.7 | 48.2 | 48.6 | 28.2 | 52.8 | 53.6 | 52.4 |

## Global-MMLU si: per category

| category | n | Qwen3.5-4B_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | Qwen3.5-4B_generate | Qwen3.5-4B_logit_5shot | Qwen3.5-4B_logit_si | gemma-3n-E4B-it_cloze | gemma-3n-E4B-it_generate | gemma-3n-E4B-it_logit_5shot | gemma-3n-E4B-it_logit_si | gemma-4-E4B-it_cloze | gemma-4-E4B-it_generate | gemma-4-E4B-it_logit_5shot | gemma-4-E4B-it_logit_si |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Business | 102 | 37.3 | 52.9 | 54.9 | 40.2 | 43.1 | 42.2 | 27.5 | 53.9 | 56.9 | 51.0 | 36.3 | 55.9 | 55.9 | 53.9 |
| Humanities | 668 | 37.3 | 42.8 | 46.9 | 37.0 | 37.6 | 37.3 | 30.1 | 42.4 | 42.7 | 45.1 | 26.9 | 47.0 | 46.1 | 46.9 |
| Medical | 216 | 50.0 | 48.6 | 54.2 | 50.5 | 47.7 | 44.0 | 27.8 | 49.1 | 47.7 | 49.5 | 23.1 | 54.6 | 54.2 | 56.0 |
| Other | 126 | 43.7 | 55.6 | 56.3 | 42.9 | 45.2 | 47.6 | 42.9 | 56.3 | 56.3 | 53.2 | 22.2 | 56.3 | 55.6 | 57.9 |
| STEM | 449 | 45.7 | 41.9 | 51.9 | 45.4 | 48.1 | 41.9 | 33.9 | 41.9 | 42.5 | 39.2 | 22.3 | 51.7 | 53.7 | 50.6 |
| Social Sciences | 439 | 52.4 | 50.6 | 57.2 | 52.4 | 54.0 | 50.3 | 26.7 | 50.1 | 48.3 | 50.3 | 23.5 | 57.9 | 55.1 | 57.6 |

## Global-MMLU si: culturally sensitive (CS) / agnostic (CA) / unlabelled (-)

| cultural_sensitivity | n | Qwen3.5-4B_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | Qwen3.5-4B_generate | Qwen3.5-4B_logit_5shot | Qwen3.5-4B_logit_si | gemma-3n-E4B-it_cloze | gemma-3n-E4B-it_generate | gemma-3n-E4B-it_logit_5shot | gemma-3n-E4B-it_logit_si | gemma-4-E4B-it_cloze | gemma-4-E4B-it_generate | gemma-4-E4B-it_logit_5shot | gemma-4-E4B-it_logit_si |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - | 1603 | 43.8 | 45.7 | 51.2 | 43.7 | 45.0 | 41.9 | 30.6 | 45.7 | 45.3 | 46.7 | 25.5 | 51.3 | 51.3 | 50.3 |
| CA | 277 | 47.7 | 46.6 | 57.8 | 47.7 | 47.7 | 47.3 | 28.2 | 46.2 | 48.7 | 41.9 | 21.3 | 59.2 | 54.9 | 59.2 |
| CS | 120 | 42.5 | 52.5 | 50.0 | 43.3 | 45.0 | 44.2 | 35.8 | 52.5 | 49.2 | 50.0 | 25.0 | 50.0 | 50.8 | 60.0 |

## Position bias on SinhalaMMLU dev (share of predictions per displayed position)

| model | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B_logit | 11.0 | 7.9 | 27.7 | 53.4 |
| gemma-3n-E4B-it_logit | 14.0 | 29.6 | 32.7 | 23.7 |
| gemma-4-E4B-it_logit | 22.4 | 31.0 | 26.3 | 20.4 |
| Qwen3.5-4B_generate | 10.5 | 8.8 | 29.2 | 51.4 |
| Qwen3.5-4B_logit_5shot | 11.2 | 22.0 | 38.0 | 28.8 |
| Qwen3.5-4B_logit_si | 9.0 | 18.9 | 47.3 | 24.7 |
| gemma-3n-E4B-it_cloze | 27.9 | 25.1 | 23.2 | 23.7 |
| gemma-3n-E4B-it_generate | 13.6 | 31.4 | 34.5 | 20.5 |
| gemma-3n-E4B-it_logit_5shot | 22.0 | 30.8 | 32.8 | 14.4 |
| gemma-3n-E4B-it_logit_si | 14.3 | 30.9 | 35.1 | 19.7 |
| gemma-4-E4B-it_cloze | 25.5 | 24.7 | 23.8 | 26.0 |
| gemma-4-E4B-it_generate | 22.6 | 32.9 | 26.8 | 17.6 |
| gemma-4-E4B-it_logit_5shot | 22.8 | 30.7 | 29.0 | 17.6 |
| gemma-4-E4B-it_logit_si | 21.4 | 33.4 | 26.1 | 19.1 |
