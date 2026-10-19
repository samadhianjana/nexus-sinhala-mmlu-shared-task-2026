# Baseline comparison

Accuracy in %. `orig` = original option order; `vote` = mean probabilities over cyclic option permutations; `±` = std of accuracy across permutations (position sensitivity). Random = 25.

## Overview

| model | params loaded | tok/char | SinhalaMMLU dev orig | vote | ± | macro-subj | Global-MMLU si | Belebele si | label mass | consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-4-E4B-it_logit | 7.94B | 0.40 | 55.2 | 54.5 | 1.1 | 55.4 | 52.0 | 79.8 | 0.99 | 0.49 |
| sft_gmmlu_v1_step-357_logit | 7.46B | 0.40 | 49.6 | 52.7 | 1.1 | 49.7 | 54.1 | - | 1.00 | 0.35 |
| sft_gmmlu_v1_step-714_logit | 7.46B | 0.40 | 52.3 | 53.2 | 0.6 | 52.4 | 56.9 | - | 1.00 | 0.57 |

## SinhalaMMLU dev: per subject

| subject | n | gemma-4-E4B-it_logit | sft_gmmlu_v1_step-357_logit | sft_gmmlu_v1_step-714_logit |
|---|---:|---:|---:|---:|
| Arts | 100 | 56.0 | 41.0 | 47.0 |
| Buddhism | 162 | 51.9 | 42.6 | 48.8 |
| Catholicism | 132 | 50.0 | 50.0 | 53.0 |
| Christianity | 130 | 62.3 | 60.8 | 66.2 |
| Civics | 123 | 74.8 | 68.3 | 69.1 |
| Eastern music | 143 | 45.5 | 39.9 | 37.8 |
| Geography | 116 | 64.7 | 57.8 | 60.3 |
| Health and Physical Science | 130 | 69.2 | 59.2 | 64.6 |
| History | 151 | 39.1 | 39.1 | 38.4 |
| Islam | 108 | 59.3 | 57.4 | 59.3 |
| Sinhala language and literature | 155 | 46.5 | 38.1 | 48.4 |
| dancing | 108 | 34.3 | 25.9 | 27.8 |
| drama and Theatre | 128 | 55.5 | 57.8 | 55.5 |
| science | 164 | 66.5 | 58.5 | 57.9 |

## SinhalaMMLU dev: per domain

| category | n | gemma-4-E4B-it_logit | sft_gmmlu_v1_step-357_logit | sft_gmmlu_v1_step-714_logit |
|---|---:|---:|---:|---:|
| humanities | 1162 | 50.2 | 46.0 | 48.1 |
| language | 155 | 46.5 | 38.1 | 48.4 |
| social_science | 369 | 69.6 | 61.8 | 64.8 |
| stem | 164 | 66.5 | 58.5 | 57.9 |

## SinhalaMMLU dev: per grade

| grade | n | gemma-4-E4B-it_logit | sft_gmmlu_v1_step-357_logit | sft_gmmlu_v1_step-714_logit |
|---|---:|---:|---:|---:|
| 6 | 584 | 59.9 | 52.7 | 54.5 |
| 7 | 634 | 53.0 | 47.8 | 52.4 |
| 8 | 632 | 53.0 | 48.6 | 50.3 |

## Global-MMLU si: per category

| category | n | gemma-4-E4B-it_logit | sft_gmmlu_v1_step-357_logit | sft_gmmlu_v1_step-714_logit |
|---|---:|---:|---:|---:|
| Business | 102 | 54.9 | 57.8 | 54.9 |
| Humanities | 668 | 46.9 | 49.3 | 53.3 |
| Medical | 216 | 54.2 | 56.5 | 56.9 |
| Other | 126 | 56.3 | 61.1 | 60.3 |
| STEM | 449 | 51.9 | 53.9 | 57.7 |
| Social Sciences | 439 | 57.2 | 57.9 | 61.0 |

## Global-MMLU si: culturally sensitive (CS) / agnostic (CA) / unlabelled (-)

| cultural_sensitivity | n | gemma-4-E4B-it_logit | sft_gmmlu_v1_step-357_logit | sft_gmmlu_v1_step-714_logit |
|---|---:|---:|---:|---:|
| - | 1603 | 51.2 | 54.1 | 57.0 |
| CA | 277 | 57.8 | 54.9 | 58.5 |
| CS | 120 | 50.0 | 53.3 | 51.7 |

## Position bias on SinhalaMMLU dev (share of predictions per displayed position)

| model | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| gemma-4-E4B-it_logit | 22.4 | 31.0 | 26.3 | 20.4 |
| sft_gmmlu_v1_step-357_logit | 10.2 | 33.7 | 26.0 | 30.1 |
| sft_gmmlu_v1_step-714_logit | 27.3 | 26.4 | 25.7 | 20.7 |
