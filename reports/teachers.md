# Baseline comparison

Accuracy in %. `orig` = original option order; `vote` = mean probabilities over cyclic option permutations; `±` = std of accuracy across permutations (position sensitivity). Random = 25.

## Overview

| model | params loaded | tok/char | SinhalaMMLU dev orig | vote | ± | macro-subj | Global-MMLU si | Belebele si | label mass | consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-4-E4B-it_logit | 7.94B | 0.40 | 55.2 | 54.5 | 1.1 | 55.4 | 52.0 | 79.8 | 0.99 | 0.49 |
| teacher_Qwen3.5-27B_gen | 26.90B | 0.53 | 64.8 | 64.8 | 0.0 | 64.9 | - | - | nan | 1.00 |
| teacher_Qwen3.5-27B_logit | 26.90B | 0.53 | 64.6 | 64.6 | 0.0 | 64.8 | 74.1 | - | 1.00 | 1.00 |
| teacher_gemma-4-26B-A4B-it_gen | 25.81B | 0.40 | 68.6 | 68.6 | 0.0 | 68.6 | - | - | nan | 1.00 |
| teacher_gemma-4-26B-A4B-it_logit | 25.81B | 0.40 | 68.3 | 68.3 | 0.0 | 68.3 | 66.5 | - | 1.00 | 1.00 |
| teacher_gemma-4-31B-it_gen | 31.27B | 0.40 | 73.5 | 73.5 | 0.0 | 73.4 | - | - | nan | 1.00 |
| teacher_gemma-4-31B-it_logit | 31.27B | 0.40 | 73.6 | 73.6 | 0.0 | 73.5 | 75.8 | - | 1.00 | 1.00 |

## SinhalaMMLU dev: per subject

| subject | n | gemma-4-E4B-it_logit | teacher_Qwen3.5-27B_gen | teacher_Qwen3.5-27B_logit | teacher_gemma-4-26B-A4B-it_gen | teacher_gemma-4-26B-A4B-it_logit | teacher_gemma-4-31B-it_gen | teacher_gemma-4-31B-it_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arts | 100 | 56.0 | 61.0 | 62.0 | 69.0 | 69.0 | 67.0 | 67.0 |
| Buddhism | 162 | 51.9 | 58.6 | 58.6 | 69.8 | 69.8 | 76.5 | 77.2 |
| Catholicism | 132 | 50.0 | 73.5 | 73.5 | 75.0 | 76.5 | 80.3 | 82.6 |
| Christianity | 130 | 62.3 | 76.9 | 76.9 | 73.8 | 73.1 | 80.8 | 80.0 |
| Civics | 123 | 74.8 | 81.3 | 80.5 | 82.9 | 82.1 | 84.6 | 83.7 |
| Eastern music | 143 | 45.5 | 55.2 | 53.8 | 58.7 | 57.3 | 60.1 | 60.1 |
| Geography | 116 | 64.7 | 74.1 | 74.1 | 74.1 | 72.4 | 84.5 | 84.5 |
| Health and Physical Science | 130 | 69.2 | 74.6 | 74.6 | 76.9 | 76.9 | 81.5 | 81.5 |
| History | 151 | 39.1 | 45.7 | 45.0 | 62.9 | 60.3 | 62.9 | 62.9 |
| Islam | 108 | 59.3 | 75.0 | 75.0 | 77.8 | 77.8 | 86.1 | 87.0 |
| Sinhala language and literature | 155 | 46.5 | 56.8 | 55.5 | 57.4 | 59.4 | 65.2 | 65.2 |
| dancing | 108 | 34.3 | 36.1 | 39.8 | 41.7 | 39.8 | 45.4 | 45.4 |
| drama and Theatre | 128 | 55.5 | 60.9 | 60.2 | 63.3 | 64.1 | 69.5 | 68.8 |
| science | 164 | 66.5 | 78.0 | 78.0 | 77.4 | 77.4 | 83.5 | 83.5 |

## SinhalaMMLU dev: per domain

| category | n | gemma-4-E4B-it_logit | teacher_Qwen3.5-27B_gen | teacher_Qwen3.5-27B_logit | teacher_gemma-4-26B-A4B-it_gen | teacher_gemma-4-26B-A4B-it_logit | teacher_gemma-4-31B-it_gen | teacher_gemma-4-31B-it_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| humanities | 1162 | 50.2 | 60.2 | 60.2 | 65.9 | 65.4 | 70.1 | 70.3 |
| language | 155 | 46.5 | 56.8 | 55.5 | 57.4 | 59.4 | 65.2 | 65.2 |
| social_science | 369 | 69.6 | 76.7 | 76.4 | 78.0 | 77.2 | 83.5 | 83.2 |
| stem | 164 | 66.5 | 78.0 | 78.0 | 77.4 | 77.4 | 83.5 | 83.5 |

## SinhalaMMLU dev: per grade

| grade | n | gemma-4-E4B-it_logit | teacher_Qwen3.5-27B_gen | teacher_Qwen3.5-27B_logit | teacher_gemma-4-26B-A4B-it_gen | teacher_gemma-4-26B-A4B-it_logit | teacher_gemma-4-31B-it_gen | teacher_gemma-4-31B-it_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 584 | 59.9 | 66.8 | 66.6 | 71.4 | 70.9 | 77.6 | 77.6 |
| 7 | 634 | 53.0 | 63.9 | 63.6 | 68.9 | 68.9 | 73.7 | 74.0 |
| 8 | 632 | 53.0 | 63.8 | 63.9 | 65.8 | 65.3 | 69.6 | 69.6 |

## Global-MMLU si: per category

| category | n | gemma-4-E4B-it_logit | teacher_Qwen3.5-27B_gen | teacher_Qwen3.5-27B_logit | teacher_gemma-4-26B-A4B-it_gen | teacher_gemma-4-26B-A4B-it_logit | teacher_gemma-4-31B-it_gen | teacher_gemma-4-31B-it_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Business | 102 | 54.9 | - | 73.5 | - | 68.6 | - | 78.4 |
| Humanities | 668 | 46.9 | - | 66.0 | - | 58.2 | - | 70.2 |
| Medical | 216 | 54.2 | - | 76.9 | - | 70.8 | - | 75.9 |
| Other | 126 | 56.3 | - | 74.6 | - | 68.3 | - | 77.8 |
| STEM | 449 | 51.9 | - | 78.2 | - | 68.2 | - | 76.2 |
| Social Sciences | 439 | 57.2 | - | 80.6 | - | 74.3 | - | 82.9 |

## Global-MMLU si: culturally sensitive (CS) / agnostic (CA) / unlabelled (-)

| cultural_sensitivity | n | gemma-4-E4B-it_logit | teacher_Qwen3.5-27B_gen | teacher_Qwen3.5-27B_logit | teacher_gemma-4-26B-A4B-it_gen | teacher_gemma-4-26B-A4B-it_logit | teacher_gemma-4-31B-it_gen | teacher_gemma-4-31B-it_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| - | 1603 | 51.2 | - | 73.7 | - | 66.7 | - | 76.2 |
| CA | 277 | 57.8 | - | 77.3 | - | 68.6 | - | 77.3 |
| CS | 120 | 50.0 | - | 70.8 | - | 59.2 | - | 67.5 |

## Position bias on SinhalaMMLU dev (share of predictions per displayed position)

| model | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| gemma-4-E4B-it_logit | 22.4 | 31.0 | 26.3 | 20.4 |
| teacher_Qwen3.5-27B_gen | 21.8 | 31.3 | 26.4 | 20.5 |
| teacher_Qwen3.5-27B_logit | 22.0 | 31.1 | 26.4 | 20.5 |
| teacher_gemma-4-26B-A4B-it_gen | 24.9 | 29.9 | 25.0 | 20.2 |
| teacher_gemma-4-26B-A4B-it_logit | 24.9 | 30.4 | 25.0 | 19.7 |
| teacher_gemma-4-31B-it_gen | 24.9 | 30.1 | 25.8 | 19.1 |
| teacher_gemma-4-31B-it_logit | 24.8 | 29.9 | 26.1 | 19.2 |
