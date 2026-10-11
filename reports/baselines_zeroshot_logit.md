# Baseline comparison

Accuracy in %. `orig` = original option order; `vote` = mean probabilities over cyclic option permutations; `±` = std of accuracy across permutations (position sensitivity). Random = 25.

## Overview

| model | params loaded | tok/char | SinhalaMMLU dev orig | vote | ± | macro-subj | Global-MMLU si | Belebele si | label mass | consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct_logit | 3.21B | 1.02 | 28.3 | 29.9 | 2.6 | 28.2 | 24.7 | 29.1 | 1.00 | 0.00 |
| Phi-4-mini-instruct_logit | 3.84B | 0.42 | 31.5 | 29.1 | 2.4 | 31.4 | 28.6 | 39.9 | 0.98 | 0.07 |
| Qwen3-4B-Instruct-2507_logit | 4.02B | 0.85 | 34.6 | 37.2 | 0.8 | 34.5 | 37.5 | 56.0 | 1.00 | 0.21 |
| Qwen3-4B_logit | 4.02B | 0.85 | 34.3 | 35.1 | 1.5 | 34.2 | 31.8 | 50.4 | 1.00 | 0.15 |
| Qwen3.5-4B_logit | 4.21B | 0.53 | 40.6 | 47.6 | 1.5 | 40.5 | 44.2 | 65.7 | 1.00 | 0.19 |
| SmolLM3-3B_logit | 3.08B | 1.02 | 28.9 | 29.4 | 2.2 | 29.0 | 26.9 | 29.8 | 0.62 | 0.01 |
| gemma-3-4b-it_logit | 4.30B | 0.40 | 39.0 | 40.2 | 1.3 | 39.1 | 35.4 | 55.7 | 1.00 | 0.19 |
| gemma-3n-E4B-it_logit | 7.82B | 0.40 | 50.3 | 51.7 | 1.5 | 50.5 | 46.2 | 73.8 | 1.00 | 0.39 |
| gemma-4-E4B-it_logit | 7.94B | 0.40 | 55.2 | 54.5 | 1.1 | 55.4 | 52.0 | 79.8 | 0.99 | 0.49 |
| granite-4.2-3b_logit | 3.66B | 1.04 | 29.1 | 29.2 | 1.7 | 29.4 | 27.4 | 30.6 | 0.99 | 0.04 |

## SinhalaMMLU dev: per subject

| subject | n | Llama-3.2-3B-Instruct_logit | Phi-4-mini-instruct_logit | Qwen3-4B-Instruct-2507_logit | Qwen3-4B_logit | Qwen3.5-4B_logit | SmolLM3-3B_logit | gemma-3-4b-it_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | granite-4.2-3b_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arts | 100 | 26.0 | 27.0 | 27.0 | 27.0 | 33.0 | 28.0 | 38.0 | 49.0 | 56.0 | 30.0 |
| Buddhism | 162 | 27.2 | 34.0 | 30.2 | 32.7 | 29.6 | 22.8 | 29.6 | 46.3 | 51.9 | 27.8 |
| Catholicism | 132 | 25.0 | 28.0 | 27.3 | 31.1 | 31.8 | 23.5 | 42.4 | 47.0 | 50.0 | 31.8 |
| Christianity | 130 | 26.2 | 32.3 | 30.8 | 30.0 | 47.7 | 28.5 | 40.0 | 46.9 | 62.3 | 26.2 |
| Civics | 123 | 34.1 | 34.1 | 47.2 | 43.1 | 61.8 | 36.6 | 56.1 | 66.7 | 74.8 | 30.1 |
| Eastern music | 143 | 30.8 | 33.6 | 35.7 | 32.9 | 26.6 | 32.2 | 38.5 | 49.7 | 45.5 | 22.4 |
| Geography | 116 | 26.7 | 29.3 | 24.1 | 44.0 | 37.1 | 30.2 | 37.1 | 55.2 | 64.7 | 34.5 |
| Health and Physical Science | 130 | 23.8 | 39.2 | 50.0 | 46.9 | 53.1 | 30.0 | 48.5 | 61.5 | 69.2 | 34.6 |
| History | 151 | 22.5 | 25.2 | 27.8 | 31.8 | 35.1 | 27.2 | 29.1 | 34.4 | 39.1 | 29.1 |
| Islam | 108 | 25.0 | 28.7 | 34.3 | 34.3 | 41.7 | 25.0 | 37.0 | 52.8 | 59.3 | 34.3 |
| Sinhala language and literature | 155 | 30.3 | 28.4 | 35.5 | 29.7 | 37.4 | 29.7 | 33.5 | 43.2 | 46.5 | 23.2 |
| dancing | 108 | 30.6 | 32.4 | 34.3 | 25.9 | 29.6 | 31.5 | 31.5 | 40.7 | 34.3 | 26.9 |
| drama and Theatre | 128 | 34.4 | 32.0 | 40.6 | 33.6 | 44.5 | 32.0 | 41.4 | 53.1 | 55.5 | 33.6 |
| science | 164 | 32.3 | 34.8 | 38.4 | 36.6 | 58.5 | 29.3 | 45.1 | 60.4 | 66.5 | 27.4 |

## SinhalaMMLU dev: per domain

| category | n | Llama-3.2-3B-Instruct_logit | Phi-4-mini-instruct_logit | Qwen3-4B-Instruct-2507_logit | Qwen3-4B_logit | Qwen3.5-4B_logit | SmolLM3-3B_logit | gemma-3-4b-it_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | granite-4.2-3b_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| humanities | 1162 | 27.5 | 30.5 | 31.9 | 31.2 | 35.3 | 27.7 | 36.1 | 46.4 | 50.2 | 28.9 |
| language | 155 | 30.3 | 28.4 | 35.5 | 29.7 | 37.4 | 29.7 | 33.5 | 43.2 | 46.5 | 23.2 |
| social_science | 369 | 28.2 | 34.4 | 40.9 | 44.7 | 50.9 | 32.2 | 47.4 | 61.2 | 69.6 | 33.1 |
| stem | 164 | 32.3 | 34.8 | 38.4 | 36.6 | 58.5 | 29.3 | 45.1 | 60.4 | 66.5 | 27.4 |

## SinhalaMMLU dev: per grade

| grade | n | Llama-3.2-3B-Instruct_logit | Phi-4-mini-instruct_logit | Qwen3-4B-Instruct-2507_logit | Qwen3-4B_logit | Qwen3.5-4B_logit | SmolLM3-3B_logit | gemma-3-4b-it_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | granite-4.2-3b_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 584 | 29.6 | 31.8 | 37.3 | 37.7 | 44.3 | 31.7 | 43.7 | 52.9 | 59.9 | 29.1 |
| 7 | 634 | 29.3 | 32.6 | 32.8 | 33.6 | 39.4 | 28.1 | 37.7 | 49.7 | 53.0 | 29.7 |
| 8 | 632 | 25.9 | 29.9 | 33.9 | 31.8 | 38.4 | 27.2 | 35.9 | 48.6 | 53.0 | 28.6 |

## Global-MMLU si: per category

| category | n | Llama-3.2-3B-Instruct_logit | Phi-4-mini-instruct_logit | Qwen3-4B-Instruct-2507_logit | Qwen3-4B_logit | Qwen3.5-4B_logit | SmolLM3-3B_logit | gemma-3-4b-it_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | granite-4.2-3b_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Business | 102 | 27.5 | 32.4 | 44.1 | 37.3 | 37.3 | 25.5 | 43.1 | 52.9 | 54.9 | 30.4 |
| Humanities | 668 | 25.0 | 28.3 | 33.8 | 31.1 | 37.3 | 28.9 | 36.1 | 42.8 | 46.9 | 27.8 |
| Medical | 216 | 27.3 | 29.2 | 38.0 | 25.5 | 50.0 | 26.9 | 31.5 | 48.6 | 54.2 | 35.2 |
| Other | 126 | 27.0 | 31.7 | 34.1 | 31.7 | 43.7 | 26.2 | 42.1 | 55.6 | 56.3 | 24.6 |
| STEM | 449 | 24.1 | 27.6 | 41.0 | 34.7 | 45.7 | 26.1 | 33.2 | 41.9 | 51.9 | 29.4 |
| Social Sciences | 439 | 22.3 | 28.2 | 38.7 | 31.4 | 52.4 | 25.3 | 35.1 | 50.6 | 57.2 | 21.0 |

## Global-MMLU si: culturally sensitive (CS) / agnostic (CA) / unlabelled (-)

| cultural_sensitivity | n | Llama-3.2-3B-Instruct_logit | Phi-4-mini-instruct_logit | Qwen3-4B-Instruct-2507_logit | Qwen3-4B_logit | Qwen3.5-4B_logit | SmolLM3-3B_logit | gemma-3-4b-it_logit | gemma-3n-E4B-it_logit | gemma-4-E4B-it_logit | granite-4.2-3b_logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - | 1603 | 23.8 | 28.3 | 37.1 | 31.2 | 43.8 | 26.6 | 34.8 | 45.7 | 51.2 | 27.0 |
| CA | 277 | 28.9 | 28.5 | 41.9 | 34.7 | 47.7 | 25.6 | 38.3 | 46.6 | 57.8 | 29.2 |
| CS | 120 | 27.5 | 33.3 | 32.5 | 32.5 | 42.5 | 33.3 | 37.5 | 52.5 | 50.0 | 28.3 |

## Position bias on SinhalaMMLU dev (share of predictions per displayed position)

| model | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct_logit | 0.2 | 4.5 | 93.0 | 2.2 |
| Phi-4-mini-instruct_logit | 35.2 | 37.2 | 23.6 | 4.0 |
| Qwen3-4B-Instruct-2507_logit | 11.8 | 16.1 | 29.7 | 42.4 |
| Qwen3-4B_logit | 29.5 | 41.9 | 14.3 | 14.4 |
| Qwen3.5-4B_logit | 11.0 | 7.9 | 27.7 | 53.4 |
| SmolLM3-3B_logit | 19.6 | 0.5 | 76.0 | 3.9 |
| gemma-3-4b-it_logit | 21.9 | 32.6 | 35.8 | 9.7 |
| gemma-3n-E4B-it_logit | 14.0 | 29.6 | 32.7 | 23.7 |
| gemma-4-E4B-it_logit | 22.4 | 31.0 | 26.3 | 20.4 |
| granite-4.2-3b_logit | 69.2 | 13.6 | 12.4 | 4.8 |
