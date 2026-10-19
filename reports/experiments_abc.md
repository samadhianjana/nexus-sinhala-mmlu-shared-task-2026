# Baseline comparison

Accuracy in %. `orig` = original option order; `vote` = mean probabilities over cyclic option permutations; `±` = std of accuracy across permutations (position sensitivity). Random = 25.

## Overview

| model | params loaded | tok/char | SinhalaMMLU dev orig | vote | ± | macro-subj | Global-MMLU si | Belebele si | label mass | consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-4-E4B-it_logit | 7.94B | 0.40 | 55.2 | 54.5 | 1.1 | 55.4 | 52.0 | 79.8 | 0.99 | 0.49 |
| cpt_pilot_step-114_logit | 7.46B | 0.40 | 54.1 | 54.9 | 1.1 | 54.1 | 51.9 | - | 1.00 | 0.46 |
| cpt_pilot_step-228_logit | 7.46B | 0.40 | 53.9 | 55.5 | 1.5 | 54.1 | 51.7 | - | 1.00 | 0.45 |
| devcv_lora_all | 7.46B | 0.40 | 56.3 | 57.4 | 0.7 | 56.4 | - | - | 1.00 | 0.54 |
| sft_gmmlu_lora_logit | 7.46B | 0.40 | 56.2 | 57.2 | 0.5 | 56.2 | 58.1 | - | 1.00 | 0.53 |

## SinhalaMMLU dev: per subject

| subject | n | gemma-4-E4B-it_logit | cpt_pilot_step-114_logit | cpt_pilot_step-228_logit | devcv_lora_all | sft_gmmlu_lora_logit |
|---|---:|---:|---:|---:|---:|---:|
| Arts | 100 | 56.0 | 55.0 | 53.0 | 54.0 | 52.0 |
| Buddhism | 162 | 51.9 | 50.0 | 47.5 | 54.3 | 53.7 |
| Catholicism | 132 | 50.0 | 51.5 | 52.3 | 59.8 | 53.0 |
| Christianity | 130 | 62.3 | 61.5 | 61.5 | 60.0 | 63.1 |
| Civics | 123 | 74.8 | 76.4 | 75.6 | 73.2 | 74.8 |
| Eastern music | 143 | 45.5 | 47.6 | 47.6 | 43.4 | 44.1 |
| Geography | 116 | 64.7 | 62.1 | 62.9 | 64.7 | 62.1 |
| Health and Physical Science | 130 | 69.2 | 62.3 | 63.1 | 69.2 | 70.8 |
| History | 151 | 39.1 | 41.7 | 39.7 | 38.4 | 39.1 |
| Islam | 108 | 59.3 | 60.2 | 61.1 | 62.0 | 63.0 |
| Sinhala language and literature | 155 | 46.5 | 45.2 | 45.8 | 47.7 | 52.9 |
| dancing | 108 | 34.3 | 27.8 | 29.6 | 35.2 | 33.3 |
| drama and Theatre | 128 | 55.5 | 50.8 | 50.8 | 60.2 | 58.6 |
| science | 164 | 66.5 | 65.9 | 66.5 | 67.7 | 67.1 |

## SinhalaMMLU dev: per domain

| category | n | gemma-4-E4B-it_logit | cpt_pilot_step-114_logit | cpt_pilot_step-228_logit | devcv_lora_all | sft_gmmlu_lora_logit |
|---|---:|---:|---:|---:|---:|---:|
| humanities | 1162 | 50.2 | 49.5 | 49.1 | 51.7 | 50.9 |
| language | 155 | 46.5 | 45.2 | 45.8 | 47.7 | 52.9 |
| social_science | 369 | 69.6 | 66.9 | 67.2 | 69.1 | 69.4 |
| stem | 164 | 66.5 | 65.9 | 66.5 | 67.7 | 67.1 |

## SinhalaMMLU dev: per grade

| grade | n | gemma-4-E4B-it_logit | cpt_pilot_step-114_logit | cpt_pilot_step-228_logit | devcv_lora_all | sft_gmmlu_lora_logit |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 584 | 59.9 | 59.4 | 59.9 | 62.5 | 61.0 |
| 7 | 634 | 53.0 | 50.5 | 51.3 | 53.0 | 53.2 |
| 8 | 632 | 53.0 | 52.7 | 51.1 | 53.8 | 54.9 |

## Global-MMLU si: per category

| category | n | gemma-4-E4B-it_logit | cpt_pilot_step-114_logit | cpt_pilot_step-228_logit | devcv_lora_all | sft_gmmlu_lora_logit |
|---|---:|---:|---:|---:|---:|---:|
| Business | 102 | 54.9 | 61.8 | 58.8 | - | 61.8 |
| Humanities | 668 | 46.9 | 45.7 | 45.5 | - | 49.9 |
| Medical | 216 | 54.2 | 54.2 | 52.8 | - | 63.4 |
| Other | 126 | 56.3 | 54.0 | 51.6 | - | 56.3 |
| STEM | 449 | 51.9 | 53.0 | 53.7 | - | 59.7 |
| Social Sciences | 439 | 57.2 | 56.5 | 56.9 | - | 66.3 |

## Global-MMLU si: culturally sensitive (CS) / agnostic (CA) / unlabelled (-)

| cultural_sensitivity | n | gemma-4-E4B-it_logit | cpt_pilot_step-114_logit | cpt_pilot_step-228_logit | devcv_lora_all | sft_gmmlu_lora_logit |
|---|---:|---:|---:|---:|---:|---:|
| - | 1603 | 51.2 | 51.3 | 51.3 | - | 57.7 |
| CA | 277 | 57.8 | 56.7 | 56.0 | - | 61.4 |
| CS | 120 | 50.0 | 49.2 | 46.7 | - | 56.7 |

## Position bias on SinhalaMMLU dev (share of predictions per displayed position)

| model | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| gemma-4-E4B-it_logit | 22.4 | 31.0 | 26.3 | 20.4 |
| cpt_pilot_step-114_logit | 21.3 | 33.5 | 25.2 | 20.0 |
| cpt_pilot_step-228_logit | 21.1 | 33.8 | 25.3 | 19.8 |
| devcv_lora_all | 28.4 | 24.3 | 23.6 | 23.6 |
| sft_gmmlu_lora_logit | 29.5 | 23.8 | 26.1 | 20.6 |
