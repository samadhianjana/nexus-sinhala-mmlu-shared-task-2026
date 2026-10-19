# Experiment run scripts

Each script is the exact queue that produced a reported result or a submission. They run from any directory
(`cd` to the repo root themselves), skip steps whose outputs already exist, and log to `results/logs/`.
Reusable tools (data building, evaluation, overlap checks) live one level up in `scripts/` and in `eval/`, `train/`,
`inference/`.

| Script | Date | Produced | Where reported |
|---|---|---|---|
| `run_baselines.sh` | 10 Oct | Zero-shot baselines of 10 candidate models (dev, Global-MMLU `si` val, Belebele) | `reports/baselines_zeroshot_logit.md` |
| `run_experiments.sh` | 10 Oct | Prompting experiments on the top 3 models: Sinhala instruction, fixed 5-shot, generation, cloze | `reports/experiments_prompting.md` |
| `run_phase1.sh` | 10 Oct | Text-only Gemma 4 E4B (7.463B) check; offline submission pipeline test on dev; first training-speed benchmark | `reports/throughput.jsonl` |
| `run_bench_speedups.sh` | 10 Oct | Training speed with Liger loss, selective recomputation, `torch.compile` | `reports/throughput.jsonl` |
| `run_bench_gemma_compile.sh` | 11 Oct | Gemma 4 E4B speed after the compile recompile-limit fix | `reports/throughput.jsonl` |
| `run_bench_memory.sh` | 11 Oct | Speed/memory with 8-bit AdamW and every-2nd-layer recomputation | `reports/throughput.jsonl` |
| `run_sft_gmmlu.sh` | 12 Oct | Full fine-tuning on Global-MMLU (negative result: dev −2.9) | `reports/sft_gmmlu_v1.md` |
| `run_abc.sh` | 12 Oct | **Model A** (LoRA on Global-MMLU, basis of submissions 2 and 4); dev 5-fold LoRA (B); continued-pretraining pilot (C) | `reports/experiments_abc.md` |
| `download_teachers.sh` | 18 Oct | Downloads of the teacher candidates (data generation only, never inference) | — |
| `run_v2_and_teachers.sh` | 18 Oct | **v2** (LoRA on Global-MMLU + all dev; submission 3) and its test predictions; teacher evaluations | `reports/teachers.md` |
| `run_thinking_tests.sh` | 19 Oct | Thinking mode: model A and base model on the Global-MMLU academic proxy and a dev sample | PLAN.md §2b |
| `run_softlabels_pilot_sub4_kd.sh` | 19 Oct | Gemma 4 31B soft labels for Global-MMLU; synthetic MCQ pilot; **submission 4** predictions; distillation LoRA (interrupted by the 19 Oct crash) | `data/synth/pilot.stats.json`, PLAN.md |

Submissions (all chosen on dev / validation, never on test):

| # | Config | Test macro avg (easy / medium / hard) |
|---|---|---|
| 1 | `configs/submission.yaml` (zero-shot Gemma 4 E4B) | 51.59 |
| 2 | `configs/submission_lora_gmmlu_vote.yaml` (model A, 4-order vote) | 54.59 (55.30 / 49.51 / 58.97) |
| 3 | `configs/submission_v2.yaml` (v2, 4-order vote) | 54.85 (54.83 / 50.77 / 58.97) |
| 4 | `configs/submission_A_think_hard.yaml` (model A; thinking for `hard`) | 55.83 (55.30 / 49.51 / 62.68) |

The final submission is reproduced with `./run_submission.sh` at the repository root.
