# Task for the agent on the second GPU machine: option 2, reasoning-trace training for Medium questions

You are helping a team in the **ICTer 2026 SinhalaMMLU shared task** (Sinhala multiple-choice QA; total parameters of everything used at
inference must be <= 8B; open-weight models only; no closed APIs, no retrieval, no internet at inference). Our system is **base Gemma 4
E4B-it, text-only (7.463B)**, answering with its thinking mode and agreement-gated self-consistency ("candidate B", test macro avg 62.93;
easy 63.40 / medium 55.09 / hard 70.31). The leaders are about 2 points ahead, and **all of the gap is the Medium tier (grades 9-11,
mostly Sri Lankan culture/civics/history/religion facts)**. Your job is to run one fully scripted experiment and report back. You do not
decide anything that is not described here.

## What the experiment is
Reasoning-preserving fine-tuning: a LoRA (r=16) trained on the base model's **own correct thinking traces** (rejection sampling), so that
knowledge from extra data is added without the model losing its reasoning (answer-only fine-tuning destroyed it in earlier tests).
Round 1 used 1,577 traces from SinhalaMMLU dev questions and gave a held-out gain of +1.3 (CI -1.3..+3.9, not proven). Round 2 adds traces for
~2,500 Global-MMLU (Sinhala) questions (humanities / social / other / business) and 441 synthetic Wikipedia-based Sinhala MCQs.
`scripts/runs/run_rft2.sh` does everything: traces -> training -> validation on held-out questions -> automatic GO/NO-GO (rule fixed in
advance in `eval/decide_rft.py`: pooled paired gain >= +1.5 points over 700 held-out questions, neither set worse than -3) -> if GO, test
predictions for the **Medium questions only**, merged into candidate B's answers. Do not change the rule, the seeds or the settings.

## Hard rules
- The **test questions must never be used for training, tuning, prompt design or model selection**; they are only read by `inference/predict.py`.
- **Task data stays inside the team**: SinhalaMMLU dev/test and anything derived from them (`data/`, `results/`, the bundle) must not be pushed
  to any public place or git remote. Do not commit anything to git.
- No closed/hosted LLM APIs, no extra datasets beyond what the scripts download.
- Never print or store the Hugging Face token anywhere except `.env` (git-ignored). If it is missing, ask the user for it.
- You may adapt the *environment* (package versions, batch size, memory cap), not the method. Do not submit to Codabench.

## Setup (once)
1. Unpack the bundle (`rft2_bundle.tar.gz`) into a new directory and work there. It contains the code, configs, candidate B's predictions
   (`results/submissions/final_candidate_base_think_gated_sc.zip`), the base model's greedy-thinking results on the two held-out validation
   sets (`results/think_base_*`), the dev-question traces (`data/synth/rft_traces.jsonl`) and the 441 synthetic MCQs (`data/synth/pilot.jsonl`).
2. Python environment for this machine's GPU/CUDA: install a matching PyTorch build, then
   `transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 huggingface_hub==2.2.0 liger-kernel==0.8.4 torchao==0.18.0`
   plus `numpy pandas scipy pyyaml tqdm safetensors sentencepiece protobuf timm pillow tabulate`. Do not install `requirements.lock.txt`
   blindly (it pins aarch64 CUDA-13 builds from the original machine). Gemma 4 needs a transformers version with `Gemma4` support; if a
   pinned version is unavailable pick the nearest compatible one. Liger/Triton needs Python headers; if the Liger loss fails, set
   `loss: chunked` in `configs/train/rft_trace2.yaml`.
3. Create `.env` in the repo root containing one line `hf_token = <token>` (ask the user; the token's account must have accepted the terms
   of the gated dataset `naist-nlp/SinhalaMMLU`). `chmod 600 .env`.
4. Data: `python scripts/fetch_data.py` then `python scripts/prepare_data.py` then `python scripts/prepare_subsets.py` then
   `python scripts/build_sft_gmmlu.py`. Test set (public): download the dataset repo
   `SinhalaMMLU-SharedTask/Sinhala-MMLU-SharedTask-Test-Dataset` at revision `56fb2da50cfcdf0a22bfaee179370eef170fdc98` into `data/test`
   with `huggingface_hub.snapshot_download(..., repo_type="dataset", revision=..., local_dir="data/test")`.
   Check: `data/processed/smmlu_dev_sample400.jsonl` has 400 lines, `smmlu_dev_rest1450.jsonl` 1450, `gmmlu_si_val_nonacademic300.jsonl` 300,
   and `data/test` has 2,238 questions (easy 642 / medium 1,111 / hard 485).
5. Model: `python scripts/make_text_only.py google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2`
   (must print about 7.463B parameters; output `models/gemma-4-E4B-it-text`).
6. Smoke test: `python -m eval.run_eval --model models/gemma-4-E4B-it-text --data smmlu_dev_sample400 --limit 16 --method generate --thinking
   --max-new-tokens 1024 --run-name smoke`. Thinking answers must be parsed for most of the 16 questions (see `results/smoke/predictions.jsonl`,
   `pred_display` not null). Then `rm -rf results/smoke`.

## Run
Pick settings for this GPU, then run inside tmux (it must survive a disconnect), e.g.
`GPU_MEM_GB=<~85% of the GPU memory in GB> MAX_BATCH=<64 on an 80-96 GB GPU, 32 on 40-48 GB, 16 on 24 GB> TRACE_MINUTES=<see below> scripts/runs/run_rft2.sh 2>&1 | tee results/logs/run_rft2.log`.
**Time budget:** the user must finish all uploads by a fixed time today (ask them for the deadline and the current time zone). Stages on the
original machine (a slow GB10): traces `TRACE_MINUTES` (default 240), training ~1.5 h, validation ~1.2 h, Medium test run ~5.5 h. On a faster
GPU these shrink. Choose `TRACE_MINUTES` so the sum fits the deadline with at least 1 h spare. If after the traces stage the remaining time
is too short for the Medium test run, tell the user immediately (the pipeline will decide GO/NO-GO before that run).
Every stage is resumable: if the process dies, rerun the same command.

## Report back
Check progress periodically (`cat results/logs/run_rft2.log`) and tell the user: number of traces kept, the validation numbers and the
decision (`results/rft2_decision.json`), and at the end the path of the final file. If the decision is **GO** the deliverable is
`results/submissions/final_candidate_B_plus_rft2_medium.zip` (format check must print `2238 rows, 0 format errors`); also bring
`results/rft2_decision.json`, `results/logs/` and `results/submissions/rft2_medium.zip.run.json` (or the predict log). If **NO-GO**, say so and
stop; the team keeps candidate B. Never upload anything yourself.
