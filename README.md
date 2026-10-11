# SinhalaMMLU shared task (≤ 8B)

System for the ICTer 2026 SinhalaMMLU shared task: Sinhala multiple-choice QA with open-weight models totalling
≤ 8B parameters, no retrieval, no internet at inference. Built on SinhalaMMLU (Pramodya et al., EMNLP 2025):

```bibtex
@inproceedings{pramodya-etal-2025-sinhalammlu,
  title     = "{S}inhala{MMLU}: A Comprehensive Benchmark for Evaluating Multitask Language Understanding in {S}inhala",
  author    = "Pramodya, Ashmari and Nelki, Nirasha and Shalinda, Heshan and Liyanage, Chamila and Sakai, Yusuke and
               Pushpananda, Randil and Weerasinghe, Ruvan and Kamigaito, Hidetaka and Watanabe, Taro",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
  year      = "2025",
  pages     = "32931--32949",
  url       = "https://aclanthology.org/2025.emnlp-main.1673/"
}
```

**Keep this repository private until the final deadline. Never commit task data (`data/` is git-ignored).**

## Layout

| Path | Contents |
|---|---|
| `data/` | Git-ignored. `raw/` downloads, `processed/` normalised JSONL |
| `eval/` | Evaluation harness (`run_eval.py`, scorers, metrics, `summarize.py`) |
| `train/` | Fine-tuning / continued-pretraining code |
| `inference/` | Final offline submission pipeline |
| `configs/` | Candidate models, fixed data splits |
| `scripts/` | Data fetching, parameter counting, baseline runs |
| `reports/` | Aggregate results tables (no task data) |
| `MODEL_MANIFEST.md` | Every model: exact total parameters, release date, license |
| `DATA_LOG.md` | Every dataset and every closed-API data-generation use |

## Setup (DGX Spark, aarch64, CUDA 13)

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -r requirements.lock.txt
echo "hf_token = hf_..." > .env && chmod 600 .env   # read token; accept gated terms for SinhalaMMLU, Gemma 3/3n, Llama
```

Triton (used by `torch.compile`, Liger Kernel and flash-linear-attention) compiles helpers against `Python.h`.
If `python3.12-dev` is installed, nothing else is needed. Without sudo, unpack the headers locally; the code
(`eval/common.py: ensure_python_headers`) then adds them to `CPATH` automatically:

```bash
mkdir -p ~/.local/pydev && cd ~/.local/pydev
V=3.12.3-1ubuntu0.17   # any 3.12.3 Ubuntu build; the machine's apt index may list a version no longer on the mirror
for p in libpython3.12-dev python3.12-dev; do
  curl -fO "http://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/${p}_${V}_arm64.deb"; done
for f in *.deb; do dpkg -x "$f" root; done
```

Long GPU jobs: PyTorch is capped at 100 GB (`SMMLU_GPU_MEM_GB`). The Spark shares memory between CPU and GPU, so
an uncapped job triggers the kernel OOM killer (which also kills tmux) instead of a CUDA out-of-memory error.

## Commands

```bash
.venv/bin/python scripts/fetch_data.py            # SinhalaMMLU dev, Global-MMLU si, Belebele si -> data/raw
.venv/bin/python scripts/prepare_data.py          # -> data/processed/*.jsonl (fixed splits in configs/splits)
.venv/bin/python scripts/count_params.py          # exact parameter counts -> MODEL_MANIFEST.md
scripts/run_baselines.sh                          # zero-shot baselines -> results/, reports/

# One model, any setting
.venv/bin/python -m eval.run_eval --model google/gemma-4-E4B-it --data sinhalammlu_dev gmmlu_si_val \
    --method logit --n-perms 4 --lang en --run-name my_run
```

### Harness options

- `--method logit` (next-token probability over labels 1-4), `generate` (greedy decode + parse), `cloze`
  (length-normalised option likelihood).
- `--n-perms K --perm-mode cyclic|random`: score each question under K option orders. Reports original-order
  accuracy, permutation-vote accuracy, spread across orders, consistency, and position-bias tables.
- `--lang en|si` instruction language; `--chat auto|on|off`; `--fewshot-file F --n-shot N` (the same fixed
  examples for every question, as the rules require).
- Breakdowns by dataset, domain, subject, grade and (Global-MMLU) cultural sensitivity, with Wilson 95% CIs.
- Every run stores the exact model commit, library versions, git commit and the full argument set in `metrics.json`.
- `--write-submission` writes a Codabench-format `predictions.zip` (`predictions.jsonl` with
  `q_no, subject, category, level, answer`).

## Rules checklist

- Total parameters of everything loaded at inference ≤ 8B (embeddings, vision/audio towers if loaded, adapters).
- Open-weight models released before 2026-10-01 only. No API, internet or retrieval of any kind at inference.
- Fixed few-shot examples only (identical for every question). No training, tuning or prompt design on test questions.
- Closed APIs only for training-data generation, logged in `DATA_LOG.md`.
- Final pipeline runs offline from one command with pinned versions and fixed seeds.
