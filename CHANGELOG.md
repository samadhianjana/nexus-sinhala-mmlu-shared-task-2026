# Changelog

## 1.0.0 (26 November 2026)

First public release: the code and experiments behind Team Nexus's 3rd-place entry in the ICTer 2026 SinhalaMMLU Shared Task.

- **Submitted system:** text-only Gemma 4 E4B-it (7.463B parameters), zero-shot, built-in thinking mode with greedy decoding
  (`configs/submission_base_think_greedy.yaml`). No fine-tuning, no sampling, no extra data.
- **Development work included:** evaluation harness, LoRA and full fine-tuning, continued pre-training, distillation,
  reasoning-trace tuning, option-order voting and self-consistency, with the scripts and configs behind each result.
- Added `LICENSE` (Apache-2.0), `CITATION.cff` and a score-progression section in the README.
- Rewrote `DATA_LOG.md` as a complete record of datasets and model-generated data.
- Removed internal planning notes, an agent hand-off prompt and the system report PDF from the repository.
