# Mixture-of-Quants (MoQ) — Project Documentation

## Project Overview
Mixture-of-Quants (MoQ) pipeline for quantization analysis of Qwen/Qwen3.5-0.8B model. Pipeline runs KLD analysis, CE computation, MCKP optimization, and apply+evaluate stages for 19 quant types across 4 grouping modes.

## Key Files
- `moq/` — The Python library (installed via `pip install -e .`):
  - `moq_core.py` — **The single library.** Everything: constants, KLD math, datasets, model loader, C++ extension compiler, `MoQRunner` (orchestrator: teacher cache → KLD → CE → MCKP optimize → apply+evaluate), `MixedQuantRunner`, `Evaluator`, `MoQFinalQuantizer`, `MMLUEvaluator`, `RunLog`, `upload_results`.
  - `build_llama_cpp_cpu.py` — llama.cpp CPU build script
  - `quant_extension.cpp` — C++ quantization extension
- `scripts/` — Entry points and utilities:
  - `moq.py` — **THE CLI.** 7 subcommands: `pipeline`, `quant`, `eval`, `build-llama`, `map-tensors`, `run`, `upload`.
  - `run_all_goals.py` — Experimental goal runner (thin wrapper). Reads `sweep_config.yaml`, calls `moq.py run` per goal. No logic here.
  - `sweep_config.yaml` — All run_all_goals.py hyperparameters (verbatim from the original experiment).
- `configs/` — Per-model data + eval configs:
  - `tensor_maps/tensor_map_qwen.txt` — Tensor mapping file for Qwen3.5-0.8B (per-model, not library code)
  - Uniform quant eval configs: `config_2.0.txt`, `config_3.0.txt`, `config_4.0.txt`
- `tests/` — `test_tensor_registry.py` (22 tests, TensorRegistry + grouping)
- `waste/` — Obsolete files kept for reuse (not deleted): `moq_engine.py`, `moq_sweep.py`, `eval_mmlu.py`, `runlog.py`, `eval_only_cpu_smoke.py`, `fetch_moq_exp.py`, `upload_outputs.py`
- `logic.md` — Canonical English logic record (the 4 stages, grouping modes, teacher caching). Updated whenever logic changes.

## Goals
- GOAL 1: Test all 4 MoQ grouping modes (by-tensor-type, by-layer, individual, by-layer-tensor) with expanded eval datasets
- GOAL 2: Noise impact analysis — compare calib data vs dataset repo with expanded noise datasets
- GOAL 3: Dataset repo sample count sweep (100, 500, 1000 samples) with expanded eval datasets
- GOAL 4: Uniform quantization eval (config_2.0.txt, config_3.0.txt, config_4.0.txt)

## Status

> **No Qwen3.5-0.8B goal has been achieved yet.** Everything on disk under `Results/` and
> `MoQ-EXP/` is from the **LFM2.5-230M** era (pre-migration). The Qwen migration
> (2026-09-30) updated code + configs but **no run has been executed against Qwen**.

| Goal | State | Notes |
|------|-------|-------|
| GOAL 1 | ❌ not run on Qwen | LFM2.5 data exists for all 4 modes in `MoQ-EXP/`; Qwen run not started |
| GOAL 2 | ❌ not run on Qwen | LFM2.5 `calib` + `repo100` exist; Qwen run not started |
| GOAL 3 | ❌ not run on Qwen | LFM2.5 `100` + `500` exist; `1000` never run on either model |
| GOAL 4 | ❌ not started | uniform quant configs exist (`configs/`) but no eval run |

**Blocker:** `run_all_goals.py` is written for Colab paths (`/content/...`) and needs
a Colab CPU session provisioned first. Nothing has been kicked off since the migration.

## Refactor Status (2026-10-04)
The repo was modularized so that `moq/` is the library, `scripts/moq.py` is the single CLI
entry point, and `scripts/run_all_goals.py` is a thin experimental wrapper. Obsolete modules
(`moq_engine.py`, `moq_sweep.py`, `eval_mmlu.py`, `runlog.py`) were merged into
`moq/moq_core.py` and their originals moved to `waste/`. No feature was lost — only reorder.

## Notes
- `goals.md` was merged into this file; do not recreate it.
- Colab is **CPU only** — never request GPU, never pass `--gpu`.
- `AGENTS.md` is the agent-facing quick reference; keep it in sync with this file.
- `logic.md` is the canonical English logic record; update it whenever logic changes.
- HF token: stored in env `HF_TOKEN`, never baked into scripts.

## Eval Datasets
- pile-cc: `w-ahmad/open-perfectblend-500k-tokens` (chat, unchanged)
- mmlu: standard MMLU benchmark (unchanged)
- wiki: `/content/workspace/wiki.test (2).raw` (50k tokens)
- fineweb-edu: `HuggingFaceFW/fineweb-edu` (500k tokens, 1000 docs streamed)
- the-stack-python: `bigcode/the-stack` config `data/python` (500k tokens, 1000 docs streamed)
- finemath-4plus: `HuggingFaceTB/finemath` config `finemath-4plus` (500k tokens, 1000 docs streamed)
- c4-ar: `allenai/c4` config `ar` (500k tokens, 1000 docs streamed)
- c4-en: `allenai/c4` config `en` (500k tokens, 1000 docs streamed)

## Noise Impact Datasets (Goal 2)
- wiki: 50k tokens
- fineweb-edu: 50k tokens
- the-stack-python: 50k tokens
- finemath-4plus: 50k tokens
- c4-ar: 50k tokens
- c4-en: 50k tokens

## Token Budgets
- Final eval: 500,000 tokens per dataset (977 chunks × 512 seq_length)
- Noise impact: 50,000 tokens per dataset (98 chunks × 512 seq_length)
- Wiki: 50,000 tokens both phases (98 chunks × 512 seq_length)
- Chat (pile-cc): unchanged
- MMLU: unchanged

## Status
2026-09-30: Updated for Qwen/Qwen3.5-0.8B. Added HFDataset class, expanded eval/noise datasets, updated AGENTS.md.