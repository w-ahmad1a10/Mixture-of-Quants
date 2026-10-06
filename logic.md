# MoQ Pipeline — Logic (English)

This file is the canonical English record of what the MoQ pipeline does and why.
It is updated whenever the logic changes. Read this before reading code.

## The core idea

Take a teacher model (Qwen3.5-0.8B). Measure how much each group of weights
hurts the model when quantized. Then pick which quantization type each group
should get so the total stays under a bit budget, apply it, and evaluate.

## The four stages (in order)

### Stage 1 — KLD Analysis

- Loads the teacher model once.
- For each of 19 quant types, it quantizes each group's weights in-place
  (using llama.cpp quantization), runs the cached calibration data through the
  model, and measures how much the output distribution shifts vs. the teacher
  (KLD = Kullback-Leibler divergence).
- Records mean / max / p99 / p99.9 KLD per (quant_type, group).
- This is the expensive stage — that is why the teacher logits are cached
  first (see "Teacher caching" below).

### Stage 2 — CE Computation

- Takes the KLD results and normalizes them into a cost-effectiveness score
  per (group, quant_type).
- For each group, it divides each quant type's KLD by the worst quant type's
  KLD in that group, so 0 = best, 1 = worst.
- This gives a per-group elasticity curve: how bad is each quant type for this
  group, relative to the worst option.

### Stage 3 — Config Generation (MCKP)

- This is the optimization. Each group is a "stage" with options (the quant
  types, each with a cost = bits × params and a value = CE score).
- It solves a multiple-choice knapsack problem: pick exactly one quant type
  per group to minimize total CE while staying under the bit budget.
- Outputs one .txt config file per budget (5.0, 4.5, 4.0, 3.5, 3.0, 2.5 BPW).

### Stage 4 — Apply + Evaluate

- Applies each config to the teacher model in-place, then restores the
  original weights.
- Evaluates on all cached eval datasets (chat, wiki, fineweb-edu,
  the-stack-python, finemath-4plus, c4-ar, c4-en, MMLU).
- Records KLD + PPL + MMLU accuracy/agreement per config.
- Saves per-config results and a master summary.

## The four grouping modes

A "group" is what gets quantized together and measured as one unit:

1. `by-tensor-type` — 11 groups (e.g. all q_proj tensors together)
2. `by-layer` — 5 groups (layers 0-3, 4-7, 8-11, 12-15, embed)
3. `individual` — 95 groups (each tensor separately)
4. `by-layer-tensor` — 41 groups (e.g. q_proj in layers 0-3)

## Teacher caching

- The teacher model's forward pass is the expensive part.
- It is run once per eval dataset, and the resulting logits are cached as
  sparse top-K log probs per batch.
- These caches are reused for every config, so the forward pass happens once
  per dataset, not once per config.
- Caches are keyed by a hash of (model, dataset, batch size, chunk count, etc.)
  so they are reused across runs.

## Key files

- `moq/moq_core.py` — **the single library.** Everything: constants, KLD math,
  datasets, model loader, C++ extension compiler, and `MoQRunner` (the orchestrator
  that builds the teacher cache, runs the 4 stages per mode, and saves results).
  Also holds `MixedQuantRunner`, `Evaluator`, `MoQFinalQuantizer`, `MMLUEvaluator`,
  `RunLog`, `upload_results`, and all grouping/parsing utilities.
- `scripts/moq.py` — **THE CLI.** 7 subcommands: `pipeline`, `quant`, `eval`,
  `build-llama`, `map-tensors`, `run`, `upload`. The `run` subcommand invokes
  `MoQRunner`.
- `scripts/run_all_goals.py` — thin experimental wrapper. Reads
  `scripts/sweep_config.yaml` and calls `scripts/moq.py run` per goal. No logic here.
- `scripts/sweep_config.yaml` — all goal hyperparameters (verbatim from the original
  experiment).
- `quant_extension.cpp` — C++ quantization (apply_llama_quant_noise).
- `configs/tensor_maps/tensor_map_qwen.txt` — maps PyTorch module names to GGUF
  tensor names (per-model data, not library code).
- `waste/` — obsolete modules kept for reuse: `moq_engine.py`, `moq_sweep.py`,
  `eval_mmlu.py`, `runlog.py`, `eval_only_cpu_smoke.py`, `fetch_moq_exp.py`,
  `upload_outputs.py`.
