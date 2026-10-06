# AGENTS.md — Mixture-of-Quants Pipeline

## Project
Mixture-of-Quants (MoQ) pipeline for Qwen/Qwen3.5-0.8B model quantization analysis.

## Key Files
- `moq/` — The Python package (installed via `pip install -e .`):
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
- `help.md` — Pipeline help documentation + blog narrative
- `Mixture-of-Quants.md` — Project record (goals, status, results). **Canonical goals doc** — `goals.md` was merged into this file; do not recreate `goals.md`.
- `logic.md` — Canonical English logic record (the 4 stages, grouping modes, teacher caching). Updated whenever logic changes.

## Commands
- Run a single mode: `python scripts/moq.py run --model Qwen/Qwen3.5-0.8B --mapping configs/tensor_maps/tensor_map_qwen.txt --gguf-repo unsloth/Qwen3.5-0.8B-GGUF --gguf-file Qwen3.5-0.8B-BF16.gguf --llama-cpp-dir /workspace/llama.cpp --imatrix /workspace/Qwen3.5-0.8B.imatrix --dataset-repo w-ahmad/open-perfectblend-500k-tokens --dataset-split train --max-seq-length 1200 --max-samples 100 --batch-size 8 --num-chunks 13 --seed 42 --trust-remote-code --dtype float16 --quant-types ALL --bits-list 5.0 4.5 4.0 3.5 3.0 2.5 --mmlu-batch-size 8 --mode by-tensor-type --results-dir results_goal1_by-tensor-type`
- Run all goals (Colab): `python scripts/run_all_goals.py`
- Full pipeline: `python scripts/moq.py pipeline --gguf-repo ... --gguf-file ... --hf-repo ... --model ...`
- Quantize: `python scripts/moq.py quant --model ... --mapping ... --quant-config-dir ./configs/ --llama-cpp-dir ...`
- Eval: `python scripts/moq.py eval --teacher-repo ... --student-repos ... --mapping ...`
- Upload: `python scripts/moq.py upload <local_path> <hf_path> --hf-repo ...`

## Quant Types (19, excl. Q1_0, Q2_0)
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M

## Grouping Modes
1. `by-tensor-type` (11 groups)
2. `by-layer` (5 groups)
3. `individual` (95 groups)
4. `by-layer-tensor` (41 groups)

## Pipeline Metrics
- Stage 1: KLD Analysis → `analysis_{mode}.json`
- Stage 2: CE Computation → `ce_{mode}/ce_results.txt` + `param_counts.json`
- Stage 3: Config Generation → `configs_{mode}/` (.txt files per target BPW)
- Stage 4: Apply + Evaluate → `eval_{mode}_{bits}.json` + `summary_{mode}.json` + `master_summary.json`
- Logs: `Logs/{run}/metrics.jsonl`

## HF Repo
`unsloth/Qwen3.5-0.8B-GGUF` (GGUF) — contains Qwen3.5-0.8B-BF16.gguf, imatrix_unsloth.gguf_file
`Qwen/Qwen3.5-0.8B` (base model) — HF model repo
`w-ahmad/moq-lfm25-230m-results` (dataset) — contains wiki.test (2).raw (legacy, kept for reference)

## Eval Datasets
- pile-cc: `w-ahmad/open-perfectblend-500k-tokens` (chat)
- mmlu: standard MMLU benchmark
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
- Use `cmd.exe /c "D:\Miniconda3\envs\kaggle\Scripts\kaggle.exe <command> ..."`
- Windows paths (D:\...) work in cmd.exe wrapper — do NOT use /mnt/d/...
- Do NOT set PYTHONUTF8 in cmd.exe (causes issues)
- (Legacy kernel `moq-single-gpu-test` removed — see "Legacy LFM2.5-230M test kernel" below.)

## Colab (WSL) — Interactive Session

For running `eval_only_individual_colab.py` (Goal 1 final eval, individual mode).

### Setup (one-time)

```bash
# 1. Install Colab CLI (if not installed)
uv tool install git+https://github.com/googlecolab/google-colab-cli.git

# 2. Create interactive session (keep alive)
colab new -s moq-goal1 --keep
```

### Upload repo to Colab (direct upload, no zip)

Upload each file individually using `colab upload`. Do NOT zip the repo.

Files to upload (to `/content/Mixture-of-Quants/`):
- `moq/` directory (library: `moq_core.py`, `build_llama_cpp_cpu.py`, `quant_extension.cpp`)
- `scripts/` directory (CLI + goal runner: `moq.py`, `run_all_goals.py`, `sweep_config.yaml`)
- `configs/` directory (per-model data + eval configs)

Configs to upload (to `/content/Mixture-of-Quants/configs/`):
- `config_2.0.txt`
- `config_3.0.txt`
- `config_4.0.txt`
- `tensor_maps/tensor_map_qwen.txt`

Do NOT upload:
- `Results/`, `Logs/`, `MoQ-EXP/`, `__pycache__/`, `.git/`, `*.md`
- `waste/` (obsolete files kept for reference)
- `tensor_map_lfm25.txt` (obsolete for Qwen — removed)
- `eval_only_cpu_smoke.py`, `eval_only_individual.py`, `eval_only_individual_colab.py`

### Install dependencies

```bash
colab install -s moq-goal1 torch transformers huggingface_hub datasets tqdm accelerate 2>&1 | tail -3
```

### Run experiment

Copy script to WSL home first (for `colab exec -f` local path), then run:
```bash
cp "/mnt/c/Workspace/01 Projects/Mixture-of-Quants/scripts/eval_only_individual_colab.py" ~/eval_only_individual_colab.py
colab exec -s moq-goal1 -f ~/eval_only_individual_colab.py &
# Background: script may take hours on CPU
```

Or use `colab console` for direct VM shell (monitor progress in real time):
```bash
colab console -s moq-goal1
# Inside VM:
cd /content/Mixture-of-Quants
python3 eval_only_individual_colab.py
```

### Download results

```bash
colab download -s moq-goal1 /content/results_goal1_individual_eval_only/ ./moq-results/
```

### Stop session

```bash
colab stop -s moq-goal1
```

### Key paths on Colab

| Item | Path |
|------|------|
| Repo | `/content/Mixture-of-Quants/` |
| Configs | `/content/Mixture-of-Quants/configs/` |
| Tensor map | `/content/Mixture-of-Quants/configs/tensor_maps/tensor_map_qwen.txt` |
| llama.cpp binaries | `/content/workspace/llama_bin` |
| llama.cpp source | `/content/workspace/llama.cpp` |
| Imatrix | `/content/workspace/Qwen3.5-0.8B.imatrix` |
| Wiki data | `/content/workspace/wiki.test (2).raw` |
| Results | `/content/results_goal1_individual_eval_only/` |

**Note**: `moq/build_llama_cpp_cpu.py` downloads pre-built binaries from GitHub (not compiling from source). Script downloads imatrix + wiki.test from HF at runtime. Model weights and MMLU data auto-download from HF.

### Notes
- HF token: set as Colab secret or env var `HF_TOKEN` (script reads `os.environ.get("HF_TOKEN", ...)`)
- Interactive CPU session auto-terminates after ~24h; use `--keep` to prevent idle timeout (~90 min)
- CPU only — never request GPU. Do not pass `--gpu`.
- Consider `nohup` or background for long runs to avoid `colab exec` 10s poll timeout
- To run the full goal suite on Colab: `python scripts/run_all_goals.py` (reads `scripts/sweep_config.yaml`, calls `scripts/moq.py run` per goal)

## Legacy LFM2.5-230M Test Kernel (obsolete — kept for history)

The Qwen3.5-0.8B migration (2026-09-30) superseded the LFM2.5-230M test kernel. These entries are historical only; the active model is `Qwen/Qwen3.5-0.8B` with `tensor_map_qwen.txt`.

- **v9.3**: Added `gguf_base_to_ce_key` dict built from `tensor_map_lfm25.txt` to map GGUF base names (e.g., `shortconv.in_proj`) to CE report keys (e.g., `conv_in_proj`). Fixed `by_layer` handling in `else` branch for non-`blk.\d+.` tensors.
- **v9.4**: Fixed by-layer key mismatch: `param_counts` normalization (line 1266) converted hyphens to underscores, breaking lookup against `ce_results` keys which have hyphens (`layer_0-3`). Also fixed config generation lines 1337/1340 to use hyphens instead of underscores in `layer_{start}-{end}` keys.

### v9.1 Bug Fix (suffix_to_quant key mismatch)
| Test | Details |
|------|---------|
| Model loading | LiquidAI/LFM2.5-230M |
| C++ quant | Q4_0, Q5_0, Q8_0 |
| Tensor discovery | 95 mappings, 91 candidates, 11 groups |
| Chat eval | PPL=5.02, KLD=0.0 |
| MMLU eval | accuracy=0%, 5/5 questions, agreement=100%, KLD=0.0 |
| Wiki text eval | PPL=154.49, KLD=0.0 |
| Single quant+eval | Q4_0, PPL=5.10, KLD=0.0 |
| **Debug** | Mapping/config key mismatch logging added |
| **v9 Bug** | Config had ALL `bf16` values — `suffix_to_quant` keys (GGUF base names like `shortconv.in_proj`) didn't match `gguf_base_to_type` values (abstract types like `conv_in_proj`) |
| **v9.1 Fix** | Removed `_build_gguf_base_to_type` indirection; config generation now uses `gguf_base` directly to look up `suffix_to_quant` |

## Bugs Fixed
1. MMLU answer bounds: `answer_start >= len(input_ids)` → `>` with safety clamping
2. Single-token choice access: `vocab_size` bounds check in `cache_teacher_logits` (~line 164) and `evaluate_student` (~line 234)
3. **v9**: Config generation used `_build_gguf_base_to_type` which mapped GGUF base → abstract type (e.g., `shortconv.in_proj` → `conv_in_proj`), but `ce_results`/`suffix_to_quant` keys are GGUF base names. All quant types defaulted to `bf16`, which is not in `GGML_TYPE_MAP`, causing all 133 modules to be skipped.
4. **v9.1**: Fixed config generation to use `gguf_base` directly instead of `abstract_type` for `suffix_to_quant` lookups. Removed `_build_gguf_base_to_type` call entirely.
5. **v9.4**: Fixed by-layer mode: (a) Removed `param_counts` key normalization (`k.replace('-', '_')`) that broke lookup against `ce_results` keys which use hyphens (`layer_0-3`). (b) Fixed config generation `layer_{block_start}_{block_end}` → `layer_{block_start}-{block_end}` and `{ce_key}_layer_{block_start}_{block_end}` → `{ce_key}_layer_{block_start}-{block_end}` to match solver's hyphenated keys.

## Notes
- KLD=0.0 expected in test (teacher=student, no real quantization)
- MMLU accuracy=0% with 5 questions expected (tiny test set, random logits)
- HF token: stored in env (never bake into scripts)
- Kaggle auth: `C:\Users\win11pro\.kaggle\access_token` (raw KGAT_... token)

# MoQ Goals — Qwen3.5-0.8B (canonical)

> Goals live in `Mixture-of-Quants.md`. This section is the run-ready quick reference.
> The LFM2.5-230M content below is **obsolete** (superseded 2026-09-30) — kept only for history.

---

## Common / Shared Settings (Qwen3.5-0.8B)

| Param | Value |
|---|---|
| Model | `Qwen/Qwen3.5-0.8B` |
| GGUF Repo | `unsloth/Qwen3.5-0.8B-GGUF` |
| GGUF File | `Qwen3.5-0.8B-BF16.gguf` |
| Tensor Map | `tensor_map_qwen.txt` |
| Imatrix | `Qwen3.5-0.8B.imatrix` (from GGUF repo) |
| llama.cpp dir | `llama.cpp` |
| dtype | `float16` |
| seed | `42` |
| trust-remote-code | `true` |
| HF token | from env `HF_TOKEN` |
| **Quant types (ALL 19, excl. Q1_0, Q2_0)** | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ4_NL`, `IQ3_S`, `IQ2_S`, `IQ4_XS`, `IQ1_M` |
| Bits list | `5.0`, `4.5`, `4.0`, `3.5`, `3.0`, `2.5` |

---

## MoQ Grouping Modes

### Mode 1: `by-tensor-type` → 11 groups
| Group | Tensors |
|---|---|
| conv_conv | 8 |
| conv_in_proj | 8 |
| conv_out_proj | 8 |
| embed_tokens | 1 |
| feed_forward_w1 | 14 |
| feed_forward_w2 | 14 |
| feed_forward_w3 | 14 |
| self_attn_k_proj | 7 |
| self_attn_out_proj | 7 |
| self_attn_q_proj | 7 |
| self_attn_v_proj | 7 |

### Mode 2: `by-layer` → 5 groups
| Group | Tensors |
|---|---|
| layer_0-3 | 29 |
| layer_4-7 | 26 |
| layer_8-11 | 26 |
| layer_12-15 | 13 |
| layer_embed | 1 |

### Mode 3: `individual` → 95 groups

### Mode 4: `by-layer-tensor` → 41 groups

---

## Final Eval (all goals run this)

All 3 evals run after each KLD analysis.

### Eval 1 — Chat Dataset
| Param | Value |
|---|---|
| type | `chat` |
| dataset_repo | `w-ahmad/open-perfectblend-500k-tokens` |
| dataset_split | `train` |
| max_seq_length | `1200` |
| max_samples | `ALL` |
| num_chunks | `99999999` |
| batch_size | `8` |
| seed | `42` |

### Eval 2 — Text Calibration
| Param | Value |
|---|---|
| type | `text` |
| calib_data | `/content/workspace/wiki.test (2).raw` |
| max_seq_length | `512` |
| seq_length | `512` |
| max_samples | `50` |
| num_chunks | `4` (16×4=64 ≥ 50) |
| batch_size | `16` |
| seed | `42` |

### Eval 3 — MMLU
| Param | Value |
|---|---|
| type | `mmlu` |
| mmlu-subjects | `ALL` |
| max-samples | `ALL` (~14,042 questions) |
| batch-size | `8` |
| seed | `42` |

---

## Batch Size Auto-Tuning

Run with batch_size=1 first, double each run (1→2→4→8→16) until OOM or memory is well-used. Lock that value.

| Model Size | Compute | Suggested batch_size |
|---|---|---|
| 0.8B | CPU | 8 |
| 0.8B+ | CPU | 1-2 |

---

## GOAL 1 — Test All 4 Modes

**Purpose**: Run all 4 MoQ grouping modes with 100 samples from dataset repo during KLD analysis. Run final eval after each.

### KLD Analysis — All 4 Modes

| Param | Value |
|---|---|
| Dataset | `w-ahmad/open-perfectblend-500k-tokens` |
| max_samples | `100` |
| max_seq_length | `1200` |
| batch_size | `8` |
| num_chunks | `13` (8×13=104 samples ≥ 100) |
| **Total tokens** | **120,000** |
| seed | `42` |

### 4 Modes
1. `by-tensor-type` (11 groups)
2. `by-layer` (5 groups)
3. `individual` (95 groups)
4. `by-layer-tensor` (41 groups)

### Run per mode
```bash
python scripts/moq.py run \
    --model Qwen/Qwen3.5-0.8B \
    --mapping configs/tensor_maps/tensor_map_qwen.txt \
    --gguf-repo unsloth/Qwen3.5-0.8B-GGUF \
    --gguf-file Qwen3.5-0.8B-BF16.gguf \
    --llama-cpp-dir /workspace/llama.cpp \
    --imatrix /workspace/Qwen3.5-0.8B.imatrix \
    --dataset-repo w-ahmad/open-perfectblend-500k-tokens \
    --dataset-split train \
    --max-seq-length 1200 \
    --max-samples 100 \
    --batch-size 8 \
    --num-chunks 13 \
    --seed 42 \
    --trust-remote-code \
    --dtype float16 \
    --quant-types ALL \
    --bits-list 5.0 4.5 4.0 3.5 3.0 2.5 \
    --eval-configs '[\
        {"type":"chat","path":"w-ahmad/open-perfectblend-500k-tokens","name":"pile-cc"},\
        {"type":"text","path":"/workspace/wiki.test (2).raw","name":"wiki","seq_length":512,"num_chunks":98},\
        {"type":"text","hf_dataset_repo":"HuggingFaceFW/fineweb-edu","name":"fineweb-edu","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"bigcode/the-stack","name":"the-stack-python","hf_dataset_config":"data/python","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"HuggingFaceTB/finemath","name":"finemath-4plus","hf_dataset_config":"finemath-4plus","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"allenai/c4","name":"c4-ar","hf_dataset_config":"ar","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"allenai/c4","name":"c4-en","hf_dataset_config":"en","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"mmlu","name":"mmlu"}]' \
    --mmlu-batch-size 8 \
    --mode <MODE> \
    --results-dir results_goal1_<MODE>
```

---

## GOAL 2 — Calib Data vs Dataset Repo (100 samples)

**Purpose**: Compare calib data vs dataset repo (100 samples) for KLD analysis using `by-tensor-type` mode.

### Approach A — Calib Data

| Param | Value |
|---|---|
| Mode | `by-tensor-type` |
| calib_data | `/content/workspace/wiki.test (2).raw` |
| max_samples | `50` |
| max_seq_length | `512` |
| seq_length | `512` |
| batch_size | `16` |
| num_chunks | `4` |
| **Total tokens** | **25,600** |
| seed | `42` |

### Approach B — Dataset Repo (100 samples)

| Param | Value |
|---|---|
| Mode | `by-tensor-type` |
| dataset_repo | `w-ahmad/open-perfectblend-500k-tokens` |
| max_samples | `100` |
| max_seq_length | `1200` |
| batch_size | `8` |
| num_chunks | `13` |
| **Total tokens** | **120,000** |
| seed | `42` |

### Run Approach A
```bash
python scripts/moq.py run \
    --model Qwen/Qwen3.5-0.8B \
    --mapping configs/tensor_maps/tensor_map_qwen.txt \
    --gguf-repo unsloth/Qwen3.5-0.8B-GGUF \
    --gguf-file Qwen3.5-0.8B-BF16.gguf \
    --llama-cpp-dir /workspace/llama.cpp \
    --imatrix /workspace/Qwen3.5-0.8B.imatrix \
    --calib-data "/workspace/wiki.test (2).raw" \
    --max-seq-length 512 \
    --seq-length 512 \
    --max-samples 50 \
    --batch-size 16 \
    --num-chunks 4 \
    --seed 42 \
    --trust-remote-code \
    --dtype float16 \
    --quant-types ALL \
    --bits-list 5.0 4.5 4.0 3.5 3.0 2.5 \
    --eval-configs '[\
        {"type":"chat","path":"w-ahmad/open-perfectblend-500k-tokens","name":"pile-cc"},\
        {"type":"text","path":"/workspace/wiki.test (2).raw","name":"wiki","seq_length":512,"num_chunks":98},\
        {"type":"text","hf_dataset_repo":"HuggingFaceFW/fineweb-edu","name":"fineweb-edu","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"bigcode/the-stack","name":"the-stack-python","hf_dataset_config":"data/python","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"HuggingFaceTB/finemath","name":"finemath-4plus","hf_dataset_config":"finemath-4plus","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"allenai/c4","name":"c4-ar","hf_dataset_config":"ar","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"allenai/c4","name":"c4-en","hf_dataset_config":"en","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"mmlu","name":"mmlu"}]' \
    --mmlu-batch-size 8 \
    --mode by-tensor-type \
    --results-dir results_goal2_calib
```

### Run Approach B
```bash
python scripts/moq.py run \
    --model Qwen/Qwen3.5-0.8B \
    --mapping configs/tensor_maps/tensor_map_qwen.txt \
    --gguf-repo unsloth/Qwen3.5-0.8B-GGUF \
    --gguf-file Qwen3.5-0.8B-BF16.gguf \
    --llama-cpp-dir /workspace/llama.cpp \
    --imatrix /workspace/Qwen3.5-0.8B.imatrix \
    --dataset-repo w-ahmad/open-perfectblend-500k-tokens \
    --dataset-split train \
    --max-seq-length 1200 \
    --max-samples 100 \
    --batch-size 8 \
    --num-chunks 13 \
    --seed 42 \
    --trust-remote-code \
    --dtype float16 \
    --quant-types ALL \
    --bits-list 5.0 4.5 4.0 3.5 3.0 2.5 \
    --eval-configs '[\
        {"type":"chat","path":"w-ahmad/open-perfectblend-500k-tokens","name":"pile-cc"},\
        {"type":"text","path":"/workspace/wiki.test (2).raw","name":"wiki","seq_length":512,"num_chunks":98},\
        {"type":"text","hf_dataset_repo":"HuggingFaceFW/fineweb-edu","name":"fineweb-edu","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"bigcode/the-stack","name":"the-stack-python","hf_dataset_config":"data/python","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"HuggingFaceTB/finemath","name":"finemath-4plus","hf_dataset_config":"finemath-4plus","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"allenai/c4","name":"c4-ar","hf_dataset_config":"ar","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"allenai/c4","name":"c4-en","hf_dataset_config":"en","seq_length":512,"num_chunks":98,"hf_dataset_max_docs":1000},\
        {"type":"mmlu","name":"mmlu"}]' \
    --mmlu-batch-size 8 \
    --mode by-tensor-type \
    --results-dir results_goal2_repo100
```

---

## GOAL 3 — Dataset Repo Sample Count Sweep

**Purpose**: Compare 100, 500, 1000 samples from dataset repo for KLD analysis using `by-tensor-type` mode.

### All use
| Param | Value |
|---|---|
| Mode | `by-tensor-type` |
| dataset_repo | `w-ahmad/open-perfectblend-500k-tokens` |
| dataset_split | `train` |
| max_seq_length | `1200` |
| batch_size | `8` |
| seed | `42` |

### Variant A — 100 samples
| Param | Value |
|---|---|
| max_samples | `100` |
| num_chunks | `13` |
| **Total tokens** | **120,000** |

### Variant B — 500 samples
| Param | Value |
|---|---|
| max_samples | `500` |
| num_chunks | `63` |
| **Total tokens** | **600,000** |

### Variant C — 1000 samples
| Param | Value |
|---|---|
| max_samples | `1000` |
| num_chunks | `125` |
| **Total tokens** | **1,200,000** |

### Run Variant A (100 samples)
```bash
python scripts/moq.py run \
    --model Qwen/Qwen3.5-0.8B \
    --mapping configs/tensor_maps/tensor_map_qwen.txt \
    --gguf-repo unsloth/Qwen3.5-0.8B-GGUF \
    --gguf-file Qwen3.5-0.8B-BF16.gguf \
    --llama-cpp-dir /workspace/llama.cpp \
    --imatrix /workspace/Qwen3.5-0.8B.imatrix \
    --dataset-repo w-ahmad/open-perfectblend-500k-tokens \
    --dataset-split train \
    --max-seq-length 1200 \
    --max-samples 100 \
    --batch-size 8 \
    --num-chunks 13 \
    --seed 42 \
    --trust-remote-code \
    --dtype float16 \
    --quant-types ALL \
    --bits-list 5.0 4.5 4.0 3.5 3.0 2.5 \
    --eval-configs '[\
        {"type":"chat","path":"w-ahmad/open-perfectblend-500k-tokens","name":"pile-cc"},\
        {"type":"text","hf_dataset_repo":"HuggingFaceFW/fineweb-edu","name":"fineweb-edu","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"bigcode/the-stack","name":"the-stack-python","hf_dataset_config":"data/python","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"HuggingFaceTB/finemath","name":"finemath-4plus","hf_dataset_config":"finemath-4plus","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"allenai/c4","name":"c4-ar","hf_dataset_config":"ar","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"text","hf_dataset_repo":"allenai/c4","name":"c4-en","hf_dataset_config":"en","seq_length":512,"num_chunks":977,"hf_dataset_max_docs":1000},\
        {"type":"mmlu","name":"mmlu"}]' \
    --mmlu-batch-size 8 \
    --mode by-tensor-type \
    --results-dir results_goal3_100
```

### Run Variant B (500 samples)
```bash
# same as above but:
    --max-samples 500 \
    --num-chunks 63 \
    --results-dir results_goal3_500
```

### Run Variant C (1000 samples)
```bash
# same as above but:
    --max-samples 1000 \
    --num-chunks 125 \
    --results-dir results_goal3_1000
```

---

## Real Run Metrics

Full stage-by-stage metric schema lives in `Mixture-of-Quants.md` ("Pipeline Metrics").
Quick reference here:

| Stage | Output | Key metric |
|-------|--------|------------|
| 1. KLD Analysis | `analysis_{mode}.json` | `mean_kld`, `max_kld`, `p99_kld`, `p99.9_kld` per `(quant_type, group)` |
| 2. CE Computation | `ce_{mode}/ce_results.txt` + `param_counts.json` | `CE_Average` (normalized elasticity, 0-1) |
| 3. Config Generation | `configs_{mode}/` | `.txt` per target BPW (5.0 / 4.5 / 4.0 / 3.5 / 3.0) |
| 4. Apply + Evaluate | `eval_{mode}_{bits}.json`, `summary_{mode}.json`, `master_summary.json` | KLD + PPL (+ MMLU accuracy/agreement) |
| Logs | `Logs/{run}/metrics.jsonl` | RunLog dashboard entries |
