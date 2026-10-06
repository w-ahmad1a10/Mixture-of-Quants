# Mixture-of-Quants (MoQ)

Mixed-precision quantization research for **Qwen/Qwen3.5-0.8B**.

The MoQ pipeline measures how much each group of weights hurts the model when
quantized, then picks which quantization type each group should get so the total
stays under a bit budget, applies it, and evaluates the result.

## Layout

```
.
├── moq/                    # Python library (pip install -e .)
│   ├── moq_core.py         # THE library: everything (KLD, datasets, MoQRunner, ...)
│   ├── build_llama_cpp_cpu.py
│   └── quant_extension.cpp
├── scripts/                # Entry points and utilities
│   ├── moq.py              # THE CLI: pipeline / quant / eval / build-llama / map-tensors / run / upload
│   ├── run_all_goals.py    # Thin goal runner (reads sweep_config.yaml, calls moq.py run)
│   └── sweep_config.yaml   # All goal hyperparameters (verbatim from the original experiment)
├── configs/
│   └── tensor_maps/
│       └── tensor_map_qwen.txt   # Per-model tensor map (not library code)
├── tests/                  # test_tensor_registry.py (22 tests)
├── waste/                  # Obsolete files kept for reuse (moq_engine.py, moq_sweep.py, ...)
├── AGENTS.md               # Agent quick reference
├── Mixture-of-Quants.md    # Project record
├── help.md                 # Context / blog narrative
├── logic.md                # English logic record
├── pyproject.toml
└── .gitignore
```

## Install

```bash
pip install -e .
```

## Quick start

```bash
# Run a single mode — all 4 grouping modes × 6 bit budgets
python scripts/moq.py run \
    --model Qwen/Qwen3.5-0.8B \
    --mapping configs/tensor_maps/tensor_map_qwen.txt \
    --gguf-repo unsloth/Qwen3.5-0.8B-GGUF \
    --gguf-file Qwen3.5-0.8B-BF16.gguf \
    --llama-cpp-dir /workspace/llama.cpp \
    --imatrix /workspace/Qwen3.5-0.8B.imatrix \
    --dataset-repo w-ahmad/open-perfectblend-500k-tokens \
    --dataset-split train \
    --max-seq-length 1200 --max-samples 100 --batch-size 8 \
    --num-chunks 13 --seed 42 --trust-remote-code --dtype float16 \
    --quant-types ALL --bits-list 5.0 4.5 4.0 3.5 3.0 2.5 \
    --mmlu-batch-size 8 --mode by-tensor-type \
    --results-dir results_goal1_by-tensor-type
```

## CLI commands

| Command | What it does |
|---|---|
| `moq.py pipeline` | Full 5-stage MoQ generation pipeline |
| `moq.py quant` | Apply mixed-precision configs + optional eval/upload |
| `moq.py eval` | Batch evaluate student HF repos against a teacher repo |
| `moq.py build-llama` | Download & setup llama.cpp CPU binaries |
| `moq.py map-tensors` | Build GGUF ↔ PyTorch tensor name mapping |
| `moq.py run` | Run MoQ pipeline: KLD → CE → optimize → apply+evaluate |
| `moq.py upload` | Upload local results to HF dataset repo |

## Goals

- **GOAL 1** — Test all 4 MoQ grouping modes with expanded eval datasets
- **GOAL 2** — Noise impact analysis (calib data vs dataset repo)
- **GOAL 3** — Dataset repo sample count sweep (100 / 500 / 1000)
- **GOAL 4** — Uniform quantization eval (`configs/`)

## Notes

- Colab is **CPU only** — never request GPU, never pass `--gpu`.
- See `Mixture-of-Quants.md` for the full status and `AGENTS.md` for the agent quick reference.
- `logic.md` is the canonical English logic record; update it whenever logic changes.