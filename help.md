# help.md — Mixture-of-Quants: Project Context & Blog Notes

> **What this file is:** the living context document for the Mixture-of-Quants (MoQ)
> project and the blog post being written about it. Waleed dictates context in plain
> language; this file structures and interprets it. It grows as more context is shared.
>
> Maintained by: Cline (agent) · Owner: Waleed (AI researcher)
>
> **Layout:** the Python library lives in `moq/` (installed via `pip install -e .`). The single CLI entry point is `scripts/moq.py`; the experimental goal runner is `scripts/run_all_goals.py` (reads `scripts/sweep_config.yaml`). Per-model data lives in `configs/tensor_maps/`, tests in `tests/`, uniform-quant configs in `configs/`. Obsolete files are kept in `waste/`.
> Last updated: 2026-10-04 · Status: **refactored — experiment results still to be added**

---

## 1. Origin story

Waleed built Mixture-of-Quants himself, used it to produce **many models, released
them on Hugging Face**, and ran evaluations to select the best solution. The variant
that shipped and that he actually uses is **`by-tensor-type` grouping** — chosen
because it was the efficient one.

The next goal is a **blog post** explaining the project. This file collects the
context for it.

---

## 2. The problem: quantization hurts local models

- llama.cpp lets anyone quantize a model into **GGUF** format for fast local
  inference on consumer hardware.
- But quantization **degrades the model**. Typical symptoms:
  - **endless tokens** (degenerate generation that never stops),
  - **poor performance on tasks** the original model could do fine.
- The degradation is not uniform — it depends on *where* and *how hard* you quantize.

---

## 3. Why not just invent a better quantization method?

- Millions of people use llama.cpp for local AI. Creating a **new engine or new
  quant formats** is hard, slow to adopt, and risky.
- The pragmatic move: **keep the existing quants exactly as they are** and improve
  the one thing nobody automates properly — **which tensor gets which quant**.

> **Blog framing:** "We don't need a new engine. We need to spend the bit budget
> correctly with the quants we already have."

---

## 4. The key insight: sensitivity is not equal across the model

- **Some parts of a model are highly sensitive to quantization** — they can
  *collapse* when quantized heavily (this is where the endless-tokens failure
  comes from).
- **Other parts survive extreme quantization** with almost no damage.
- A single uniform quant type must pick one size for *everything*: it either wastes
  bits on tough-but-robust tensors or destroys the fragile ones.
- If bits are **allocated correctly among tensors**, you get a model that is
  **smaller AND more accurate** than any uniform quant at the same size.

---

## 5. The llama.cpp quant landscape (the toolbox MoQ mixes from)

Each existing quant has a **different way of quantizing** and a **different
bits-per-weight (BPW)**:

| Family | Examples | Character |
|---|---|---|
| Legacy (simple) | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | Simple, fast; the classics |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | Better quality than legacy, **trade speed** |
| I-quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | Allow **non-linearity**, complex, **reduce speed**; extreme compression |

(The repo's `GGML_TYPE_MAP` in `moq_core.py` is the exact list the pipeline can
assign, including very-low-bit entries like Q1_0/Q2_0.)

> **Experiment decision (2026-09-06):** Q1_0 is **EXCLUDED** from the group-type
> inner-test toolbox (owner decision). So 21 → **20 quants used**:
> 6 legacy (Q4_0..Q8_0, Q2_0), 5 K-quants (Q2_K..Q6_K), 9 I-quants.

**The catch:** the quant *method* and the *bit width* are two separate axes, and
**both decide quality** — you can't rank the methods in the abstract (see §7).

---

## 6. How Mixture-of-Quants works (the recipe)

MoQ **mixes many quants inside one model**. The pipeline, in plain language:

1. **Group** the model's tensors into candidate units. A *group* can be a single
   tensor, a whole layer, tensors of the same type, layer-blocks, etc. (four
   grouping variants were tested — see §8).
2. **Sensitivity probing (KLD):** for each group, quantize **only that group**
   with one candidate quant while **keeping every other group at original
   precision (16-bit)** — so only that one group is degraded. Measure the **KLD**
   (KL divergence) between the untouched model's token distribution and the
   degraded model's. This reveals *how sensitive* that group is to that quant.
3. **Coefficient of elasticity:** computed per group from the probe results —
   the code's "CE" stage (`run_compute_ce` in `moq_engine.py`). It summarizes how
   each group's performance responds as its precision/bits change.
4. **Budget assignment (dynamic programming):** an optimizer assigns each group
   its best quant from the probe data so the **whole model hits a target BPW
   budget** (e.g. 5.0 / 4.5 / 4.0 / 3.5 / 3.0 average bits per weight) with the
   least total damage. In the code this is the MCKP-style knapsack solver.
5. **Build & evaluate:** apply the mixed config, then score the mixed model
   against the original (teacher) with the same KLD metric — assistant tokens
   only — to confirm the size savings didn't cost quality.

---

## 7. Design principle: analysis-only selection (no hand rules)

This is the part Waleed wants the blog to get exactly right:

- **There is no rule** like "sensitive tensor ⇒ give it Q6_K" or "big layers get
  K-quants". The best quant for a group is decided **purely by measurement**.
- Reason: **the quant method and the bit width jointly decide the outcome, and the
  relationship is ambiguous / non-monotonic.**
  - A group might work **perfectly with Q2_0** but **fail with Q6_K**.
  - Sometimes performance is *poor at 6 bits* but *good at 4 bits*, or the reverse.
- Because there is no rule, **MoQ's KLD analysis tests every quant method** on
  every group and lets the data pick the winner per group, per budget.

> **Blog framing:** "In quantization, more bits is not a ladder — it's a maze.
> So we stopped guessing and started measuring."

---

## 8. What was tried, and what won

Four grouping variants were implemented and swept against each other
(all four live in `MoQRunner` in `moq/moq_core.py`):

1. `individual` — each tensor decided on its own
2. `by-layer` — whole layers as units
3. `by-tensor-type` — tensors grouped by their type/role  ← **THE CHOSEN ONE**
4. `by-layer-tensor` — layer blocks × tensor type

**Winner in practice: `by-tensor-type`**, because it was **efficient** (good
quality gains per unit of probe/analysis cost — exact trade data in §9).

---

## 9. Results & evidence  ⏳ *(to be filled — Waleed will share)*

- [ ] Sweep results across the 4 grouping variants
- [ ] KLD sensitivity findings (what collapsed, what survived)
- [ ] Quality-vs-size curves for released models
- [ ] List of models released on Hugging Face with MoQ
- [ ] Evaluation numbers used to select the final solution
- [ ] Non-monotonicity examples (a real tensor where 4-bit beat 6-bit, etc.)

---

## 10. Blog plan  ⏳ *(to be filled)*

- [ ] Audience & angle
- [ ] Outline / sections
- [ ] Which charts to include (visualize with **w1a-board**, local dashboard only)

---

## 11. How tensors & group types are discovered — no manual input

Nothing is told to the model by hand. MoQ **discovers** everything:

- Tensors are found automatically from the loaded model (`discover_quantizable_tensors`):
  - must have a weight (biases skipped), must be 2-D (1-D norms skipped),
  - name must not match the auto skip-list `SKIP_PATTERNS` (`norm`, `ln`, `rotary`,
    `bias`, layernorms, conv1d, `ln_f`, `ln_1`, `ln_2`).
- The **tensor type is the last word of the tensor's name**: e.g. `...q_proj.weight` → type `q_proj`.
  So all `q_proj` weights across all layers automatically land in one group. This is
  **architecture-agnostic** (Llama, Mistral, Qwen… whatever its naming, the suffix is the type).
- **No hard-coded type list exists** in the code (verified: `map_unit_to_gguf_type` only searches
  the mapping file for a matching name).
- The only required input files are: the **mapping file** (PyTorch name ↔ GGUF name, auto-generated
  by the `map-tensors` stage), the **imatrix** (compulsory), the model id, and calibration/eval data.
  The mapping file is a *lookup table* used to match config lines (`blk.0.attn_q.weight=IQ3_S`) back
  to PyTorch modules — not a rulebook and not manual.

---

## 12. Terminology map (blog language ↔ repo code)

| Blog term | Code artifact |
|---|---|
| Group / unit (tensor, layer, type…) | candidate unit; `discover_quantizable_tensors`, `group_by_tensor_type`, `group_by_layer`, `group_by_layer_tensor_type` (`moq/moq_core.py`) |
| Sensitivity probe (one group degraded, rest 16-bit) | `NoiseImpactAnalyzer` (`moq/moq_core.py`) |
| KLD metric (assistant tokens only) | `compute_token_kld` (`moq/moq_core.py`) |
| Coefficient of elasticity | the **CE** stage: `run_compute_ce` (`moq/moq_core.py`) |
| Bit-budget assignment (DP) | MCKP optimizer (optimize stage in `MoQRunner`, `moq/moq_core.py`) |
| Applying the mixed config | `MoQFinalQuantizer` / `MixedQuantRunner` |
| Student vs teacher evaluation | `Evaluator` (`moq/moq_core.py`) |
| Full experiment runner (all 4 variants × budgets) | `MoQRunner` (`moq/moq_core.py`), invoked via `scripts/moq.py run` |
| Chosen variant | `by-tensor-type` grouping |
| BPW budgets | `--bits-list` (defaults 5.0 / 4.5 / 4.0 / 3.5 / 3.0) |
| Quant toolbox | `GGML_TYPE_MAP` (`moq/moq_core.py`): legacy + K-quants + I-quants |
