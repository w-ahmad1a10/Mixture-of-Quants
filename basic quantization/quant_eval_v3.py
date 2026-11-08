#!/usr/bin/env python3
"""
quant_eval_v3.py — Bit allocation with 10 methods + uniform baselines.

Methods
-------
  M1  : slope(default) × noise × W
  M2  : measured(default) × noise × W
  M3  : measured(default) × W
  M4  : marginal_sequential(default, allowlist order) × W
  M5  : = M3 (alias, reused matrix)
  M6  : marginal_sequential(default, random order, seed) × W
  M7  : slope(wikitext) × noise × W
  M8  : slope(fineweb_edu) × noise × W
  M9  : slope(fineweb_math) × noise × W
  M10 : (M4 + M5 + M6) / 3

Objectives: kld, dear (1 − EAR). Both run for every method.

Uniform baselines: pure b-bit models (2,3,4,5,6,8,12) evaluated on all datasets.

Upload: mandatory to HF (HF_REPO / HF_TOKEN constants or env).
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.optimize import milp, LinearConstraint, Bounds

# ==========================================================================
# Constants
# ==========================================================================
GROUP_SIZE = 16
TOPK_CACHE = 1000
EAR_K = 20
EPS = 1e-12

# ---- HF ----
HF_TOKEN = ""                      # falls back to env HF_TOKEN
HF_REPO  = "w-ahmad/model1a_v3"    # dataset repo

# ---- Bit-space ----
BITS = [2, 3, 4, 5, 6, 8, 12]
STATE_NONE = "none"
ALL_STATES = BITS + [STATE_NONE]
UNIFORM_VARIANTS = [2, 3, 4, 5, 6, 8, 12]

BUDGETS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
METHODS = ["method1", "method2", "method3", "method4", "method5",
           "method6", "method7", "method8", "method9", "method10"]
OBJECTIVES = ["kld", "dear"]

RANDOM_SEED = 0
DEFAULT_SCAN_DATASET = "stack_python"

DATASET_SPECS = {
    "wikitext": {
        "loader": lambda: load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1",
            split="train", streaming=True
        ),
        "field": "text", "filter": None,
    },
    "stack_python": {
        "loader": lambda: load_dataset(
            "bigcode/the-stack", data_dir="data/python",
            split="train", streaming=True
        ),
        "field": "content",
        "filter": lambda ex: ex.get("lang", "Python") == "Python",
    },
    "fineweb_edu": {
        "loader": lambda: load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True
        ),
        "field": "text", "filter": None,
    },
    "fineweb_math": {
        "loader": lambda: load_dataset(
            "HuggingFaceTB/finemath", "finemath-3plus",
            split="train", streaming=True
        ),
        "field": "text", "filter": None,
    },
}


# ==========================================================================
# 1. QDQ (per-group-16 symmetric, no clipping)
# ==========================================================================
def qdq_per_group_sym(W, bits):
    out_f, in_f = W.shape
    if in_f % GROUP_SIZE != 0:
        return None, None
    Wr = W.reshape(out_f, in_f // GROUP_SIZE, GROUP_SIZE)
    bound = np.abs(Wr).max(axis=2, keepdims=True).astype(np.float32)
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    scale = bound / float(qmax)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.round(Wr / scale), qmin, qmax)
    Wh = (q * scale).reshape(W.shape).astype(np.float32)
    denom = max(float((W ** 2).sum()), 1e-30)
    noise = float(((W - Wh) ** 2).sum()) / denom
    return Wh, noise


def apply_one(model, name, W_bf16_cpu, bits):
    p = model.get_parameter(name)
    W_np = W_bf16_cpu.float().numpy()
    Wh, noise = qdq_per_group_sym(W_np, bits)
    if Wh is None:
        return None
    p.data.copy_(torch.from_numpy(Wh).to(p.device, dtype=p.dtype))
    return noise


def restore_one(model, name, W_bf16_cpu):
    p = model.get_parameter(name)
    p.data.copy_(W_bf16_cpu.to(p.device, dtype=p.dtype))


def restore_all(model, allowlist, orig_cpu):
    for name in allowlist:
        restore_one(model, name, orig_cpu[name])


# ==========================================================================
# 2. Loading
# ==========================================================================
def load_allowlist(path, model):
    if not os.path.isfile(path):
        print(f"[allowlist] ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    names = []
    with open(path) as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        if "name" not in cols or "quantizable_default" not in cols:
            print("[allowlist] ERROR: need 'name' and 'quantizable_default'",
                  file=sys.stderr)
            sys.exit(1)
        for i, row in enumerate(r, start=2):
            val = str(row["quantizable_default"]).strip().lower()
            if val == "true":
                names.append(row["name"].strip())
            elif val != "false":
                print(f"[allowlist] ERROR: line {i} bad value "
                      f"{row['quantizable_default']!r}", file=sys.stderr)
                sys.exit(1)
    if not names:
        print("[allowlist] ERROR: no rows selected", file=sys.stderr)
        sys.exit(1)
    param_map = dict(model.named_parameters())
    errs = []
    if len(set(names)) != len(names):
        errs.append("duplicate names")
    missing = [n for n in names if n not in param_map]
    if missing:
        errs.append(f"missing: {missing}")
    non2d = [n for n in names if n in param_map and param_map[n].dim() != 2]
    if non2d:
        errs.append(f"non-2D: {non2d}")
    if errs:
        for e in errs:
            print(f"[allowlist] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[allowlist] {len(names)} tensors")
    return names


def stream_sequences(dataset_name, tokenizer, seq_len, n_seqs):
    spec = DATASET_SPECS[dataset_name]
    ds = spec["loader"]()
    field = spec["field"]
    filt = spec["filter"]
    eos = tokenizer.eos_token_id or (tokenizer.pad_token_id or 0)
    buf = []
    produced = 0
    for ex in ds:
        if filt is not None and not filt(ex):
            continue
        text = ex.get(field)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        ids.append(eos)
        buf.extend(ids)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]
            produced += 1
            if produced >= n_seqs:
                return


@torch.no_grad()
def build_cache(model, sequences, device, fwd_chunk):
    all_ids, all_probs = [], []
    n = len(sequences)
    t0 = time.time()
    for i in range(0, n, fwd_chunk):
        chunk = sequences[i:i + fwd_chunk]
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(ids).logits.float()
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, k=TOPK_CACHE, dim=-1)
        all_ids.append(top_i.to(torch.int32).cpu())
        all_probs.append(top_p.cpu())
        del logits, probs, top_p, top_i
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (i // fwd_chunk) % 5 == 0:
            print(f"    [cache] {i + len(chunk)}/{n} ({time.time() - t0:.0f}s)")
    ids = torch.cat(all_ids, dim=0).reshape(-1, TOPK_CACHE)
    probs = torch.cat(all_probs, dim=0).reshape(-1, TOPK_CACHE)
    return ids, probs


# ==========================================================================
# 3. Metrics
# ==========================================================================
@torch.no_grad()
def evaluate_config(model, sequences, cache_ids, cache_probs, device, fwd_chunk):
    n = len(sequences)
    L = len(sequences[0])
    s_kld = s_kld2 = 0.0
    s_ear = s_ear2 = 0.0
    s_qres = s_q20res = 0.0
    count = 0
    for i in range(0, n, fwd_chunk):
        chunk = sequences[i:i + fwd_chunk]
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(ids).logits.float()
        probs = torch.softmax(logits, dim=-1)
        start = i * L
        end = (i + len(chunk)) * L
        c_ids = cache_ids[start:end].long().to(device)
        c_probs = cache_probs[start:end].to(device)
        flat = probs.reshape(-1, probs.shape[-1])
        q_raw = flat.gather(1, c_ids)
        p_sum = c_probs.sum(dim=1, keepdim=True)
        p = c_probs / p_sum.clamp_min(EPS)
        q_sum = q_raw.sum(dim=1, keepdim=True)
        q = q_raw / q_sum.clamp_min(EPS)
        kld = (p * torch.log((p + EPS) / (q + EPS))).sum(dim=1)
        q_res = 1.0 - q_sum.squeeze(1)
        p20_raw = c_probs[:, :EAR_K]
        q20_raw = q_raw[:, :EAR_K]
        p20 = p20_raw / p20_raw.sum(dim=1, keepdim=True).clamp_min(EPS)
        q20 = q20_raw / q20_raw.sum(dim=1, keepdim=True).clamp_min(EPS)
        ear = torch.minimum(p20, q20).sum(dim=1)
        q20_res = 1.0 - q20_raw.sum(dim=1)
        s_kld += kld.sum().item()
        s_kld2 += (kld ** 2).sum().item()
        s_ear += ear.sum().item()
        s_ear2 += (ear ** 2).sum().item()
        s_qres += q_res.sum().item()
        s_q20res += q20_res.sum().item()
        count += kld.numel()
        del logits, probs, flat, q_raw, kld, ear
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean_kld = s_kld / count
    std_kld = max(s_kld2 / count - mean_kld ** 2, 0.0) ** 0.5
    mean_ear = s_ear / count
    std_ear = max(s_ear2 / count - mean_ear ** 2, 0.0) ** 0.5
    return {
        "n_positions": count,
        "mean_kld": mean_kld, "std_kld": std_kld,
        "mean_ear": mean_ear, "std_ear": std_ear,
        "mean_q_res": s_qres / count,
        "mean_q20_res": s_q20res / count,
    }


# ==========================================================================
# 4. Phase 1a — independent scan per dataset
# ==========================================================================
def phase1a_scan(model, allowlist, orig_cpu, quantizable, scan_caches,
                 device, fwd_chunk):
    rows = []
    t0 = time.time()
    n = len(allowlist)
    for ds_name, (seqs, c_ids, c_probs) in scan_caches.items():
        print(f"\n[phase1a] scanning dataset: {ds_name}")
        for idx, name in enumerate(allowlist):
            W_bf16 = orig_cpu[name]
            for bits in BITS:
                if not quantizable[name]:
                    rows.append({"dataset": ds_name, "tensor": name, "bits": bits,
                                 "KLD": np.nan, "EAR": np.nan,
                                 "q_res": np.nan, "q20_res": np.nan, "noise": np.nan})
                    continue
                noise = apply_one(model, name, W_bf16, bits)
                if noise is None:
                    rows.append({"dataset": ds_name, "tensor": name, "bits": bits,
                                 "KLD": np.nan, "EAR": np.nan,
                                 "q_res": np.nan, "q20_res": np.nan, "noise": np.nan})
                    continue
                m = evaluate_config(model, seqs, c_ids, c_probs, device, fwd_chunk)
                rows.append({
                    "dataset": ds_name, "tensor": name, "bits": bits,
                    "KLD": m["mean_kld"], "EAR": m["mean_ear"],
                    "q_res": m["mean_q_res"], "q20_res": m["mean_q20_res"],
                    "noise": noise,
                })
                restore_one(model, name, W_bf16)
            elapsed = time.time() - t0
            print(f"  [scan:{ds_name}] {idx + 1}/{n} {name} ({elapsed:.0f}s)")
    restore_all(model, allowlist, orig_cpu)
    return pd.DataFrame(rows)


# ==========================================================================
# 5. Phase 1b — slope fit per dataset
# ==========================================================================
def fit_slopes(scan_df):
    rows = []
    for name, g in scan_df.groupby("tensor"):
        g = g.dropna(subset=["KLD", "EAR", "noise"])
        g = g[g["noise"] > 1e-20]
        if len(g) < 2:
            rows.append({"tensor": name, "slope_KLD": 0.0, "slope_dEAR": 0.0,
                         "R2_KLD": 1.0, "R2_dEAR": 1.0, "n_points": len(g)})
            continue
        nz = g["noise"].values
        kld = g["KLD"].values
        dear = 1.0 - g["EAR"].values
        s_kld = float((kld * nz).sum() / max((nz ** 2).sum(), 1e-30))
        s_dear = float((dear * nz).sum() / max((nz ** 2).sum(), 1e-30))
        r2_kld = 1.0 - ((kld - s_kld * nz) ** 2).sum() / max((kld ** 2).sum(), 1e-30)
        r2_dear = 1.0 - ((dear - s_dear * nz) ** 2).sum() / max((dear ** 2).sum(), 1e-30)
        rows.append({"tensor": name, "slope_KLD": s_kld, "slope_dEAR": s_dear,
                     "R2_KLD": float(r2_kld), "R2_dEAR": float(r2_dear),
                     "n_points": int(len(g))})
    return pd.DataFrame(rows)


# ==========================================================================
# 6. Phase 1c/1d — sequential (M4 / M6)
# ==========================================================================
def phase1_sequential(model, allowlist, orig_cpu, quantizable,
                      scan_cache, tensor_order, device, fwd_chunk, tag):
    rows = []
    clamp_k = clamp_e = 0
    t0 = time.time()
    for bits in BITS:
        restore_all(model, allowlist, orig_cpu)
        cum_kld = 0.0
        cum_ear = 1.0
        for k, name in enumerate(tensor_order):
            if not quantizable[name]:
                rows.append({"tensor": name, "bits": bits, "order": k,
                             "cum_KLD": cum_kld, "cum_EAR": cum_ear,
                             "marginal_KLD": 0.0, "marginal_dEAR": 0.0})
                continue
            W_bf16 = orig_cpu[name]
            noise = apply_one(model, name, W_bf16, bits)
            if noise is None:
                rows.append({"tensor": name, "bits": bits, "order": k,
                             "cum_KLD": cum_kld, "cum_EAR": cum_ear,
                             "marginal_KLD": 0.0, "marginal_dEAR": 0.0})
                continue
            seqs, c_ids, c_probs = scan_cache
            m = evaluate_config(model, seqs, c_ids, c_probs, device, fwd_chunk)
            new_kld = m["mean_kld"]
            new_ear = m["mean_ear"]
            mk = new_kld - cum_kld
            me = (1.0 - new_ear) - (1.0 - cum_ear)
            if mk < 0:
                mk = 0.0
                clamp_k += 1
            if me < 0:
                me = 0.0
                clamp_e += 1
            rows.append({"tensor": name, "bits": bits, "order": k,
                         "cum_KLD": new_kld, "cum_EAR": new_ear,
                         "marginal_KLD": mk, "marginal_dEAR": me})
            cum_kld = new_kld
            cum_ear = new_ear
        print(f"  [seq:{tag}] {bits}-bit done ({time.time() - t0:.0f}s)")
    restore_all(model, allowlist, orig_cpu)
    print(f"[{tag}] clamped negatives: KLD={clamp_k}, dEAR={clamp_e}")
    return pd.DataFrame(rows)


# ==========================================================================
# 7. Phase 2 — cost matrices + ILP
# ==========================================================================
def build_cost_matrix(method, objective, scan_dfs, slopes_dfs, m4_df, m6_df,
                      allowlist, param_weights, quantizable, default_dataset):
    n = len(allowlist)
    n_states = len(ALL_STATES)
    cost = np.full((n, n_states), np.inf, dtype=np.float64)
    storage = np.full((n, n_states), np.inf, dtype=np.float64)

    for i in range(n):
        for j, state in enumerate(ALL_STATES):
            if state == STATE_NONE:
                storage[i, j] = 16.0
            else:
                storage[i, j] = float(state) + 1.0

    if method in ["method1", "method2", "method3", "method5"]:
        ds = default_dataset
        kld_pivot = scan_dfs[ds].pivot_table(index="tensor", columns="bits",
                                              values="KLD", aggfunc="first")
        ear_pivot = scan_dfs[ds].pivot_table(index="tensor", columns="bits",
                                              values="EAR", aggfunc="first")
        noise_pivot = scan_dfs[ds].pivot_table(index="tensor", columns="bits",
                                                values="noise", aggfunc="first")
        slopes_by = slopes_dfs[ds].set_index("tensor").to_dict("index")
        for i, name in enumerate(allowlist):
            cost[i, -1] = 0.0
            if not quantizable[name]:
                continue
            w = param_weights[name]
            for j, state in enumerate(ALL_STATES):
                if state == STATE_NONE:
                    continue
                try:
                    mk = float(kld_pivot.loc[name, state])
                    me = float(ear_pivot.loc[name, state])
                    nz = float(noise_pivot.loc[name, state])
                except (KeyError, ValueError):
                    continue
                if any(np.isnan(x) for x in (mk, me, nz)):
                    continue
                if objective == "kld":
                    measured = mk; slope = slopes_by[name]["slope_KLD"]
                else:
                    measured = 1.0 - me; slope = slopes_by[name]["slope_dEAR"]
                if method == "method1":
                    cost[i, j] = slope * nz * w
                elif method == "method2":
                    cost[i, j] = measured * nz * w
                else:  # method3, method5
                    cost[i, j] = measured * w

    elif method in ["method7", "method8", "method9"]:
        ds = {"method7": "wikitext", "method8": "fineweb_edu",
              "method9": "fineweb_math"}[method]
        kld_pivot = scan_dfs[ds].pivot_table(index="tensor", columns="bits",
                                              values="KLD", aggfunc="first")
        ear_pivot = scan_dfs[ds].pivot_table(index="tensor", columns="bits",
                                              values="EAR", aggfunc="first")
        noise_pivot = scan_dfs[ds].pivot_table(index="tensor", columns="bits",
                                                values="noise", aggfunc="first")
        slopes_by = slopes_dfs[ds].set_index("tensor").to_dict("index")
        for i, name in enumerate(allowlist):
            cost[i, -1] = 0.0
            if not quantizable[name]:
                continue
            w = param_weights[name]
            for j, state in enumerate(ALL_STATES):
                if state == STATE_NONE:
                    continue
                try:
                    nz = float(noise_pivot.loc[name, state])
                except (KeyError, ValueError):
                    continue
                if np.isnan(nz):
                    continue
                slope = slopes_by[name]["slope_KLD"] if objective == "kld" \
                        else slopes_by[name]["slope_dEAR"]
                cost[i, j] = slope * nz * w

    elif method in ["method4", "method6"]:
        seq_df = m4_df if method == "method4" else m6_df
        piv_k = seq_df.pivot_table(index="tensor", columns="bits",
                                    values="marginal_KLD", aggfunc="first")
        piv_e = seq_df.pivot_table(index="tensor", columns="bits",
                                    values="marginal_dEAR", aggfunc="first")
        for i, name in enumerate(allowlist):
            cost[i, -1] = 0.0
            if not quantizable[name]:
                continue
            w = param_weights[name]
            for j, state in enumerate(ALL_STATES):
                if state == STATE_NONE:
                    continue
                try:
                    if objective == "kld":
                        val = float(piv_k.loc[name, state])
                    else:
                        val = float(piv_e.loc[name, state])
                except (KeyError, ValueError):
                    continue
                if np.isnan(val):
                    continue
                cost[i, j] = val * w

    elif method == "method10":
        c4, _ = build_cost_matrix("method4", objective, scan_dfs, slopes_dfs,
                                   m4_df, m6_df, allowlist, param_weights,
                                   quantizable, default_dataset)
        c5, _ = build_cost_matrix("method5", objective, scan_dfs, slopes_dfs,
                                   m4_df, m6_df, allowlist, param_weights,
                                   quantizable, default_dataset)
        c6, _ = build_cost_matrix("method6", objective, scan_dfs, slopes_dfs,
                                   m4_df, m6_df, allowlist, param_weights,
                                   quantizable, default_dataset)
        cost = (c4 + c5 + c6) / 3.0

    else:
        raise ValueError(f"unknown method {method}")

    return cost, storage


def solve_ilp(cost, storage, quantizable_list, param_counts, budget_bpw):
    n, n_states = cost.shape
    n_vars = n * n_states

    c = cost.flatten().copy()
    c[np.isinf(c)] = 1e10

    A_eq = np.zeros((n, n_vars))
    for i in range(n):
        A_eq[i, i * n_states:(i + 1) * n_states] = 1.0
    b_eq = np.ones(n)

    storage_bits = storage * np.array(param_counts).reshape(-1, 1)
    ws = storage_bits.flatten().copy()
    ws[np.isinf(ws)] = 1e10
    A_ineq = ws.reshape(1, -1)
    b_ineq = budget_bpw * float(sum(param_counts))

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    for i in range(n):
        if not quantizable_list[i]:
            for j in range(len(BITS)):
                ub[i * n_states + j] = 0.0

    res = milp(c=c,
               constraints=[LinearConstraint(A_eq, b_eq, b_eq),
                            LinearConstraint(A_ineq, -np.inf, b_ineq)],
               integrality=np.ones(n_vars), bounds=Bounds(lb, ub))
    if not res.success:
        return None, None
    x = np.asarray(res.x).reshape(n, n_states)
    predicted = 0.0
    assignment = {}
    for i, name in enumerate([f"__i{i}" for i in range(n)]):
        j = int(np.argmax(x[i]))
        assignment[name] = j
        predicted += cost[i, j]
    return assignment, float(predicted)


def phase2_solve(scan_dfs, slopes_dfs, m4_df, m6_df, allowlist, param_weights,
                 quantizable, param_counts, default_dataset, budgets):
    quantizable_list = [quantizable[nm] for nm in allowlist]
    param_counts_list = [param_counts[nm] for nm in allowlist]

    matrices = {}
    for obj in OBJECTIVES:
        for method in METHODS:
            cost, storage = build_cost_matrix(
                method, obj, scan_dfs, slopes_dfs, m4_df, m6_df,
                allowlist, param_weights, quantizable, default_dataset,
            )
            matrices[(method, obj)] = (cost, storage)

    configs = {}
    for method in METHODS:
        configs[method] = {}
        for budget in budgets:
            configs[method][budget] = {}
            for obj in OBJECTIVES:
                cost, storage = matrices[(method, obj)]
                assign_idx, pred = solve_ilp(
                    cost, storage, quantizable_list, param_counts_list, budget,
                )
                if assign_idx is None:
                    print(f"[solve] {method} b={budget:4.1f} {obj:4s} INFEASIBLE")
                    configs[method][budget][obj] = None
                    continue
                assignment = {allowlist[i]: ALL_STATES[assign_idx[allowlist[i]]]
                              for i in range(len(allowlist))}
                # fix: assign_idx uses placeholders __i{n}; rebuild cleanly
                assignment = {}
                for i, name in enumerate(allowlist):
                    assignment[name] = ALL_STATES[assign_idx[f"__i{i}"]]
                n_quant = sum(1 for v in assignment.values() if v != STATE_NONE)
                print(f"[solve] {method} b={budget:4.1f} {obj:4s} feasible "
                      f"({n_quant} quantized, pred={pred:.4e})")
                configs[method][budget][obj] = {
                    "assignment": assignment,
                    "predicted_cost": pred,
                }
    return configs


# ==========================================================================
# 8. Phase 3 — evaluation
# ==========================================================================
def apply_assignment(model, allowlist, orig_cpu, assignment):
    for name in allowlist:
        state = assignment.get(name, STATE_NONE)
        if state == STATE_NONE:
            restore_one(model, name, orig_cpu[name])
        else:
            apply_one(model, name, orig_cpu[name], int(state))


def actual_bpw(assignment, allowlist, param_counts, quantizable, total_params):
    total = 0
    for name in allowlist:
        state = assignment.get(name, STATE_NONE)
        pc = param_counts[name]
        if state == STATE_NONE or not quantizable[name]:
            total += pc * 16
        else:
            total += pc * (int(state) + 1)
    return total / total_params


def phase3a_composite(model, allowlist, orig_cpu, quantizable, configs,
                      eval_caches, eval_datasets, param_counts, total_params,
                      device, fwd_chunk):
    rows = []
    for method in METHODS:
        for budget in BUDGETS:
            for obj in OBJECTIVES:
                entry = configs.get(method, {}).get(budget, {}).get(obj)
                if not entry:
                    continue
                assignment = entry["assignment"]
                predicted = entry["predicted_cost"]
                bpw = actual_bpw(assignment, allowlist, param_counts,
                                 quantizable, total_params)
                apply_assignment(model, allowlist, orig_cpu, assignment)
                for ds_name in eval_datasets:
                    seqs, c_ids, c_probs = eval_caches[ds_name]
                    m = evaluate_config(model, seqs, c_ids, c_probs,
                                        device, fwd_chunk)
                    rows.append({
                        "method": method, "budget_target": budget,
                        "objective": obj, "dataset": ds_name,
                        "actual_bpw": bpw,
                        "predicted_cost": predicted,
                        "mean_kld": m["mean_kld"], "mean_ear": m["mean_ear"],
                        "mean_q_res": m["mean_q_res"],
                        "mean_q20_res": m["mean_q20_res"],
                    })
                    print(f"  [eval] {method} b={budget:4.1f} {obj:4s} "
                          f"ds={ds_name:<13s} bpw={bpw:.2f} "
                          f"KLD={m['mean_kld']:.4e} EAR={m['mean_ear']:.4f}")
                restore_all(model, allowlist, orig_cpu)
    return pd.DataFrame(rows)


def phase3b_uniform(model, allowlist, orig_cpu, quantizable, eval_caches,
                    eval_datasets, param_counts, total_params, device, fwd_chunk):
    rows = []
    for bits in UNIFORM_VARIANTS:
        total = 0
        for name in allowlist:
            pc = param_counts[name]
            if quantizable[name]:
                total += pc * (bits + 1)
            else:
                total += pc * 16
        bpw = total / total_params
        for name in allowlist:
            if quantizable[name]:
                apply_one(model, name, orig_cpu[name], bits)
        for ds_name in eval_datasets:
            seqs, c_ids, c_probs = eval_caches[ds_name]
            m = evaluate_config(model, seqs, c_ids, c_probs, device, fwd_chunk)
            rows.append({
                "variant": f"uniform_{bits}bit",
                "bits": bits, "dataset": ds_name, "actual_bpw": bpw,
                "mean_kld": m["mean_kld"], "mean_ear": m["mean_ear"],
                "mean_q_res": m["mean_q_res"],
                "mean_q20_res": m["mean_q20_res"],
            })
            print(f"  [uniform] {bits}-bit ds={ds_name} bpw={bpw:.2f} "
                  f"KLD={m['mean_kld']:.4e} EAR={m['mean_ear']:.4f}")
        restore_all(model, allowlist, orig_cpu)
    return pd.DataFrame(rows)


# ==========================================================================
# 9. Upload
# ==========================================================================
def upload_results(out_dir, repo_id, token):
    if not repo_id:
        print("[upload] no HF_REPO set, skipping")
        return
    if not token:
        print("[upload] no HF_TOKEN available, skipping")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset",
                          folder_path=out_dir,
                          commit_message=f"quant_eval_v3 "
                                         f"{datetime.now(timezone.utc).isoformat()}")
        print(f"[upload] done: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[upload] failed: {e}", file=sys.stderr)


# ==========================================================================
# 10. Main
# ==========================================================================
def cmd_run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for d in ("phase1", "phase2", "phase3"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)

    device = torch.device(args.device)
    print(f"[load] {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device).eval()

    allowlist = load_allowlist(args.allowlist, model)
    orig_cpu = {n: model.get_parameter(n).detach().to("cpu", copy=True)
                for n in allowlist}
    param_counts = {n: model.get_parameter(n).numel() for n in allowlist}
    total_params = sum(param_counts.values())
    param_weights = {n: param_counts[n] / total_params for n in allowlist}
    quantizable = {n: (model.get_parameter(n).shape[1] % GROUP_SIZE == 0)
                   for n in allowlist}

    with open(args.allowlist) as s, \
         open(os.path.join(args.out_dir, "allowlist_used.csv"), "w") as d:
        d.write(s.read())

    scan_ds = [x.strip() for x in args.scan_datasets.split(",") if x.strip()]
    eval_ds = [x.strip() for x in args.eval_datasets.split(",") if x.strip()]
    if args.default_dataset not in scan_ds:
        print(f"[error] default_dataset {args.default_dataset} not in scan_datasets",
              file=sys.stderr)
        sys.exit(1)

    all_ds = sorted(set(scan_ds) | set(eval_ds))
    max_seqs = max(args.scan_n_seqs, args.n_seqs)

    print(f"[cache] building caches for {all_ds} at {max_seqs} seqs")
    caches_full = {}
    for ds in all_ds:
        print(f"[cache] {ds}")
        seqs = list(stream_sequences(ds, tokenizer, args.seq_len, max_seqs))
        c_ids, c_probs = build_cache(model, seqs, device, args.fwd_chunk)
        caches_full[ds] = (seqs, c_ids, c_probs)

    def slice_cache(ds, n):
        seqs, c_ids, c_probs = caches_full[ds]
        return (seqs[:n],
                c_ids[:n * args.seq_len],
                c_probs[:n * args.seq_len])

    scan_caches = {ds: slice_cache(ds, args.scan_n_seqs) for ds in scan_ds}
    eval_caches = {ds: slice_cache(ds, args.n_seqs) for ds in eval_ds}

    # ---- Phase 1a ----
    p1a = os.path.join(args.out_dir, "phase1", "scan_all.parquet")
    if os.path.exists(p1a) and args.resume:
        scan_df = pd.read_parquet(p1a)
    else:
        scan_df = phase1a_scan(model, allowlist, orig_cpu, quantizable,
                                scan_caches, device, args.fwd_chunk)
        scan_df.to_parquet(p1a, index=False)
    scan_dfs = {ds: scan_df[scan_df["dataset"] == ds].drop(columns=["dataset"])
                for ds in scan_ds}

    # ---- Phase 1b ----
    slopes_dfs = {}
    for ds in scan_ds:
        p = os.path.join(args.out_dir, "phase1", f"slopes_{ds}.parquet")
        if os.path.exists(p) and args.resume:
            slopes_dfs[ds] = pd.read_parquet(p)
        else:
            slopes_dfs[ds] = fit_slopes(scan_dfs[ds])
            slopes_dfs[ds].to_parquet(p, index=False)

    # ---- Phase 1c / 1d ----
    m4_df = m6_df = None
    p_m4 = os.path.join(args.out_dir, "phase1", "method4_sequential.parquet")
    p_m6 = os.path.join(args.out_dir, "phase1", "method6_random.parquet")

    if os.path.exists(p_m4) and args.resume:
        m4_df = pd.read_parquet(p_m4)
    else:
        order = list(allowlist)
        m4_df = phase1_sequential(
            model, allowlist, orig_cpu, quantizable,
            scan_caches[args.default_dataset], order, device,
            args.fwd_chunk, tag="M4",
        )
        m4_df.to_parquet(p_m4, index=False)

    if os.path.exists(p_m6) and args.resume:
        m6_df = pd.read_parquet(p_m6)
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        order = list(allowlist)
        rng.shuffle(order)
        m6_df = phase1_sequential(
            model, allowlist, orig_cpu, quantizable,
            scan_caches[args.default_dataset], order, device,
            args.fwd_chunk, tag="M6",
        )
        m6_df.to_parquet(p_m6, index=False)

    # tensor meta
    meta_df = pd.DataFrame([
        {"tensor": n, "param_count": param_counts[n],
         "param_weight": param_weights[n], "quantizable": quantizable[n]}
        for n in allowlist
    ])
    meta_df.to_parquet(os.path.join(args.out_dir, "phase1", "tensor_meta.parquet"),
                       index=False)

    # ---- Phase 2 ----
    configs_path = os.path.join(args.out_dir, "phase2", "all_configs.json")
    if os.path.exists(configs_path) and args.resume:
        with open(configs_path) as f:
            configs = json.load(f)
        configs = {m: {float(b): v for b, v in mv.items()}
                   for m, mv in configs.items()}
    else:
        configs = phase2_solve(scan_dfs, slopes_dfs, m4_df, m6_df, allowlist,
                                param_weights, quantizable, param_counts,
                                args.default_dataset, BUDGETS)
        with open(configs_path, "w") as f:
            json.dump(configs, f, indent=2)

    # ---- Phase 3a ----
    p3a = os.path.join(args.out_dir, "phase3", "composite_results.csv")
    if os.path.exists(p3a) and args.resume:
        composite_df = pd.read_csv(p3a)
    else:
        composite_df = phase3a_composite(
            model, allowlist, orig_cpu, quantizable, configs, eval_caches,
            eval_ds, param_counts, total_params, device, args.fwd_chunk,
        )
        composite_df.to_csv(p3a, index=False)

    # ---- Phase 3b ----
    p3b = os.path.join(args.out_dir, "phase3", "uniform_results.csv")
    if os.path.exists(p3b) and args.resume:
        uniform_df = pd.read_csv(p3b)
    else:
        uniform_df = phase3b_uniform(
            model, allowlist, orig_cpu, quantizable, eval_caches, eval_ds,
            param_counts, total_params, device, args.fwd_chunk,
        )
        uniform_df.to_csv(p3b, index=False)

    # ---- run_config + upload ----
    run_config = {
        "model": args.model, "dtype": args.dtype, "device": str(device),
        "seq_len": args.seq_len, "scan_n_seqs": args.scan_n_seqs,
        "n_seqs": args.n_seqs, "fwd_chunk": args.fwd_chunk,
        "scan_datasets": scan_ds, "eval_datasets": eval_ds,
        "default_dataset": args.default_dataset, "random_seed": RANDOM_SEED,
        "bits": BITS, "budgets": BUDGETS, "methods": METHODS,
        "objectives": OBJECTIVES, "uniform_variants": UNIFORM_VARIANTS,
        "n_tensors": len(allowlist), "total_params": total_params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    if not args.no_upload:
        token = HF_TOKEN or os.environ.get("HF_TOKEN", "")
        upload_results(args.out_dir, HF_REPO, token)


# ==========================================================================
# CLI
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--out-dir", default="./results_v3")
    ap.add_argument("--dtype", default="float16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--scan-n-seqs", type=int, default=30)
    ap.add_argument("--n-seqs", type=int, default=100)
    ap.add_argument("--fwd-chunk", type=int, default=16)
    ap.add_argument("--scan-datasets",
                    default="wikitext,stack_python,fineweb_edu,fineweb_math")
    ap.add_argument("--eval-datasets",
                    default="wikitext,stack_python,fineweb_edu,fineweb_math")
    ap.add_argument("--default-dataset", default=DEFAULT_SCAN_DATASET,
                    help="Dataset used by M1-M6 and M10")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()