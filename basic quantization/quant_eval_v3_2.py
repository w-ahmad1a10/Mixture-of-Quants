#!/usr/bin/env python3
"""
quant_eval_v3_2.py — KLD truncation + aggregation ablation.

Methods
-------
  method1b : slope(aggregation) × weight_noise × [W]
  method3b : measured(aggregation) × [W]

Aggregations
------------
  mean    : mean over per-position KLD
  p99     : 99th percentile over per-position KLD
  p99.9   : 99.9th percentile over per-position KLD
  combined: (min-max normalized mean + p99 + p99.9) / 3  per tensor per bit

Scan truncations (top-K support)
--------------------------------
  K = 100, 1000, 5000   (one forward per (tensor, bits); 3 KLDs derived)

Eval truncation is always K=1000.

No calibration, no uniform baselines, no dear objective.
Weight-space noise (‖ΔW‖²_F / ‖W‖²_F), same as v3.
RAM-based precompute (v3 style).

Upload: subfolder per run under HF_REPO/datasets path.
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
EAR_K = 20
EPS = 1e-12

TOPK_SCAN = 5000     # cache size for stack_python
TOPK_EVAL = 1000     # cache size for eval datasets (and eval slice of stack_python)

BITS = [2, 3, 4, 5, 6, 8, 12]
STATE_NONE = "none"
ALL_STATES = BITS + [STATE_NONE]

SCAN_TOPKS = [100, 1000, 5000]
AGGREGATIONS = ["mean", "p99", "p99.9", "combined"]

BUDGETS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
METHODS = ["method1b", "method3b"]
OBJECTIVES = ["kld"]

SCAN_DATASETS = ["stack_python"]
EVAL_DATASETS = ["wikitext", "stack_python", "fineweb_edu", "fineweb_math"]

HF_TOKEN = ""
HF_REPO = "w-ahmad/quant_eval_v3_2"

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


# ==========================================================================
# 2. RAM precompute (v3 style)
# ==========================================================================
def precompute_quantized(allowlist, orig_cpu, quantizable, device):
    """Compute and cache every (tensor, bits) QDQ in CPU RAM.
    Returns {name: {bits: (Wh_cpu_float32_tensor, noise_float)}}."""
    t0 = time.time()
    precomputed = {}
    total_tensors = len(allowlist)
    for idx, name in enumerate(allowlist):
        precomputed[name] = {}
        if not quantizable[name]:
            continue
        W_bf16 = orig_cpu[name]
        W_np = W_bf16.float().numpy()
        for bits in BITS:
            Wh, noise = qdq_per_group_sym(W_np, bits)
            if Wh is None:
                continue
            precomputed[name][bits] = (torch.from_numpy(Wh), noise)
        if (idx + 1) % 10 == 0 or idx == total_tensors - 1:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total_tensors - idx - 1)
            total_mb = sum(
                t.numel() * t.element_size()
                for d in precomputed.values()
                for (t, _) in d.values()
            ) / 1e6
            print(f"  [precompute] {idx + 1}/{total_tensors} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s, {total_mb:.0f} MB)")
    return precomputed


def apply_precomputed(model, name, Wh_cpu):
    p = model.get_parameter(name)
    p.data.copy_(Wh_cpu.to(p.device, dtype=p.dtype))


def restore_one(model, name, W_bf16_cpu):
    p = model.get_parameter(name)
    p.data.copy_(W_bf16_cpu.to(p.device, dtype=p.dtype))


def restore_all(model, allowlist, orig_cpu):
    for name in allowlist:
        restore_one(model, name, orig_cpu[name])


# ==========================================================================
# 3. Loading
# ==========================================================================
def load_allowlist(path, model):
    if not os.path.isfile(path):
        print(f"[allowlist] ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    names = []
    with open(path) as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        if "name" not in cols or "quantizable_default" not in cols:
            print("[allowlist] ERROR: missing columns", file=sys.stderr)
            sys.exit(1)
        for i, row in enumerate(r, start=2):
            val = str(row["quantizable_default"]).strip().lower()
            if val == "true":
                names.append(row["name"].strip())
            elif val != "false":
                print(f"[allowlist] ERROR: line {i}", file=sys.stderr)
                sys.exit(1)
    if not names:
        print("[allowlist] ERROR: empty", file=sys.stderr)
        sys.exit(1)
    param_map = dict(model.named_parameters())
    errs = []
    if len(set(names)) != len(names):
        errs.append("duplicates")
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
def build_cache(model, sequences, device, fwd_chunk, topk):
    all_ids, all_probs = [], []
    n = len(sequences)
    t0 = time.time()
    for i in range(0, n, fwd_chunk):
        chunk = sequences[i:i + fwd_chunk]
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(ids).logits.float()
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, k=topk, dim=-1)
        all_ids.append(top_i.to(torch.int32).cpu())
        all_probs.append(top_p.cpu())
        del logits, probs, top_p, top_i
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (i // fwd_chunk) % 5 == 0:
            print(f"    [cache] {i + len(chunk)}/{n} ({time.time() - t0:.0f}s)")
    ids = torch.cat(all_ids, dim=0).reshape(-1, topk)
    probs = torch.cat(all_probs, dim=0).reshape(-1, topk)
    return ids, probs


# ==========================================================================
# 4. Scan one (tensor, bits) — one forward, three K aggregations
# ==========================================================================
@torch.no_grad()
def scan_one_tensor_bit(model, seqs, c_ids_full, c_probs_full, device, fwd_chunk):
    """
    One forward pass; compute per-position KLD at each K in SCAN_TOPKS.
    Also returns EAR, q_res, q20_res at K=1000.
    """
    n = len(seqs)
    L = len(seqs[0])
    per_pos_kld = {K: [] for K in SCAN_TOPKS}
    per_pos_ear_1000 = []
    per_pos_qres_1000 = []
    per_pos_q20res_1000 = []

    for i in range(0, n, fwd_chunk):
        chunk = seqs[i:i + fwd_chunk]
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(ids).logits.float()
        probs = torch.softmax(logits, dim=-1)

        start = i * L
        end = (i + len(chunk)) * L
        flat = probs.reshape(-1, probs.shape[-1])

        for K in SCAN_TOPKS:
            c_ids_K = c_ids_full[start:end, :K].long().to(device)
            c_probs_K = c_probs_full[start:end, :K].to(device)

            q_raw = flat.gather(1, c_ids_K)
            p_sum = c_probs_K.sum(dim=1, keepdim=True)
            p = c_probs_K / p_sum.clamp_min(EPS)
            q_sum = q_raw.sum(dim=1, keepdim=True)
            q = q_raw / q_sum.clamp_min(EPS)
            kld = (p * torch.log((p + EPS) / (q + EPS))).sum(dim=1)
            per_pos_kld[K].append(kld.cpu())

            if K == 1000:
                q_res = 1.0 - q_sum.squeeze(1)
                per_pos_qres_1000.append(q_res.cpu())

                p20_raw = c_probs_K[:, :EAR_K]
                q20_raw = q_raw[:, :EAR_K]
                p20 = p20_raw / p20_raw.sum(dim=1, keepdim=True).clamp_min(EPS)
                q20 = q20_raw / q20_raw.sum(dim=1, keepdim=True).clamp_min(EPS)
                ear = torch.minimum(p20, q20).sum(dim=1)
                per_pos_ear_1000.append(ear.cpu())

                q20_res = 1.0 - q20_raw.sum(dim=1)
                per_pos_q20res_1000.append(q20_res.cpu())

            del c_ids_K, c_probs_K, q_raw, p_sum, p, q_sum, q, kld

        del logits, probs, flat
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {}
    for K in SCAN_TOPKS:
        arr = torch.cat(per_pos_kld[K]).numpy()
        out[f"mean_KLD_K{K}"] = float(arr.mean())
        out[f"p99_KLD_K{K}"] = float(np.percentile(arr, 99))
        out[f"p99.9_KLD_K{K}"] = float(np.percentile(arr, 99.9))
    out["EAR"] = float(torch.cat(per_pos_ear_1000).numpy().mean())
    out["q_res"] = float(torch.cat(per_pos_qres_1000).numpy().mean())
    out["q20_res"] = float(torch.cat(per_pos_q20res_1000).numpy().mean())
    return out


# ==========================================================================
# 5. Phase 1a — scan
# ==========================================================================
def phase1a_scan(model, allowlist, orig_cpu, quantizable, precomputed,
                 scan_cache, device, fwd_chunk):
    seqs, c_ids_full, c_probs_full = scan_cache
    c_ids_full = c_ids_full.to(device)
    c_probs_full = c_probs_full.to(device)
    if device.type == "cuda":
        mb = (c_ids_full.numel() + c_probs_full.numel()) * 4 / 1e6
        print(f"[phase1a] cache on GPU ({mb:.0f} MB)")

    rows = []
    t0 = time.time()
    n = len(allowlist)
    for idx, name in enumerate(allowlist):
        for bits in BITS:
            if not quantizable[name]:
                rows.append({"tensor": name, "bits": bits,
                             **{f"{k}_K{K}": np.nan for k in ("mean_KLD", "p99_KLD", "p99.9_KLD")
                                for K in SCAN_TOPKS},
                             "EAR": np.nan, "q_res": np.nan, "q20_res": np.nan})
                continue
            entry = precomputed[name].get(bits)
            if entry is None:
                rows.append({"tensor": name, "bits": bits,
                             **{f"{k}_K{K}": np.nan for k in ("mean_KLD", "p99_KLD", "p99.9_KLD")
                                for K in SCAN_TOPKS},
                             "EAR": np.nan, "q_res": np.nan, "q20_res": np.nan})
                continue
            Wh_cpu, _ = entry
            apply_precomputed(model, name, Wh_cpu)
            m = scan_one_tensor_bit(model, seqs, c_ids_full, c_probs_full,
                                     device, fwd_chunk)
            rows.append({"tensor": name, "bits": bits, **m})
            restore_one(model, name, orig_cpu[name])
        elapsed = time.time() - t0
        print(f"  [scan] {idx + 1}/{n} {name} ({elapsed:.0f}s)")

    del c_ids_full, c_probs_full
    if device.type == "cuda":
        torch.cuda.empty_cache()
    restore_all(model, allowlist, orig_cpu)
    return pd.DataFrame(rows)


# ==========================================================================
# 6. Phase 1b — slopes + combined_norm
# ==========================================================================
def fit_slopes_for_K(scan_df, noise_df, K):
    mean_col = f"mean_KLD_K{K}"
    p99_col = f"p99_KLD_K{K}"
    p999_col = f"p99.9_KLD_K{K}"
    merged = scan_df[["tensor", "bits", mean_col, p99_col, p999_col]].merge(
        noise_df, on=["tensor", "bits"], how="inner"
    )
    rows = []
    for tensor, g in merged.groupby("tensor"):
        g = g.dropna(subset=[mean_col, p99_col, p999_col, "noise"])
        g = g[g["noise"] > 1e-30]
        if len(g) < 2:
            rows.append({"tensor": tensor,
                         "slope_mean": 0.0, "R2_mean": 1.0,
                         "slope_p99": 0.0, "R2_p99": 1.0,
                         "slope_p99.9": 0.0, "R2_p99.9": 1.0,
                         "n_points": len(g)})
            continue
        nz = g["noise"].values

        def fit(vals):
            s = float((vals * nz).sum() / max((nz ** 2).sum(), 1e-30))
            r2 = 1.0 - ((vals - s * nz) ** 2).sum() / max((vals ** 2).sum(), 1e-30)
            return s, float(r2)

        s_m, r2_m = fit(g[mean_col].values)
        s_p99, r2_p99 = fit(g[p99_col].values)
        s_p999, r2_p999 = fit(g[p999_col].values)
        rows.append({"tensor": tensor,
                     "slope_mean": s_m, "R2_mean": r2_m,
                     "slope_p99": s_p99, "R2_p99": r2_p99,
                     "slope_p99.9": s_p999, "R2_p99.9": r2_p999,
                     "n_points": int(len(g))})
    return pd.DataFrame(rows)


def compute_combined_norm(scan_df, K):
    mean_col = f"mean_KLD_K{K}"
    p99_col = f"p99_KLD_K{K}"
    p999_col = f"p99.9_KLD_K{K}"
    rows = []
    for tensor, g in scan_df.groupby("tensor"):
        g = g.dropna(subset=[mean_col, p99_col, p999_col]).copy()
        g = g.sort_values("bits").reset_index(drop=True)
        if len(g) < 2:
            for _, r in g.iterrows():
                rows.append({"tensor": tensor, "bits": int(r["bits"]),
                             "combined_norm": 0.0})
            continue
        norms = {}
        for agg, col in [("mean", mean_col), ("p99", p99_col),
                         ("p99.9", p999_col)]:
            vals = g[col].values
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin < 1e-30:
                norms[agg] = np.zeros_like(vals)
            else:
                norms[agg] = (vals - vmin) / (vmax - vmin)
        combined = (norms["mean"] + norms["p99"] + norms["p99.9"]) / 3.0
        for j, row in g.iterrows():
            rows.append({"tensor": tensor, "bits": int(row["bits"]),
                         "combined_norm": float(combined[j])})
    return pd.DataFrame(rows)


def fit_slope_combined(combined_df, noise_df):
    merged = combined_df.merge(noise_df, on=["tensor", "bits"], how="inner")
    rows = []
    for tensor, g in merged.groupby("tensor"):
        g = g.dropna(subset=["combined_norm", "noise"])
        g = g[g["noise"] > 1e-30]
        if len(g) < 2:
            rows.append({"tensor": tensor,
                         "slope_combined": 0.0, "R2_combined": 1.0})
            continue
        nz = g["noise"].values
        vals = g["combined_norm"].values
        s = float((vals * nz).sum() / max((nz ** 2).sum(), 1e-30))
        r2 = 1.0 - ((vals - s * nz) ** 2).sum() / max((vals ** 2).sum(), 1e-30)
        rows.append({"tensor": tensor, "slope_combined": s,
                     "R2_combined": float(r2)})
    return pd.DataFrame(rows)


# ==========================================================================
# 7. Phase 2 — ILP
# ==========================================================================
def build_cost_matrix(method, aggregation, K, scan_df, slopes_df,
                      combined_df, noise_df, allowlist, param_weights,
                      quantizable, use_w):
    n = len(allowlist)
    n_states = len(ALL_STATES)
    cost = np.full((n, n_states), np.inf, dtype=np.float64)
    storage = np.full((n, n_states), np.inf, dtype=np.float64)
    for i in range(n):
        for j, state in enumerate(ALL_STATES):
            storage[i, j] = 16.0 if state == STATE_NONE else float(state) + 1.0

    noise_pivot = noise_df.pivot_table(
        index="tensor", columns="bits", values="noise", aggfunc="first"
    )
    slopes_by = slopes_df.set_index("tensor").to_dict("index")

    if method == "method1b":
        if aggregation == "combined":
            slope_key = "slope_combined"
        else:
            slope_key = f"slope_{aggregation}"

        for i, name in enumerate(allowlist):
            cost[i, -1] = 0.0
            if not quantizable[name]:
                continue
            w = param_weights[name] if use_w else 1.0
            slope = slopes_by.get(name, {}).get(slope_key, 0.0)
            for j, state in enumerate(ALL_STATES):
                if state == STATE_NONE:
                    continue
                try:
                    nz = float(noise_pivot.loc[name, state])
                except (KeyError, ValueError):
                    continue
                if np.isnan(nz):
                    continue
                cost[i, j] = slope * nz * w
    elif method == "method3b":
        if aggregation == "combined":
            piv = combined_df.pivot_table(
                index="tensor", columns="bits",
                values="combined_norm", aggfunc="first"
            )
        else:
            col = f"{aggregation}_KLD_K{K}"
            piv = scan_df.pivot_table(
                index="tensor", columns="bits", values=col, aggfunc="first"
            )
        for i, name in enumerate(allowlist):
            cost[i, -1] = 0.0
            if not quantizable[name]:
                continue
            w = param_weights[name] if use_w else 1.0
            for j, state in enumerate(ALL_STATES):
                if state == STATE_NONE:
                    continue
                try:
                    val = float(piv.loc[name, state])
                except (KeyError, ValueError):
                    continue
                if np.isnan(val):
                    continue
                cost[i, j] = val * w
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
    state_idx = [int(np.argmax(x[i])) for i in range(n)]
    predicted = float(sum(cost[i, state_idx[i]] for i in range(n)))
    return state_idx, predicted


# ==========================================================================
# 8. Phase 3a — composite eval
# ==========================================================================
def apply_assignment(model, allowlist, orig_cpu, assignment, precomputed):
    for name in allowlist:
        state = assignment.get(name, STATE_NONE)
        if state == STATE_NONE:
            restore_one(model, name, orig_cpu[name])
        else:
            Wh_cpu, _ = precomputed[name][int(state)]
            apply_precomputed(model, name, Wh_cpu)


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


@torch.no_grad()
def eval_composite(model, seqs, c_ids, c_probs, device, fwd_chunk, topk=1000):
    n = len(seqs)
    L = len(seqs[0])
    all_kld = []
    all_ear = []
    all_qres = []
    all_q20res = []

    for i in range(0, n, fwd_chunk):
        chunk = seqs[i:i + fwd_chunk]
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(ids).logits.float()
        probs = torch.softmax(logits, dim=-1)

        start = i * L
        end = (i + len(chunk)) * L
        c_ids_s = c_ids[start:end, :topk].long().to(device)
        c_probs_s = c_probs[start:end, :topk].to(device)

        flat = probs.reshape(-1, probs.shape[-1])
        q_raw = flat.gather(1, c_ids_s)

        p_sum = c_probs_s.sum(dim=1, keepdim=True)
        p = c_probs_s / p_sum.clamp_min(EPS)
        q_sum = q_raw.sum(dim=1, keepdim=True)
        q = q_raw / q_sum.clamp_min(EPS)
        kld = (p * torch.log((p + EPS) / (q + EPS))).sum(dim=1)
        q_res = 1.0 - q_sum.squeeze(1)

        p20_raw = c_probs_s[:, :EAR_K]
        q20_raw = q_raw[:, :EAR_K]
        p20 = p20_raw / p20_raw.sum(dim=1, keepdim=True).clamp_min(EPS)
        q20 = q20_raw / q20_raw.sum(dim=1, keepdim=True).clamp_min(EPS)
        ear = torch.minimum(p20, q20).sum(dim=1)
        q20_res = 1.0 - q20_raw.sum(dim=1)

        all_kld.append(kld.cpu())
        all_ear.append(ear.cpu())
        all_qres.append(q_res.cpu())
        all_q20res.append(q20_res.cpu())

        del logits, probs, flat, q_raw, kld, ear
        if device.type == "cuda":
            torch.cuda.empty_cache()

    kld_arr = torch.cat(all_kld).numpy()
    return {
        "mean_kld": float(kld_arr.mean()),
        "p99_kld": float(np.percentile(kld_arr, 99)),
        "p99.9_kld": float(np.percentile(kld_arr, 99.9)),
        "mean_ear": float(torch.cat(all_ear).numpy().mean()),
        "mean_q_res": float(torch.cat(all_qres).numpy().mean()),
        "mean_q20_res": float(torch.cat(all_q20res).numpy().mean()),
    }


# ==========================================================================
# 9. Upload
# ==========================================================================
def upload_results(out_dir, repo_id, token, run_name):
    if not repo_id:
        print("[upload] no HF_REPO set, skipping")
        return
    if not token:
        print("[upload] no HF_TOKEN, skipping")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=repo_id, repo_type="dataset",
            folder_path=out_dir, path_in_repo=run_name,
            ignore_patterns=["*.pyc", "__pycache__/**"],
            commit_message=f"quant_eval_v3_2 {run_name} "
                           f"{datetime.now(timezone.utc).isoformat()}",
        )
        print(f"[upload] done: https://huggingface.co/datasets/{repo_id}/tree/main/{run_name}")
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

    use_w = args.use_w
    print(f"[config] use_w = {use_w}")

    with open(args.allowlist) as s, \
         open(os.path.join(args.out_dir, "allowlist_used.csv"), "w") as d:
        d.write(s.read())

    # ---- precompute in RAM ----
    print("[precompute] computing all (tensor, bits) QDQ once ...")
    precomputed = precompute_quantized(allowlist, orig_cpu, quantizable, device)

    # ---- build caches ----
    # stack_python needs top-5000 (scan + eval on it)
    # other eval datasets top-1000
    max_seqs = max(args.scan_n_seqs, args.n_seqs)
    all_ds = sorted(set(SCAN_DATASETS) | set(EVAL_DATASETS))
    print(f"[cache] building caches for {all_ds} at {max_seqs} seqs")
    caches_full = {}
    for ds in all_ds:
        topk = TOPK_SCAN if ds == "stack_python" else TOPK_EVAL
        print(f"[cache] {ds} (topk={topk})")
        seqs = list(stream_sequences(ds, tokenizer, args.seq_len, max_seqs))
        c_ids, c_probs = build_cache(model, seqs, device, args.fwd_chunk, topk)
        caches_full[ds] = (seqs, c_ids, c_probs, topk)

    def slice_cache(ds, n, requested_topk):
        seqs, c_ids, c_probs, cache_topk = caches_full[ds]
        k = min(requested_topk, cache_topk)
        return (seqs[:n],
                c_ids[:n * args.seq_len, :k],
                c_probs[:n * args.seq_len, :k])

    # Scan cache: stack_python at top-5000, first scan_n_seqs
    scan_cache = slice_cache("stack_python", args.scan_n_seqs, TOPK_SCAN)
    # Eval caches: every dataset at top-1000, first n_seqs
    eval_caches = {ds: slice_cache(ds, args.n_seqs, TOPK_EVAL)
                   for ds in EVAL_DATASETS}

    # ---- Phase 1a — scan ----
    p1a = os.path.join(args.out_dir, "phase1", "scan_v3_2.parquet")
    if os.path.exists(p1a) and args.resume:
        scan_df = pd.read_parquet(p1a)
    else:
        scan_df = phase1a_scan(model, allowlist, orig_cpu, quantizable,
                                precomputed, scan_cache, device, args.fwd_chunk)
        scan_df.to_parquet(p1a, index=False)

    # ---- noise table ----
    noise_rows = []
    for name in allowlist:
        for bits, (_, noise) in precomputed[name].items():
            noise_rows.append({"tensor": name, "bits": bits, "noise": noise})
    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_parquet(os.path.join(args.out_dir, "phase1", "noise_w.parquet"),
                        index=False)

    # ---- Phase 1b — slopes + combined_norm per K ----
    slopes_by_k = {}
    combined_by_k = {}
    for K in SCAN_TOPKS:
        sp = os.path.join(args.out_dir, "phase1", f"slopes_v3_2_K{K}.parquet")
        cp = os.path.join(args.out_dir, "phase1", f"combined_norm_v3_2_K{K}.parquet")
        if os.path.exists(sp) and os.path.exists(cp) and args.resume:
            slopes_by_k[K] = pd.read_parquet(sp)
            combined_by_k[K] = pd.read_parquet(cp)
        else:
            combined_df = compute_combined_norm(scan_df, K)
            combined_df.to_parquet(cp, index=False)
            slopes_df = fit_slopes_for_K(scan_df, noise_df, K)
            comb_slopes = fit_slope_combined(combined_df, noise_df)
            slopes_df = slopes_df.merge(comb_slopes, on="tensor", how="left")
            slopes_df.to_parquet(sp, index=False)
            slopes_by_k[K] = slopes_df
            combined_by_k[K] = combined_df

    # tensor meta
    pd.DataFrame([
        {"tensor": n, "param_count": param_counts[n],
         "param_weight": param_weights[n], "quantizable": quantizable[n]}
        for n in allowlist
    ]).to_parquet(os.path.join(args.out_dir, "phase1", "tensor_meta.parquet"),
                  index=False)

    # ---- Phase 2 — ILP ----
    configs_path = os.path.join(args.out_dir, "phase2", "all_configs_v3_2.json")
    if os.path.exists(configs_path) and args.resume:
        with open(configs_path) as f:
            configs = json.load(f)
        configs = {k: {float(b): v for b, v in mv.items()}
                   for k, mv in configs.items()}
    else:
        quantizable_list = [quantizable[nm] for nm in allowlist]
        param_counts_list = [param_counts[nm] for nm in allowlist]
        configs = {}
        t0 = time.time()
        for K in SCAN_TOPKS:
            for agg in AGGREGATIONS:
                for method in METHODS:
                    key = f"K{K}_agg{agg}_{method}"
                    configs[key] = {}
                    cost, storage = build_cost_matrix(
                        method, agg, K, scan_df, slopes_by_k[K],
                        combined_by_k[K], noise_df, allowlist, param_weights,
                        quantizable, use_w,
                    )
                    for budget in BUDGETS:
                        state_idx, pred = solve_ilp(
                            cost, storage, quantizable_list,
                            param_counts_list, budget,
                        )
                        if state_idx is None:
                            print(f"[solve] {key} b={budget:4.1f} INFEASIBLE")
                            configs[key][budget] = None
                            continue
                        assignment = {allowlist[i]: ALL_STATES[state_idx[i]]
                                      for i in range(len(allowlist))}
                        n_quant = sum(1 for v in assignment.values()
                                      if v != STATE_NONE)
                        print(f"[solve] {key} b={budget:4.1f} feasible "
                              f"({n_quant} quant, pred={pred:.4e})")
                        configs[key][budget] = {
                            "assignment": assignment,
                            "predicted_cost": pred,
                        }
        with open(configs_path, "w") as f:
            json.dump(configs, f, indent=2)
        print(f"[phase2] {len(configs)} config groups, {time.time() - t0:.0f}s")

    # ---- Phase 3a — composite eval ----
    p3a = os.path.join(args.out_dir, "phase3", "composite_results_v3_2.csv")
    if os.path.exists(p3a) and args.resume:
        comp_df = pd.read_csv(p3a)
    else:
        rows = []
        t0 = time.time()
        for key, budget_map in configs.items():
            # parse key: K{K}_agg{agg}_{method}
            parts = key.split("_")
            K_val = int(parts[0][1:])
            agg = parts[1][3:]
            method = "_".join(parts[2:])

            for budget in BUDGETS:
                entry = budget_map.get(budget)
                if entry is None:
                    continue
                assignment = entry["assignment"]
                predicted = entry["predicted_cost"]
                bpw = actual_bpw(assignment, allowlist, param_counts,
                                 quantizable, total_params)
                apply_assignment(model, allowlist, orig_cpu, assignment, precomputed)
                for ds_name in EVAL_DATASETS:
                    seqs, c_ids, c_probs = eval_caches[ds_name]
                    m = eval_composite(model, seqs, c_ids, c_probs,
                                        device, args.fwd_chunk, topk=TOPK_EVAL)
                    rows.append({
                        "K": K_val, "aggregation": agg, "method": method,
                        "budget_target": budget, "dataset": ds_name,
                        "actual_bpw": bpw, "predicted_cost": predicted,
                        **m,
                    })
                    print(f"  [eval] {key} b={budget:4.1f} ds={ds_name:<13s} "
                          f"bpw={bpw:.2f} KLD={m['mean_kld']:.4e} "
                          f"EAR={m['mean_ear']:.4f}")
                restore_all(model, allowlist, orig_cpu)
        comp_df = pd.DataFrame(rows)
        comp_df.to_csv(p3a, index=False)
        print(f"[phase3] {len(comp_df)} rows ({time.time() - t0:.0f}s)")

    # ---- run_config + upload ----
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = args.model.replace("/", "_")
    run_name = f"quant_eval_v3_2_{model_short}_{timestamp}"

    run_config = {
        "model": args.model, "dtype": args.dtype, "device": str(device),
        "seq_len": args.seq_len, "scan_n_seqs": args.scan_n_seqs,
        "n_seqs": args.n_seqs, "fwd_chunk": args.fwd_chunk,
        "scan_datasets": SCAN_DATASETS, "eval_datasets": EVAL_DATASETS,
        "scan_topks": SCAN_TOPKS, "eval_topk": TOPK_EVAL,
        "aggregations": AGGREGATIONS,
        "bits": BITS, "budgets": BUDGETS, "methods": METHODS,
        "objectives": OBJECTIVES,
        "n_tensors": len(allowlist), "total_params": total_params,
        "use_w_cost": use_w,
        "noise_definition": "weight_space_relative_no_division",
        "precompute_storage": "ram",
        "scan_eval_overlap_note": "stack_python scan=first 30 seqs, "
                                  "eval=first 100 seqs (30 seq overlap)",
        "timestamp": timestamp,
        "run_name": run_name,
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    if not args.no_upload:
        token = HF_TOKEN or os.environ.get("HF_TOKEN", "")
        upload_results(args.out_dir, HF_REPO, token, run_name)


# ==========================================================================
# CLI
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--out-dir", default="./results_v3_2")
    ap.add_argument("--dtype", default="float16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--scan-n-seqs", type=int, default=30)
    ap.add_argument("--n-seqs", type=int, default=100)
    ap.add_argument("--fwd-chunk", type=int, default=4,
                    help="batch size per forward; reduce if OOM")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--use-w", action="store_true",
                    help="Enable param_weight multiplier in cost (default: OFF)")
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()