#!/usr/bin/env python3
"""
quant_eval_v2.py — Bit allocation via per-tensor sensitivity scan + ILP knapsack.

Phases
------
  scan         Per-tensor QDQ scan across {2,3,4,5,6,8,12} bits, measure KLD/EAR/noise.
  fit          Origin-constrained linear slope fit per tensor.
  solve        Build cost matrices (3 methods × 2 objectives) + ILP per budget.
  eval         Apply composite assignments, measure actual KLD/EAR.
  upload       Push everything to HF (repo created if not exists).

All phases are idempotent and resumable: existing outputs are loaded, not overwritten.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import product

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

# ---- Fill here, or via env ----
HF_TOKEN = ""          # e.g. "hf_xxx"; if empty, falls back to os.environ["HF_TOKEN"]
HF_REPO = "w-ahmad/model1a"           # e.g. "user/quant-bit-allocation"

BITS = [2, 3, 4, 5, 6, 8, 12]
STATE_NONE = "none"
ALL_STATES = BITS + [STATE_NONE]
BUDGETS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 8.0]
METHODS = ["method1", "method2", "method3"]
OBJECTIVES = ["kld", "dear"]

DATASET_SPECS = {
    "wikitext": {
        "loader": lambda: load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1",
            split="train", streaming=True
        ),
        "field": "text",
        "filter": None,
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
        "field": "text",
        "filter": None,
    },
    "fineweb_math": {
        "loader": lambda: load_dataset(
            "HuggingFaceTB/finemath", "finemath-3plus",
            split="train", streaming=True
        ),
        "field": "text",
        "filter": None,
    },
}


# ==========================================================================
# 1. Per-group-16 symmetric QDQ (the only scheme used here)
# ==========================================================================
def qdq_per_group_sym(W, bits):
    """
    Symmetric per-group-16 quantization, no clipping.
    W: np.float32 [out, in]
    Returns (W_hat: np.float32 [out, in], noise: float) or (None, None) if not quantizable.
    """
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
    """Apply QDQ to a single tensor in-place. Returns noise or None."""
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
# 2. Model / allowlist / data loading
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
            print("[allowlist] ERROR: need 'name' and 'quantizable_default' columns",
                  file=sys.stderr)
            sys.exit(1)
        for i, row in enumerate(r, start=2):
            val = str(row["quantizable_default"]).strip().lower()
            if val == "true":
                names.append(row["name"].strip())
            elif val != "false":
                print(f"[allowlist] ERROR: line {i} bad value {row['quantizable_default']!r}",
                      file=sys.stderr)
                sys.exit(1)

    if not names:
        print("[allowlist] ERROR: no rows selected", file=sys.stderr)
        sys.exit(1)

    param_map = dict(model.named_parameters())
    errs = []
    if len(set(names)) != len(names):
        errs.append(f"duplicate names")
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
# 3. Metric evaluation
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
# 4. Phase 1 — per-tensor scan
# ==========================================================================
def phase1_scan(model, allowlist, orig_cpu, param_counts, quantizable,
                scan_caches, scan_datasets, seq_len, device, fwd_chunk):
    rows = []
    n = len(allowlist)
    t0 = time.time()

    for idx, name in enumerate(allowlist):
        W_bf16 = orig_cpu[name]
        for bits in BITS:
            if not quantizable[name]:
                rows.append({
                    "tensor": name, "bits": bits,
                    "KLD": np.nan, "EAR": np.nan,
                    "q_res": np.nan, "q20_res": np.nan, "noise": np.nan,
                })
                continue
            noise = apply_one(model, name, W_bf16, bits)
            if noise is None:
                rows.append({
                    "tensor": name, "bits": bits,
                    "KLD": np.nan, "EAR": np.nan,
                    "q_res": np.nan, "q20_res": np.nan, "noise": np.nan,
                })
                continue

            klds, ears, qres, q20res = [], [], [], []
            for ds_name in scan_datasets:
                seqs, c_ids, c_probs = scan_caches[ds_name]
                m = evaluate_config(model, seqs, c_ids, c_probs, device, fwd_chunk)
                klds.append(m["mean_kld"])
                ears.append(m["mean_ear"])
                qres.append(m["mean_q_res"])
                q20res.append(m["mean_q20_res"])

            rows.append({
                "tensor": name, "bits": bits,
                "KLD": float(np.mean(klds)),
                "EAR": float(np.mean(ears)),
                "q_res": float(np.mean(qres)),
                "q20_res": float(np.mean(q20res)),
                "noise": noise,
            })
            restore_one(model, name, W_bf16)

        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (n - idx - 1)
        print(f"  [scan] {idx + 1}/{n}  {name}  ({elapsed:.0f}s elapsed, {eta:.0f}s ETA)")

    restore_all(model, allowlist, orig_cpu)
    return pd.DataFrame(rows)


# ==========================================================================
# 5. Phase 1.5 — origin-constrained slope fit
# ==========================================================================
def phase15_fit(scan_df):
    rows = []
    for name, g in scan_df.groupby("tensor"):
        g = g.dropna(subset=["KLD", "EAR", "noise"])
        g = g[g["noise"] > 1e-20]
        if len(g) < 2:
            rows.append({
                "tensor": name,
                "slope_KLD": 0.0, "slope_dEAR": 0.0,
                "R2_KLD": 1.0, "R2_dEAR": 1.0, "n_points": len(g),
            })
            continue
        nz = g["noise"].values
        kld = g["KLD"].values
        dear = 1.0 - g["EAR"].values
        s_kld = float((kld * nz).sum() / max((nz ** 2).sum(), 1e-30))
        s_dear = float((dear * nz).sum() / max((nz ** 2).sum(), 1e-30))
        r2_kld = 1.0 - ((kld - s_kld * nz) ** 2).sum() / max((kld ** 2).sum(), 1e-30)
        r2_dear = 1.0 - ((dear - s_dear * nz) ** 2).sum() / max((dear ** 2).sum(), 1e-30)
        rows.append({
            "tensor": name,
            "slope_KLD": s_kld, "slope_dEAR": s_dear,
            "R2_KLD": float(r2_kld), "R2_dEAR": float(r2_dear),
            "n_points": int(len(g)),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 6. Phase 2 — cost matrices + ILP
# ==========================================================================
def _cost_for(scan_pivot, noise_pivot, slopes_by_tensor, name, state, weight,
              method, objective):
    """Compute the scalar cost for (tensor, state) under method/objective."""
    if state == STATE_NONE:
        return 0.0

    try:
        m_val = float(scan_pivot.loc[name, state])
        n_val = float(noise_pivot.loc[name, state])
    except (KeyError, ValueError):
        return np.inf
    if np.isnan(m_val) or np.isnan(n_val):
        return np.inf

    if objective == "kld":
        measured = m_val
        slope = slopes_by_tensor[name]["slope_KLD"]
    else:  # dear
        ear_row = scan_pivot.loc[name]  # placeholder
        measured = np.nan  # filled below
        slope = slopes_by_tensor[name]["slope_dEAR"]

    if method == "method1":
        return slope * n_val * weight
    elif method == "method2":
        return measured * n_val * weight
    else:  # method3
        return measured * weight


def build_cost_and_storage(scan_df, slopes_df, allowlist, param_counts,
                           param_weights, quantizable, method, objective):
    """Build cost [n, n_states] and storage [n, n_states] (storage in bits/weight)."""
    # Pivot measured metrics
    kld_pivot = scan_df.pivot_table(index="tensor", columns="bits", values="KLD", aggfunc="first")
    ear_pivot = scan_df.pivot_table(index="tensor", columns="bits", values="EAR", aggfunc="first")
    noise_pivot = scan_df.pivot_table(index="tensor", columns="bits", values="noise", aggfunc="first")
    slopes_by_tensor = slopes_df.set_index("tensor").to_dict("index")

    n = len(allowlist)
    n_states = len(ALL_STATES)
    cost = np.full((n, n_states), np.inf, dtype=np.float64)
    storage = np.full((n, n_states), np.inf, dtype=np.float64)

    for i, name in enumerate(allowlist):
        w = param_weights[name]
        for j, state in enumerate(ALL_STATES):
            if state == STATE_NONE:
                cost[i, j] = 0.0
                storage[i, j] = 16.0
                continue
            if not quantizable[name]:
                continue  # keep inf
            # get measured
            try:
                m_kld = float(kld_pivot.loc[name, state])
                m_ear = float(ear_pivot.loc[name, state])
                n_val = float(noise_pivot.loc[name, state])
            except (KeyError, ValueError):
                continue
            if any(np.isnan(x) for x in (m_kld, m_ear, n_val)):
                continue

            if objective == "kld":
                measured = m_kld
                slope = slopes_by_tensor[name]["slope_KLD"]
            else:
                measured = 1.0 - m_ear
                slope = slopes_by_tensor[name]["slope_dEAR"]

            if method == "method1":
                cost[i, j] = slope * n_val * w
            elif method == "method2":
                cost[i, j] = measured * n_val * w
            else:  # method3
                cost[i, j] = measured * w

            storage[i, j] = float(state) + 1.0  # b bits data + 1 bit scale, per weight

    return cost, storage


def solve_ilp(cost, storage, quantizable, param_counts, allowlist, budget_bpw):
    """Multiple-choice knapsack via scipy.optimize.milp."""
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
        if not quantizable[allowlist[i]]:
            for j in range(len(BITS)):  # quantized states only
                ub[i * n_states + j] = 0.0

    integrality = np.ones(n_vars)

    res = milp(
        c=c,
        constraints=[
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ineq, -np.inf, b_ineq),
        ],
        integrality=integrality,
        bounds=Bounds(lb, ub),
    )
    if not res.success:
        return None, None

    x = np.asarray(res.x).reshape(n, n_states)
    assignment = {}
    predicted = 0.0
    for i, name in enumerate(allowlist):
        j = int(np.argmax(x[i]))
        assignment[name] = ALL_STATES[j]
        predicted += cost[i, j]
    return assignment, float(predicted)


def phase2_solve(scan_df, slopes_df, allowlist, param_counts, param_weights,
                 quantizable, budgets):
    configs = {}
    for method in METHODS:
        configs[method] = {}
        for budget in budgets:
            configs[method][budget] = {}
            for objective in OBJECTIVES:
                cost, storage = build_cost_and_storage(
                    scan_df, slopes_df, allowlist, param_counts,
                    param_weights, quantizable, method, objective,
                )
                assign, pred = solve_ilp(
                    cost, storage, quantizable,
                    [param_counts[nm] for nm in allowlist],
                    allowlist, budget,
                )
                if assign is None:
                    print(f"[solve] {method} budget={budget:2d} {objective:4s} INFEASIBLE")
                    configs[method][budget][objective] = None
                else:
                    n_quant = sum(1 for v in assign.values() if v != STATE_NONE)
                    print(f"[solve] {method} budget={budget:2d} {objective:4s} "
                          f"feasible  ({n_quant} quantized, pred={pred:.4e})")
                    configs[method][budget][objective] = {
                        "assignment": assign,
                        "predicted_cost": pred,
                    }
    return configs


# ==========================================================================
# 7. Phase 3 — composite eval
# ==========================================================================
def apply_assignment(model, allowlist, orig_cpu, assignment):
    for name in allowlist:
        state = assignment.get(name, STATE_NONE)
        if state == STATE_NONE:
            restore_one(model, name, orig_cpu[name])
        else:
            apply_one(model, name, orig_cpu[name], int(state))


def actual_bpw(assignment, allowlist, param_counts, total_params):
    total = 0
    for name in allowlist:
        state = assignment.get(name, STATE_NONE)
        pc = param_counts[name]
        if state == STATE_NONE:
            total += pc * 16
        else:
            total += pc * (int(state) + 1)
    return total / total_params


def phase3_eval(model, allowlist, orig_cpu, configs, eval_caches,
                eval_datasets, param_counts, total_params, device, fwd_chunk):
    rows = []
    for method in METHODS:
        for budget in BUDGETS:
            for objective in OBJECTIVES:
                entry = configs.get(method, {}).get(budget, {}).get(objective)
                if not entry:
                    continue
                assign = entry["assignment"]
                predicted = entry["predicted_cost"]
                bpw = actual_bpw(assign, allowlist, param_counts, total_params)

                apply_assignment(model, allowlist, orig_cpu, assign)

                for ds_name in eval_datasets:
                    seqs, c_ids, c_probs = eval_caches[ds_name]
                    m = evaluate_config(model, seqs, c_ids, c_probs, device, fwd_chunk)
                    rows.append({
                        "method": method,
                        "budget_target": budget,
                        "objective": objective,
                        "dataset": ds_name,
                        "actual_bpw": bpw,
                        "predicted_cost": predicted,
                        "mean_kld": m["mean_kld"],
                        "mean_ear": m["mean_ear"],
                        "mean_q_res": m["mean_q_res"],
                        "mean_q20_res": m["mean_q20_res"],
                    })
                    print(f"  [eval] {method} b={budget:2d} {objective:4s} "
                          f"ds={ds_name:<13s} bpw={bpw:.2f} "
                          f"KLD={m['mean_kld']:.4e} EAR={m['mean_ear']:.4f}")

                restore_all(model, allowlist, orig_cpu)

    return pd.DataFrame(rows)


# ==========================================================================
# 8. Upload
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
        api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=out_dir,
            commit_message=f"quant_eval_v2 {datetime.now(timezone.utc).isoformat()}",
        )
        print(f"[upload] done: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[upload] failed: {e}", file=sys.stderr)


# ==========================================================================
# 9. Main
# ==========================================================================
def cmd_run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    phase1_dir = os.path.join(args.out_dir, "phase1")
    phase2_dir = os.path.join(args.out_dir, "phase2")
    phase3_dir = os.path.join(args.out_dir, "phase3")
    for d in (phase1_dir, phase2_dir, phase3_dir):
        os.makedirs(d, exist_ok=True)

    # ---- load ----
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
    orig_cpu = {name: model.get_parameter(name).detach().to("cpu", copy=True)
                for name in allowlist}
    param_counts = {name: model.get_parameter(name).numel() for name in allowlist}
    total_params = sum(param_counts.values())
    param_weights = {name: param_counts[name] / total_params for name in allowlist}
    quantizable = {name: (model.get_parameter(name).shape[1] % GROUP_SIZE == 0)
                   for name in allowlist}

    with open(args.allowlist) as src, \
         open(os.path.join(args.out_dir, "allowlist_used.csv"), "w") as dst:
        dst.write(src.read())

    # ---- caches ----
    scan_datasets = [d.strip() for d in args.scan_datasets.split(",") if d.strip()]
    eval_datasets = [d.strip() for d in args.eval_datasets.split(",") if d.strip()]
    all_datasets = sorted(set(scan_datasets) | set(eval_datasets))
    max_seqs = max(args.scan_n_seqs, args.n_seqs)

    print(f"[cache] building caches for {all_datasets} at {max_seqs} seqs each")
    caches_full = {}
    for ds in all_datasets:
        print(f"[cache] streaming {ds}")
        seqs = list(stream_sequences(ds, tokenizer, args.seq_len, max_seqs))
        print(f"[cache] {ds}: {len(seqs)} seqs, forward pass ...")
        c_ids, c_probs = build_cache(model, seqs, device, args.fwd_chunk)
        caches_full[ds] = (seqs, c_ids, c_probs)

    scan_caches = {}
    for ds in scan_datasets:
        seqs, c_ids, c_probs = caches_full[ds]
        n = args.scan_n_seqs
        scan_caches[ds] = (
            seqs[:n],
            c_ids[:n * args.seq_len],
            c_probs[:n * args.seq_len],
        )

    eval_caches = {}
    for ds in eval_datasets:
        seqs, c_ids, c_probs = caches_full[ds]
        n = args.n_seqs
        eval_caches[ds] = (
            seqs[:n],
            c_ids[:n * args.seq_len],
            c_probs[:n * args.seq_len],
        )

    # ---- Phase 1 ----
    scan_path = os.path.join(phase1_dir, "scan.parquet")
    if os.path.exists(scan_path) and args.resume:
        print(f"[phase1] loading {scan_path}")
        scan_df = pd.read_parquet(scan_path)
    else:
        print("[phase1] scanning")
        scan_df = phase1_scan(
            model, allowlist, orig_cpu, param_counts, quantizable,
            scan_caches, scan_datasets, args.seq_len, device, args.fwd_chunk,
        )
        scan_df.to_parquet(scan_path, index=False)
        print(f"[phase1] wrote {scan_path}")

    # ---- Phase 1.5 ----
    slopes_path = os.path.join(phase1_dir, "slopes.parquet")
    if os.path.exists(slopes_path) and args.resume:
        print(f"[fit] loading {slopes_path}")
        slopes_df = pd.read_parquet(slopes_path)
    else:
        print("[fit] fitting slopes")
        slopes_df = phase15_fit(scan_df)
        slopes_df.to_parquet(slopes_path, index=False)
        print(f"[fit] wrote {slopes_path}")

    # tensor metadata
    meta_df = pd.DataFrame([
        {"tensor": n, "param_count": param_counts[n],
         "param_weight": param_weights[n], "quantizable": quantizable[n]}
        for n in allowlist
    ])
    meta_df.to_parquet(os.path.join(phase1_dir, "tensor_meta.parquet"), index=False)

    # ---- Phase 2 ----
    configs_path = os.path.join(phase2_dir, "all_configs.json")
    if os.path.exists(configs_path) and args.resume:
        print(f"[solve] loading {configs_path}")
        with open(configs_path) as f:
            configs = json.load(f)
        # JSON keys are strings; convert budget keys back to int
        configs = {
            m: {int(b): v for b, v in mv.items()}
            for m, mv in configs.items()
        }
    else:
        print("[solve] building costs + ILP")
        configs = phase2_solve(
            scan_df, slopes_df, allowlist, param_counts, param_weights,
            quantizable, BUDGETS,
        )
        with open(configs_path, "w") as f:
            json.dump(configs, f, indent=2)
        print(f"[solve] wrote {configs_path}")

    # ---- Phase 3 ----
    composite_path = os.path.join(phase3_dir, "composite_results.csv")
    if os.path.exists(composite_path) and args.resume:
        print(f"[eval] loading {composite_path}")
        composite_df = pd.read_csv(composite_path)
    else:
        print("[eval] composite evaluation")
        composite_df = phase3_eval(
            model, allowlist, orig_cpu, configs, eval_caches, eval_datasets,
            param_counts, total_params, device, args.fwd_chunk,
        )
        composite_df.to_csv(composite_path, index=False)
        print(f"[eval] wrote {composite_path}")

    # ---- upload ----
    run_config = {
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "seq_len": args.seq_len,
        "scan_n_seqs": args.scan_n_seqs,
        "n_seqs": args.n_seqs,
        "fwd_chunk": args.fwd_chunk,
        "scan_datasets": scan_datasets,
        "eval_datasets": eval_datasets,
        "bits": BITS,
        "budgets": BUDGETS,
        "methods": METHODS,
        "objectives": OBJECTIVES,
        "n_tensors": len(allowlist),
        "total_params": total_params,
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
    ap.add_argument("--out-dir", default="./results_v2")
    ap.add_argument("--dtype", default="float16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--scan-n-seqs", type=int, default=30,
                    help="sequences per scan dataset for Phase 1")
    ap.add_argument("--n-seqs", type=int, default=100,
                    help="sequences per eval dataset for Phase 3")
    ap.add_argument("--fwd-chunk", type=int, default=16)
    ap.add_argument("--scan-datasets", default="stack_python",
                    help="comma-separated datasets used in Phase 1 scan")
    ap.add_argument("--eval-datasets",
                    default="wikitext,stack_python,fineweb_edu,fineweb_math",
                    help="comma-separated datasets used in Phase 3 evaluation")
    ap.add_argument("--resume", action="store_true",
                    help="Load existing intermediate files, skip completed phases")
    ap.add_argument("--no-upload", action="store_true",
                    help="Skip HF upload")

    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()