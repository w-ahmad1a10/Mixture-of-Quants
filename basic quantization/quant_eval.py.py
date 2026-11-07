#!/usr/bin/env python3
"""
quant_eval.py — Fake-quantization damage evaluation.

Subcommands
-----------
  inventory   Produce tensors.csv from model.named_parameters().
  evaluate    Apply 24 fake-quant configs, measure KLD/EAR against the
              original model, upload results to HF.
  selftest    Compare our symmetric 8-bit QDQ against torch.quantize_per_tensor.

Allowlist contract
------------------
The same tensors.csv produced by `inventory` is passed to `evaluate`.
The user edits ONLY the `quantizable_default` column (True/False).
`evaluate` reads names where `quantizable_default == True`, and hard-aborts
on missing column, invalid values, duplicate names, non-existent names,
non-2D tensors, or tied aliases. No rows are deleted. No second file.
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
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# Global constants
# --------------------------------------------------------------------------
GROUP_SIZE = 16
TOPK_CACHE = 1000
EAR_K = 20
EPS = 1e-12
CLIP_PCT = 99.9

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
# 1. Manual QDQ (NumPy, float32)
# ==========================================================================
def qdq_np(W, bits, symmetric, granularity, clip_pct):
    """
    Fake-quantize W (np.float32, 2-D) and dequantize.
    Returns np.float32 array of same shape, or None if per-group is impossible.
    """
    out_f, in_f = W.shape

    if granularity == "per_group":
        if in_f % GROUP_SIZE != 0:
            return None
        Wr = W.reshape(out_f, in_f // GROUP_SIZE, GROUP_SIZE)
        axis = 2
    elif granularity == "per_channel":
        Wr = W
        axis = 1
    elif granularity == "per_tensor":
        Wr = W.reshape(-1)
        axis = 0
    else:
        raise ValueError(f"unknown granularity {granularity}")

    # --- clip (identical for symmetric and asymmetric) ---
    if clip_pct is None:
        bound = np.abs(Wr).max(axis=axis, keepdims=True).astype(np.float32)
    else:
        bound = np.percentile(
            np.abs(Wr), clip_pct, axis=axis, keepdims=True
        ).astype(np.float32)

    Wc = np.clip(Wr, -bound, bound)

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
        scale = bound / float(qmax)
        scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        q = np.clip(np.round(Wc / scale), qmin, qmax)
        Wh = q * scale
    else:
        qmax = (1 << bits) - 1
        qmin = 0
        w_min = Wc.min(axis=axis, keepdims=True)
        w_max = Wc.max(axis=axis, keepdims=True)
        rng = w_max - w_min
        scale = rng / float(qmax - qmin)
        scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        zp = np.clip(np.round(qmin - w_min / scale), qmin, qmax).astype(np.float32)
        q = np.clip(np.round(Wc / scale) + zp, qmin, qmax)
        Wh = (q - zp) * scale
        degenerate = (rng == 0)
        if degenerate.any():
            Wh = np.where(degenerate, w_min, Wh)

    return Wh.reshape(W.shape).astype(np.float32)


def apply_config(model, allowlist, orig_cpu, cfg):
    """Write fake-quantized weights into the model in-place. Returns skipped names."""
    skipped = []
    for name in allowlist:
        p = model.get_parameter(name)
        W_np = orig_cpu[name].float().numpy()
        Wh = qdq_np(
            W_np, cfg["bits"], cfg["symmetric"],
            cfg["granularity"], cfg["clip_pct"],
        )
        if Wh is None:
            skipped.append(name)
            continue
        p.data.copy_(torch.from_numpy(Wh).to(p.device, dtype=p.dtype))
    return skipped


def restore_config(model, allowlist, orig_cpu):
    for name in allowlist:
        p = model.get_parameter(name)
        p.data.copy_(orig_cpu[name].to(p.device, dtype=p.dtype))


# ==========================================================================
# 2. Selftest — MSE vs torch.quantize_per_tensor
# ==========================================================================
def run_selftest(out_dir):
    torch.manual_seed(0)
    W = torch.randn(512, 512, dtype=torch.float32)

    # ---- unclipped symmetric 8-bit ----
    scale = float(W.abs().max()) / 127.0
    q = torch.quantize_per_tensor(W, scale, 0, torch.qint8)
    W_torch = q.dequantize()
    mse_torch = float(((W_torch - W) ** 2).mean())

    Wh_ours_np = qdq_np(W.numpy(), 8, True, "per_tensor", None)
    mse_ours = float(((torch.from_numpy(Wh_ours_np) - W) ** 2).mean())
    rel = abs(mse_ours - mse_torch) / max(mse_torch, 1e-12)
    pass_unclipped = rel < 1e-4

    # ---- clipped symmetric 8-bit (torch reference path) ----
    bound_t = torch.quantile(W.abs().flatten(), CLIP_PCT / 100.0)
    Wc_t = torch.clamp(W, -bound_t, bound_t)
    scale_c = float(bound_t) / 127.0
    q_c = torch.quantize_per_tensor(Wc_t, scale_c, 0, torch.qint8)
    W_torch_c = q_c.dequantize()
    mse_torch_c = float(((W_torch_c - W) ** 2).mean())

    Wh_ours_c_np = qdq_np(W.numpy(), 8, True, "per_tensor", CLIP_PCT)
    mse_ours_c = float(((torch.from_numpy(Wh_ours_c_np) - W) ** 2).mean())
    rel_c = abs(mse_ours_c - mse_torch_c) / max(mse_torch_c, 1e-12)
    pass_clipped = rel_c < 1e-4

    result = {
        "unclipped": {
            "mse_ours": mse_ours,
            "mse_torch": mse_torch,
            "relative_diff": rel,
            "pass": pass_unclipped,
        },
        "clipped_p99.9": {
            "mse_ours": mse_ours_c,
            "mse_torch": mse_torch_c,
            "relative_diff": rel_c,
            "pass": pass_clipped,
        },
        "pass": bool(pass_unclipped and pass_clipped),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "selftest.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[selftest] unclipped  MSE ours={mse_ours:.6e}  torch={mse_torch:.6e}  "
          f"rel={rel:.2e}  {'PASS' if pass_unclipped else 'FAIL'}")
    print(f"[selftest] clipped    MSE ours={mse_ours_c:.6e}  torch={mse_torch_c:.6e}  "
          f"rel={rel_c:.2e}  {'PASS' if pass_clipped else 'FAIL'}")
    return result


# ==========================================================================
# 3. Inventory
# ==========================================================================
def cmd_inventory(args):
    print(f"[inventory] loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)  # noqa: F841
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    id_to_names = {}
    for name, p in model.named_parameters():
        id_to_names.setdefault(id(p), []).append(name)

    rows = []
    seen_ids = set()
    for name, p in model.named_parameters():
        assert id(p) not in seen_ids, f"named_parameters() yielded duplicate id for {name}"
        seen_ids.add(id(p))
        aliases = [n for n in id_to_names[id(p)] if n != name]
        is_2d = p.dim() == 2
        is_weight = name.endswith(".weight")
        rows.append({
            "name": name,
            "shape": "x".join(str(s) for s in p.shape),
            "numel": p.numel(),
            "dtype": str(p.dtype),
            "tied": len(aliases) > 0,
            "aliases": ";".join(aliases),
            "quantizable_default": is_2d and is_weight,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "shape", "numel", "dtype", "tied",
                        "aliases", "quantizable_default"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[inventory] {len(rows)} unique parameters -> {args.out}")
    print(f"[inventory] tied: {sum(1 for r in rows if r['tied'])} "
          f"(aliases shown in 'aliases' column)")
    print(f"[inventory] quantizable_default=True count: "
          f"{sum(1 for r in rows if r['quantizable_default'])}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================================================
# 4. Allowlist — read `quantizable_default == True` from the inventory CSV
# ==========================================================================
def load_allowlist(path, model):
    """
    Read `tensors.csv`, select names with quantizable_default == True,
    and validate. Hard-aborts on any inconsistency.
    """
    if not os.path.isfile(path):
        print(f"[allowlist] ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    names = []
    with open(path) as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        if "name" not in cols:
            print("[allowlist] ERROR: missing 'name' column", file=sys.stderr)
            sys.exit(1)
        if "quantizable_default" not in cols:
            print("[allowlist] ERROR: missing 'quantizable_default' column. "
                  "Pass the tensors.csv produced by `inventory`.",
                  file=sys.stderr)
            sys.exit(1)

        for i, row in enumerate(r, start=2):  # header is line 1
            raw = row["quantizable_default"]
            val = str(raw).strip().lower() if raw is not None else ""
            if val == "true":
                names.append(row["name"].strip())
            elif val == "false":
                continue
            else:
                print(f"[allowlist] ERROR: line {i} "
                      f"(name={row.get('name')!r}) has "
                      f"quantizable_default={raw!r}; must be 'True' or 'False'",
                      file=sys.stderr)
                sys.exit(1)

    if not names:
        print("[allowlist] ERROR: no rows with quantizable_default=True",
              file=sys.stderr)
        sys.exit(1)

    param_map = dict(model.named_parameters())
    errs = []

    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        errs.append(f"duplicate names: {dup}")

    missing = [n for n in names if n not in param_map]
    if missing:
        errs.append(f"names not in model.named_parameters(): {missing}")

    non2d = [n for n in names if n in param_map and param_map[n].dim() != 2]
    if non2d:
        errs.append(f"non-2D tensors (refusing to quantize): {non2d}")

    first_name = {}
    for n, p in model.named_parameters():
        first_name.setdefault(id(p), n)
    aliases_in_allowlist = [
        n for n in names
        if n in param_map and first_name[id(param_map[n])] != n
    ]
    if aliases_in_allowlist:
        errs.append(f"tied aliases in allowlist (would be silently skipped): "
                    f"{aliases_in_allowlist}")

    if errs:
        for e in errs:
            print(f"[allowlist] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[allowlist] {len(names)} tensors selected "
          f"(quantizable_default=True)")
    return names


# ==========================================================================
# 5. Streaming data -> fixed-length sequences (no padding)
# ==========================================================================
def stream_sequences(dataset_name, tokenizer, seq_len, n_seqs):
    spec = DATASET_SPECS[dataset_name]
    ds = spec["loader"]()
    field = spec["field"]
    filt = spec["filter"]

    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id or 0

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


# ==========================================================================
# 6. Baseline cache
# ==========================================================================
@torch.no_grad()
def build_cache(model, sequences, device, fwd_chunk):
    all_ids, all_probs = [], []
    n = len(sequences)
    t0 = time.time()
    for i in range(0, n, fwd_chunk):
        chunk = sequences[i:i + fwd_chunk]
        input_ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(input_ids).logits.float()
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, k=TOPK_CACHE, dim=-1)
        all_ids.append(top_i.to(torch.int32).cpu())
        all_probs.append(top_p.cpu())
        del logits, probs, top_p, top_i
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (i // fwd_chunk) % 5 == 0:
            print(f"  [cache] {i + len(chunk)}/{n} seqs "
                  f"({time.time() - t0:.1f}s)")
    ids = torch.cat(all_ids, dim=0).reshape(-1, TOPK_CACHE)
    probs = torch.cat(all_probs, dim=0).reshape(-1, TOPK_CACHE)
    return ids, probs


# ==========================================================================
# 7. Metrics
# ==========================================================================
@torch.no_grad()
def evaluate_config(model, sequences, cache_ids, cache_probs,
                    device, fwd_chunk):
    n = len(sequences)
    L = len(sequences[0])

    s_kld = s_kld2 = 0.0
    s_ear = s_ear2 = 0.0
    s_qres = s_pres = 0.0
    s_q20res = s_p20res = 0.0
    count = 0

    for i in range(0, n, fwd_chunk):
        chunk = sequences[i:i + fwd_chunk]
        input_ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(input_ids).logits.float()
        probs = torch.softmax(logits, dim=-1)

        start = i * L
        end = (i + len(chunk)) * L
        c_ids = cache_ids[start:end].long().to(device)
        c_probs = cache_probs[start:end].to(device)

        flat = probs.reshape(-1, probs.shape[-1])
        q_raw = flat.gather(1, c_ids)

        # ---- KLD on top-1000 ----
        p_sum = c_probs.sum(dim=1, keepdim=True)
        p = c_probs / p_sum.clamp_min(EPS)
        q_sum = q_raw.sum(dim=1, keepdim=True)
        q = q_raw / q_sum.clamp_min(EPS)
        kld = (p * torch.log((p + EPS) / (q + EPS))).sum(dim=1)
        p_res = 1.0 - p_sum.squeeze(1)
        q_res = 1.0 - q_sum.squeeze(1)

        # ---- EAR on top-20 (normalized) ----
        p20_raw = c_probs[:, :EAR_K]
        q20_raw = q_raw[:, :EAR_K]
        p20 = p20_raw / p20_raw.sum(dim=1, keepdim=True).clamp_min(EPS)
        q20 = q20_raw / q20_raw.sum(dim=1, keepdim=True).clamp_min(EPS)
        ear = torch.minimum(p20, q20).sum(dim=1)
        p20_res = 1.0 - p20_raw.sum(dim=1)
        q20_res = 1.0 - q20_raw.sum(dim=1)

        s_kld += kld.sum().item()
        s_kld2 += (kld ** 2).sum().item()
        s_ear += ear.sum().item()
        s_ear2 += (ear ** 2).sum().item()
        s_qres += q_res.sum().item()
        s_pres += p_res.sum().item()
        s_q20res += q20_res.sum().item()
        s_p20res += p20_res.sum().item()
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
        "mean_q_res": s_qres / count, "mean_p_res": s_pres / count,
        "mean_q20_res": s_q20res / count, "mean_p20_res": s_p20res / count,
    }


# ==========================================================================
# 8. Config grid
# ==========================================================================
def make_configs():
    cfgs = []
    cid = 0
    for sym in (True, False):
        for bits in (4, 8):
            for gran in ("per_tensor", "per_channel", "per_group"):
                for clip_pct in (None, CLIP_PCT):
                    cid += 1
                    cfgs.append({
                        "config_id": cid,
                        "symmetric": sym,
                        "bits": bits,
                        "granularity": gran,
                        "clipping": "none" if clip_pct is None else "p99.9",
                        "clip_pct": clip_pct,
                    })
    return cfgs


# ==========================================================================
# 9. Upload
# ==========================================================================
def upload_results(out_dir, repo_id):
    if not repo_id:
        print("[upload] no --hf-repo, skipping")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[upload] HF_TOKEN not set, skipping")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=out_dir,
            commit_message=f"quant_eval {datetime.now(timezone.utc).isoformat()}",
        )
        print(f"[upload] done: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[upload] failed: {e}", file=sys.stderr)


# ==========================================================================
# 10. Evaluate
# ==========================================================================
def cmd_evaluate(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # selftest gate
    st = run_selftest(args.out_dir)
    if not st["pass"]:
        print("[evaluate] selftest failed — refusing to start", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    print(f"[evaluate] loading {args.model} ({args.dtype}) on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device).eval()

    # ---- global tensor count (id-dedup guard) ----
    total_tensors = 0
    seen_ids = set()
    id_names = {}
    for name, p in model.named_parameters():
        assert id(p) not in seen_ids, f"duplicate parameter id yielded: {name}"
        seen_ids.add(id(p))
        total_tensors += 1
        id_names.setdefault(id(p), []).append(name)

    tied_aliases = []
    for _, names in id_names.items():
        if len(names) > 1:
            tied_aliases.extend(names[1:])

    # ---- allowlist from tensors.csv (quantizable_default == True) ----
    allowlist = load_allowlist(args.allowlist, model)
    allowed_tensors = len(allowlist)
    print(f"[evaluate] total_tensors={total_tensors}  "
          f"allowed={allowed_tensors}  tied_aliases={len(tied_aliases)}")

    # ---- snapshot originals on CPU bf16 for restore ----
    print("[evaluate] snapshotting allowlisted originals to CPU ...")
    orig_cpu = {}
    for name in allowlist:
        p = model.get_parameter(name)
        orig_cpu[name] = p.detach().to("cpu", copy=True)

    # ---- copy the exact allowlist file used (reproducibility) ----
    with open(args.allowlist, "r") as src, \
         open(os.path.join(args.out_dir, "allowlist_used.csv"), "w") as dst:
        dst.write(src.read())

    configs = make_configs()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows = []
    skipped_by_config = {}

    for ds_name in datasets:
        print(f"\n[evaluate] === dataset: {ds_name} ===")
        try:
            seqs = list(stream_sequences(
                ds_name, tokenizer, args.seq_len, args.n_seqs
            ))
        except Exception as e:
            print(f"[evaluate] failed to stream {ds_name}: {e}", file=sys.stderr)
            continue
        if len(seqs) < args.n_seqs:
            print(f"[evaluate] WARNING: only {len(seqs)} seqs available "
                  f"(requested {args.n_seqs})")
        if not seqs:
            print(f"[evaluate] no sequences for {ds_name}, skipping")
            continue
        print(f"[evaluate] streaming produced {len(seqs)} sequences "
              f"of length {args.seq_len}")

        print("[evaluate] building baseline cache ...")
        cache_ids, cache_probs = build_cache(
            model, seqs, device, args.fwd_chunk
        )
        cache_mb = (cache_ids.numel() * 4 + cache_probs.numel() * 4) / 1e6
        print(f"[evaluate] cache: {cache_ids.shape[0]} positions, "
              f"{cache_mb:.1f} MB")

        for cfg in configs:
            t0 = time.time()
            skipped = apply_config(model, allowlist, orig_cpu, cfg)
            if skipped:
                skipped_by_config[cfg["config_id"]] = skipped
            res = evaluate_config(
                model, seqs, cache_ids, cache_probs, device, args.fwd_chunk
            )
            restore_config(model, allowlist, orig_cpu)

            row = {
                "config_id": cfg["config_id"],
                "symmetry": "symmetric" if cfg["symmetric"] else "asymmetric",
                "bits": cfg["bits"],
                "granularity": cfg["granularity"],
                "clipping": cfg["clipping"],
                "dataset": ds_name,
                "n_seqs": len(seqs),
                **res,
            }
            rows.append(row)
            print(f"  cfg {cfg['config_id']:02d} {row['symmetry']:<10} "
                  f"b={cfg['bits']} {cfg['granularity']:<11} "
                  f"clip={cfg['clipping']:<5} "
                  f"kld={res['mean_kld']:.4e} ear={res['mean_ear']:.4f} "
                  f"({time.time() - t0:.1f}s)")

        del cache_ids, cache_probs, seqs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- write results ----
    if rows:
        fields = list(rows[0].keys())
        with open(os.path.join(args.out_dir, "results_per_config.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    global_report = {
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "seq_len": args.seq_len,
        "n_seqs": args.n_seqs,
        "fwd_chunk": args.fwd_chunk,
        "datasets": datasets,
        "total_tensors": total_tensors,
        "allowed_tensors": allowed_tensors,
        "duplicate_ids_found": 0,
        "tied_aliases_skipped": tied_aliases,
        "skipped_per_group_indivisible": skipped_by_config,
        "num_configs": len(configs),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "results_global.json"), "w") as f:
        json.dump(global_report, f, indent=2)

    with open(os.path.join(args.out_dir, "cache_manifest.json"), "w") as f:
        json.dump({
            "datasets": datasets,
            "seq_len": args.seq_len,
            "n_seqs": args.n_seqs,
            "topk": TOPK_CACHE,
            "ear_k": EAR_K,
            "note": "cache not uploaded (regenerable); manifest for reproducibility",
        }, f, indent=2)

    print(f"\n[evaluate] wrote {len(rows)} result rows to {args.out_dir}")
    upload_results(args.out_dir, args.hf_repo)


# ==========================================================================
# 11. CLI
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Fake-quantization damage evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_inv = sub.add_parser("inventory")
    p_inv.add_argument("--model", required=True)
    p_inv.add_argument("--out", default="tensors.csv")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--allowlist", required=True,
                        help="tensors.csv produced by `inventory`. "
                             "Rows with quantizable_default=True are quantized.")
    p_eval.add_argument("--out-dir", default="./results")
    p_eval.add_argument("--hf-repo", default=None,
                        help="dataset repo id, e.g. user/quant-eval-results")
    p_eval.add_argument("--datasets", default=",".join(DATASET_SPECS.keys()))
    p_eval.add_argument("--seq-len", type=int, default=2048)
    p_eval.add_argument("--n-seqs", type=int, default=200)
    p_eval.add_argument("--fwd-chunk", type=int, default=4)
    p_eval.add_argument("--dtype", default="float16",
                        choices=["bfloat16", "float16", "float32"])
    p_eval.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")

    p_st = sub.add_parser("selftest")
    p_st.add_argument("--out-dir", default="./results")

    args = ap.parse_args()

    if args.cmd == "inventory":
        cmd_inventory(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "selftest":
        res = run_selftest(args.out_dir)
        sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()