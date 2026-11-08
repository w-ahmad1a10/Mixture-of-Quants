#!/usr/bin/env python3
"""
####
noise_scan.py — Report QDQ noise per tensor per bit-width.

Same QDQ scheme as quant_eval_v2.py:
  - per-group-16, symmetric, no clipping
  - 16 data values × b bits + 16 scale bits per group

Output: CSV with columns  tensor, param_count, quantizable, bits, noise
where  noise = ||W - W_hat||_F^2 / ||W||_F^2

No model forward pass, no metrics. Pure weight-space.
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

GROUP_SIZE = 16
BITS_DEFAULT = [2, 3, 4, 5, 6, 8, 12]


def qdq_per_group_sym(W, bits):
    """Return W_hat (np.float32) and noise, or (None, None) if not quantizable."""
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


def load_allowlist(path, model):
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
    missing = [n for n in names if n not in param_map]
    if missing:
        print(f"[allowlist] ERROR: missing: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[allowlist] {len(names)} tensors")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--out", default="noise_scan.csv")
    ap.add_argument("--bits", default="2,3,4,5,6,8,12",
                    help="comma-separated bit-widths")
    ap.add_argument("--dtype", default="float16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    bits_list = [int(b.strip()) for b in args.bits.split(",") if b.strip()]
    print(f"[noise] bits = {bits_list}")

    print(f"[load] {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True, low_cpu_mem_usage=True,
    )

    allowlist = load_allowlist(args.allowlist, model)

    rows = []
    for idx, name in enumerate(allowlist):
        p = model.get_parameter(name)
        shape = tuple(p.shape)
        pc = p.numel()
        quantizable = (p.dim() == 2 and p.shape[1] % GROUP_SIZE == 0)

        if not quantizable:
            for b in bits_list:
                rows.append({
                    "tensor": name, "shape": "x".join(map(str, shape)),
                    "param_count": pc, "quantizable": False,
                    "bits": b, "noise": np.nan,
                })
            print(f"  [{idx + 1}/{len(allowlist)}] {name}  SKIP (not per-group-16)")
            continue

        W_np = p.detach().float().cpu().numpy()
        for b in bits_list:
            _, noise = qdq_per_group_sym(W_np, b)
            rows.append({
                "tensor": name, "shape": "x".join(map(str, shape)),
                "param_count": pc, "quantizable": True,
                "bits": b, "noise": noise,
            })
        print(f"  [{idx + 1}/{len(allowlist)}] {name}  {shape}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n[noise] wrote {args.out}  ({len(df)} rows)")
    summary = df.dropna().groupby("bits")["noise"].agg(["mean", "median", "max"])
    print("\nsummary (mean / median / max noise per bit):")
    print(summary.to_string())


if __name__ == "__main__":
    main()