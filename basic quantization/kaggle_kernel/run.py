#!/usr/bin/env python3
"""Kaggle entrypoint: clone the repo (per global workflow) and run the user's script unchanged."""
import os
import subprocess
import sys

REPO_URL = "https://github.com/w-ahmad1a10/Mixture-of-Quants.git"
REPO_DIR = "/kaggle/working/Mixture-of-Quants"
SCRIPT = os.path.join(REPO_DIR, "basic quantization", "quant_eval.py.py")
ALLOWLIST = os.path.join(REPO_DIR, "basic quantization", "tensor.csv")

# HF token from attached private dataset
hf_token_path = "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt"
if os.path.isfile(hf_token_path):
    try:
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()
        print("[run] HF_TOKEN loaded from dataset", flush=True)
    except Exception as e:
        print(f"[run] could not read HF token: {e}", file=sys.stderr)

# Download latest code from the online code site (git), per global workflow.
if not os.path.isdir(REPO_DIR):
    print("[run] cloning repo...", flush=True)
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    if r.returncode != 0:
        print("[run] git clone failed", file=sys.stderr)
        sys.exit(r.returncode)

os.chdir(REPO_DIR)
print("[run] cwd:", os.getcwd(), flush=True)

cmd = [
    sys.executable, SCRIPT, "evaluate",
    "--model", "LiquidAI/LFM2.5-230M-Base",
    "--allowlist", ALLOWLIST,
    "--out-dir", os.path.join(REPO_DIR, "results"),
    "--hf-repo", "w-ahmad/Quant-study",
]
print("[run] executing:", " ".join(cmd), flush=True)
rc = subprocess.run(cmd).returncode
print(f"[run] exit code: {rc}", flush=True)
print("QUANT_EVAL_DONE" if rc == 0 else "QUANT_EVAL_FAILED", flush=True)
sys.exit(rc)
