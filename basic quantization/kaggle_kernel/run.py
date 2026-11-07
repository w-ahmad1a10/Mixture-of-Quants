#!/usr/bin/env python3
"""Kaggle entrypoint. Invokes the user's quant_eval.py.py CLI unchanged."""
import os
import subprocess
import sys

# Load HF token from the attached private dataset (if mounted).
hf_token_path = "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt"
if os.path.isfile(hf_token_path):
    try:
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()
        print("[run] HF_TOKEN loaded from dataset")
    except Exception as e:
        print(f"[run] could not read HF token: {e}", file=sys.stderr)

cmd = [
    sys.executable, "quant_eval.py.py", "evaluate",
    "--model", "LiquidAI/LFM2.5-230M-Base",
    "--allowlist", "tensors.csv",
    "--out-dir", "./results",
    "--hf-repo", "w-ahmad/Quant-study",
]
print("[run] executing:", " ".join(cmd), flush=True)
rc = subprocess.run(cmd).returncode
print(f"[run] exit code: {rc}", flush=True)
print("QUANT_EVAL_DONE" if rc == 0 else "QUANT_EVAL_FAILED", flush=True)
sys.exit(rc)
