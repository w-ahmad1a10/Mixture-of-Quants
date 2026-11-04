"""Thin Kaggle runner for quantization (CPU)."""
import os, subprocess, sys

REPO_URL = "https://github.com/w-ahmad1a10/Mixture-of-Quants.git"
REPO_DIR = "/kaggle/working/Mixture-of-Quants"


def run(cmd):
    print("RUNNING:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)
    run([sys.executable, "-m", "pip", "install", "-q",
         "torch", "transformers", "datasets", "huggingface_hub", "tqdm", "accelerate"])
    run([sys.executable, "basic quantization/quantize.py",
         "--model", "Qwen/Qwen3-0.6B",
         "--dtype", "float16",
         "--hf-repo", "w-ahmad/basic-quantization-models",
         "--output-dir", "/kaggle/working/models"])
    print("QUANT_DONE", flush=True)


if __name__ == "__main__":
    main()
