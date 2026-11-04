"""Thin Kaggle runner: clone repo, install deps, run basic quantization, upload results."""
import os
import subprocess
import sys

REPO_URL = "https://github.com/w-ahmad1a10/Mixture-of-Quants.git"
REPO_DIR = "/kaggle/working/Mixture-of-Quants"
TOKEN_PATHS = [
    "/kaggle/input/hf-token-store/hf_token.txt",
    "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt",
]


def run(cmd):
    print("RUNNING:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)
    run([sys.executable, "-m", "pip", "install", "-q",
         "torch", "transformers", "datasets", "huggingface_hub", "tqdm", "accelerate"])
    run([sys.executable, "basic quantization/run_basic.py",
         "--model", "Qwen/Qwen3-0.6B",
         "--dtype", "float16",
         "--mmlu-samples", "100",
         "--mmlu-batch-size", "4",
         "--text-chunks", "500",
         "--seq-length", "512",
         "--max-docs", "50",
         "--output", "/kaggle/working/basic_results.json"])
    # upload results
    hf_token = None
    for tp in TOKEN_PATHS:
        if os.path.exists(tp):
            with open(tp) as f:
                hf_token = f.read().strip()
            break
    if hf_token:
        from huggingface_hub import HfApi, create_repo
        repo_id = "deepresearch1a10/basic-quantization-results"
        create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj="/kaggle/working/basic_results.json",
            path_in_repo="basic_results.json",
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
        )
        print("Uploaded results to", repo_id)
    else:
        print("HF token file not found; skipping upload.")
    print("BASIC_QUANTIZATION_DONE")


if __name__ == "__main__":
    main()
