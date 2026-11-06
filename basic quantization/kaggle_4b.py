"""Thin Kaggle runner for Qwen3.5 4B merged experiment."""
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
    wiki = "/kaggle/working/wiki.test (2).raw"
    if not os.path.exists(wiki):
        try:
            run([sys.executable, "-c",
                 "from huggingface_hub import hf_hub_download; "
                 "hf_hub_download(repo_id='w-ahmad/moq-lfm25-230m-results', filename='wiki.test (2).raw', "
                 "local_dir='/kaggle/working', repo_type='dataset')"])
        except Exception as e:
            print("Wiki download failed:", e)
    run([sys.executable, "basic quantization/run_4b.py",
         "--model", "LiquidAI/LFM3-1.7B",
         "--dtype", "float16",
         "--text-chunks", "500",
         "--seq-length", "512",
         "--max-docs", "50",
         "--output", "/kaggle/working/results_4b.json"])
    print("EXPERIMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
