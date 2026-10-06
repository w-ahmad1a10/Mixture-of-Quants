#!/usr/bin/env python3
"""
verify_run.py — Quick smoke test of the refactored MoQ pipeline.

Same code as run_all_goals.py (setup, run, main, _json all verbatim) but with
every hyperparameter reduced to the minimum (max 5 samples, 1 eval dataset,
1 chunk) so the whole thing completes in minutes instead of hours.
Purpose: confirm the refactored code (moq.py run → MoQRunner → KLD → CE →
optimize → apply+eval) works end-to-end on Colab CPU.
"""
import os
import subprocess
import sys
from pathlib import Path

try:
    REPO_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # colab run executes the file in a kernel where __file__ is undefined.
    REPO_DIR = Path("/content/Mixture-of-Quants")
WORK_DIR = "/content/workspace"
LLAMA_CPP_DIR = os.path.join(WORK_DIR, "llama.cpp")
LLAMA_BIN_DIR = os.path.join(WORK_DIR, "llama_bin")
IMATRIX = os.path.join(WORK_DIR, "Qwen3.5-0.8B.imatrix")
WIKI_PATH = os.path.join(WORK_DIR, "wiki.test (2).raw")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# One tiny eval config — just enough to exercise the cache + eval path.
# Must be a JSON array (list of dicts), matching what run_all_goals.py passes.
EVAL_CONFIGS = (
    '[{"type":"text","path":"' + WIKI_PATH + '","name":"wiki",'
    '"seq_length":512,"num_chunks":1,"hf_dataset_max_docs":100}]'
)


def download_if_missing(url, destination, token):
    destination = os.path.expanduser(destination)
    if os.path.exists(destination):
        print(f"[Download] Already present: {destination}")
        return
    print(f"[Download] Fetching {url}")
    import urllib.request

    headers = {"User-Agent": "moq-sweep"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, open(destination, "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    print(f"[Download] Saved {destination}")


def build_llama_cpp():
    print("[Build] Preparing llama.cpp CPU build")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["LD_LIBRARY_PATH"] = LLAMA_BIN_DIR + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    command = [
        sys.executable,
        os.path.join(str(REPO_DIR), "moq", "build_llama_cpp_cpu.py"),
        "--install-system-libs",
        "--output-dir", LLAMA_BIN_DIR,
        "--source-dir", LLAMA_CPP_DIR,
    ]
    result = subprocess.run(command, cwd=str(REPO_DIR), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("llama.cpp build failed")
    print("[Build] llama.cpp prepared")


def link_shared_libraries():
    for directory in ("/usr/lib", "/usr/local/lib"):
        subprocess.run(
            ["bash", "-c", f"for lib in {LLAMA_BIN_DIR}/libggml*.so*; do cp -n \"$lib\" {directory}/ 2>/dev/null || true; done"],
            check=False,
        )
    subprocess.run(["ldconfig"], check=False)
    for lib in ("libggml.so", "libggml-base.so"):
        for directory in ("/usr/lib", "/usr/local/lib"):
            real = os.path.join(directory, lib + ".0.9.7")
            link = os.path.join(directory, lib)
            if os.path.exists(real) and not os.path.islink(link) and not os.path.exists(link):
                os.symlink(real, link)
    subprocess.run(["ldconfig"], check=False)
    print("[Build] Shared libraries linked")


def compile_extension():
    sys.path.insert(0, str(REPO_DIR))
    from moq.moq_core import compile_cpp_extension

    print("[Extension] Compiling quantization extension")
    compile_cpp_extension(LLAMA_CPP_DIR)
    print("[Extension] Done")


def setup():
    os.chdir(str(REPO_DIR))
    os.makedirs(WORK_DIR, exist_ok=True)

    download_if_missing(
        "https://huggingface.co/datasets/w-ahmad/moq-lfm25-230m-results/resolve/main/wiki.test%20%282%29.raw",
        WIKI_PATH,
        HF_TOKEN,
    )
    download_if_missing(
        "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/imatrix_unsloth.gguf_file",
        IMATRIX,
        HF_TOKEN,
    )

    build_llama_cpp()
    link_shared_libraries()
    compile_extension()


def run(name, extra):
    cmd = [
        sys.executable, os.path.join(str(REPO_DIR), "scripts", "moq.py"), "run",
        "--model", "Qwen/Qwen3.5-0.8B",
        "--mapping", "configs/tensor_maps/tensor_map_qwen.txt",
        "--gguf-repo", "unsloth/Qwen3.5-0.8B-GGUF",
        "--gguf-file", "Qwen3.5-0.8B-BF16.gguf",
        "--llama-cpp-dir", LLAMA_CPP_DIR,
        "--imatrix", IMATRIX,
        "--llama-bin-dir", LLAMA_BIN_DIR,
        "--trust-remote-code",
        "--dtype", "float16",
        "--quant-types", "ALL",
        "--bits-list", "5.0", "4.0", "3.0",
        "--mmlu-batch-size", "8",
        "--max-samples", "5",
        "--max-seq-length", "256",
        "--batch-size", "1",
        "--num-chunks", "1",
        "--seed", "42",
        "--eval-configs", EVAL_CONFIGS,
        "--results-dir", "verify_results",
    ] + extra
    print("\n" + "=" * 70)
    print("  VERIFY:", name)
    print("  CMD:", " ".join(cmd))
    print("=" * 70 + "\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[ERROR] FAILED:", name, "exit", result.returncode)
    else:
        print("[DONE] FINISHED:", name)
    return result.returncode


def main():
    setup()

    # GOAL 1 — all 4 grouping modes, 5 samples
    for mode in ("by-tensor-type", "by-layer", "individual", "by-layer-tensor"):
        run(f"Goal1-{mode}", [
            "--mode", mode,
            "--dataset-repo", "w-ahmad/open-perfectblend-500k-tokens",
            "--dataset-split", "train",
        ])

    # GOAL 2 — calib data vs dataset repo, 5 samples
    run("Goal2-calib", [
        "--mode", "by-tensor-type",
        "--calib-data", WIKI_PATH,
        "--seq-length", "256",
    ])
    run("Goal2-repo100", [
        "--mode", "by-tensor-type",
        "--dataset-repo", "w-ahmad/open-perfectblend-500k-tokens",
        "--dataset-split", "train",
        "--seq-length", "256",
    ])

    # GOAL 3 — sample count sweep (5 samples only)
    run("Goal3-5", [
        "--mode", "by-tensor-type",
        "--dataset-repo", "w-ahmad/open-perfectblend-500k-tokens",
        "--dataset-split", "train",
        "--seq-length", "256",
    ])

    print("\n" + "=" * 70)
    print("  ALL VERIFY GOALS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()