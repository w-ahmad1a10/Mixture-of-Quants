#!/usr/bin/env python3
"""
run_all_goals.py — Experimental goal runner (thin wrapper).

Keeps all hyperparameters in scripts/sweep_config.yaml and calls the
canonical CLI (scripts/moq.py run) for every goal. No logic here.
"""
import os
import subprocess
import sys
from pathlib import Path

import yaml

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
HF_TOKEN = "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"

CONFIG_PATH = REPO_DIR / "scripts" / "sweep_config.yaml"


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


def run(name, extra, common):
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
        "--dtype", common["dtype"],
        "--quant-types", common["quant_types"],
        "--bits-list", *common["bits_list"],
        "--mmlu-batch-size", str(common["mmlu_batch_size"]),
        "--hf-upload-repo", "w-ahmad/moq-qwenoo",
        "--hf-token", HF_TOKEN,
    ] + extra
    print("\n" + "=" * 70)
    print("  RUNNING:", name)
    print("  CMD:", " ".join(cmd))
    print("=" * 70 + "\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[ERROR] FAILED:", name, "exit", result.returncode)
    else:
        print("[DONE] FINISHED:", name)
    return result.returncode


def run_quant(name, extra):
    cmd = [
        sys.executable, os.path.join(str(REPO_DIR), "scripts", "moq.py"), "quant",
        "--model", "Qwen/Qwen3.5-0.8B",
        "--mapping", "configs/tensor_maps/tensor_map_qwen.txt",
        "--llama-cpp-dir", LLAMA_CPP_DIR,
        "--imatrix", IMATRIX,
        "--llama-bin-dir", LLAMA_BIN_DIR,
        "--trust-remote-code",
        "--dtype", "float16",
        "--quant-types", "ALL",
        "--evaluate-now",
        "--hf-token", HF_TOKEN,
        "--results-dir", "results_goal4_" + name,
    ] + extra
    print("\n" + "=" * 70)
    print("  RUNNING:", name)
    print("  CMD:", " ".join(cmd))
    print("=" * 70 + "\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[ERROR] FAILED:", name, "exit", result.returncode)
    else:
        print("[DONE] FINISHED:", name)
    return result.returncode


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    common = cfg["common"]

    setup()

    # GOAL 1 — all 4 grouping modes
    for g in cfg["goal1"]:
        run(f"Goal1-{g['name']}", [
            "--dataset-repo", g["dataset_repo"],
            "--dataset-split", g["dataset_split"],
            "--max-seq-length", str(g["max_seq_length"]),
            "--max-samples", str(g["max_samples"]),
            "--batch-size", str(g["batch_size"]),
            "--num-chunks", str(g["num_chunks"]),
            "--seed", str(g["seed"]),
            "--eval-configs", _json(cfg, g["eval_configs"]),
            "--results-dir", g["results_dir"],
        ], common)

    # GOAL 2 — calib data vs dataset repo
    for g in cfg["goal2"]:
        args = [
            "--mode", g["mode"],
            "--max-seq-length", str(g["max_seq_length"]),
            "--seq-length", str(g.get("seq_length", 512)),
            "--max-samples", str(g["max_samples"]),
            "--batch-size", str(g["batch_size"]),
            "--num-chunks", str(g["num_chunks"]),
            "--seed", str(g["seed"]),
            "--eval-configs", _json(cfg, g["eval_configs"]),
            "--results-dir", g["results_dir"],
        ]
        if "calib_data" in g:
            args += ["--calib-data", g["calib_data"]]
        elif "hf_dataset_repo" in g:
            args += ["--hf-dataset-repo", g["hf_dataset_repo"]]
            if g.get("hf_dataset_config"):
                args += ["--hf-dataset-config", g["hf_dataset_config"]]
            args += ["--hf-dataset-split", g.get("hf_dataset_split", "train")]
        else:
            args += ["--dataset-repo", g["dataset_repo"], "--dataset-split", g["dataset_split"]]
        run(f"Goal2-{g['name']}", args, common)

    # GOAL 3 — sample count sweep
    for g in cfg["goal3"]:
        run(f"Goal3-{g['name']}", [
            "--dataset-repo", g["dataset_repo"],
            "--dataset-split", g["dataset_split"],
            "--max-seq-length", str(g["max_seq_length"]),
            "--max-samples", str(g["max_samples"]),
            "--batch-size", str(g["batch_size"]),
            "--num-chunks", str(g["num_chunks"]),
            "--seed", str(g["seed"]),
            "--eval-configs", _json(cfg, g["eval_configs"]),
            "--results-dir", g["results_dir"],
        ], common)

    # GOAL 4 — uniform quantization baseline (apply + evaluate only)
    for g in cfg["goal4"]:
        run_quant(g["name"], [
            "--quant-config", g["quant_config"],
            "--eval-configs", _json(cfg, g["eval_configs"]),
        ])

    print("\n" + "=" * 70)
    print("  ALL GOALS FINISHED")
    print("=" * 70)


def _json(cfg, key):
    import json
    return json.dumps(cfg[key])


if __name__ == "__main__":
    main()