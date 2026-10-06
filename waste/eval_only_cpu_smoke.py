import argparse
import gc
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = Path("/content/Mixture-of-Quants")
WORK_DIR = Path("/content/workspace")
RESULTS_DIR = WORK_DIR / "cpu_smoke_results"
CACHE_DIR = WORK_DIR / "cpu_smoke_cache"
LLAMA_CPP_DIR = WORK_DIR / "llama.cpp"
LLAMA_BIN_DIR = WORK_DIR / "llama_bin"
IMATRIX = WORK_DIR / "LFM2.5-230M.imatrix"
WIKI_PATH = WORK_DIR / "wiki.test (2).raw"
MODEL = "LiquidAI/LFM2.5-230M"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = torch.device("cpu")
LOG_PATH = WORK_DIR / "eval_progress.log"


def log(message):
    print(message, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except Exception:
        pass


def download_if_missing(url, destination, token):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        log(f"[Download] Already present: {destination}")
        return
    log(f"[Download] Fetching {url}")
    import urllib.request

    headers = {"User-Agent": "moq-cpu-smoke"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    log(f"[Download] Saved {destination}")


def build_llama_cpp():
    log("[Build] Preparing llama.cpp CPU build")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["LD_LIBRARY_PATH"] = str(LLAMA_BIN_DIR) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    command = [
        "python3",
        str(REPO_DIR / "build_llama_cpp_cpu.py"),
        "--install-system-libs",
        "--output-dir",
        str(LLAMA_BIN_DIR),
        "--source-dir",
        str(LLAMA_CPP_DIR),
    ]
    result = subprocess.run(command, cwd=REPO_DIR, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        log(result.stderr[-2000:])
        raise RuntimeError("llama.cpp build failed")
    log("[Build] llama.cpp prepared")


def link_shared_libraries():
    for directory in (Path("/usr/lib"), Path("/usr/local/lib")):
        subprocess.run(
            ["bash", "-c", f"for lib in {LLAMA_BIN_DIR}/libggml*.so*; do cp -n \"$lib\" {directory}/ 2>/dev/null || true; done"],
            check=False,
        )
    subprocess.run(["ldconfig"], check=False)


def compile_extension():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from moq.moq_core import GGML_TYPE_MAP, compile_cpp_extension

    log("[Extension] Compiling quantization extension")
    return compile_cpp_extension(str(LLAMA_CPP_DIR)), GGML_TYPE_MAP


def parse_config(path):
    config = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_DIR / "configs_uniform_eval/config_2.0.txt"))
    parser.add_argument("--mmlu-samples", type=int, default=2)
    parser.add_argument("--cache-tokens", type=int, default=100)
    args = parser.parse_args()

    os.chdir(REPO_DIR)
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("[Setup] CPU smoke mode")
    log(f"[Setup] Device: {DEVICE}")
    log(f"[Setup] Cache tokens: {args.cache_tokens}")
    log(f"[Setup] MMLU samples: {args.mmlu_samples}")
    log(f"[Setup] Config: {args.config}")

    download_if_missing(
        "https://huggingface.co/datasets/w-ahmad/moq-lfm25-230m-results/resolve/main/wiki.test%20%282%29.raw",
        WIKI_PATH,
        HF_TOKEN,
    )
    download_if_missing(
        "https://huggingface.co/unsloth/LFM2.5-230M-GGUF/resolve/main/imatrix_unsloth.gguf_file",
        IMATRIX,
        HF_TOKEN,
    )

    build_llama_cpp()
    link_shared_libraries()
    quant_ext, ggml_type_map = compile_extension()

    from transformers import AutoTokenizer
    from moq.moq_core import load_model
    from moq.moq_engine import Evaluator
    from moq.eval_mmlu import MMLUEvaluator

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    eval_args = argparse.Namespace(
        model=MODEL,
        calib_data=str(WIKI_PATH),
        dataset_repo=None,
        dataset_split="train",
        max_samples=1,
        max_seq_length=args.cache_tokens,
        seq_length=args.cache_tokens,
        num_chunks=1,
        batch_size=1,
        llama_cpp_dir=str(LLAMA_CPP_DIR),
        hf_token=HF_TOKEN or None,
        teacher_cache_dir=str(CACHE_DIR / "text"),
        skip_teacher_cache=False,
        seed=42,
        dtype="float32",
    )
    evaluator = Evaluator(eval_args)
    evaluator.dataset_name = "wiki_100_token_smoke"
    evaluator.input_device = DEVICE

    mmlu_evaluator = MMLUEvaluator(
        model_name=MODEL,
        tokenizer=tokenizer,
        device=DEVICE,
        max_samples=args.mmlu_samples,
        batch_size=1,
        cache_dir=str(CACHE_DIR / "mmlu"),
        skip_cache=False,
    )
    mmlu_evaluator.dataset_name = f"mmlu_{args.mmlu_samples}_samples"

    log("[Model] Loading teacher on CPU")
    teacher_model, input_device = load_model(
        MODEL,
        torch.float32,
        True,
        HF_TOKEN or None,
        DEVICE,
    )
    teacher_model.eval()
    teacher_model.config.use_cache = False
    log(f"[Model] Loaded on {input_device}")

    log("[Cache] Teacher logits for tiny text and MMLU budgets")
    evaluator.cache_teacher_logits(teacher_model)
    mmlu_evaluator.cache_teacher_logits(teacher_model)

    config_path = Path(args.config)
    quant_config = parse_config(config_path)
    mapping = {}
    with (REPO_DIR / "tensor_map_lfm25.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            mapping[key.strip()] = value.strip()

    model, _ = load_model(
        MODEL,
        torch.float32,
        True,
        HF_TOKEN or None,
        DEVICE,
    )
    model.eval()
    model.config.use_cache = False

    originals = []
    quantized = 0
    skipped = 0
    log("[Quantize] Applying one uniform config")
    for full_name, module in model.named_modules():
        if not hasattr(module, "weight") or not isinstance(module.weight, nn.Parameter):
            continue
        gguf_name = mapping.get(full_name)
        quant_type = quant_config.get(gguf_name) if gguf_name else None
        if not quant_type or quant_type not in ggml_type_map:
            skipped += 1
            continue
        try:
            weight_cpu = module.weight.data.detach().cpu()
            quantized_weight = quant_ext.apply_llama_quant_noise(
                weight_cpu,
                gguf_name,
                ggml_type_map[quant_type],
                str(IMATRIX),
            )
            originals.append((module, module.weight.data))
            module.weight.data = quantized_weight.to(DEVICE).to(module.weight.dtype)
            quantized += 1
        except Exception as error:
            skipped += 1
            log(f"[Warn] Skipped {full_name}: {error}")
    log(f"[Quantize] Quantized {quantized}, skipped {skipped}")

    metrics = {}
    log("[Evaluate] Text cache")
    metrics["wiki_100_token_smoke"] = evaluator.evaluate_student(model)
    log("[Evaluate] MMLU smoke samples")
    metrics[f"mmlu_{args.mmlu_samples}_samples"] = mmlu_evaluator.evaluate_student(model)

    for module, original in originals:
        module.weight.data = original
    del originals
    del model
    gc.collect()

    result = {
        "mode": "cpu_smoke",
        "model": MODEL,
        "device": str(DEVICE),
        "cache_tokens": args.cache_tokens,
        "mmlu_samples": args.mmlu_samples,
        "config_file": config_path.name,
        "quantized": quantized,
        "skipped": skipped,
        "metrics": metrics,
    }
    output_path = RESULTS_DIR / "cpu_smoke_result.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    log(f"[Done] Saved {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
