#!/usr/bin/env python3
"""
moq_sweep.py
Sweep all 4 grouping variants, produce PyTorch mixed-precision models,
evaluate them, save ALL results, upload ONLY results to HF Hub.
NO GGUF generation. Models are deleted after eval; results are kept forever.

Now supports multiple evaluation datasets, WandB logging, and global teacher cache.
"""
import argparse, gc, json, math, os, random, re, sys, glob, shutil, time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from moq_core import (
    GGML_TYPE_MAP, TEACHER_TOPK, compile_cpp_extension,
    compute_token_kld, extract_assistant_log_probs,
    ChatDataset, chat_collate_fn, CalibrationDataset, text_collate_fn,
    parse_mapping_file, parse_quant_config, load_model
)
from moq_engine import (
    NoiseImpactAnalyzer, group_by_layer_tensor_type,
    run_compute_ce, MoQFinalQuantizer, Evaluator,
    GGUFMetadataExtractor, map_unit_to_gguf_type,
    run_extract_weights, DEFAULT_QUANT_BITS
)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _upload_results_to_hf(local_dir: str, repo_id: str, token: str, path_in_repo: str = ""):
    """Upload all files in local_dir to HF Hub under path_in_repo."""
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
            print(f"[HF Upload] Created repo: {repo_id}")
        except Exception as e:
            print(f"[HF Upload] Repo exists or skipped: {e}")
        for root, _, files in os.walk(local_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel = os.path.relpath(local_path, local_dir)
                repo_path = f"{path_in_repo}/{rel}" if path_in_repo else rel
                print(f"[HF Upload] {local_path} -> {repo_id}:{repo_path}")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                )
        print(f"[HF Upload] Done: {local_dir}")
    except Exception as e:
        print(f"[HF Upload] Failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MoQSweep:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_root = _ensure_dir(args.results_dir)
        self.quant_ext = compile_cpp_extension(args.llama_cpp_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code, token=args.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ─── Parse evaluation configs ────────────────────────────────────────
        self.eval_configs = self._parse_eval_configs(args)
        # ─── Load teacher model once and cache logits for all eval datasets ──
        self.teacher_model = None
        self.evaluators = []
        self._prepare_teacher_cache()

    def _parse_eval_configs(self, args):
        """Build list of eval config dicts from --eval-configs JSON or fallback to KLD dataset."""
        if args.eval_configs:
            configs = json.loads(args.eval_configs)
            # Fill missing fields with defaults from main args
            for cfg in configs:
                cfg.setdefault("split", "train")
                cfg.setdefault("max_samples", args.max_samples)
                cfg.setdefault("max_seq_length", args.max_seq_length)
                cfg.setdefault("seq_length", args.seq_length)
                cfg.setdefault("num_chunks", args.num_chunks)
                cfg.setdefault("batch_size", args.batch_size)
                if "name" not in cfg:
                    cfg["name"] = cfg["path"].replace("/", "_").replace(".txt", "")
            return configs
        else:
            # Single default from KLD args
            default = {
                "type": "chat" if args.dataset_repo else "text",
                "path": args.dataset_repo or args.calib_data,
                "split": args.dataset_split,
                "max_samples": args.max_samples,
                "max_seq_length": args.max_seq_length,
                "seq_length": args.seq_length,
                "num_chunks": args.num_chunks,
                "batch_size": args.batch_size,
                "name": "default"
            }
            if default["path"] is None:
                raise ValueError("No calibration dataset provided (neither --calib-data nor --dataset-repo).")
            return [default]

    def _prepare_teacher_cache(self):
        """Load the teacher model once and cache its logits for all eval datasets."""
        print("\n" + "="*70)
        print("  PREPARING TEACHER CACHE (global, reused for all sweeps)")
        print("="*70)

        # 1. Load teacher model
        torch_dtype = getattr(torch, self.args.dtype)
        self.teacher_model, input_device = load_model(
            self.args.model, torch_dtype, self.args.trust_remote_code,
            self.args.hf_token, self.device
        )
        # 2. For each eval config, create an Evaluator and cache teacher logits
        for cfg in self.eval_configs:
            eval_args = self._build_eval_args(cfg)
            evaluator = Evaluator(eval_args)
            # We need to set the evaluator's input_device to that of teacher model
            evaluator.input_device = input_device
            # Cache teacher logits using the teacher model
            evaluator.cache_teacher_logits(self.teacher_model)
            # Store evaluator with a name
            evaluator.dataset_name = cfg.get("name", cfg["path"])
            self.evaluators.append(evaluator)

        print(f"[Teacher] Cached logits for {len(self.evaluators)} datasets.")
        # Keep teacher_model for later quantization (we'll modify it in-place)

    def _build_eval_args(self, cfg):
        """Build argparse.Namespace for Evaluator from a config dict."""
        a = argparse.Namespace(**vars(self.args))
        a.model = self.args.model
        a.calib_data = cfg["path"] if cfg["type"] == "text" else None
        a.dataset_repo = cfg["path"] if cfg["type"] == "chat" else None
        a.dataset_split = cfg.get("split", "train")
        a.max_samples = cfg.get("max_samples")
        a.max_seq_length = cfg.get("max_seq_length", 2048)
        a.seq_length = cfg.get("seq_length", 512)
        a.num_chunks = cfg.get("num_chunks", 50)
        a.batch_size = cfg.get("batch_size", 1)
        # Unique cache dir per dataset to avoid collisions
        ds_name = cfg.get("name", cfg["path"].replace("/", "_"))
        a.teacher_cache_dir = os.path.join(self.results_root, "teacher_cache", ds_name)
        a.skip_teacher_cache = False  # We want to cache
        return a

    # ─── KLD, CE, Optimize (unchanged except minor path adjustments) ──────

    def _kld_args_for_mode(self, mode: str, block_size: int = 4):
        a = argparse.Namespace(**vars(self.args))
        a.quant_mode = mode
        a.layer_block_size = block_size if mode == 'by-layer-tensor' else 4
        a.output = os.path.join(self.results_root, f"analysis_{mode}.json")
        a.mapping = self.args.mapping
        a.teacher_cache_dir = os.path.join(self.results_root, "teacher_cache")
        a.skip_teacher_cache = False
        a.imatrix = getattr(self.args, 'imatrix', None)
        return a

    def _ce_args_for_mode(self, mode: str):
        a = argparse.Namespace()
        a.input_json = os.path.join(self.results_root, f"analysis_{mode}.json")
        a.tensor_map = self.args.mapping
        a.gguf_repo = self.args.gguf_repo
        a.gguf_file = self.args.gguf_file
        a.output_dir = _ensure_dir(os.path.join(self.results_root, f"ce_{mode}"))
        a.token = self.args.hf_token
        a.layer_block_size = self.args.layer_block_size if mode == 'by-layer-tensor' else 4
        return a

    def _optimize_args_for_mode(self, mode: str):
        ce_dir = os.path.join(self.results_root, f"ce_{mode}")
        a = argparse.Namespace()
        a.ce_results = os.path.join(ce_dir, "ce_results.txt")
        a.param_counts = os.path.join(ce_dir, "param_counts.json")
        a.output_dir = _ensure_dir(os.path.join(self.results_root, f"configs_{mode}"))
        a.base_gguf = None
        a.imatrix = None
        a.llama_bin_dir = self.args.llama_bin_dir
        a.bits_list = [float(b) for b in self.args.bits_list]
        a.model_name = f"{self.args.model_name}_{mode}"
        a.tensor_files_dir = self._get_weights_file()
        a.layer_block_size = self.args.layer_block_size if mode == 'by-layer-tensor' else 4
        a.dry_run = True,
        a.mapping = self.args.mapping
        return a

    def _get_weights_file(self):
        cache_path = os.path.join(self.results_root, "weights_map.txt")
        if os.path.exists(cache_path):
            return cache_path
        ew_args = argparse.Namespace(
            repo_id=self.args.gguf_repo, output=None, token=self.args.hf_token,
            file=self.args.gguf_file, output_dir=self.results_root
        )
        path = run_extract_weights(ew_args)
        if path != cache_path:
            shutil.copy(path, cache_path)
        return cache_path

    def run_kld(self, mode: str):
        print(f"\n{'='*70}")
        print(f"  [SWEEP] KLD ANALYSIS :: {mode}")
        print(f"{'='*70}")
        kld_args = self._kld_args_for_mode(mode)
        analyzer = NoiseImpactAnalyzer(kld_args, self.quant_ext)
        analyzer.run()
        print(f"[SWEEP] KLD saved: {kld_args.output}")
        return kld_args.output

    def run_ce(self, mode: str):
        print(f"\n{'='*70}")
        print(f"  [SWEEP] CE COMPUTATION :: {mode}")
        print(f"{'='*70}")
        ce_args = self._ce_args_for_mode(mode)
        run_compute_ce(ce_args)
        print(f"[SWEEP] CE saved in: {ce_args.output_dir}")
        return ce_args.output_dir

    def run_optimize(self, mode: str):
        print(f"\n{'='*70}")
        print(f"  [SWEEP] CONFIG GENERATION :: {mode}")
        print(f"{'='*70}")
        opt_args = self._optimize_args_for_mode(mode)
        quantizer = MoQFinalQuantizer(opt_args)
        configs = quantizer.generate_moq_tensor_files(quantizer.parse_ce_report(opt_args.ce_results))
        print(f"[SWEEP] Generated {len(configs)} configs: {[c['bits'] for c in configs]}")
        return configs

    # ─── NEW APPLY+EVAL (with multiple datasets, WandB, global cache) ──────

    def apply_and_evaluate(self, config_path: str, bits: float, mode: str):
        """Apply quant config to the globally cached teacher model, evaluate on all cached datasets, restore, log to WandB."""
        print(f"\n{'='*70}")
        print(f"  [SWEEP] APPLY + EVAL :: {mode} @ {bits} BPW")
        print(f"{'='*70}")

        # Use the globally loaded teacher model (we will modify it in-place and restore)
        model = self.teacher_model  # Already on device

        # ── 1. Apply quantization config (in-place) ──
        mapping = parse_mapping_file(self.args.mapping)
        quant_cfg = parse_quant_config(config_path)
        originals = []
        quantized = 0
        skipped = 0
        for full_name, module in tqdm(list(model.named_modules()), desc="Quantizing"):
            if not hasattr(module, 'weight') or not isinstance(module.weight, nn.Parameter):
                continue
            gguf_name = mapping.get(full_name)
            if not gguf_name:
                skipped += 1; continue
            qt = quant_cfg.get(gguf_name)
            if not qt or qt not in GGML_TYPE_MAP:
                skipped += 1; continue
            try:
                w_cpu = module.weight.data.detach().cpu()
                q_weight = self.quant_ext.apply_llama_quant_noise(
                    w_cpu, gguf_name, GGML_TYPE_MAP[qt],
                    str(self.args.imatrix) if self.args.imatrix else ""
                )
                originals.append((module, module.weight.data))
                module.weight.data = q_weight.to(module.weight.device).to(module.weight.dtype)
                quantized += 1
            except Exception as e:
                print(f"  [Warn] Skip {full_name}: {e}")
                skipped += 1
        print(f"  Quantized {quantized}, skipped {skipped}")

        # ── 2. Evaluate on all cached datasets ──
        all_metrics = {}
        for evaluator in self.evaluators:
            ds_name = evaluator.dataset_name
            print(f"\n  Evaluating dataset: {ds_name}")
            metrics = evaluator.evaluate_student(model)
            all_metrics[ds_name] = metrics if metrics else {"error": "No metrics"}

        # ── 3. Restore original weights ──
        for module, orig in originals:
            module.weight.data = orig
        print(f"  [Cleanup] Restored original weights")

        # ── 4. Save combined results ──
        result = {
            "mode": mode,
            "bits": bits,
            "config_file": os.path.basename(config_path),
            "model": self.args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_datasets": [e.dataset_name for e in self.evaluators],
            "metrics": all_metrics
        }
        result_path = os.path.join(self.results_root, f"eval_{mode}_{bits}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  [Result] Saved: {result_path}")

        # ── 5. WandB logging ──
        if self.args.use_wandb:
            import wandb
            run_name = self.args.wandb_run_name or f"{mode}_{bits}bpw"
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=run_name,
                config={
                    "mode": mode,
                    "bits": bits,
                    "quant_config": os.path.basename(config_path),
                    "eval_datasets": self.eval_configs,
                }
            )
            for ds_name, metrics in all_metrics.items():
                if metrics and "error" not in metrics:
                    wandb.log({f"eval/{ds_name}/{k}": v for k, v in metrics.items()})
                else:
                    wandb.log({f"eval/{ds_name}/error": 1})
            wandb.finish()
            print(f"  [WandB] Logged run: {run_name}")

        # ── 6. Upload results ──
        if self.args.hf_upload_repo:
            _upload_results_to_hf(
                self.results_root, self.args.hf_upload_repo,
                self.args.hf_token, path_in_repo=f"results_{mode}_{bits}"
            )

        return result

    # ─── RUN SINGLE MODE (unchanged loop) ──────────────────────────────────

    def run_single_mode(self, mode: str):
        print(f"\n{'#'*70}")
        print(f"#  SWEEPING MODE: {mode}")
        print(f"{'#'*70}")

        # Stage 2: KLD
        kld_json = self.run_kld(mode)

        # Stage 3: CE
        ce_dir = self.run_ce(mode)

        # Stage 4: Optimize (dry-run → .txt configs)
        configs = self.run_optimize(mode)

        # Stage 5: Apply each config + evaluate (using global cache)
        all_evals = []
        for cfg in configs:
            eval_result = self.apply_and_evaluate(cfg['path'], cfg['bits'], mode)
            all_evals.append(eval_result)

        # Save mode summary
        summary_path = os.path.join(self.results_root, f"summary_{mode}.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "mode": mode,
                "kld_file": kld_json,
                "ce_dir": ce_dir,
                "configs": [c['path'] for c in configs],
                "evaluations": all_evals
            }, f, indent=2)
        print(f"[SWEEP] Summary saved: {summary_path}")
        return all_evals

    def run(self):
        modes = ['by-tensor-type', 'by-layer', 'individual', 'by-layer-tensor']
        all_results = {}
        for mode in modes:
            try:
                results = self.run_single_mode(mode)
                all_results[mode] = results
            except Exception as e:
                print(f"[SWEEP ERROR] Mode {mode} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[mode] = {"error": str(e)}

        # Final master summary
        master_path = os.path.join(self.results_root, "master_summary.json")
        with open(master_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"  MASTER SUMMARY: {master_path}")
        print(f"{'='*70}")

        # Upload everything one last time
        if self.args.hf_upload_repo:
            _upload_results_to_hf(
                self.results_root, self.args.hf_upload_repo,
                self.args.hf_token, path_in_repo="final_results"
            )

        print("\n🎉 SWEEP COMPLETE — ALL RESULTS KEPT, MODELS DELETED")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MoQ Sweep — Test all 4 grouping variants, keep all results, delete models."
    )
    # Model & data
    parser.add_argument("--model", type=str, required=True, help="HF model ID")
    parser.add_argument("--mapping", type=str, required=True, help="tensor_map.txt path")
    parser.add_argument("--gguf-repo", type=str, required=True, help="GGUF repo for metadata")
    parser.add_argument("--gguf-file", type=str, required=True, help="GGUF filename")
    parser.add_argument("--llama-cpp-dir", type=str, required=True, help="llama.cpp source dir")
    parser.add_argument("--imatrix", type=str, default=None, help="Imatrix GGUF path")
    # Calibration (KLD/CE) – still needed for MCKP, but eval can be overridden
    parser.add_argument("--calib-data", type=str, default=None, help="Raw text calibration file")
    parser.add_argument("--dataset-repo", type=str, default=None, help="HF dataset repo for chat")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunks-at-once", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    # Sweep config
    parser.add_argument("--quant-types", nargs='+', default=['Q8_0','Q6_K'])
    parser.add_argument("--bits-list", nargs='+', type=float, default=[5.0,4.5,4.0,3.5,3.0], help="Target BPW budgets to generate configs for")
    parser.add_argument("--layer-block-size", type=int, default=4, help="Layer block size for by-layer-tensor")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=['float16','bfloat16','float32'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-upload-repo", type=str, default=None, help="HF repo to upload ALL results to")
    parser.add_argument("--results-dir", type=str, default="sweep_results", help="Local dir to keep ALL results forever")
    parser.add_argument("--model-name", type=str, default="MoQ-Sweep")
    parser.add_argument("--llama-bin-dir", type=str, default="/workspace/llama_bin", help="Compiled llama.cpp binaries directory")

    # ─── NEW ARGUMENTS ──────────────────────────────────────────────────────────
    parser.add_argument("--eval-configs", type=str, default=None,
                        help='JSON list of eval dataset configs: [{"type":"text|chat","path":"...","name":"..."}, ...]')
    parser.add_argument("--use-wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="MoQ-Sweep", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity/username")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Override run name (default: <mode>_<bits>bpw)")

    args = parser.parse_args()

    if not args.dataset_repo and not args.calib_data and not args.eval_configs:
        parser.error("Must provide either --calib-data, --dataset-repo, or --eval-configs")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sweep = MoQSweep(args)
    sweep.run()


if __name__ == "__main__":
    main()
