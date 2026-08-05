#!/usr/bin/env python3
"""
moq.py
Single entry point for the entire MoQ (Mixture of Quants) system.

Commands:
  pipeline     Run the full 5-stage MoQ generation pipeline
  quant        Apply mixed-precision configs + optional eval/upload
  eval         Batch evaluate student HF repos against a teacher repo
  build-llama  Download & setup llama.cpp CPU binaries
  map-tensors  Build GGUF <-> PyTorch tensor name mapping

Usage:
  python moq.py pipeline --gguf-repo ... --hf-repo ... --model ...
  python moq.py quant --model ... --mapping tensor_map.txt --quant-config-dir ./configs/
  python moq.py eval --teacher-repo ... --student-repos ... --mapping tensor_map.txt
  python moq.py build-llama --install-system-libs
  python moq.py map-tensors --gguf-repo ... --gguf-file ... --hf-repo ...
"""
import argparse, gc, json, os, random, subprocess, sys, glob, time

import torch

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR (from MoQ.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_cmd(cmd, desc=""):
    print(f"\n{'='*70}\n  {desc}\n{'='*70}\nRUNNING: {' '.join(cmd)}\n{'-'*70}")
    subprocess.run(cmd, check=True)

def run_pipeline(args):
    os.makedirs(args.eval_output_dir, exist_ok=True)
    os.makedirs(args.final_output_dir, exist_ok=True)
    if not args.skip_map:
        cmd = [sys.executable, '-c',
               'from moq_engine import run_map_tensors; import argparse; a=argparse.Namespace(); '
               + '; '.join([f'setattr(a,"{k}",getattr(args,"{k}"))' for k in ['gguf_repo','gguf_file','hf_repo','gguf_revision','hf_revision','token','output']])
               + '; run_map_tensors(a)']
        # Actually just call the function directly
        from moq_engine import run_map_tensors
        map_args = argparse.Namespace(
            gguf_repo=args.gguf_repo, gguf_file=args.gguf_file, hf_repo=args.hf_repo,
            gguf_revision='main', hf_revision='main', token=args.token,
            output=args.tensor_map_output
        )
        run_map_tensors(map_args)
        print(f"\n{'='*70}\n  STAGE 1: Map Tensors — DONE\n{'='*70}")
    if not args.skip_kld:
        from moq_engine import NoiseImpactAnalyzer
        from moq_core import compile_cpp_extension
        quant_ext = compile_cpp_extension(args.llama_cpp_dir)
        analyzer = NoiseImpactAnalyzer(args, quant_ext)
        analyzer.run()
        print(f"\n{'='*70}\n  STAGE 2: Analyze KLD — DONE\n{'='*70}")
    if not args.skip_ce:
        from moq_engine import run_compute_ce
        ce_args = argparse.Namespace(
            input_json=args.kld_output_json, tensor_map=args.tensor_map_output,
            gguf_repo=args.gguf_repo, gguf_file=args.gguf_file,
            output_dir=args.eval_output_dir, token=args.token,
            layer_block_size=getattr(args, 'layer_block_size', 4)
        )
        run_compute_ce(ce_args)
        print(f"\n{'='*70}\n  STAGE 3: Compute CE — DONE\n{'='*70}")
    weights_file = None
    if not args.skip_weights:
        from moq_engine import run_extract_weights
        ew_args = argparse.Namespace(
            repo_id=args.gguf_repo, output=None, token=args.token,
            file=args.gguf_file, output_dir=args.eval_output_dir
        )
        weights_path = run_extract_weights(ew_args)
        weights_file = weights_path
        print(f"\n{'='*70}\n  STAGE 4: Extract Weights — DONE\n{'='*70}")
    if not weights_file:
        weights_candidates = glob.glob(os.path.join(args.eval_output_dir, '*_weights.txt'))
        if weights_candidates:
            weights_file = max(weights_candidates, key=os.path.getmtime)
        else:
            sys.exit("❌ FATAL: No *_weights.txt found.")
    if not args.skip_quant:
        from moq_engine import MoQFinalQuantizer
        quant_args = argparse.Namespace(
            ce_results=os.path.join(args.eval_output_dir, 'ce_results.txt'),
            param_counts=os.path.join(args.eval_output_dir, 'param_counts.json'),
            output_dir=args.final_output_dir,
            base_gguf=args.base_gguf_local,
            imatrix=args.final_imatrix,
            llama_bin_dir=args.llama_bin_dir,
            bits_list=args.bits_list,
            model_name=args.model_name,
            tensor_files_dir=weights_file,
            layer_block_size=getattr(args, 'layer_block_size', 4),
            dry_run=args.dry_run,
            mapping=args.tensor_map_output
        )
        MoQFinalQuantizer(quant_args).run()
        print(f"\n{'='*70}\n  STAGE 5: Optimize & Quantize — DONE\n{'='*70}")
    print("\n" + "="*70 + "\n  🎉 MoQ PIPELINE COMPLETE 🎉\n" + "="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MoQ Unified — Mixed Quantization Pipeline & Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python moq.py pipeline --gguf-repo bartowski/Qwen_Qwen3-8B-GGUF \
      --gguf-file Qwen_Qwen3-8B-Q4_K_S.gguf --hf-repo Qwen/Qwen3-8B \
      --model Qwen/Qwen3-8B --calib-data wiki.test.raw \
      --llama-cpp-dir /workspace/llama.cpp --base-gguf-local base.gguf \
      --final-imatrix imatrix.gguf --quant-mode by-layer-tensor --layer-block-size 4

  python moq.py quant --model Qwen/Qwen3-8B --mapping tensor_map.txt \
      --quant-config-dir ./configs/ --llama-cpp-dir /workspace/llama.cpp \
      --calib-data wiki.test.raw --evaluate-now

  python moq.py eval --teacher-repo org/teacher --student-repos org/s1 org/s2 \
      --local-model-dir /tmp/models --mapping tensor_map.txt \
      --llama-cpp-dir /workspace/llama.cpp --calib-data wiki.test.raw
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ─── pipeline ───
    pl = subparsers.add_parser("pipeline", help="Run full MoQ generation pipeline")
    pl.add_argument('--gguf-repo', required=True)
    pl.add_argument('--gguf-file', required=True)
    pl.add_argument('--hf-repo', required=True)
    pl.add_argument('--token', default=None)
    pl.add_argument('--tensor-map-output', default='tensor_map.txt')
    pl.add_argument('--skip-map', action='store_true')
    pl.add_argument('--model', default=None)
    pl.add_argument('--llama-cpp-dir', default=None)
    pl.add_argument('--imatrix-kld', default=None)
    pl.add_argument('--quant-types', nargs='+', default=['Q8_0', 'Q6_K', 'Q5_K', 'Q4_K', 'Q3_K', 'Q2_K', 'IQ4_XS', 'IQ3_S', 'IQ2_S'])
    pl.add_argument('--quant-mode', default='by-tensor-type')
    pl.add_argument('--layer-block-size', type=int, default=4)
    pl.add_argument('--num-chunks', type=int, default=50)
    pl.add_argument('--chunks-at-once', type=int, default=5)
    pl.add_argument('--dtype', default='bfloat16')
    pl.add_argument('--kld-output-json', default='analysis_results.json')
    pl.add_argument('--seed', type=int, default=42)
    pl.add_argument('--trust-remote-code', action='store_true')
    pl.add_argument('--skip-kld', action='store_true')
    pl.add_argument('--calib-data', default=None)
    pl.add_argument('--seq-length', type=int, default=512)
    pl.add_argument('--dataset-repo', default=None)
    pl.add_argument('--dataset-split', default='train')
    pl.add_argument('--max-samples', type=int, default=None)
    pl.add_argument('--max-seq-length', type=int, default=2048)
    pl.add_argument('--eval-output-dir', default='evaluation')
    pl.add_argument('--skip-ce', action='store_true')
    pl.add_argument('--skip-weights', action='store_true')
    pl.add_argument('--base-gguf-local', default=None)
    pl.add_argument('--final-imatrix', default=None)
    pl.add_argument('--llama-bin-dir', default='/workspace/llama_bin')
    pl.add_argument('--bits-list', nargs='+', type=float, default=[5.0, 4.5, 4.0, 3.5])
    pl.add_argument('--model-name', default='MoQ-Model')
    pl.add_argument('--final-output-dir', default='quantized')
    pl.add_argument('--dry-run', action='store_true')
    pl.add_argument('--skip-quant', action='store_true')

    # ─── quant ───
    q = subparsers.add_parser("quant", help="Mixed-precision quantization")
    q.add_argument("--model", type=str, required=True)
    q.add_argument("--mapping", type=str, required=True)
    q.add_argument("--quant-config", type=str, default=None)
    q.add_argument("--quant-config-dir", type=str, default=None)
    q.add_argument("--llama-cpp-dir", type=str, required=True)
    q.add_argument("--imatrix", type=str, default=None)
    q.add_argument("--calib-data", type=str, default=None)
    q.add_argument("--dataset-repo", type=str, default=None)
    q.add_argument("--dataset-split", type=str, default="train")
    q.add_argument("--max-samples", type=int, default=None)
    q.add_argument("--max-seq-length", type=int, default=2048)
    q.add_argument("--seq-length", type=int, default=512)
    q.add_argument("--num-chunks", type=int, default=50)
    q.add_argument("--batch-size", type=int, default=1)
    q.add_argument("--dtype", type=str, default="bfloat16", choices=['float16', 'bfloat16', 'float32'])
    q.add_argument("--output", type=str, default="mixed_quant_kld_batch.json")
    q.add_argument("--trust-remote-code", action="store_true")
    q.add_argument("--hf-token", type=str, default=None)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--evaluate-now", action="store_true")
    q.add_argument("--upload", action="store_true")
    q.add_argument("--hf-upload-repo", type=str, default=None)
    q.add_argument("--teacher-cache-dir", type=str, default="teacher_cache")
    q.add_argument("--skip-teacher-cache", action="store_true")

    # ─── eval ───
    e = subparsers.add_parser("eval", help="Batch evaluation of student repos")
    e.add_argument("--teacher-repo", type=str, required=True)
    e.add_argument("--student-repos", type=str, nargs="+", required=True)
    e.add_argument("--local-model-dir", type=str, required=True)
    e.add_argument("--mapping", type=str, required=True)
    e.add_argument("--llama-cpp-dir", type=str, required=True)
    e.add_argument("--imatrix", type=str, default=None)
    e.add_argument("--calib-data", type=str, default=None)
    e.add_argument("--dataset-repo", type=str, default=None)
    e.add_argument("--dataset-split", type=str, default="train")
    e.add_argument("--max-samples", type=int, default=None)
    e.add_argument("--max-seq-length", type=int, default=2048)
    e.add_argument("--seq-length", type=int, default=512)
    e.add_argument("--num-chunks", type=int, default=50)
    e.add_argument("--batch-size", type=int, default=1)
    e.add_argument("--dtype", type=str, default="bfloat16", choices=['float16', 'bfloat16', 'float32'])
    e.add_argument("--output", type=str, default="batch_eval_results.json")
    e.add_argument("--trust-remote-code", action="store_true")
    e.add_argument("--hf-token", type=str, default=None)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--teacher-cache-dir", type=str, default="teacher_cache")
    e.add_argument("--skip-teacher-cache", action="store_true")
    e.add_argument("--keep-models", action="store_true")

    # ─── build-llama ───
    bl = subparsers.add_parser("build-llama", help="Build llama.cpp CPU binaries")
    bl.add_argument("--output-dir", default="/workspace/llama_bin")
    bl.add_argument("--repo", default='ggml-org/llama.cpp')
    bl.add_argument("--source-dir", default='/workspace/llama.cpp')
    bl.add_argument("--system-lib-dir", default='/usr/lib')
    bl.add_argument("--skip-download", action="store_true")
    bl.add_argument("--install-system-libs", action="store_true")
    bl.add_argument("--no-clone-source", action="store_true")

    # ─── map-tensors ───
    mt = subparsers.add_parser("map-tensors", help="Map GGUF tensor names to PyTorch")
    mt.add_argument("--gguf-repo", required=True)
    mt.add_argument("--gguf-file", required=True)
    mt.add_argument("--hf-repo", required=True)
    mt.add_argument("--gguf-revision", default="main")
    mt.add_argument("--hf-revision", default="main")
    mt.add_argument("--token", default=None)
    mt.add_argument("-o", "--output", default="tensor_map.txt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validate dataset source for eval/quant commands
    if args.command in ("quant", "eval"):
        has_text = getattr(args, 'calib_data', None) is not None
        has_chat = getattr(args, 'dataset_repo', None) is not None
        if not has_text and not has_chat:
            parser.error("Must provide either --calib-data or --dataset-repo")

    if args.command in ("quant", "eval", "pipeline"):
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.command == "pipeline":
        run_pipeline(args)

    elif args.command == "quant":
        from moq_engine import MixedQuantRunner
        runner = MixedQuantRunner(args)
        runner.run()

    elif args.command == "eval":
        from moq_engine import Evaluator
        Evaluator.run_batch(args)

    elif args.command == "build-llama":
        from moq_engine import run_build_llama
        sys.exit(run_build_llama(args))

    elif args.command == "map-tensors":
        from moq_engine import run_map_tensors
        run_map_tensors(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
