#!/usr/bin/env python3
"""
moq_engine.py
All pipeline stages and evaluation engines in one file.
"""
import argparse, gc, json, math, os, random, re, sys, glob, warnings, time, hashlib
from collections import defaultdict
from pathlib import Path
import struct
import subprocess
from typing import Dict, List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from moq_core import (
    GGML_TYPE_MAP, TEACHER_TOPK, SKIP_PATTERNS, compile_cpp_extension,
    compute_token_kld, extract_assistant_log_probs,
    ChatDataset, chat_collate_fn, CalibrationDataset, text_collate_fn,
    parse_mapping_file, BucketBatchSampler
)

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: KLD ANALYSIS ENGINE (from analyze_kld.py)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_quantizable_tensors(model: nn.Module, mapping: dict):
    module_lookup = {name: mod for name, mod in model.named_modules()}
    candidates = []
    for pt_name, gguf_name in mapping.items():
        module = module_lookup.get(pt_name)
        if module is None or not hasattr(module, 'weight'):
            continue
        w = module.weight
        if w.dim() < 2:
            continue
        nl = pt_name.lower()
        if any(p in nl for p in SKIP_PATTERNS):
            continue
        candidates.append((pt_name, module, w.numel()))
    return sorted(candidates, key=lambda x: x[0])

def group_by_tensor_type(candidates):
    groups = defaultdict(list)
    for name, module, n_params in candidates:
        parts = name.split('.')
        tensor_type = parts[-1] if parts else name
        groups[tensor_type].append((name, module, n_params))
    return dict(groups)

def group_by_layer(candidates):
    layers = defaultdict(list)
    for name, module, n_params in candidates:
        nl = name.lower()
        if 'embed' in nl:
            layers['layer_embed'].append((name, module, n_params))
            continue
        if 'lm_head' in nl:
            layers['layer_lm_head'].append((name, module, n_params))
            continue
        found_layer = False
        for pattern in ['layers', 'layer', 'blocks', 'h', 'encoder.layer']:
            match = re.search(rf'{pattern}\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))
                group_start = (layer_idx // 4) * 4
                group_name = f"layer_{group_start}-{group_start+3}"
                layers[group_name].append((name, module, n_params))
                found_layer = True
                break
        if not found_layer:
            parent = '.'.join(name.split('.')[:-1]) if '.' in name else 'other'
            layers[parent].append((name, module, n_params))
    return dict(sorted(layers.items(), key=lambda x: (
        0, int(x[0].split('_')[1].split('-')[0])
    ) if x[0].startswith('layer_') and x[0] not in ['layer_embed', 'layer_lm_head'] else (1, x[0])))

def group_by_layer_tensor_type(candidates, block_size=4):
    """Hybrid grouping: tensor type + layer block. e.g. q_proj_layer_0-3"""
    groups = defaultdict(list)
    for name, module, n_params in candidates:
        nl = name.lower()
        if 'embed' in nl:
            groups['embed_tokens'].append((name, module, n_params))
            continue
        if 'lm_head' in nl:
            groups['lm_head'].append((name, module, n_params))
            continue
        found_layer = False
        for pattern in ['layers', 'layer', 'blocks', 'h', 'encoder.layer']:
            match = re.search(rf'{pattern}\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))
                group_start = (layer_idx // block_size) * block_size
                group_end = group_start + block_size - 1
                parts = name.split('.')
                tensor_type = parts[-1] if parts else name
                group_name = f"{tensor_type}_layer_{group_start}-{group_end}"
                groups[group_name].append((name, module, n_params))
                found_layer = True
                break
        if not found_layer:
            parts = name.split('.')
            tensor_type = parts[-1] if parts else name
            groups[tensor_type].append((name, module, n_params))

    def sort_key(item):
        key = item[0]
        if '_layer_' in key:
            parts = key.split('_layer_')
            start = int(parts[1].split('-')[0])
            return (0, start, parts[0])
        return (1, 0, key)
    return dict(sorted(groups.items(), key=sort_key))

class NoiseImpactAnalyzer:
    def __init__(self, args, quant_ext):
        self.args = args
        self.quant_ext = quant_ext
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"[Pipeline] Using {torch.cuda.device_count()} GPUs with device_map='auto'")
            self.device = torch.device("cuda")
            self.input_device = self.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.input_device = self.device
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        torch_dtype = getattr(torch, args.dtype)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch_dtype, trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=True, device_map="auto",
            )
            self.input_device = next(self.model.parameters()).device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch_dtype, trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = False
        self.mapping = parse_mapping_file(args.mapping)
        self.all_candidates = discover_quantizable_tensors(self.model, self.mapping)
        self._module_lookup = {name: mod for name, mod, _ in self.all_candidates}
        if args.dataset_repo:
            self.dataset = ChatDataset(
                args.dataset_repo, args.dataset_split, self.tokenizer,
                args.max_samples, args.max_seq_length
            )
            sampler = BucketBatchSampler(
                self.dataset.lengths, batch_size=args.chunks_at_once, shuffle=False, seed=args.seed
            )
            self.dataloader = DataLoader(
                self.dataset, batch_sampler=sampler,
                collate_fn=lambda b: chat_collate_fn(b, self.tokenizer),
                pin_memory=(self.device.type == "cuda"), num_workers=4
            )
        else:
            if not args.calib_data:
                sys.exit("FATAL: Must provide either --dataset-repo or --calib-data")
            with open(args.calib_data, 'r', encoding='utf-8') as f:
                texts = [t for t in f.read().split('\n') if t.strip()]
            self.dataset = CalibrationDataset(texts, self.tokenizer, args.seq_length)
            sampler = BucketBatchSampler(
                self.dataset.lengths, batch_size=args.chunks_at_once, shuffle=False, seed=args.seed
            )
            self.dataloader = DataLoader(
                self.dataset, batch_sampler=sampler,
                collate_fn=lambda b: text_collate_fn(b, self.tokenizer),
                pin_memory=(self.device.type == "cuda"), num_workers=4
            )

    def _get_teacher_cache_path(self):
        os.makedirs(self.args.teacher_cache_dir, exist_ok=True)
        model_name = self.args.model.replace("/", "__")
        ds_name = (self.args.dataset_repo or os.path.basename(self.args.calib_data or "text")).replace("/", "__")
        cfg_str = (
            f"{model_name}_{ds_name}_"
            f"ms{self.args.max_samples or 0}_"
            f"msl{self.args.max_seq_length}_"
            f"nc{self.args.num_chunks}_"
            f"ca{self.args.chunks_at_once}_"
            f"seed{self.args.seed}_"
            f"dtype{self.args.dtype}_"
            f"topk{TEACHER_TOPK}_"
            f"bucketed_v1"
        )
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
        return os.path.join(self.args.teacher_cache_dir, f"teacher_{cfg_hash}.pt")

    def _cache_teacher_logits_to_gpu(self):
        cache_path = self._get_teacher_cache_path()
        if not self.args.skip_teacher_cache and os.path.exists(cache_path):
            print(f"[Cache] Found teacher cache: {cache_path}")
            try:
                cached_batches = torch.load(cache_path, map_location=self.input_device, weights_only=False)
                if isinstance(cached_batches, list) and len(cached_batches) > 0:
                    if "teacher_assistant_log_probs" not in cached_batches[0] or \
                       not isinstance(cached_batches[0]["teacher_assistant_log_probs"][0], tuple):
                        print("[Cache] Old dense format detected — invalidating, will recompute...")
                        os.remove(cache_path)
                    else:
                        num_batches = len(cached_batches)
                        cache_mb = sum(
                            sum(
                                vals.element_size() * vals.nelement() + inds.element_size() * inds.nelement()
                                for vals, inds in b["teacher_assistant_log_probs"]
                            )
                            for b in cached_batches
                        ) / 1e6
                        print(f"[Cache] Loaded {num_batches} batches ({cache_mb:.1f} MB) from disk — skipping teacher forward pass")
                        return cached_batches, num_batches
            except Exception as e:
                print(f"[Cache] Failed to load cache ({e}), recomputing...")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        batch_iter = iter(self.dataloader)
        cached_batches = []
        num_batches = 0
        global_actual_chunks = min(self.args.num_chunks, len(self.dataset))
        global_num_batches = math.ceil(global_actual_chunks / self.args.chunks_at_once)
        with torch.inference_mode():
            for _ in tqdm(range(global_num_batches), desc="Caching teacher logits (forward pass)"):
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                input_ids_gpu = batch["input_ids"].to(self.input_device, non_blocking=True)
                attention_mask_gpu = batch["attention_mask"].to(self.input_device, non_blocking=True)
                logits = self.model(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu).logits
                log_probs = F.log_softmax(logits, dim=-1)
                eval_mask = batch.get("eval_mask")
                teacher_assistant_log_probs = extract_assistant_log_probs(
                    log_probs, eval_mask, batch["attention_mask"]
                )
                teacher_sparse = []
                vocab_size = log_probs.size(-1)
                k = min(TEACHER_TOPK, vocab_size)
                for t in teacher_assistant_log_probs:
                    vals, inds = torch.topk(t, k=k, dim=-1, sorted=False)
                    teacher_sparse.append((vals, inds.to(torch.int32)))
                cached_batches.append({
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "teacher_assistant_log_probs": teacher_sparse,
                    "eval_mask": eval_mask,
                })
                num_batches += 1
        if not self.args.skip_teacher_cache:
            try:
                torch.save(cached_batches, cache_path)
                print(f"[Cache] Saved teacher cache to {cache_path}")
            except Exception as e:
                print(f"[Cache] Warning: failed to save cache ({e})")
        return cached_batches, num_batches

    def run(self):
        t0 = time.time()
        cached_batches, num_batches = self._cache_teacher_logits_to_gpu()
        t_cache = time.time()
        if num_batches > 0:
            sample_size = sum(
                sum(
                    vals.element_size() * vals.nelement() + inds.element_size() * inds.nelement()
                    for vals, inds in b["teacher_assistant_log_probs"]
                )
                for b in cached_batches
            ) / 1e9
            print(f"[Perf] {num_batches} batches ready ({sample_size:.2f} GB in GPU VRAM) in {t_cache - t0:.1f}s")
        if self.args.quant_mode == 'by-tensor-type':
            raw_groups = group_by_tensor_type(self.all_candidates)
        elif self.args.quant_mode == 'by-layer':
            raw_groups = group_by_layer(self.all_candidates)
        elif self.args.quant_mode == 'by-layer-tensor':
            raw_groups = group_by_layer_tensor_type(self.all_candidates, getattr(self.args, 'layer_block_size', 4))
        else:
            raw_groups = {name: [(name, None, None)] for name, _, _ in self.all_candidates}
        groups = {unit_name: set(t[0] for t in tensor_list) for unit_name, tensor_list in raw_groups.items()}
        kld_accumulators = defaultdict(list)
        total_groups = len(self.args.quant_types) * len(groups)
        group_idx = 0
        with torch.inference_mode():
            for quant_type in self.args.quant_types:
                t_quant_start = time.time()
                ggml_type_id = GGML_TYPE_MAP[quant_type]
                imatrix_str = str(self.args.imatrix) if self.args.imatrix else ""
                is_iq = quant_type.startswith("IQ")
                for unit_name, tensor_names in tqdm(groups.items(), desc=f"Quant {quant_type}"):
                    originals = {}
                    for name in tensor_names:
                        module = self._module_lookup[name]
                        gguf_name = self.mapping[name]
                        if is_iq:
                            if not imatrix_str or not self.quant_ext.has_imatrix(gguf_name, imatrix_str):
                                continue
                        w_cpu = module.weight.data.detach().cpu()
                        q_weight = self.quant_ext.apply_llama_quant_noise(
                            w_cpu, gguf_name, ggml_type_id, imatrix_str
                        )
                        originals[name] = module.weight.data
                        module.weight.data = q_weight.to(module.weight.device).to(module.weight.dtype)
                    if not originals:
                        continue
                    for batch_idx in range(num_batches):
                        batch_data = cached_batches[batch_idx]
                        input_ids = batch_data["input_ids"].to(self.input_device, non_blocking=True)
                        attention_mask = batch_data["attention_mask"].to(self.input_device, non_blocking=True)
                        teacher_assistant_log_probs = batch_data["teacher_assistant_log_probs"]
                        eval_mask = batch_data.get("eval_mask")
                        if eval_mask is not None:
                            eval_mask = eval_mask.to(self.input_device, non_blocking=True)
                        student_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                        student_log_probs = F.log_softmax(student_logits, dim=-1)
                        student_assistant_log_probs = extract_assistant_log_probs(
                            student_log_probs, eval_mask, attention_mask
                        )
                        if student_assistant_log_probs and teacher_assistant_log_probs:
                            for i, (teacher_vals, teacher_inds) in enumerate(teacher_assistant_log_probs):
                                teacher_vals = teacher_vals.to(student_logits.device, non_blocking=True)
                                teacher_inds = teacher_inds.to(student_logits.device, non_blocking=True).long()
                                student_vals = student_assistant_log_probs[i].gather(1, teacher_inds)
                                mask = teacher_vals > -16.0
                                teacher_probs = torch.exp(teacher_vals) * mask
                                kld_contrib = teacher_probs * (teacher_vals - student_vals)
                                kld_per_token = torch.clamp(kld_contrib.sum(dim=-1), min=0.0)
                                kld_accumulators[(quant_type, unit_name)].append(kld_per_token)
                    key = (quant_type, unit_name)
                    if kld_accumulators[key]:
                        kld_accumulators[key] = [torch.cat(kld_accumulators[key]).cpu()]
                    for name, orig_data in originals.items():
                        self._module_lookup[name].weight.data = orig_data
                    del originals
                    group_idx += 1
                    if group_idx % max(1, total_groups // 10) == 0:
                        elapsed = time.time() - t0
                        remain = elapsed / group_idx * (total_groups - group_idx) if group_idx > 0 else 0
                        print(f"[Perf] {group_idx}/{total_groups} groups | {elapsed:.1f}s elapsed | ~{remain:.0f}s left")
                print(f"[Perf] Quant type {quant_type} done in {time.time() - t_quant_start:.1f}s")
        results = {}
        for quant_type in self.args.quant_types:
            for unit_name in groups.keys():
                kld_list = kld_accumulators.get((quant_type, unit_name), [])
                if not kld_list:
                    metrics = {
                        "mean_kld": float('nan'), "max_kld": float('nan'),
                        "p99_kld": float('nan'), "p99.9_kld": float('nan')
                    }
                else:
                    all_kld = kld_list[0].float()
                    metrics = {
                        "mean_kld": all_kld.mean().item(),
                        "max_kld": all_kld.max().item(),
                        "p99_kld": torch.quantile(all_kld, 0.99).item(),
                        "p99.9_kld": torch.quantile(all_kld, 0.999).item()
                    }
                results[f"{quant_type}/{unit_name}"] = metrics
        with open(self.args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {self.args.output}")
        print(f"[Perf] Total time: {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: EVALUATION ENGINE (from eval.py)
# ═══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=getattr(args, 'trust_remote_code', False),
            token=getattr(args, 'hf_token', None)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.mode = 'chat' if getattr(args, 'dataset_repo', None) else 'text'
        self._build_dataloader()
        self.input_device = None
        self.num_batches = 0

    def _build_dataloader(self):
        if self.mode == 'chat':
            self.dataset = ChatDataset(
                self.args.dataset_repo,
                getattr(self.args, 'dataset_split', 'train'),
                self.tokenizer,
                getattr(self.args, 'max_samples', None),
                getattr(self.args, 'max_seq_length', 2048)
            )
            self.collate_fn = lambda b: chat_collate_fn(b, self.tokenizer)
        else:
            with open(self.args.calib_data, 'r', encoding='utf-8') as f:
                texts = [t for t in f.read().split('\n') if t.strip()]
            self.dataset = CalibrationDataset(
                texts, self.tokenizer, getattr(self.args, 'seq_length', 512)
            )
            self.collate_fn = lambda b: text_collate_fn(b, self.tokenizer)
            
        sampler = BucketBatchSampler(
            self.dataset.lengths,
            batch_size=getattr(self.args, 'batch_size', 1),
            shuffle=False,
            seed=getattr(self.args, 'seed', 42)
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            pin_memory=(self.device.type == "cuda"),
            num_workers=4
        )

    def _get_teacher_cache_path(self):
        cache_dir = getattr(self.args, 'teacher_cache_dir', 'teacher_cache')
        os.makedirs(cache_dir, exist_ok=True)
        model_name = self.args.model.replace("/", "__")
        ds_name = (getattr(self.args, 'dataset_repo', None) or os.path.basename(
            getattr(self.args, 'calib_data', "text") or "text")).replace("/", "__")
        cfg_str = (
            f"{model_name}_{ds_name}_"
            f"ms{getattr(self.args, 'max_samples', 0) or 0}_"
            f"msl{getattr(self.args, 'max_seq_length', 2048)}_"
            f"nc{getattr(self.args, 'num_chunks', 50)}_"
            f"bs{getattr(self.args, 'batch_size', 1)}_"
            f"seed{self.args.seed}_"
            f"dtype{getattr(self.args, 'dtype', 'bfloat16')}_"
            f"topk{TEACHER_TOPK}_"
            f"bucketed_v1"
        )
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
        return os.path.join(cache_dir, f"teacher_{cfg_hash}.pt")

    def cache_teacher_logits(self, model):
        print("\n" + "="*70)
        print("  PHASE 1: TEACHER LOGIT CACHING")
        print("="*70)
        cache_path = self._get_teacher_cache_path()
        self.input_device = next(model.parameters()).device
        if not getattr(self.args, 'skip_teacher_cache', False) and os.path.exists(cache_path):
            print(f"[Cache] Found teacher cache: {cache_path}")
            try:
                cached_batches = torch.load(cache_path, map_location=self.input_device, weights_only=False)
                if isinstance(cached_batches, list) and len(cached_batches) > 0:
                    if "teacher_assistant_log_probs" not in cached_batches[0] or \
                       not isinstance(cached_batches[0]["teacher_assistant_log_probs"][0], tuple):
                        print("[Cache] Old dense format detected — invalidating, will recompute...")
                        os.remove(cache_path)
                    else:
                        self.num_batches = len(cached_batches)
                        cache_mb = sum(
                            sum(
                                vals.element_size() * vals.nelement() + inds.element_size() * inds.nelement()
                                for vals, inds in b["teacher_assistant_log_probs"]
                            )
                            for b in cached_batches
                        ) / 1e6
                        print(f"[Cache] Loaded {self.num_batches} batches ({cache_mb:.1f} MB in GPU VRAM) from disk — skipping teacher forward pass")
                        self.cached_batches = cached_batches
                        return
            except Exception as e:
                print(f"[Cache] Failed to load cache ({e}), recomputing...")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        actual_chunks = min(getattr(self.args, 'num_chunks', 50), len(self.dataset))
        global_num_batches = math.ceil(actual_chunks / getattr(self.args, 'batch_size', 1))
        batch_iter = iter(self.dataloader)
        cached_batches = []
        with torch.inference_mode():
            for _ in tqdm(range(global_num_batches), desc="[Teacher] Forward"):
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                input_ids = batch["input_ids"].to(self.input_device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.input_device, non_blocking=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                log_probs = F.log_softmax(logits, dim=-1)
                eval_mask = batch.get("eval_mask")
                teacher_assistant_log_probs = extract_assistant_log_probs(
                    log_probs, eval_mask, attention_mask
                )
                teacher_sparse = []
                vocab_size = log_probs.size(-1)
                k = min(TEACHER_TOPK, vocab_size)
                for t in teacher_assistant_log_probs:
                    vals, inds = torch.topk(t, k=k, dim=-1, sorted=False)
                    teacher_sparse.append((vals, inds.to(torch.int32)))
                cached_batches.append({
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "teacher_assistant_log_probs": teacher_sparse,
                    "eval_mask": eval_mask,
                })
        self.num_batches = len(cached_batches)
        self.cached_batches = cached_batches
        if not getattr(self.args, 'skip_teacher_cache', False):
            try:
                torch.save(cached_batches, cache_path)
                print(f"[Cache] Saved teacher cache to {cache_path}")
            except Exception as e:
                print(f"[Cache] Warning: failed to save cache ({e})")
        print("[Teacher] Caching complete.")

    def evaluate_student(self, model):
        if not hasattr(self, 'cached_batches') or self.num_batches == 0:
            raise RuntimeError("Teacher logits not cached. Call cache_teacher_logits() first.")
        kld_accumulator = []
        total_ce = 0.0
        total_ppl_tokens = 0
        if self.input_device is None:
            self.input_device = next(model.parameters()).device
        with torch.inference_mode():
            for batch_idx in tqdm(range(self.num_batches), desc="  Evaluating"):
                batch_data = self.cached_batches[batch_idx]
                input_ids = batch_data["input_ids"].to(self.input_device, non_blocking=True)
                attention_mask = batch_data["attention_mask"].to(self.input_device, non_blocking=True)
                teacher_assistant_log_probs = batch_data["teacher_assistant_log_probs"]
                eval_mask = batch_data.get("eval_mask")
                if eval_mask is not None:
                    eval_mask = eval_mask.to(self.input_device, non_blocking=True)
                student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                student_assistant_log_probs = extract_assistant_log_probs(
                    student_log_probs, eval_mask, attention_mask
                )
                if student_assistant_log_probs and teacher_assistant_log_probs:
                    for i, (teacher_vals, teacher_inds) in enumerate(teacher_assistant_log_probs):
                        teacher_vals = teacher_vals.to(student_logits.device, non_blocking=True)
                        teacher_inds = teacher_inds.to(student_logits.device, non_blocking=True).long()
                        student_vals = student_assistant_log_probs[i].gather(1, teacher_inds)
                        mask = teacher_vals > -16.0
                        teacher_probs = torch.exp(teacher_vals) * mask
                        kld_contrib = teacher_probs * (teacher_vals - student_vals)
                        kld_per_token = torch.clamp(kld_contrib.sum(dim=-1), min=0.0)
                        kld_accumulator.append(kld_per_token.cpu())
                shift_logits = student_logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                if eval_mask is not None:
                    shift_mask = eval_mask[:, 1:].contiguous().bool()
                else:
                    shift_mask = attention_mask[:, 1:].contiguous().bool()
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                flat_mask = shift_mask.view(-1)
                if flat_mask.any():
                    ce = F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask], reduction='sum')
                    total_ce += ce.item()
                    total_ppl_tokens += flat_mask.sum().item()
        return self.aggregate(kld_accumulator, total_ce, total_ppl_tokens)

    def aggregate(self, kld_accumulator, total_ce, total_ppl_tokens):
        if not kld_accumulator:
            print(f"  [Err] No KLD data")
            return None
        all_kld = torch.cat(kld_accumulator).float()
        metrics = {
            "mean_kld": all_kld.mean().item(),
            "max_kld": all_kld.max().item(),
            "p99_kld": torch.quantile(all_kld, 0.99).item(),
            "p99.9_kld": torch.quantile(all_kld, 0.999).item(),
            "total_tokens": all_kld.numel()
        }
        if total_ppl_tokens > 0:
            metrics["ppl"] = math.exp(total_ce / total_ppl_tokens)
        else:
            metrics["ppl"] = float('inf')
        print(f"\n  ── Results ──")
        print(f"    Mean KLD   : {metrics['mean_kld']:.6f}")
        print(f"    P99 KLD    : {metrics['p99_kld']:.6f}")
        print(f"    P99.9 KLD  : {metrics['p99.9_kld']:.6f}")
        print(f"    Max KLD    : {metrics['max_kld']:.6f}")
        print(f"    PPL        : {metrics['ppl']:.4f}")
        print(f"    Tokens     : {metrics['total_tokens']}")
        return metrics

    def cleanup(self):
        if hasattr(self, 'cached_batches'):
            del self.cached_batches
        gc.collect()

    @staticmethod
    def run_batch(args):
        print(f"\n{'='*70}")
        print("  BATCH REPO EVALUATION")
        print(f"{'='*70}")
        print(f"Teacher : {args.teacher_repo}")
        print(f"Students: {len(args.student_repos)} repos")
        print(f"Local cache dir: {args.local_model_dir}")
        os.makedirs(args.local_model_dir, exist_ok=True)
        teacher_args = argparse.Namespace(**vars(args))
        teacher_args.model = args.teacher_repo
        evaluator = Evaluator(teacher_args)
        torch_dtype = getattr(torch, args.dtype)
        from moq_core import load_model
        teacher_model, _ = load_model(
            args.teacher_repo, torch_dtype, args.trust_remote_code,
            getattr(args, 'hf_token', None), evaluator.device
        )
        evaluator.cache_teacher_logits(teacher_model)
        del teacher_model
        gc.collect()
        if evaluator.device.type == "cuda":
            torch.cuda.empty_cache()
        all_results = []
        for student_repo in args.student_repos:
            print(f"\n{'='*70}")
            print(f"  Student: {student_repo}")
            print(f"{'='*70}")
            t0 = time.time()
            local_path = os.path.join(args.local_model_dir, student_repo.replace("/", "__"))
            try:
                print(f"[Student] Loading {student_repo}...")
                student_model, _ = load_model(
                    student_repo, torch_dtype, args.trust_remote_code,
                    getattr(args, 'hf_token', None), evaluator.device
                )
                student_model.save_pretrained(local_path)
                evaluator.tokenizer.save_pretrained(local_path)
                metrics = evaluator.evaluate_student(student_model)
                metrics["student_repo"] = student_repo
                metrics["elapsed_sec"] = time.time() - t0
                all_results.append(metrics)
                print(f"\n  ── Results for {student_repo} ──")
                print(f"    Mean KLD   : {metrics['mean_kld']:.6f}")
                print(f"    P99 KLD    : {metrics['p99_kld']:.6f}")
                print(f"    P99.9 KLD  : {metrics['p99.9_kld']:.6f}")
                print(f"    Max KLD    : {metrics['max_kld']:.6f}")
                print(f"    PPL        : {metrics['ppl']:.4f}")
                print(f"    Tokens     : {metrics['total_tokens']}")
            except Exception as e:
                print(f"[Student] Failed: {e}")
                all_results.append({"student_repo": student_repo, "error": str(e)})
            if 'student_model' in dir():
                del student_model
            gc.collect()
            if evaluator.device.type == "cuda":
                torch.cuda.empty_cache()
            if os.path.exists(local_path) and not getattr(args, 'keep_models', False):
                shutil.rmtree(local_path, ignore_errors=True)
                print(f"[Student] Deleted local copy: {local_path}")
        with open(args.output, 'w') as f:
            json.dump({
                "teacher_repo": args.teacher_repo,
                "student_repos": args.student_repos,
                "results": all_results
            }, f, indent=4)
        print(f"\n[Done] Results saved to {args.output}")
        evaluator.cleanup()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: MIXED-PRECISION QUANT RUNNER (from quant.py)
# ═══════════════════════════════════════════════════════════════════════════════

class MixedQuantRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Setup] Compiling C++ extension...")
        self.quant_ext = compile_cpp_extension(args.llama_cpp_dir)
        print("[Setup] Parsing mapping...")
        self.mapping = parse_mapping_file(args.mapping)
        print(f"  Mapping: {len(self.mapping)} entries")
        if args.quant_config_dir:
            self.config_files = sorted(glob.glob(os.path.join(args.quant_config_dir, "*.txt")))
            if not self.config_files:
                sys.exit(f"No .txt files found in {args.quant_config_dir}")
        elif args.quant_config:
            self.config_files = [args.quant_config]
        else:
            sys.exit("Must provide --quant-config or --quant-config-dir")
        print(f"  Configs to evaluate: {len(self.config_files)}")

    def _apply_single_config(self, model, config_path):
        quant_config = parse_quant_config(config_path)
        print(f"\n[Config] Applying: {os.path.basename(config_path)} ({len(quant_config)} rules)")
        originals = []
        quantized_count = 0
        skipped_count = 0
        for full_name, module in tqdm(list(model.named_modules()), desc="  Quantizing"):
            if not hasattr(module, 'weight') or not isinstance(module.weight, nn.Parameter):
                continue
            gguf_name = self.mapping.get(full_name)
            if not gguf_name:
                skipped_count += 1
                continue
            quant_type = quant_config.get(gguf_name)
            if not quant_type:
                skipped_count += 1
                continue
            if quant_type not in GGML_TYPE_MAP:
                print(f"  [Warn] Unknown quant type '{quant_type}' for {gguf_name}. Skipping.")
                skipped_count += 1
                continue
            ggml_type_id = GGML_TYPE_MAP[quant_type]
            w_cpu = module.weight.data.detach().cpu()
            try:
                q_weight = self.quant_ext.apply_llama_quant_noise(
                    w_cpu, gguf_name, ggml_type_id, self.imatrix_str
                )
                originals.append((module, module.weight.data))
                module.weight.data = q_weight.to(module.weight.device).to(module.weight.dtype)
                quantized_count += 1
            except Exception as e:
                print(f"  [Err] Failed to quantize {full_name} ({gguf_name}): {e}")
                skipped_count += 1
        print(f"  [Config] Quantized {quantized_count}, skipped {skipped_count}")
        return originals

    def _restore_weights(self, originals):
        for module, orig_data in originals:
            module.weight.data = orig_data
        print(f"  [Config] Restored {len(originals)} tensors to clean state")

    def _upload_model(self, model, config_path, metrics=None):
        try:
            from huggingface_hub import create_repo
            repo_id = getattr(self.args, 'hf_upload_repo', None)
            if not repo_id:
                print("[Upload] Warning: --hf-upload-repo not set, skipping upload.")
                return
            print(f"[Upload] Pushing model to {repo_id}...")
            create_repo(repo_id, exist_ok=True, token=self.args.hf_token)
            model.push_to_hub(repo_id, token=self.args.hf_token)
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model, trust_remote_code=self.args.trust_remote_code,
                token=self.args.hf_token
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.push_to_hub(repo_id, token=self.args.hf_token)
            print(f"[Upload] Successfully uploaded to {repo_id}")
        except Exception as e:
            print(f"[Upload] Failed to upload model: {e}")

    def run(self):
        print("\n" + "="*70)
        print("  PHASE 0: LOAD MODEL ONCE")
        print("="*70)
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model, trust_remote_code=self.args.trust_remote_code,
            token=self.args.hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        torch_dtype = getattr(torch, self.args.dtype)
        from moq_core import load_model
        model, input_device = load_model(
            self.args.model, torch_dtype, self.args.trust_remote_code,
            self.args.hf_token, self.device
        )
        self.imatrix_str = str(self.args.imatrix) if self.args.imatrix else ""
        evaluator = None
        if self.args.evaluate_now:
            print("\n[Eval] --evaluate-now set; initializing evaluator...")
            evaluator = Evaluator(self.args)
            evaluator.cache_teacher_logits(model)
        all_results = []
        for config_path in self.config_files:
            print(f"\n{'='*70}")
            print(f"  QUANTIZING: {os.path.basename(config_path)}")
            print(f"{'='*70}")
            originals = self._apply_single_config(model, config_path)
            metrics = None
            if evaluator:
                print(f"\n  Evaluating {os.path.basename(config_path)}...")
                metrics = evaluator.evaluate_student(model)
                if metrics:
                    metrics["config_file"] = os.path.basename(config_path)
                    all_results.append(metrics)
            if self.args.upload:
                self._upload_model(model, config_path, metrics)
            self._restore_weights(originals)
            del originals
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if self.args.evaluate_now and all_results:
            print("\n" + "="*70)
            print("  PHASE 3: SAVE COMBINED RESULTS")
            print("="*70)
            with open(self.args.output, 'w') as f:
                json.dump({
                    "model": self.args.model,
                    "evaluated_configs": len(all_results),
                    "results": all_results
                }, f, indent=4)
            print(f"[Done] Combined results saved to {self.args.output}")
        print("\n[Cleanup] Unloading model...")
        del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if evaluator:
            evaluator.cleanup()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: COMPUTE CE (from compute_ce.py)
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np

DEFAULT_QUANT_BITS = {
    "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5,
    "Q3_K": 3.4375, "Q2_K": 2.625,
    "Q5_1": 6.0, "Q5_0": 5.5,
    "Q4_1": 5.0, "Q4_0": 4.5,
    "Q1_0": 1.125, "Q2_0": 2.25,
    "IQ4_NL": 4.5, "IQ4_XS": 4.25, "IQ3_S": 3.4375,
    "IQ3_XXS": 3.0625, "IQ2_S": 2.5625,
    "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625,
    "IQ1_M": 1.75, "IQ1_S": 1.5625,
    "BF16": 16.0, "F16": 16.0, "F32": 32.0,
}

class GGUFMetadataExtractor:
    def __init__(self, repo_id, filename, branch="main", token=None):
        self.repo_id, self.filename, self.branch, self.token = repo_id, filename, branch, token

    @staticmethod
    def _get_tensor_type(name):
        cleaned = name.strip()
        while cleaned.endswith(".weight") or cleaned.endswith(".bias"):
            cleaned = cleaned.rsplit('.', 1)[0]
        return re.sub(r'^blk\.\d+\.', '', cleaned)

    def extract(self):
        url = f"https://huggingface.co/{self.repo_id}/resolve/{self.branch}/{self.filename}"
        HEADER_SIZE = 16 * 1024 * 1024
        headers = {"Range": f"bytes=0-{HEADER_SIZE - 1}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = __import__('requests').get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        data, data_len, offset = resp.content, len(resp.content), 0

        def check(n):
            if offset + n > data_len:
                raise ValueError("Header truncated.")
        def read_u32():
            nonlocal offset
            check(4)
            v = struct.unpack("<I", data[offset:offset+4])[0]
            offset += 4
            return v
        def read_u64():
            nonlocal offset
            check(8)
            v = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            return v
        def read_str():
            nonlocal offset
            length = read_u64()
            check(length)
            s = data[offset:offset+length].decode('utf-8')
            offset += length
            return s

        check(4)
        if data[offset:offset+4] != b'GGUF':
            raise ValueError("Not a GGUF file")
        offset += 4
        version = read_u32()
        tensor_count = read_u64() if version >= 3 else read_u32()
        metadata_kv_count = read_u64() if version >= 3 else read_u32()

        for _ in range(metadata_kv_count):
            key = read_str()
            vtype = read_u32()
            if vtype in (0, 1, 7):
                offset += 1
            elif vtype in (2, 3):
                offset += 2
            elif vtype in (4, 5, 6):
                offset += 4
            elif vtype == 8:
                read_str()
            elif vtype == 9:
                arr_type = read_u32()
                arr_len = read_u64()
                if arr_type == 8:
                    for _ in range(arr_len):
                        read_str()
                else:
                    elem_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1}
                    offset += arr_len * elem_sizes.get(arr_type, 0)
            elif vtype in (10, 11, 12):
                offset += 8

        raw_tensors = {}
        n_blocks = 0
        for _ in range(tensor_count):
            name = read_str()
            n_dims = read_u32()
            dims = [read_u64() for _ in range(n_dims)]
            offset += 12
            params = 1
            for d in dims:
                params *= d
            raw_tensors[name] = params
            if name.startswith("blk."):
                parts = name.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    n_blocks = max(n_blocks, int(parts[1]) + 1)

        type_params = {}
        for full_name, params in raw_tensors.items():
            ttype = self._get_tensor_type(full_name)
            type_params[ttype] = type_params.get(ttype, 0) + params
        return type_params, n_blocks

def map_unit_to_gguf_type(unit_name, tensor_map):
    base_unit = re.sub(r'_layer_\d+-\d+$', '', unit_name)
    gguf_types = set()
    for pt_name, gguf_name in tensor_map.items():
        if pt_name.endswith(f".{base_unit}") or pt_name == base_unit:
            clean = gguf_name.replace('.weight', '').replace('.bias', '')
            clean = re.sub(r'^blk\.\d+\.', '', clean)
            gguf_types.add(clean)
    return list(gguf_types)[0] if gguf_types else unit_name

def run_compute_ce(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_json) as f:
        results = json.load(f)
    tensor_map = {}
    with open(args.tensor_map) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                pt, gguf = line.split('=', 1)
                tensor_map[pt.strip()] = gguf.strip()
    extractor = GGUFMetadataExtractor(args.gguf_repo, args.gguf_file, token=args.token)
    param_counts, n_blocks = extractor.extract()
    data = []
    for key, metrics in results.items():
        qt, unit = key.split('/', 1)
        gguf_type = map_unit_to_gguf_type(unit, tensor_map)
        bpw = DEFAULT_QUANT_BITS.get(qt, 16.0)
        data.append({
            "Layer": unit, "Quant": qt, "BPW": bpw,
            "Mean KLD": metrics.get("mean_kld", 0),
            "99.0% KLD": metrics.get("p99_kld", 0),
            "99.9% KLD": metrics.get("p99.9_kld", 0)
        })
    df = pd.DataFrame(data)
    ce_dfs = []
    for layer, group in df.groupby("Layer"):
        g = group.copy()
        for col in ["Mean KLD", "99.0% KLD", "99.9% KLD"]:
            w = g[col].max()
            g[f"CE_{col}"] = g[col] / w if w > 0 else 0.0
        g["CE_Average"] = g[[f"CE_Mean KLD", f"CE_99.0% KLD", f"CE_99.9% KLD"]].mean(axis=1)
        ce_dfs.append(g)
    final_df = pd.concat(ce_dfs)
    final_df["Model"] = final_df.apply(lambda row: f"MoQ-{row['Layer']}-{row['Quant']}.gguf", axis=1)
    final_df = final_df.rename(columns={
        "CE_Mean KLD": "CE_Mean_KLD",
        "CE_99.0% KLD": "CE_990_KLD",
        "CE_99.9% KLD": "CE_999_KLD"
    })
    out_path = os.path.join(args.output_dir, "ce_results.txt")
    param_path = os.path.join(args.output_dir, "param_counts.json")
    final_param_counts = {}
    for unit in final_df["Layer"].unique():
        gguf_type = map_unit_to_gguf_type(unit, tensor_map)
        if '_layer_' in unit:
            match = re.search(r'_layer_(\d+)-(\d+)$', unit)
            if match and n_blocks > 0:
                start, end = int(match.group(1)), int(match.group(2))
                block_size = end - start + 1
                params = param_counts.get(gguf_type, 0) / n_blocks * block_size
            else:
                params = param_counts.get(gguf_type, 0)
        else:
            params = param_counts.get(gguf_type, 0)
        final_param_counts[unit] = int(params)
    with open(param_path, "w") as f:
        json.dump(final_param_counts, f, indent=2)
    with open(out_path, "w") as f:
        f.write("CE ELASTICITY ANALYSIS REPORT\n")
        for layer, group in final_df.groupby("Layer"):
            f.write("\n" + "="*50 + f"\n  TENSOR TYPE : {layer.upper()}\n" + "="*50 + "\n\n")
            f.write(f"--- FULL CE REPORT ({layer.upper()}) ---\n\n")
            cols = ["Model", "BPW", "CE_Mean_KLD", "CE_990_KLD", "CE_999_KLD", "CE_Average"]
            f.write(group[cols].to_string(index=False) + "\n")
    print(f"✅ Saved CE results to {out_path}")
    print(f"✅ Saved param counts to {param_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E: EXTRACT WEIGHTS (from extract_weights.py)
# ═══════════════════════════════════════════════════════════════════════════════

import urllib.request

HF_API = "https://huggingface.co"
HF_CDN = "https://huggingface.co"
GGUF_MAGIC_INT = 0x46554747

GGML_TYPE_NAMES = {
    0: 'f32', 1: 'f16', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1', 8: 'q8_0', 9: 'q8_1',
    10: 'q2_k', 11: 'q3_k', 12: 'q4_k', 13: 'q5_k', 14: 'q6_k', 15: 'q8_k', 16: 'iq2_xxs',
    17: 'iq2_xs', 18: 'iq3_xxs', 19: 'iq1_s', 20: 'iq4_nl', 21: 'iq3_s', 22: 'iq2_s',
    23: 'iq4_xs', 24: 'i8', 25: 'i16', 26: 'i32', 27: 'i64', 28: 'f64', 29: 'iq1_m', 30: 'bf16',
    41: 'q1_0', 42: 'q2_0'
}

def _hf_api_request(url, token=None):
    headers = {'User-Agent': 'moq-weight-mapper/1.0'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def _get_repo_files(repo_id, token=None):
    return [s['rfilename'] for s in _hf_api_request(f"{HF_API}/api/models/{repo_id}", token).get('siblings', [])]

def _pick_gguf_file(files, prefer=None):
    gguf_files = [f for f in files if f.endswith('.gguf')]
    if not gguf_files:
        raise ValueError("No .gguf files found.")
    if prefer:
        matches = [f for f in gguf_files if prefer.lower() in f.lower()]
        if matches:
            return matches[0]
    for f in gguf_files:
        if '-00001-of-' in f or f.count('-of-') == 0:
            return f
    return gguf_files[0]

def _fetch_bytes(repo_id, filename, start, length, token=None):
    url = f"{HF_CDN}/{repo_id}/resolve/main/{filename}"
    headers = {'Range': f'bytes={start}-{start + length - 1}', 'User-Agent': 'moq-weight-mapper/1.0'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()

class _RangeReader:
    CHUNK = 512 * 1024
    def __init__(self, repo_id, filename, token=None):
        self.repo_id, self.filename, self.token = repo_id, filename, token
        self.pos, self.buf, self.buf_start = 0, b'', 0
    def _fill(self, need):
        have = self.buf_start + len(self.buf) - self.pos
        if have >= need:
            return
        fetch_start, fetch_len = self.pos, max(need, self.CHUNK)
        data = _fetch_bytes(self.repo_id, self.filename, fetch_start, fetch_len, self.token)
        self.buf, self.buf_start = data, fetch_start
    def read(self, n):
        self._fill(n)
        offset = self.pos - self.buf_start
        result = self.buf[offset:offset + n]
        self.pos += len(result)
        return result
    def read_fmt(self, fmt):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))
    def read_u32(self):
        return self.read_fmt('<I')[0]
    def read_u64(self):
        return self.read_fmt('<Q')[0]
    def read_string(self):
        length = self.read_u64()
        return self.read(length).decode('utf-8', errors='replace')
    def read_value(self, vtype):
        if vtype in (0,1,7):
            return self.read_fmt('<B')[0] if vtype!=1 else self.read_fmt('<b')[0]
        if vtype in (2,3):
            return self.read_fmt('<H')[0] if vtype==2 else self.read_fmt('<h')[0]
        if vtype in (4,5,6):
            return self.read_fmt('<I')[0] if vtype==4 else (self.read_fmt('<i')[0] if vtype==5 else self.read_fmt('<f')[0])
        if vtype == 8:
            return self.read_string()
        if vtype in (10,11,12):
            return self.read_fmt('<Q')[0] if vtype==10 else (self.read_fmt('<q')[0] if vtype==11 else self.read_fmt('<d')[0])
        if vtype == 9:
            elem_type, count = self.read_u32(), self.read_u64()
            return [self.read_value(elem_type) for _ in range(count)]
        raise ValueError(f"Unknown value type: {vtype}")

def _parse_gguf_tensors(reader):
    if reader.read_u32() != GGUF_MAGIC_INT:
        raise ValueError("Not a GGUF file")
    version, tensor_count, metadata_kv_count = reader.read_u32(), reader.read_u64(), reader.read_u64()
    for _ in range(metadata_kv_count):
        reader.read_string()
        reader.read_value(reader.read_u32())
    tensors = []
    for _ in range(tensor_count):
        name, n_dims = reader.read_string(), reader.read_u32()
        dims = [reader.read_u64() for _ in range(n_dims)]
        ggml_type, offset = reader.read_u32(), reader.read_u64()
        tensors.append((name, ggml_type))
    return tensors

def run_extract_weights(args):
    files = _get_repo_files(args.repo_id, args.token)
    gguf_file = args.file if args.file else _pick_gguf_file(files)
    reader = _RangeReader(args.repo_id, gguf_file, args.token)
    tensors = _parse_gguf_tensors(reader)
    lines = [f"{name}={GGML_TYPE_NAMES.get(ggml_type, f'type_{ggml_type}')}" for name, ggml_type in tensors]
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'Tensor_Files'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = args.output or f"{args.repo_id.split('/')[-1]}_weights.txt"
    final_output_path = output_dir / output_filename
    with open(final_output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"✓ Done! Written {len(lines)} weight entries to: {final_output_path}")
    return str(final_output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F: OPTIMIZE & QUANTIZE (from optimize_quantize.py)
# ═══════════════════════════════════════════════════════════════════════════════

class MoQFinalQuantizer:
    def __init__(self, args):
        self.ce_results_file = args.ce_results
        self.output_dir = args.output_dir
        self.base_gguf_path = args.base_gguf
        self.imatrix_path = args.imatrix
        self.llama_bin_dir = args.llama_bin_dir
        self.bits_list = [float(b) for b in args.bits_list]
        self.model_name = args.model_name
        self.tensor_files_dir = args.tensor_files_dir
        self.dry_run = args.dry_run
        self.layer_block_size = getattr(args, 'layer_block_size', 4)
        with open(args.param_counts, 'r') as f:
            self.param_counts = json.load(f)
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_layer_tensors(self):
        tensor_names = []
        with open(self.tensor_files_dir, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                tensor_names.append(line.split('=')[0].strip())
        return tensor_names

    def parse_ce_report(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        tensor_data = {}
        sections = re.split(r'={50,}\s+TENSOR TYPE : ([\w_]+)', content)
        for i in range(1, len(sections), 2):
            tensor_name = sections[i].lower()
            section_body = sections[i + 1]
            report_match = re.search(r'--- FULL CE REPORT.*?---', section_body, re.DOTALL)
            if not report_match:
                continue
            report_text = section_body[report_match.end():].split('---')[0]
            data_points = []
            for line in report_text.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        filename = parts[0]
                        quant_match = re.search(r'-((?:IQ|Q)\d+[\w_]*?)\.gguf$', filename, re.IGNORECASE)
                        quant_name = quant_match.group(1) if quant_match else "unknown"
                        bpw_val = float(parts[1])
                        ce_val = float(parts[-1])
                        data_points.append({"quant": quant_name, "bpw": bpw_val, "ce": ce_val})
                    except ValueError:
                        continue
            data_points.append({"quant": "bf16", "bpw": 16.0, "ce": 0.0})
            data_points.sort(key=lambda x: x['bpw'])
            tensor_data[tensor_name] = data_points
        return tensor_data

    def solve_mckp(self, stages, budget, scale=10000):
        max_w = int(round(budget * scale))
        n = len(stages)
        dp = np.full(max_w + 1, np.inf)
        dp[0] = 0.0
        parent = np.full((n, max_w + 1), -1, dtype=int)
        for i, stage in enumerate(stages):
            next_dp = np.full(max_w + 1, np.inf)
            for w in range(max_w + 1):
                if dp[w] == np.inf:
                    continue
                for opt_idx, opt in enumerate(stage["options"]):
                    cost = int(round(opt["bpw"] * stage["param_weight"] * scale))
                    nw = w + cost
                    if nw <= max_w:
                        val = dp[w] + opt["ce"]
                        if val < next_dp[nw]:
                            next_dp[nw] = val
                            parent[i][nw] = opt_idx
            dp = next_dp
        best_idx = int(np.argmin(dp))
        best_ce = dp[best_idx]
        if best_ce == np.inf:
            return None, None, None
        combo = []
        w = best_idx
        for i in range(n - 1, -1, -1):
            opt_idx = parent[i][w]
            opt = stages[i]["options"][opt_idx]
            combo.append({"tensor": stages[i]["name"], "quant": opt["quant"], "bpw": opt["bpw"], "ce": opt["ce"]})
            w -= int(round(opt["bpw"] * stages[i]["param_weight"] * scale))
        combo.reverse()
        return best_ce, best_idx / scale, combo

    def generate_moq_tensor_files(self, ce_results):
        valid_tensors = [name for name in ce_results if name in self.param_counts]
        total_params = sum(self.param_counts[name] for name in valid_tensors)
        if total_params == 0:
            raise ValueError("No valid param counts found.")
        stages = []
        for name in valid_tensors:
            options = ce_results[name]
            weight = self.param_counts[name] / total_params
            stages.append({"name": name, "options": options, "param_weight": weight})
        layer_tensors = self._load_layer_tensors()
        generated_files = []
        block_size = self.layer_block_size
        for bits in self.bits_list:
            best_ce, budget_used, combo = self.solve_mckp(stages, bits, scale=10000)
            if best_ce is None:
                continue
            suffix_to_quant = {item['tensor']: item['quant'] for item in combo}
            tensor_file_content = []
            for fq_name in layer_tensors:
                clean = fq_name.replace('.weight', '').replace('.bias', '')
                layer_match = re.match(r'^blk\.(\d+)\.', clean)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    base_name = re.sub(r'^blk\.\d+\.', '', clean)
                    group_start = (layer_idx // block_size) * block_size
                    group_end = group_start + block_size - 1
                    hybrid_key = f"{base_name}_layer_{group_start}-{group_end}"
                    if hybrid_key in suffix_to_quant:
                        tensor_file_content.append(f"{fq_name}={suffix_to_quant[hybrid_key]}")
                    elif base_name in suffix_to_quant:
                        tensor_file_content.append(f"{fq_name}={suffix_to_quant[base_name]}")
                    else:
                        tensor_file_content.append(f"{fq_name}=bf16")
                else:
                    if clean in suffix_to_quant:
                        tensor_file_content.append(f"{fq_name}={suffix_to_quant[clean]}")
                    else:
                        tensor_file_content.append(f"{fq_name}=bf16")
            tensor_filename = f"MoQ_{self.model_name}_tensors_{bits}.txt"
            tensor_path = os.path.join(self.output_dir, tensor_filename)
            with open(tensor_path, 'w') as f:
                f.write('\n'.join(tensor_file_content) + '\n')
            generated_files.append({'bits': bits, 'path': tensor_path})
        return generated_files

    def run_quantization(self, tensor_file_path, output_name):
        out_file = os.path.join(self.output_dir, f"{output_name}.gguf")
        cmd = [
            f"{self.llama_bin_dir}/llama-quantize", "--imatrix", self.imatrix_path,
            "--tensor-type-file", tensor_file_path, self.base_gguf_path, out_file, 'Q8_0'
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        return out_file

    def run(self):
        ce_results = self.parse_ce_report(self.ce_results_file)
        generated_files = self.generate_moq_tensor_files(ce_results)
        if self.dry_run:
            return
        if not self.base_gguf_path or not self.imatrix_path:
            return
        for file_info in generated_files:
            output_name = f"MoQ-{self.model_name}-{file_info['bits']}"
            self.run_quantization(file_info['path'], output_name)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION G: MAP TENSORS (from map_tensors.py)
# ═══════════════════════════════════════════════════════════════════════════════

import requests

GGUF_MAGIC = 0x46554747

def _resolve_token(cli_token):
    if cli_token:
        return cli_token
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

def _raise_for_status(resp, url):
    if resp.status_code in (200, 206):
        return
    if resp.status_code == 401:
        raise RuntimeError(f"HTTP 401 Unauthorized fetching: {url}\nCheck repo/filename spelling or token.")
    if resp.status_code == 404:
        raise RuntimeError(f"HTTP 404 Not Found fetching: {url}")
    raise RuntimeError(f"HTTP {resp.status_code} fetching: {url}")

class _GGUFRemoteReader:
    _SCALAR = {
        0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
        4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('<?', 1),
        10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
    }
    def __init__(self, repo_id, filename, revision="main", token=None):
        self.url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "moq-tensor-mapper/1.0"})
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self._buf = bytearray()
    def _ensure(self, need):
        have = len(self._buf)
        if need <= have:
            return
        fetch_to = max(need, have + 2 * 1024 * 1024)
        r = self.session.get(self.url, headers={"Range": f"bytes={have}-{fetch_to - 1}"}, stream=True, timeout=30)
        _raise_for_status(r, self.url)
        self._buf.extend(r.content)
    def _read(self, off, n):
        self._ensure(off + n)
        return bytes(self._buf[off:off + n])
    def _u32(self, off):
        return struct.unpack('<I', self._read(off, 4))[0], off + 4
    def _u64(self, off):
        return struct.unpack('<Q', self._read(off, 8))[0], off + 8
    def _str(self, off):
        length, off = self._u64(off)
        return self._read(off, length).decode('utf-8'), off + length
    def _skip_value(self, off, vtype):
        if vtype in self._SCALAR:
            return off + self._SCALAR[vtype][1]
        if vtype == 8:
            length, off = self._u64(off)
            return off + length
        if vtype == 9:
            arr_type, off = self._u32(off)
            arr_len, off = self._u64(off)
            for _ in range(arr_len):
                off = self._skip_value(off, arr_type)
            return off
        raise ValueError(f"Unknown GGUF metadata type {vtype}")
    def read_header(self):
        off = 0
        magic, off = self._u32(off)
        if magic != GGUF_MAGIC:
            raise ValueError("Not a GGUF file.")
        version, off = self._u32(off)
        tensor_count, off = self._u64(off)
        kv_count, off = self._u64(off)
        print(f"  GGUF v{version} | {tensor_count} tensors | {kv_count} metadata keys", file=sys.stderr)
        architecture = None
        for _ in range(kv_count):
            key, off = self._str(off)
            vtype, off = self._u32(off)
            if key == "general.architecture" and vtype == 8:
                architecture, off = self._str(off)
            else:
                off = self._skip_value(off, vtype)
        names = []
        for _ in range(tensor_count):
            name, off = self._str(off)
            n_dims, off = self._u32(off)
            for _ in range(n_dims):
                _, off = self._u64(off)
            _, off = self._u32(off)
            _, off = self._u64(off)
            names.append(name)
        return architecture, names

def _fetch_gguf_tensor_names(repo, filename, revision, token):
    print(f"\nGGUF  -> {repo}/{filename}", file=sys.stderr)
    reader = _GGUFRemoteReader(repo, filename, revision, token)
    architecture, names = reader.read_header()
    if not architecture:
        raise RuntimeError("Could not find 'general.architecture' key.")
    print(f"  architecture: {architecture} | {len(names)} tensors fetched", file=sys.stderr)
    return architecture, names

def _fetch_transformers_tensor_names(repo, revision, token):
    print(f"\nPyTorch -> {repo}", file=sys.stderr)
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, revision=revision, token=token)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    names = [n for n, _ in model.named_parameters()] + [n for n, _ in model.named_buffers()]
    print(f"  {len(names)} tensors fetched (no weights downloaded)", file=sys.stderr)
    return sorted(names)

def _match_via_tensor_mapping(gguf_names, hf_names, architecture):
    import gguf
    arch_name_to_enum = {name: enum for enum, name in gguf.MODEL_ARCH_NAMES.items()}
    arch_enum = arch_name_to_enum.get(architecture)
    if arch_enum is None:
        raise RuntimeError(f"Architecture '{architecture}' not recognized.")
    gguf_set = set(gguf_names)
    block_ids = [int(n.split(".")[1]) for n in gguf_names if n.startswith("blk.") and n.split(".")[1].isdigit()]
    n_blocks = (max(block_ids) + 1) if block_ids else 0
    name_map = gguf.TensorNameMap(arch_enum, n_blocks)
    gguf_to_hf = {}
    for hf_name in hf_names:
        gguf_name = name_map.get_name(hf_name, try_suffixes=(".weight", ".bias"))
        if gguf_name is not None and gguf_name in gguf_set:
            gguf_to_hf[gguf_name] = hf_name
    lines = []
    for gguf_name in sorted(gguf_names):
        hf_name = gguf_to_hf.get(gguf_name)
        if hf_name:
            lines.append(f"{gguf_name} = {hf_name}")
        else:
            lines.append(f"# {gguf_name} = ???  # <-- UNMATCHED")
    return lines

def _build_fixed_lines(mapping_lines):
    fixed = []
    for line in mapping_lines:
        stripped = line.strip()
        if stripped.startswith('#') or not stripped or '=' not in line:
            fixed.append(line)
            continue
        gguf_name, hf_name = line.split('=', 1)
        hf_name = hf_name.strip()
        if hf_name.endswith('.weight'):
            hf_name = hf_name[:-7]
        elif hf_name.endswith('.bias'):
            hf_name = hf_name[:-5]
        hf_name = hf_name.replace('model.language_model.', 'model.', 1)
        fixed.append(f"{hf_name} = {gguf_name.strip()}")
    return fixed

def run_map_tensors(args):
    token = _resolve_token(args.token)
    architecture, gguf_names = _fetch_gguf_tensor_names(args.gguf_repo, args.gguf_file, args.gguf_revision, token)
    hf_names = _fetch_transformers_tensor_names(args.hf_repo, args.hf_revision, token)
    mapping = _match_via_tensor_mapping(gguf_names, hf_names, architecture)
    fixed_lines = _build_fixed_lines(mapping)
    with open(args.output, "w") as f:
        f.write("# PyTorch_Module_Name = GGUF_Name\n" + "\n".join(fixed_lines) + "\n")
    print(f"\nDone: mapped -> {args.output}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION H: BUILD LLAMA.CPP (from build_llama_cpp_cpu.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_command(cmd, check=True, capture_output=False):
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip() if capture_output else None

def _find_latest_release(repo):
    print(f"🔍 Finding latest {repo} release...")
    result = subprocess.run(
        ["curl", "-s", f"https://api.github.com/repos/{repo}/releases/latest"],
        capture_output=True, text=True
    )
    latest = json.loads(result.stdout)
    build = latest["tag_name"]
    print(f"✅ Latest release: {build}")
    return latest, build

def _find_ubuntu_asset(assets, repo, build):
    asset_names = [a["name"] for a in assets]
    print(f"   Available assets: {[a for a in asset_names if 'ubuntu' in a.lower()]}")
    exact_match_file = f"llama-{build}-bin-ubuntu-x64.tar.gz"
    if exact_match_file in asset_names:
        return exact_match_file
    for asset in assets:
        name = asset["name"]
        if ("ubuntu" in name.lower() and "x64" in name.lower() and
            "cuda" not in name.lower() and "vulkan" not in name.lower() and
            "openvino" not in name.lower()):
            return name
    raise RuntimeError("No suitable Ubuntu x64 asset found")

def _download_file(url, output_path):
    cmd = f'wget --show-progress "{url}" -O "{output_path}"'
    print(f"Downloading: {url}")
    _run_command(cmd, check=True)

def _extract_tarball(tarball_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    _run_command(f'tar -xzf "{tarball_path}" -C "{extract_dir}"', check=True)

def _fix_nested_structure(extract_dir):
    _run_command(f'find {extract_dir} -mindepth 2 -type f -name "*.so*" -exec mv -n {{}} {extract_dir}/ \\;')
    _run_command(f'find {extract_dir} -mindepth 2 -type f -name "llama-*" -exec mv -n {{}} {extract_dir}/ \\;')
    _run_command(f'chmod +x {extract_dir}/llama-*')

def _install_system_libs(extract_dir, system_lib_dir):
    _run_command(f'cp -n {extract_dir}/*.so* {system_lib_dir} 2>/dev/null || true')
    _run_command('ldconfig 2>/dev/null || true')

def _verify_installation(extract_dir):
    binaries = ['llama-quantize', 'llama-imatrix']
    missing = []
    for binary in binaries:
        path = os.path.join(extract_dir, binary)
        if not os.path.exists(path):
            missing.append(binary)
    if missing:
        print(f"\n⚠️  Some binaries not found: {missing}")
        print("Listing all binaries:")
        _run_command(f'ls {extract_dir}/llama-*')
        return False
    print(f"\n✅ SETUP COMPLETE — llama-imatrix and llama-quantize are ready")
    return True

def run_build_llama(args):
    print(f"\n=== llama.cpp CPU Build ===")
    print(f"Repo       : {args.repo}")
    print(f"Output dir : {args.output_dir}")
    print(f"Source dir : {args.source_dir}")
    latest, build = _find_latest_release(args.repo)
    assets = latest.get('assets', [])
    try:
        filename = _find_ubuntu_asset(assets, args.repo, build)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("\nAll available assets:")
        for a in assets:
            print(f"  - {a['name']}")
        return 1
    print(f"✅ Downloading: {filename}")
    url = f"https://github.com/{args.repo}/releases/download/{build}/{filename}"
    tarball_path = os.path.join('/tmp', filename)
    if not args.skip_download or not os.path.exists(tarball_path):
        _download_file(url, tarball_path)
    else:
        print(f"Using existing tarball: {tarball_path}")
    if not os.path.exists(tarball_path) or os.path.getsize(tarball_path) < 1000:
        raise RuntimeError(f"❌ Download failed or file too small: {filename}")
    print(f"✅ Downloaded {os.path.getsize(tarball_path)/1e6:.1f} MB")
    print("\n=== Extracting ===")
    _extract_tarball(tarball_path, args.output_dir)
    _fix_nested_structure(args.output_dir)
    if args.install_system_libs:
        print("\n=== Installing System Libraries ===")
        _install_system_libs(args.output_dir, args.system_lib_dir)
    print("\n=== Verification ===")
    print("Binaries extracted:")
    _run_command(f'ls {args.output_dir}/llama-* | head -20')
    success = _verify_installation(args.output_dir)
    if not args.no_clone_source and not os.path.exists(args.source_dir):
        print(f"\n📥 Cloning llama.cpp source to {args.source_dir}...")
        _run_command(f'git clone https://github.com/{args.repo} {args.source_dir}')
        print(f"✅ Source cloned")
        requirements_file = os.path.join(args.source_dir, 'requirements.txt')
        if os.path.exists(requirements_file):
            print("\n📦 Installing Python requirements...")
            _run_command(f'uv pip install --index-strategy unsafe-best-match -r {requirements_file}')
            print("✅ Requirements installed")
    print("\n✅ SETUP COMPLETE. Binaries and scripts are ready.")
    print(f"   Binaries: {args.output_dir}")
    print(f"   Source:   {args.source_dir}")
    return 0 if success else 1
