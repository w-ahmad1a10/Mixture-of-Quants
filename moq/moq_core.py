#!/usr/bin/env python3
"""
moq_core.py
Shared core utilities for MoQ pipeline and testing.
Zero logic changes — extracted from original analyze_kld.py and eval.py.
"""
import argparse, gc, json, math, os, random, re, subprocess, sys, glob, warnings, time, hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.cpp_extension import load
from tqdm import tqdm
from datasets import load_dataset

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
GGML_TYPE_MAP = {
    "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8,
    "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
    "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19,
    "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23, "IQ1_M": 29,
    "Q1_0": 41, "Q2_0": 42
}

TEACHER_TOPK = 1000

SKIP_PATTERNS = (
    'norm', 'ln', 'rotary', 'bias', 'input_layernorm', 'ssm_conv1d', 'conv1d',
    'post_attention_layernorm', 'norm.weight', 'ln_f', 'ln_1', 'ln_2', 'conv.conv'
)

# ═══════════════════════════════════════════════════════════════════════════════
# C++ EXTENSION COMPILER
# ═══════════════════════════════════════════════════════════════════════════════
def compile_cpp_extension(llama_cpp_dir: str):
    llama_cpp_dir = os.path.abspath(llama_cpp_dir)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_candidates = [
        os.path.join(base_dir, "quant_extension.cpp"),
        os.path.join(os.getcwd(), "quant_extension.cpp"),
        "quant_extension.cpp",
    ]
    sources = None
    for s in source_candidates:
        if os.path.exists(s):
            sources = [s]
            break
    if sources is None:
        sys.exit("quant_extension.cpp not found!")
    # Locate the ggml headers wherever they actually live. Different builds
    # clone to different paths and older trees put ggml.h directly in
    # ggml/src, so don't hard-code ggml/include — search a list of candidate
    # dirs under the given llama.cpp dir, and if none has it, walk up the
    # directory tree looking for a llama.cpp checkout that does.
    header_candidates = [
        f"{llama_cpp_dir}/ggml/include",
        f"{llama_cpp_dir}/include",
        f"{llama_cpp_dir}/ggml/src",
        f"{llama_cpp_dir}/ggml",
        f"{llama_cpp_dir}/include/ggml",
        f"{llama_cpp_dir}/include/gguf",
    ]
    if not any(os.path.exists(os.path.join(c, "ggml.h")) for c in header_candidates):
        parent = os.path.dirname(llama_cpp_dir)
        for _ in range(5):
            if not parent or parent == os.path.dirname(parent):
                break
            for sub in ("llama.cpp", "llama"):
                candidate = os.path.join(parent, sub)
                if any(os.path.exists(os.path.join(candidate, d, "ggml.h"))
                       for d in ("ggml/include", "include", "ggml/src", "")):
                    header_candidates = [
                        os.path.join(candidate, "ggml/include"),
                        os.path.join(candidate, "include"),
                        os.path.join(candidate, "ggml/src"),
                        candidate,
                    ]
                    break
            parent = os.path.dirname(parent)
    include_dirs = [c for c in header_candidates if os.path.isdir(c)] or header_candidates
    search_paths = ["/usr/lib", "/usr/local/lib", "/lib", "/usr/lib/x86_64-linux-gnu"]
    # Pre-built llama.cpp binaries live next to the source checkout, named
    # <source_dir's parent>/llama_bin. Derive it from llama_cpp_dir so we
    # don't have to guess the path (Colab uses /content/workspace, local
    # builds use /workspace, etc.).
    derived_bin_dir = os.path.join(os.path.dirname(llama_cpp_dir), "llama_bin")
    search_paths = [derived_bin_dir] + search_paths
    search_paths.extend([
        os.path.join(derived_bin_dir, "llama-b8121"),
        os.path.join(derived_bin_dir, "llama-b8121", "lib"),
        os.path.join(derived_bin_dir, "lib"),
    ])
    build_dir = os.path.join(llama_cpp_dir, "build")
    if os.path.isdir(build_dir):
        search_paths.extend([
            build_dir,
            os.path.join(build_dir, "bin"),
            os.path.join(build_dir, "lib"),
            os.path.join(build_dir, "ggml", "src"),
            os.path.join(build_dir, "ggml", "src", "ggml"),
        ])
    libs = []
    lib_dirs = []
    for path in search_paths:
        if os.path.isdir(path):
            found = glob.glob(os.path.join(path, "libggml*.so*"))
            if found:
                libs.extend(found)
                lib_dirs.append(path)
    ldflags = []
    seen = set()
    # Add -L flags for every directory that actually holds a libggml*.so,
    # otherwise the linker reports "cannot find -lggml" even though the
    # -l flags are present.
    for d in lib_dirs:
        ldflags.append(f"-L{d}")
    # Add -L flags for build directories so linker can find .so files
    if os.path.isdir(build_dir):
        ldflags.append(f"-L{build_dir}/ggml/src")
        ldflags.append(f"-L{build_dir}/ggml/src/ggml")
        ldflags.append(f"-L{build_dir}/lib")
    for lib in libs:
        name = os.path.basename(lib)
        if name.startswith("lib") and (name.endswith(".so") or name.endswith(".so.0")):
            lib_name = name[3:].split('.')[0]
            if lib_name not in seen:
                seen.add(lib_name)
                ldflags.append(f"-l{lib_name}")
    if not ldflags:
        sys.exit("No libggml*.so files found!")
    ldflags.append("-lgomp")
    # Copy the libggml shared objects into the system library directories so
    # the linker can resolve -lggml-base / -lggml even when the build's -L
    # flags aren't on the default search path. This is what the CPU smoke
    # script does before compiling the extension.
    for directory in ("/usr/lib", "/usr/local/lib"):
        for lib in libs:
            target = os.path.join(directory, os.path.basename(lib))
            if os.path.exists(target):
                continue
            try:
                subprocess.run(["cp", "-n", lib, directory], check=False)
            except Exception:
                pass
    subprocess.run(["ldconfig"], check=False)
    cflags = ["-O3", "-std=c++17", "-fPIC", "-Wno-unused-function", "-fopenmp"]
    return load(
        name="llama_quant_ext", sources=sources, extra_include_paths=include_dirs,
        extra_cflags=cflags, extra_ldflags=ldflags, verbose=False
    )

# ═══════════════════════════════════════════════════════════════════════════════
# KLD MATH
# ═══════════════════════════════════════════════════════════════════════════════
def compute_token_kld(student_logits, teacher_log_probs, eval_mask=None, attention_mask=None, return_cpu=True):
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    mask = teacher_log_probs > -16.0
    teacher_probs = torch.exp(teacher_log_probs) * mask
    kld_contrib = teacher_probs * (teacher_log_probs - student_log_probs)
    kld_per_token = kld_contrib.sum(dim=-1)
    kld_per_token = torch.clamp(kld_per_token, min=0.0)

    valid_klds = []
    batch_size = kld_per_token.size(0)
    for i in range(batch_size):
        m = eval_mask[i] if eval_mask is not None else attention_mask[i].bool()
        if m.any():
            valid_klds.append(kld_per_token[i][m])

    if valid_klds:
        result = torch.cat(valid_klds)
        if return_cpu:
            return result.cpu()
        return result
    return torch.tensor([], dtype=torch.float32)

def extract_assistant_log_probs(log_probs, eval_mask=None, attention_mask=None):
    assistant_log_probs = []
    batch_size = log_probs.size(0)
    for i in range(batch_size):
        if eval_mask is not None:
            m = eval_mask[i].bool()
        else:
            m = attention_mask[i].bool()
        if m.any():
            assistant_log_probs.append(log_probs[i][m])
    return assistant_log_probs

# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET BATCH SAMPLER (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
class BucketBatchSampler(Sampler):
    """
    Groups samples by length into buckets to minimize padding within batches.
    """
    def __init__(self, lengths, batch_size, shuffle=False, seed=42, drop_last=False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        
        # Define bucket boundaries (covers up to 8192 tokens)
        boundaries = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192]
        self.buckets = {b: [] for b in boundaries}
        
        for idx, length in enumerate(self.lengths):
            assigned = False
            for b in boundaries:
                if length <= b:
                    self.buckets[b].append(idx)
                    assigned = True
                    break
            if not assigned:
                # Fallback for extremely long sequences
                self.buckets[boundaries[-1]].append(idx)
                
        # Remove empty buckets
        self.buckets = {k: v for k, v in self.buckets.items() if v}
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        batches = []
        # Sort bucket keys to ensure deterministic order when shuffle=False
        sorted_keys = sorted(self.buckets.keys())
        for b in sorted_keys:
            indices = self.buckets[b]
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]
            
            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
                
        if self.shuffle:
            # Shuffle the order of the batches themselves
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
            
        for batch in batches:
            yield batch
            
    def __len__(self):
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total

# ═══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
class ChatDataset(Dataset):
    def __init__(self, dataset_repo, split, tokenizer, max_samples=None, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        print(f"Loading dataset {dataset_repo} ({split})...")
        ds = load_dataset(dataset_repo, split=split)
        if max_samples and max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))
        for row in tqdm(ds, desc="Formatting multi-turn chats", disable=False):
            convs = row.get("conversations", [])
            if not convs:
                continue
            messages = []
            for m in convs:
                role = m.get("from", "")
                if role == "human":
                    messages.append({"role": "user", "content": m["value"]})
                elif role == "gpt":
                    messages.append({"role": "assistant", "content": m["value"]})
            if not messages or not any(m["role"] == "assistant" for m in messages):
                continue
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(full_ids) > max_seq_length:
                full_ids = full_ids[:max_seq_length]
            eval_mask = torch.zeros(len(full_ids), dtype=torch.bool)
            prev_len = 0
            for i, msg in enumerate(messages):
                partial_msgs = messages[:i + 1]
                partial_text = tokenizer.apply_chat_template(partial_msgs, tokenize=False, add_generation_prompt=False)
                partial_ids = tokenizer.encode(partial_text, add_special_tokens=False)[:max_seq_length]
                if msg["role"] == "assistant" and prev_len < len(partial_ids):
                    start = prev_len
                    end = min(len(partial_ids), len(full_ids))
                    eval_mask[start:end] = True
                prev_len = len(partial_ids)
                if prev_len >= len(full_ids):
                    break
            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "eval_mask": eval_mask[:len(full_ids)]
            })
        # Track lengths for bucketing
        self.lengths = [len(s["input_ids"]) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def chat_collate_fn(batch, tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    eval_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = 1
        eval_mask[i, :seq_len] = item["eval_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "eval_mask": eval_mask}

class CalibrationDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=512):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []
        all_ids = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        for i in range(0, len(all_ids), seq_length):
            chunk = all_ids[i:i + self.seq_length]
            if len(chunk) == self.seq_length:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
            elif len(chunk) > self.seq_length // 2:
                chunk = chunk + [tokenizer.pad_token_id] * (self.seq_length - len(chunk))
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
        # Track lengths for bucketing
        self.lengths = [len(s) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx]}

class HFDataset(Dataset):
    def __init__(self, hf_repo, config, split, tokenizer, seq_length=512,
                 max_documents=1000, text_column="text", token=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []
        all_ids = []
        ds = load_dataset(hf_repo, config, split=split, streaming=True, token=token)
        count = 0
        for row in ds:
            if count >= max_documents:
                break
            text = row.get(text_column, "")
            if not text or not isinstance(text, str):
                count += 1
                continue
            all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            count += 1
        for i in range(0, len(all_ids), seq_length):
            chunk = all_ids[i:i + seq_length]
            if len(chunk) == seq_length:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
            elif len(chunk) > seq_length // 2:
                chunk = chunk + [tokenizer.pad_token_id] * (seq_length - len(chunk))
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
        self.lengths = [len(s) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx]}


def text_collate_fn(batch, tokenizer):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# ═══════════════════════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════════════════════
def parse_mapping_file(path: str):
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue
            left, right = line.split('=', 1)
            mapping[left.strip()] = right.strip()
    return mapping

def parse_quant_config(path: str) -> Dict[str, str]:
    config = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue
            left, right = line.split('=', 1)
            config[left.strip()] = right.strip()
    return config

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER (shared between eval and quant)
# ═══════════════════════════════════════════════════════════════════════════════
def load_model(model_path, torch_dtype, trust_remote_code, hf_token, device):
    try:
        import accelerate
        has_accelerate = True
    except ImportError:
        has_accelerate = False
    if has_accelerate and device.type == "cuda":
        print(f"  [Loader] Using accelerate device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True, device_map="auto", token=hf_token
        )
        input_device = next(model.parameters()).device
    else:
        print(f"  [Loader] Loading on {device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True, token=hf_token
        )
        model.to(device)
        input_device = device
    model.eval()
    model.config.use_cache = False
    print(f"  [Loader] Ready: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, input_device

#!/usr/bin/env python3
"""
moq_engine.py
All pipeline stages and evaluation engines in one file.
"""
import argparse, gc, json, math, os, random, re, sys, glob, warnings, time, hashlib, threading, queue
from collections import defaultdict
from pathlib import Path
import struct
import subprocess
from typing import Dict, List, Tuple
import shutil
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


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
        if w.dim() == 2 and (w.shape[0] < 8 or w.shape[1] < 8):
            continue
        if any(pat in pt_name.lower() for pat in SKIP_PATTERNS) or any(pat in gguf_name.lower() for pat in SKIP_PATTERNS):
            continue
        candidates.append((pt_name, module, w.numel()))
    return sorted(candidates, key=lambda x: x[0])

def group_by_tensor_type(candidates):
    groups = defaultdict(list)
    for name, module, n_params in candidates:
        parts = name.split('.')
        if len(parts) >= 2 and parts[-2] in ('self_attn', 'conv', 'feed_forward', 'mlp'):
            tensor_type = f"{parts[-2]}_{parts[-1]}"
        else:
            tensor_type = parts[-1]
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
                if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
                    tensor_type = '_'.join(parts[-2:])
                else:
                    tensor_type = parts[-1]
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
        elif getattr(args, 'hf_dataset_repo', None):
            self.dataset = HFDataset(
                args.hf_dataset_repo, getattr(args, 'hf_dataset_config', None),
                getattr(args, 'hf_dataset_split', 'train'), self.tokenizer,
                args.seq_length,
                max_documents=getattr(args, 'hf_dataset_max_docs', 1000),
                text_column=getattr(args, 'hf_dataset_text_column', 'text'),
                token=getattr(args, 'hf_token', None),
            )
            sampler = BucketBatchSampler(
                self.dataset.lengths, batch_size=args.chunks_at_once, shuffle=False, seed=args.seed
            )
            self.dataloader = DataLoader(
                self.dataset, batch_sampler=sampler,
                collate_fn=lambda b: text_collate_fn(b, self.tokenizer),
                pin_memory=(self.device.type == "cuda"), num_workers=4
            )
        else:
            if not args.calib_data:
                sys.exit("FATAL: Must provide either --dataset-repo, --hf-dataset-repo, or --calib-data")
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
        ds_name = (self.args.hf_dataset_repo or self.args.dataset_repo or os.path.basename(self.args.calib_data or "text")).replace("/", "__")
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
                cached_batches = torch.load(cache_path, map_location="cpu", weights_only=False)
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
                del logits
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
        overlap_enabled = getattr(self.args, "overlap", False)
        quant_type_order = self.args.quant_types
        with torch.inference_mode():
            # --- Optional pipelined pre-quantization (depth-1) ---
            # A background thread pre-quantizes the NEXT quant type for the SAME group
            # while the main thread runs the forward pass on the CURRENT quant type.
            # At most one pre-quantized result is ever in flight (depth-1).
            preq_q = queue.Queue(maxsize=2)
            preq_stop = threading.Event()
            def _prequantize_worker():
                while not preq_stop.is_set():
                    try:
                        item = preq_q.get_nowait()
                    except queue.Empty:
                        preq_stop.wait(0.05)
                        continue
                    if item is None:
                        break
                    (qt, un, origs, is_iq, imatrix_str) = item
                    ggml_type_id = GGML_TYPE_MAP[qt]
                    out = {}
                    for name in list(origs.keys()):
                            gguf_name = self.mapping[name]
                            q_weight = self.quant_ext.apply_llama_quant_noise(
                                origs[name], gguf_name, ggml_type_id, imatrix_str
                            )
                            out[name] = q_weight.cpu()
                    try:
                        preq_q.put((qt, un, out), block=False)
                    except queue.Full:
                        pass  # result not consumed yet; main thread will quantize on the fly
            preq_thread = None
            if overlap_enabled:
                preq_thread = threading.Thread(target=_prequantize_worker, daemon=True)
                preq_thread.start()
            for quant_type in quant_type_order:
                t_quant_start = time.time()
                ggml_type_id = GGML_TYPE_MAP[quant_type]
                imatrix_str = str(self.args.imatrix) if self.args.imatrix else ""
                is_iq = quant_type.startswith("IQ")
                for unit_name, tensor_names in tqdm(groups.items(), desc=f"Quant {quant_type}"):
                    originals = {}
                    skipped = 0
                    # ① Snapshot originals to CPU in one batched transfer (was N round trips)
                    for name in tensor_names:
                        module = self._module_lookup[name]
                        gguf_name = self.mapping[name]
                        if is_iq:
                            if not imatrix_str or not self.quant_ext.has_imatrix(gguf_name, imatrix_str):
                                skipped += 1
                                continue
                        originals[name] = module.weight.data.detach().cpu()
                    if not originals:
                        continue
                    # ② Dispatch the NEXT quant type for this group to the background thread
                    if overlap_enabled and quant_type != quant_type_order[-1]:
                        idx_qt = quant_type_order.index(quant_type)
                        next_qt = quant_type_order[idx_qt + 1]
                        if not (next_qt.startswith("IQ") and (not imatrix_str or any(
                            not self.quant_ext.has_imatrix(self.mapping[n], imatrix_str) for n in originals
                        ))):
                            try:
                                preq_q.put_nowait((next_qt, unit_name, dict(originals), next_qt.startswith("IQ"), imatrix_str))
                            except queue.Full:
                                pass  # pipeline full; this group's next quant will be quantized on the fly
                    # ③ Consume any pre-quantized result for this (quant_type, unit_name)
                    preq_result = None
                    if overlap_enabled:
                        while True:
                            try:
                                cand = preq_q.get_nowait()
                            except queue.Empty:
                                break
                            if cand[0] == quant_type and cand[1] == unit_name:
                                preq_result = cand[2]
                            else:
                                try:
                                    preq_q.put(cand, block=False)
                                except queue.Full:
                                    pass  # stale result, discard
                    # ④ Quantize current quant type for the whole group
                    if preq_result is not None:
                        for name, q_weight in preq_result.items():
                            self._module_lookup[name].weight.data = q_weight.to(self._module_lookup[name].weight.device).to(self._module_lookup[name].weight.dtype)
                    else:
                        for name in list(originals.keys()):
                            module = self._module_lookup[name]
                            gguf_name = self.mapping[name]
                            try:
                                q_weight = self.quant_ext.apply_llama_quant_noise(
                                    originals[name], gguf_name, ggml_type_id, imatrix_str
                                )
                                module.weight.data = q_weight.to(module.weight.device).to(module.weight.dtype)
                            except Exception as e:
                                skipped += 1
                                print(f"  [WARN] Skipping {name}: {e}")
                    if skipped == len(tensor_names):
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
                        device = student_logits.device
                        del student_logits
                        student_assistant_log_probs = extract_assistant_log_probs(
                            student_log_probs, eval_mask, attention_mask
                        )
                        if student_assistant_log_probs and teacher_assistant_log_probs:
                            for i, (teacher_vals, teacher_inds) in enumerate(teacher_assistant_log_probs):
                                teacher_vals = teacher_vals.to(device, non_blocking=True)
                                teacher_inds = teacher_inds.to(device, non_blocking=True).long()
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
            if overlap_enabled:
                preq_stop.set()
                # drain any remaining pre-quantized results (they are for groups already processed)
                while True:
                    try:
                        preq_q.get_nowait()
                    except queue.Empty:
                        break
                preq_thread.join(timeout=30)
                del preq_q, preq_stop, preq_thread
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
            calib = getattr(self.args, 'calib_data', None)
            hf_repo = getattr(self.args, 'hf_dataset_repo', None)
            if hf_repo:
                self.dataset = HFDataset(
                    hf_repo,
                    getattr(self.args, 'hf_dataset_config', None),
                    getattr(self.args, 'hf_dataset_split', 'train'),
                    self.tokenizer,
                    seq_length=getattr(self.args, 'seq_length', 512),
                    max_documents=getattr(self.args, 'hf_dataset_max_docs', 1000),
                    text_column=getattr(self.args, 'hf_dataset_text_column', 'text'),
                    token=getattr(self.args, 'hf_token', None),
                )
            else:
                with open(calib, 'r', encoding='utf-8') as f:
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
                cached_batches = torch.load(cache_path, map_location="cpu", weights_only=False)
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
            if any(p in full_name.lower() for p in SKIP_PATTERNS) or \
               any(p in gguf_name.lower() for p in SKIP_PATTERNS):
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: TENSOR REGISTRY — single source of truth for all 4 modes
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_tensor_map(tensor_map_path):
    tensor_map = {}
    with open(tensor_map_path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue
            pt_name, gguf_name = line.split('=', 1)
            tensor_map[pt_name.strip()] = gguf_name.strip()
    return tensor_map


def _get_ce_key(pt_name):
    parts = pt_name.split('.')
    if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
        return '_'.join(parts[-2:])
    return parts[-1]


def _get_layer_idx(gguf_name):
    m = re.match(r'^blk\.(\d+)\.', gguf_name)
    return int(m.group(1)) if m else None


class TensorRegistry:
    def __init__(self, tensor_map, param_counts, n_blocks):
        self.registry = {}
        self.by_pt_name = {}
        self.by_gguf_name = {}
        self.by_ce_key = defaultdict(list)
        self.by_layer_idx = defaultdict(list)
        self.param_counts = dict(param_counts)
        self.n_blocks = n_blocks

        for pt_name, gguf_name in tensor_map.items():
            ce_key = _get_ce_key(pt_name)
            layer_idx = _get_layer_idx(gguf_name)
            tensor_type = GGUFMetadataExtractor._get_tensor_type(gguf_name)
            if layer_idx is not None and n_blocks > 0:
                layer_key = f"{tensor_type}:{layer_idx}"
                param_count = int(param_counts.get(layer_key, param_counts.get(tensor_type, 0)) / n_blocks)
            else:
                param_count = int(param_counts.get(tensor_type, 0))
            entry = {
                'pt_name': pt_name,
                'gguf_name': gguf_name,
                'ce_key': ce_key,
                'layer_idx': layer_idx,
                'tensor_type': tensor_type,
                'param_count': param_count,
            }
            self.registry[pt_name] = entry
            self.by_pt_name[pt_name] = entry
            self.by_gguf_name[gguf_name] = entry
            self.by_ce_key[ce_key].append(entry)
            if layer_idx is not None:
                self.by_layer_idx[layer_idx].append(entry)

    @classmethod
    def from_files(cls, tensor_map_path, gguf_repo, gguf_file, token=None):
        tensor_map = _parse_tensor_map(tensor_map_path)
        extractor = GGUFMetadataExtractor(gguf_repo, gguf_file, token=token)
        param_counts, n_blocks = extractor.extract()
        return cls(tensor_map, param_counts, n_blocks)

    def get_entry(self, pt_name):
        return self.registry.get(pt_name)

    def get_by_gguf_name(self, gguf_name):
        return self.by_gguf_name.get(gguf_name)

    def entries_by_ce_key(self, ce_key):
        return self.by_ce_key.get(ce_key, [])

    def entries_by_layer_range(self, start, end):
        result = []
        for idx in range(start, end + 1):
            result.extend(self.by_layer_idx.get(idx, []))
        return result


def _group_key_for(pt_name, registry, mode, block_size=4):
    entry = registry.registry.get(pt_name)
    if entry is None:
        return pt_name
    layer_idx = entry['layer_idx']
    ce_key = entry['ce_key']

    if mode == 'individual':
        return pt_name
    elif mode == 'by-tensor-type':
        return ce_key
    elif mode == 'by-layer':
        if layer_idx is not None:
            start = (layer_idx // block_size) * block_size
            end = start + block_size - 1
            return f"layer_{start}-{end}"
        return ce_key
    elif mode == 'by-layer-tensor':
        if layer_idx is not None:
            start = (layer_idx // block_size) * block_size
            end = start + block_size - 1
            return f"{ce_key}_layer_{start}-{end}"
        return ce_key
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_group_param_counts(registry, mode, block_size=4):
    group_params = defaultdict(int)
    for pt_name, entry in registry.registry.items():
        if any(pat in pt_name.lower() for pat in SKIP_PATTERNS) or any(pat in entry['gguf_name'].lower() for pat in SKIP_PATTERNS):
            continue
        gk = _group_key_for(pt_name, registry, mode, block_size)
        group_params[gk] += entry['param_count']
    return dict(group_params)


def build_group_tensors(registry, mode, block_size=4):
    group_tensors = defaultdict(list)
    for pt_name, entry in registry.registry.items():
        if any(pat in pt_name.lower() for pat in SKIP_PATTERNS) or any(pat in entry['gguf_name'].lower() for pat in SKIP_PATTERNS):
            continue
        gk = _group_key_for(pt_name, registry, mode, block_size)
        group_tensors[gk].append(entry['gguf_name'])
    return dict(group_tensors)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F: OPTIMIZE & QUANTIZE (from optimize_quantize.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_compute_ce(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_json) as f:
        results = json.load(f)
    registry = TensorRegistry.from_files(args.tensor_map, args.gguf_repo, args.gguf_file, token=args.token)

    sample_key = next(iter(results.keys())) if results else ""
    if '/' in sample_key:
        _, unit = sample_key.split('/', 1)
    else:
        unit = sample_key
    if '_layer_' in unit:
        mode = 'by-layer-tensor'
    elif unit.startswith('layer_'):
        mode = 'by-layer'
    elif '.' in unit:
        mode = 'individual'
    else:
        mode = 'by-tensor-type'

    block_size = getattr(args, 'layer_block_size', 4)
    group_param_counts = build_group_param_counts(registry, mode, block_size)

    data = []
    for key, metrics in results.items():
        qt, unit = key.split('/', 1)
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
        final_param_counts[unit] = group_param_counts.get(unit, 0)
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
        self.mapping_path = getattr(args, 'mapping', None)
        self.mode = getattr(args, 'quant_mode', 'by-tensor-type')
        with open(args.param_counts, 'r') as f:
            self.param_counts = json.load(f)

        self.group_tensors = {}
        if self.mapping_path and os.path.exists(self.mapping_path):
            tensor_map = _parse_tensor_map(self.mapping_path)
            registry = TensorRegistry(tensor_map, {}, 16)
            self.group_tensors = build_group_tensors(registry, self.mode, self.layer_block_size)

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
        sections = re.split(r'={50,}\s+TENSOR TYPE : ([^\n]+)', content)
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
            quantized = set()
            for group_key, quant in suffix_to_quant.items():
                full_names = self._resolve_group_tensors(group_key)
                for fq_name in full_names:
                    tensor_file_content.append(f"{fq_name}={quant}")
                    quantized.add(fq_name)

            for fq_name in layer_tensors:
                if fq_name in quantized:
                    continue
                if any(pat in fq_name.lower() for pat in SKIP_PATTERNS):
                    continue
                tensor_file_content.append(f"{fq_name}=bf16")

            tensor_filename = f"MoQ_{self.model_name}_tensors_{bits}.txt"
            tensor_path = os.path.join(self.output_dir, tensor_filename)
            with open(tensor_path, 'w') as f:
                f.write('\n'.join(tensor_file_content) + '\n')
            generated_files.append({'bits': bits, 'path': tensor_path})
        return generated_files

    def _resolve_group_tensors(self, group_key):
        if not self.group_tensors:
            if getattr(self, '_resolved_group_tensors', None) is None:
                self._resolved_group_tensors = {}
                if self.mapping_path and os.path.exists(self.mapping_path):
                    tensor_map = _parse_tensor_map(self.mapping_path)
                    registry = TensorRegistry(tensor_map, {}, 16)
                    for pt_name, entry in registry.registry.items():
                        if any(pat in pt_name.lower() for pat in SKIP_PATTERNS) or any(pat in entry['gguf_name'].lower() for pat in SKIP_PATTERNS):
                            continue
                        gk = _group_key_for(pt_name, registry, self.mode, self.layer_block_size)
                        self._resolved_group_tensors.setdefault(gk, []).append(entry['gguf_name'])
            return self._resolved_group_tensors.get(group_key, [])
        if self.mode == 'by-layer-tensor' and group_key not in self.group_tensors:
            ce_key = group_key
            m = re.match(r'^(.*)_layer_\d+-\d+$', group_key)
            if m:
                ce_key = m.group(1)
            for candidate in (ce_key, f"layer_{group_key}"):
                if candidate in self.group_tensors:
                    return self.group_tensors[candidate]
        return self.group_tensors.get(group_key, [])

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

#!/usr/bin/env python3
"""
eval_mmlu.py
MMLU zero-shot final eval for MoQ pipeline.
Loads LFM2.5-230M from HuggingFace, caches teacher choice-probs,
then evaluates each quant config on MMLU.
"""
import argparse, gc, json, math, os, random, sys, glob, hashlib
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset



class MMLUEvalDataset:
    def __init__(self, tokenizer, subjects=None, max_samples=None):
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.questions = []
        self._load_questions(subjects)

    def _load_questions(self, subjects):
        dataset = load_dataset("cais/mmlu", "all", split="test")
        for i, row in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            if subjects and row.get("subject") not in subjects:
                continue

            question_text = row["question"].strip()
            choices = row["choices"]
            answer_idx = int(row["answer"])

            prompt = (
                f"{question_text}\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
                f"Answer:"
            )

            base_enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = base_enc["input_ids"][0]
            attention_mask = base_enc["attention_mask"][0]

            answer_enc = self.tokenizer("Answer:", return_tensors="pt", add_special_tokens=False)
            answer_ids = answer_enc["input_ids"][0]

            answer_start = None
            for pos in range(len(input_ids) - len(answer_ids) + 1):
                if torch.all(input_ids[pos:pos + len(answer_ids)] == answer_ids):
                    answer_start = pos + len(answer_ids)
                    break

            if answer_start is None or answer_start > len(input_ids):
                continue

            choice_tokens = []
            for letter in ["A", "B", "C", "D"]:
                enc = self.tokenizer(" " + letter, return_tensors="pt", add_special_tokens=False)
                choice_tokens.append(enc["input_ids"][0])

            max_choice_len = max(len(tok) for tok in choice_tokens)
            if answer_start + max_choice_len > len(input_ids):
                answer_start = len(input_ids) - max_choice_len
            if answer_start < 0:
                continue

            self.questions.append({
                "question_id": i,
                "subject": row.get("subject", "unknown"),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "answer_start": answer_start,
                "choice_tokens": choice_tokens,
                "answer_idx": answer_idx,
            })

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        return {
            "input_ids": q["input_ids"],
            "attention_mask": q["attention_mask"],
            "answer_start": q["answer_start"],
            "choice_tokens": q["choice_tokens"],
            "answer_idx": q["answer_idx"],
            "question_id": q["question_id"],
            "subject": q["subject"],
        }


class MMLUEvaluator:
    def __init__(self, model_name, tokenizer, device, subjects=None,
                 max_samples=None, batch_size=8, cache_dir="teacher_cache",
                 skip_cache=False):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.subjects = subjects
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.skip_cache = skip_cache
        os.makedirs(cache_dir, exist_ok=True)
        self.dataset = MMLUEvalDataset(tokenizer, subjects, max_samples)
        self.cached_teacher = None
        self.input_device = None

    def _get_cache_path(self):
        model_safe = self.model_name.replace("/", "__")
        subs = "_".join(self.subjects or ["all"])[:50]
        cfg_str = f"{model_safe}_mmlu_s{subs}_m{self.max_samples or 0}_b{self.batch_size}"
        h = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"mmlu_teacher_{h}.pt")

    def cache_teacher_logits(self, model):
        cache_path = self._get_cache_path()
        self.input_device = next(model.parameters()).device

        if not self.skip_cache and os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location=self.input_device, weights_only=False)
                if isinstance(data, list) and len(data) == len(self.dataset):
                    self.cached_teacher = data
                    print(f"[MMLU] Loaded teacher cache: {cache_path}")
                    return
            except Exception:
                pass

        cached = []
        model.eval()
        with torch.inference_mode():
            for start in tqdm(range(0, len(self.dataset), self.batch_size), desc="[MMLU] Teacher forward"):
                end = min(start + self.batch_size, len(self.dataset))
                batch = [self.dataset[i] for i in range(start, end)]

                max_len = max(q["input_ids"].size(0) for q in batch)
                ids = torch.stack([
                    F.pad(q["input_ids"], (0, max_len - q["input_ids"].size(0)), value=self.tokenizer.pad_token_id)
                    for q in batch
                ]).to(self.input_device)
                mask = torch.stack([
                    F.pad(q["attention_mask"], (0, max_len - q["attention_mask"].size(0)), value=0)
                    for q in batch
                ]).to(self.input_device)

                logits = model(input_ids=ids, attention_mask=mask).logits
                log_probs = F.log_softmax(logits, dim=-1)

                for bi, q in enumerate(batch):
                    answer_start = q["answer_start"]
                    vocab_size = log_probs.size(-1)
                    lp_choices = []
                    for tok in q["choice_tokens"]:
                        if tok.numel() == 1:
                            tid = tok.item()
                            if tid < vocab_size and answer_start < log_probs.size(1):
                                lp_choices.append(log_probs[bi, answer_start, tid].item())
                            else:
                                lp_choices.append(float("-inf"))
                        else:
                            s = 0.0
                            ok = True
                            for ti, tid in enumerate(tok):
                                if answer_start + ti < log_probs.size(1):
                                    s += log_probs[bi, answer_start + ti, tid].item()
                                else:
                                    ok = False
                                    break
                            lp_choices.append(s if ok else float("-inf"))

                    probs = F.softmax(torch.tensor(lp_choices), dim=0)
                    cached.append({
                        "probs": probs,
                        "answer_idx": q["answer_idx"],
                        "subject": q["subject"],
                    })

        if not self.skip_cache:
            try:
                torch.save(cached, cache_path)
                print(f"[MMLU] Saved teacher cache: {cache_path}")
            except Exception as e:
                print(f"[MMLU] Warning: could not save cache ({e})")

        self.cached_teacher = cached
        print(f"[MMLU] Teacher cached ({len(cached)} questions)")

    def evaluate_student(self, model, label="student"):
        if self.cached_teacher is None:
            raise RuntimeError("MMLU Teacher not cached. Call cache_teacher_logits() first.")

        kld_list = []
        ce_sum = 0.0
        ce_tokens = 0
        correct = 0
        agreement = 0
        total = 0

        model.eval()
        with torch.inference_mode():
            for start in tqdm(range(0, len(self.dataset), self.batch_size), desc=f"[MMLU] {label}"):
                end = min(start + self.batch_size, len(self.dataset))
                batch = [self.dataset[i] for i in range(start, end)]

                max_len = max(q["input_ids"].size(0) for q in batch)
                ids = torch.stack([
                    F.pad(q["input_ids"], (0, max_len - q["input_ids"].size(0)), value=self.tokenizer.pad_token_id)
                    for q in batch
                ]).to(self.input_device)
                mask = torch.stack([
                    F.pad(q["attention_mask"], (0, max_len - q["attention_mask"].size(0)), value=0)
                    for q in batch
                ]).to(self.input_device)

                logits = model(input_ids=ids, attention_mask=mask).logits
                log_probs = F.log_softmax(logits, dim=-1)

                for bi, q in enumerate(batch):
                    answer_start = q["answer_start"]
                    vocab_size = log_probs.size(-1)

                    lp_choices = []
                    for tok in q["choice_tokens"]:
                        if tok.numel() == 1:
                            tid = tok.item()
                            if tid < vocab_size and answer_start < log_probs.size(1):
                                lp_choices.append(log_probs[bi, answer_start, tid].item())
                            else:
                                lp_choices.append(float("-inf"))
                        else:
                            s = 0.0
                            ok = True
                            for ti, tid in enumerate(tok):
                                if answer_start + ti < log_probs.size(1):
                                    s += log_probs[bi, answer_start + ti, tid].item()
                                else:
                                    ok = False
                                    break
                            lp_choices.append(s if ok else float("-inf"))

                    student_probs = F.softmax(torch.tensor(lp_choices, device=self.input_device), dim=0)
                    teacher_probs = self.cached_teacher[start + bi]["probs"].to(self.input_device)

                    t = torch.clamp(teacher_probs, min=1e-10)
                    s = torch.clamp(student_probs, min=1e-10)
                    kld = torch.sum(teacher_probs * torch.log(t / s))
                    kld_list.append(kld.item())

                    teacher_choice = torch.argmax(teacher_probs).item()
                    student_choice = torch.argmax(student_probs).item()
                    if student_choice == q["answer_idx"]:
                        correct += 1
                    if teacher_choice == student_choice:
                        agreement += 1
                    total += 1

                    answer_token_id = q["choice_tokens"][q["answer_idx"]]
                    if answer_token_id.numel() == 1:
                        ce_sum += -log_probs[bi, answer_start, answer_token_id.item()].item()
                        ce_tokens += 1
                    else:
                        for ti, tid in enumerate(answer_token_id):
                            if answer_start + ti < log_probs.size(1):
                                ce_sum += -log_probs[bi, answer_start + ti, tid].item()
                                ce_tokens += 1

        kld_t = torch.tensor(kld_list)
        ppl = math.exp(ce_sum / ce_tokens) if ce_tokens > 0 else float("inf")

        return {
            "mean_kld": kld_t.mean().item(),
            "median_kld": kld_t.median().item(),
            "max_kld": kld_t.max().item(),
            "p90_kld": float(torch.quantile(kld_t, 0.90).item()) if len(kld_t) > 1 else kld_t.item(),
            "p99_kld": float(torch.quantile(kld_t, 0.99).item()) if len(kld_t) > 1 else kld_t.item(),
            "p99.9_kld": float(torch.quantile(kld_t, 0.999).item()) if len(kld_t) > 1 else kld_t.item(),
            "ppl": ppl,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
            "agreement": agreement / total if total > 0 else 0.0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2.5-230M")
    parser.add_argument("--mapping", type=str, required=True,
                        help="Path to tensor_map file (HF name -> GGUF name)")
    parser.add_argument("--quant-configs", type=str, required=True,
                        help="Glob pattern for quant config .txt files")
    parser.add_argument("--imatrix", type=str, default=None)
    parser.add_argument("--llama-cpp-dir", type=str, default="llama.cpp")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--mmlu-subjects", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="mmlu_results.json")
    parser.add_argument("--cache-dir", type=str, default="teacher_cache")
    parser.add_argument("--skip-teacher-cache", action="store_true")
    args = parser.parse_args()

    if args.mmlu_subjects:
        subjects = [s.strip() for s in args.mmlu_subjects.split(",")]
    else:
        subjects = None

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"\n{'='*70}")
    print(f"  MMLU FINAL EVAL")
    print(f"{'='*70}")
    print(f"Model    : {args.model}")
    print(f"Device   : {device}")
    print(f"Seed     : {args.seed}")
    print(f"Subjects : {subjects or 'all'}")
    print(f"Samples  : {args.max_samples or 'all'}")
    print(f"Batch    : {args.batch_size}")

    print("\n[1] Loading model...")
    model = load_model(
        args.model, dtype, device,
        getattr(args, 'trust_remote_code', False),
        getattr(args, 'hf_token', None)
    )

    print("[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=getattr(args, 'trust_remote_code', False),
        token=getattr(args, 'hf_token', None)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[3] Setting up MMLU evaluator...")
    evaluator = MMLUEvaluator(
        model_name=args.model,
        tokenizer=tokenizer,
        device=device,
        subjects=subjects,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        skip_cache=args.skip_teacher_cache,
    )

    print("[4] Caching teacher logits...")
    evaluator.cache_teacher_logits(model)

    print("\n[5] Baseline teacher eval...")
    teacher_metrics = evaluator.evaluate_student(model, label="teacher_BF16")
    print(f"\n  Teacher (BF16):")
    print(f"    Accuracy : {teacher_metrics['accuracy']:.4f} ({teacher_metrics['correct']}/{teacher_metrics['total']})")
    print(f"    PPL      : {teacher_metrics['ppl']:.4f}")
    print(f"    Mean KLD : {teacher_metrics['mean_kld']:.6f}")
    print(f"    Median   : {teacher_metrics['median_kld']:.6f}")
    print(f"    Max KLD  : {teacher_metrics['max_kld']:.6f}")
    print(f"    P90      : {teacher_metrics['p90_kld']:.6f}")
    print(f"    P99      : {teacher_metrics['p99_kld']:.6f}")
    print(f"    P99.9    : {teacher_metrics['p99.9_kld']:.6f}")
    print(f"    Agree    : {teacher_metrics['agreement']:.4f}")

    config_files = sorted(glob.glob(args.quant_configs))
    if not config_files:
        print(f"[Err] No config files found: {args.quant_configs}")
        return 1

    print(f"\n[6] Evaluating {len(config_files)} quant configs...")
    mapping = parse_mapping_file(args.mapping)
    quant_ext = compile_cpp_extension(args.llama_cpp_dir)
    imatrix_str = str(args.imatrix) if args.imatrix else ""

    results = {"model": args.model, "teacher": teacher_metrics, "configs": []}

    for cfg_path in config_files:
        cfg_name = os.path.basename(cfg_path)
        print(f"\n  ── {cfg_name} ──")
        quant_cfg = parse_quant_config(cfg_path)

        originals = []
        quantized = 0
        for full_name, module in tqdm(list(model.named_modules()), desc="Quantize", leave=False):
            if not hasattr(module, 'weight') or not isinstance(module.weight, nn.Parameter):
                continue
            gguf_name = mapping.get(full_name)
            if not gguf_name:
                continue
            qt = quant_cfg.get(gguf_name)
            if not qt or qt not in GGML_TYPE_MAP:
                continue
            try:
                w_cpu = module.weight.data.detach().cpu()
                q_weight = quant_ext.apply_llama_quant_noise(
                    w_cpu, gguf_name, GGML_TYPE_MAP[qt], imatrix_str
                )
                originals.append((module, module.weight.data))
                module.weight.data = q_weight.to(module.weight.device).to(module.weight.dtype)
                quantized += 1
            except Exception:
                pass

        print(f"    Quantized: {quantized}")
        metrics = evaluator.evaluate_student(model, label=cfg_name)

        for module, orig in originals:
            module.weight.data = orig
        print(f"    Restored")

        print(f"    Accuracy : {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        print(f"    PPL      : {metrics['ppl']:.4f}")
        print(f"    Mean KLD : {metrics['mean_kld']:.6f}")
        print(f"    Median   : {metrics['median_kld']:.6f}")
        print(f"    Max KLD  : {metrics['max_kld']:.6f}")
        print(f"    P90      : {metrics['p90_kld']:.6f}")
        print(f"    P99      : {metrics['p99_kld']:.6f}")
        print(f"    P99.9    : {metrics['p99.9_kld']:.6f}")
        print(f"    Agree    : {metrics['agreement']:.4f}")

        results["configs"].append({"config_file": cfg_name, "config_path": cfg_path, **metrics})

        del originals
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
runlog.py — Local-only run logger. Replaces wandb entirely.
Repos ONLY write files here; the dashboard does everything else.

    run = RunLog("glu-silu-100L", config={"lr": 3e-4})   # creates Logs/<name>/
    run.log({"loss": 4.2, "lr": 3e-4}, step=100)
    run.log_table("grad_rms", columns=["tensor","rms"], rows=[[...],[...]])
    run.finish()

Writes: Logs/<run_name>/metrics.jsonl , config.yaml , tables/<name>.csv
"""
import csv, json, os, time
from datetime import datetime


class RunLog:
    def __init__(self, run_name=None, config=None, base_dir=None):
        if base_dir is None:
            base_dir = os.path.join("Logs")
        run_name = run_name or "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        # unique-ify
        self.run_dir = os.path.join(base_dir, run_name)
        i = 2
        while os.path.exists(self.run_dir):
            self.run_dir = os.path.join(base_dir, f"{run_name}_{i}")
            i += 1
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "tables"), exist_ok=True)

        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")
        self._step = 0
        self._t0 = time.time()
        self._finished = False

        cfg = dict(config or {})
        cfg.setdefault("_meta", {})
        cfg["_meta"].update({
            "run_name": os.path.basename(self.run_dir),
            "started_utc": datetime.utcnow().isoformat() + "Z",
            "logger": "runlog/1.0 (local dashboard system)",
        })
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            try:
                import yaml
                yaml.safe_dump(cfg, f, default_flow_style=False)
            except Exception:
                json.dump(cfg, f, indent=2, default=str)
        print(f"[runlog] logging to {self.run_dir}")

    # ---- core ----
    def log(self, metrics: dict, step: int | None = None):
        if self._finished:
            return
        self._step = self._step + 1 if step is None else step
        entry = {"step": self._step,
                 "timestamp": time.time(),
                 **{k: v for k, v in metrics.items()}}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_table(self, name: str, columns: list, rows: list):
        """Replaces wandb.Table. Saved as CSV next to metrics; dashboard shows it."""
        path = os.path.join(self.run_dir, "tables", f"{name}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            w.writerows(rows)

    def log_artifact(self, local_path: str, copy_into_run: bool = True):
        """Register a file produced by the run. Small files are copied into the run dir."""
        if copy_into_run and os.path.getsize(local_path) < 50 * 1024 * 1024:
            import shutil
            shutil.copy2(local_path, os.path.join(self.run_dir, os.path.basename(local_path)))
        with open(os.path.join(self.run_dir, "artifacts.txt"), "a") as f:
            f.write(local_path + "\n")

    def finish(self):
        if not self._finished:
            self.log({"_runtime_seconds": time.time() - self._t0, "_finished": True})
            self._finished = True
            print(f"[runlog] finished — view in dashboard: {self.run_dir}")

    # ---- wandb-style aliases so old code needs minimal edits ----
    def init(self, *a, **k):          # wandb.init(...) -> already done in ctor
        return self

    @staticmethod
    def Table(columns, rows):         # wandb.Table(...) -> pass straight to log_table
        return {"columns": list(columns), "rows": [list(r) for r in rows]}



#!/usr/bin/env python3
"""
moq_sweep.py
Run all 4 grouping variants, produce PyTorch mixed-precision models,
evaluate them, save ALL results, upload ONLY results to HF Hub.
NO GGUF generation. Models are deleted after eval; results are kept forever.

Now supports multiple evaluation datasets, local runlog logging, and global teacher cache.
"""
import argparse, gc, json, math, os, random, re, sys, glob, shutil, time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


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

class MoQRunner:
    # CHANGE 1: Shared global cache root for final evaluation caches
    FINAL_EVAL_CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "moq_final_eval")

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

        # ─── Parse evaluation configs ──────────────────────────────────────────────
        self.eval_configs = self._parse_eval_configs(args)
        # ─── Load teacher model once and cache logits for all eval datasets ──────
        self.teacher_model = None
        self.evaluators = []
        self._prepare_teacher_cache()

    def _parse_eval_configs(self, args):
        """Build list of eval config dicts from --eval-configs JSON or fallback to KLD dataset."""
        if args.eval_configs:
            configs = json.loads(args.eval_configs)
            for cfg in configs:
                cfg_type = cfg.get("type", "chat" if cfg.get("dataset_repo") else "text")
                if cfg_type == "mmlu":
                    cfg.setdefault("name", cfg.get("name", "mmlu"))
                    cfg.setdefault("batch_size", args.mmlu_batch_size if hasattr(args, 'mmlu_batch_size') else 8)
                    cfg.setdefault("max_samples", None)
                else:
                    cfg.setdefault("split", "train")
                    cfg.setdefault("max_samples", args.max_samples)
                    cfg.setdefault("max_seq_length", args.max_seq_length)
                    cfg.setdefault("seq_length", args.seq_length)
                    cfg.setdefault("num_chunks", args.num_chunks)
                    cfg.setdefault("batch_size", args.batch_size)
                    cfg.setdefault("hf_dataset_repo", cfg.get("hf_dataset_repo"))
                    cfg.setdefault("hf_dataset_config", cfg.get("hf_dataset_config"))
                    cfg.setdefault("hf_dataset_split", cfg.get("hf_dataset_split", "train"))
                    cfg.setdefault("hf_dataset_max_docs", cfg.get("hf_dataset_max_docs", 1000))
                    cfg.setdefault("hf_dataset_text_column", cfg.get("hf_dataset_text_column", "text"))
                    if "name" not in cfg:
                        cfg["name"] = cfg.get("path", "unknown").replace("/", "_").replace(".txt", "")
            return configs
        else:
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

        # 2. For each eval config, create evaluator and cache teacher logits
        for cfg in self.eval_configs:
            cfg_type = cfg.get("type", "chat" if cfg.get("dataset_repo") else "text")

            if cfg_type == "mmlu":
                class MMluArgs:
                    mmlu_subjects = cfg.get("subjects")
                    max_samples = cfg.get("max_samples")
                    mmlu_batch_size = cfg.get("batch_size", 8)
                    mmlu_name = cfg.get("name", "mmlu")
                    teacher_cache_dir = os.path.join(self.results_root, "teacher_cache", "mmlu")
                    skip_teacher_cache = False

                evaluator = MMLUEvaluator(
                    model_name=self.args.model,
                    tokenizer=self.tokenizer,
                    device=input_device,
                    subjects=cfg.get("subjects"),
                    max_samples=cfg.get("max_samples"),
                    batch_size=cfg.get("batch_size", 8),
                    # CHANGE 1: Use shared global cache root instead of results_root
                    cache_dir=os.path.join(self.FINAL_EVAL_CACHE_ROOT, "mmlu"),
                    skip_cache=False,
                )
                evaluator.dataset_name = cfg.get("name", "mmlu")
            else:
                eval_args = self._build_eval_args(cfg)
                evaluator = Evaluator(eval_args)
                evaluator.input_device = input_device
                evaluator.dataset_name = cfg.get("name", cfg.get("path"))

            evaluator.cache_teacher_logits(self.teacher_model)
            self.evaluators.append(evaluator)

        print(f"[Teacher] Cached logits for {len(self.evaluators)} datasets.")

    def _build_eval_args(self, cfg):
        """Build argparse.Namespace for Evaluator from a config dict."""
        a = argparse.Namespace(**vars(self.args))
        a.model = self.args.model
        a.calib_data = cfg.get("path") if cfg["type"] == "text" else None
        a.dataset_repo = cfg.get("path") if cfg["type"] == "chat" else None
        a.dataset_split = cfg.get("split", "train")
        a.max_samples = cfg.get("max_samples")
        a.max_seq_length = cfg.get("max_seq_length", 2048)
        a.seq_length = cfg.get("seq_length", 512)
        a.num_chunks = cfg.get("num_chunks", 50)
        a.batch_size = cfg.get("batch_size", 1)
        a.hf_dataset_repo = cfg.get("hf_dataset_repo")
        a.hf_dataset_config = cfg.get("hf_dataset_config")
        a.hf_dataset_split = cfg.get("hf_dataset_split", "train")
        a.hf_dataset_max_docs = cfg.get("hf_dataset_max_docs", 1000)
        a.hf_dataset_text_column = cfg.get("hf_dataset_text_column", "text")
        # Unique cache dir per dataset to avoid collisions.
        # HF-dataset configs have no "path"; fall back to their repo id.
        _name_key = cfg.get("name") or cfg.get("path") or cfg.get("hf_dataset_repo") or "dataset"
        ds_name = _name_key.replace("/", "_")
        # CHANGE 1: Use shared global cache root instead of results_root
        a.teacher_cache_dir = os.path.join(self.FINAL_EVAL_CACHE_ROOT, ds_name)
        a.skip_teacher_cache = False  # We want to cache
        return a

    # ──────────────────── CHANGE 2: Offload final eval caches to CPU ─────────────

    def _offload_final_eval_caches_to_cpu(self):
        """Move only the final evaluation cached batches to CPU to free VRAM during KLD analysis."""
        for evaluator in self.evaluators:
            if not hasattr(evaluator, 'cached_batches'):
                continue
            for batch in evaluator.cached_batches:
                # Move input_ids and attention_mask to CPU
                if "input_ids" in batch and batch["input_ids"].is_cuda:
                    batch["input_ids"] = batch["input_ids"].cpu()
                if "attention_mask" in batch and batch["attention_mask"].is_cuda:
                    batch["attention_mask"] = batch["attention_mask"].cpu()
                # eval_mask can be None; only move if it's a tensor
                if "eval_mask" in batch and batch["eval_mask"] is not None and batch["eval_mask"].is_cuda:
                    batch["eval_mask"] = batch["eval_mask"].cpu()

                # Move the sparse teacher log-probs (vals, inds) to CPU
                if "teacher_assistant_log_probs" in batch:
                    new_probs = []
                    for vals, inds in batch["teacher_assistant_log_probs"]:
                        new_probs.append((vals.cpu(), inds.cpu()))
                    batch["teacher_assistant_log_probs"] = new_probs
        print("[Memory] Offloaded final eval caches to CPU.")

    # ──────────────────── KLD, CE, Optimize ──────────────────────────────────────

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
        a.dry_run = self.args.dry_run
        a.mapping = self.args.mapping
        a.quant_mode = mode
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

    # ──────────────────── APPLY+EVAL (with multiple datasets, global cache) ────

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
            if any(p in full_name.lower() for p in SKIP_PATTERNS) or \
               any(p in gguf_name.lower() for p in SKIP_PATTERNS):
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

        # CHANGE 3: Clean up memory from quantized weights that were just replaced
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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

        # ── 5. Local logging (runlog → dashboard) ──
        run_name = self.args.run_name or f"{mode}_{bits}bpw"
        run = RunLog(
            run_name=run_name,
            config={
                "mode": mode,
                "bits": bits,
                "quant_config": os.path.basename(config_path),
                "eval_datasets": self.eval_configs,
            }
        )
        for ds_name, metrics in all_metrics.items():
            if metrics and "error" not in metrics:
                run.log({f"eval/{ds_name}/{k}": v for k, v in metrics.items()})
            else:
                run.log({f"eval/{ds_name}/error": 1})
        run.finish()
        print(f"  [runlog] Logged run: {run_name}")

        # ── 6. Upload results ──
        if self.args.hf_upload_repo:
            _upload_results_to_hf(
                self.results_root, self.args.hf_upload_repo,
                self.args.hf_token, path_in_repo=f"results_{mode}_{bits}"
            )

        return result

    # ──────────────────── RUN SINGLE MODE ────────────────────────────────────────

    def run_single_mode(self, mode: str):
        print(f"\n{'#'*70}")
        print(f"#  SWEEPING MODE: {mode}")
        print(f"{'#'*70}")

        if getattr(self.args, "skip_kld", False):
            kld_json = os.path.join(self.results_root, f"analysis_{mode}.json")
            print(f"[skip-kld] Reusing {kld_json}")
        else:
            kld_json = self.run_kld(mode)

        if getattr(self.args, "skip_ce", False):
            ce_dir = os.path.join(self.results_root, f"ce_{mode}")
            print(f"[skip-ce] Reusing {ce_dir}")
        else:
            ce_dir = self.run_ce(mode)

        if getattr(self.args, "skip_optimize", False):
            configs_dir = os.path.join(self.results_root, f"configs_{mode}")
            configs = []
            for p in sorted(glob.glob(os.path.join(configs_dir, "*.txt"))):
                m = re.search(r"_(\d+\.\d+)\.txt$", os.path.basename(p))
                configs.append({"path": p, "bits": float(m.group(1)) if m else 0.0})
            print(f"[skip-optimize] Loaded {len(configs)} configs from {configs_dir}")
        else:
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

    def _run_from_configs_dir(self):
        """Apply each pre-made .txt config in --configs-dir, evaluate on all cached datasets."""
        cfg_dir = self.args.configs_dir
        if not os.path.isdir(cfg_dir):
            raise FileNotFoundError(f"--configs-dir not found: {cfg_dir}")
        cfg_files = sorted(glob.glob(os.path.join(cfg_dir, "*.txt")))
        if not cfg_files:
            raise ValueError(f"No .txt configs in {cfg_dir}")

        self._offload_final_eval_caches_to_cpu()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        all_results = {}
        for cfg_path in cfg_files:
            stem = os.path.splitext(os.path.basename(cfg_path))[0]
            m = re.search(r"(\d+\.\d+)", stem)
            bits = float(m.group(1)) if m else 0.0
            print(f"\n{'='*70}\n  [configs-dir] {stem} @ {bits} bpw\n{'='*70}")
            all_results[stem] = self.apply_and_evaluate(cfg_path, bits, mode=stem)

        master_path = os.path.join(self.results_root, "master_summary.json")
        with open(master_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[configs-dir] master_summary.json -> {master_path}")

        if self.args.hf_upload_repo:
            _upload_results_to_hf(
                self.results_root, self.args.hf_upload_repo,
                self.args.hf_token, path_in_repo="final_results"
            )
        return all_results

    def run(self):
        # Expand the "ALL" shorthand to the full quant-type toolbox
        # (excludes Q1_0/Q2_0, matching the original experiment's 19 types).
        if self.args.quant_types == ['ALL'] or self.args.quant_types == 'ALL':
            self.args.quant_types = [k for k in GGML_TYPE_MAP if k not in ('Q1_0', 'Q2_0')]

        # ── Pre-made configs: skip KLD/CE/optimize, apply+evaluate only ──
        if getattr(self.args, "configs_dir", None):
            return self._run_from_configs_dir()

        all_modes = ['by-tensor-type', 'by-layer', 'individual', 'by-layer-tensor']
        if getattr(self.args, 'mode', None):
            modes = [self.args.mode]
        else:
            modes = self.args.modes.split(',') if self.args.modes else all_modes
        modes = [m.strip() for m in modes if m.strip() in all_modes]
        if not modes:
            modes = all_modes

        # ─── After teacher cache is prepared, offload it to CPU before KLD ───
        # CHANGE 2: Offload final eval caches to CPU now
        self._offload_final_eval_caches_to_cpu()

        # CHANGE 3: Clean up GPU memory after offloading
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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

def upload_results(local_path, hf_path, repo_id, token=None):
    """Upload a local file or folder to an HF dataset repo.

    Usage:
      from moq.moq_core import upload_results
      upload_results("results/", "goal1/results/", "w-ahmad/moq-lfm25-230m-outputs")
    """
    from pathlib import Path
    from huggingface_hub import HfApi

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"local path does not exist: {local_path}")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        raise ValueError("No HF token provided and no token file found at ~/.cache/huggingface/token")

    api = HfApi(token=token)
    if local_path.is_file():
        url = api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
    elif local_path.is_dir():
        hf_path = hf_path.rstrip("/")
        url = api.upload_folder(
            folder_path=str(local_path),
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
    else:
        raise ValueError(f"unsupported path type: {local_path}")
    print(f"[Upload] {local_path} -> {repo_id}:{hf_path}")
    return url
