#!/usr/bin/env python3
"""
moq_core.py
Shared core utilities for MoQ pipeline and testing.
Zero logic changes — extracted from original analyze_kld.py and eval.py.
"""
import argparse, gc, json, math, os, random, re, sys, glob, warnings, time, hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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
    'post_attention_layernorm', 'norm.weight', 'ln_f', 'ln_1', 'ln_2'
)

# ═══════════════════════════════════════════════════════════════════════════════
# C++ EXTENSION COMPILER
# ═══════════════════════════════════════════════════════════════════════════════
def compile_cpp_extension(llama_cpp_dir: str):
    llama_cpp_dir = os.path.abspath(llama_cpp_dir)
    sources = ["quant_extension.cpp"] if os.path.exists("quant_extension.cpp") else ["Mixture-of-Quants/quant_extension.cpp"]
    include_dirs = [
        f"{llama_cpp_dir}/ggml/include",
        f"{llama_cpp_dir}/include",
        f"{llama_cpp_dir}/ggml/src"
    ]
    search_paths = ["/usr/lib", "/usr/local/lib", "/lib", "/usr/lib/x86_64-linux-gnu"]
    libs = []
    for path in search_paths:
        libs.extend(glob.glob(os.path.join(path, "libggml*.so")))
    ldflags = []
    seen = set()
    for lib in libs:
        name = os.path.basename(lib)
        if name.startswith("lib") and name.endswith(".so"):
            lib_name = name[3:-3]
            if lib_name not in seen:
                seen.add(lib_name)
                ldflags.append(f"-l{lib_name}")
    if not ldflags:
        sys.exit("No libggml*.so files found!")
    ldflags.append("-lgomp")
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
