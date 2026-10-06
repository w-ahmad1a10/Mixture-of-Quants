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

from moq.moq_core import (
    parse_mapping_file, parse_quant_config, GGML_TYPE_MAP,
    compile_cpp_extension
)


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


def load_model(model_name, dtype, device, trust_remote_code=False, token=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        token=token,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
    )
    if torch.cuda.device_count() == 1:
        model = model.to(device)
    return model


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
