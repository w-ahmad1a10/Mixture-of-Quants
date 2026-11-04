"""Run basic affine quantization experiments and report KLD, PPL, and EAR."""
import argparse, gc, json, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moq.moq_core import SKIP_PATTERNS, load_model
from affine import quantize_tensor, layer_params, quantize_with_params

TEACHER_TOPK = 10


def build_text_samples(tokenizer, repo, config, split, seq_length, max_docs, num_chunks, hf_token=None):
    ds = load_dataset(repo, config, split=split, streaming=True, token=hf_token) if config else load_dataset(repo, split=split, streaming=True, token=hf_token)
    all_ids = []
    count = 0
    for row in ds:
        if count >= max_docs:
            break
        text = row.get("text", "")
        if text and isinstance(text, str):
            all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        count += 1
    samples = []
    for i in range(0, len(all_ids), seq_length):
        chunk = all_ids[i:i + seq_length]
        if len(chunk) == seq_length:
            samples.append(torch.tensor(chunk, dtype=torch.long))
        elif len(chunk) > seq_length // 2:
            chunk = chunk + [tokenizer.pad_token_id] * (seq_length - len(chunk))
            samples.append(torch.tensor(chunk, dtype=torch.long))
        if len(samples) >= num_chunks:
            break
    return samples


def eval_text(model, samples, tokenizer, device, cached=None):
    model.eval()
    ce_sum = 0.0
    ce_tokens = 0
    kld_vals = []
    ear_vals = []
    mass_vals = []
    with torch.inference_mode():
        for idx, s in enumerate(samples):
            ids = s.unsqueeze(0).to(device)
            mask = (ids != tokenizer.pad_token_id).long()
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits, dim=-1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous().bool()
            flat_l = shift_logits.view(-1, shift_logits.size(-1))
            flat_y = shift_labels.view(-1)
            flat_m = shift_mask.view(-1)
            if flat_m.any():
                ce = F.cross_entropy(flat_l[flat_m], flat_y[flat_m], reduction="sum")
                ce_sum += ce.item()
                ce_tokens += int(flat_m.sum().item())
            if cached is not None and idx < len(cached):
                tvals, tinds = cached[idx]
                tvals = tvals.to(device)
                tinds = tinds.to(device).long()
                if tinds.dim() == 2:
                    for i in range(tinds.size(0)):
                        s_lp = lp[0, i].gather(0, tinds[i])
                        t_p = torch.exp(tvals[i])
                        mask_t = tvals[i] > -16.0
                        t_p = t_p * mask_t
                        s_p = torch.exp(s_lp)
                        kld = (t_p * (tvals[i] - s_lp)).sum()
                        kld = torch.clamp(kld, min=0.0)
                        kld_vals.append(kld.item())
                        ear = torch.sum(torch.minimum(t_p, s_p)).item()
                        ear_vals.append(ear)
                        mass_vals.append(t_p.sum().item())
    ppl = math.exp(ce_sum / ce_tokens) if ce_tokens > 0 else float("inf")
    top10_mass = (sum(mass_vals) / len(mass_vals)) if mass_vals else 0.0
    ear_avg = (sum(ear_vals) / len(ear_vals)) if ear_vals else 0.0
    return {
        "mean_kld": (sum(kld_vals) / len(kld_vals)) if kld_vals else 0.0,
        "ppl": ppl,
        "ear": ear_avg,
        "top10_mass": top10_mass,
        "normalized_ear": (ear_avg / top10_mass) if top10_mass > 0 else 0.0,
        "tokens": ce_tokens,
    }


def eval_mmlu(model, tokenizer, device, max_samples, orig_probs=None, hf_token=None):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test", token=hf_token)
    correct = 0
    total = 0
    kld_sum = 0.0
    ce_sum = 0.0
    ce_tokens = 0
    with torch.inference_mode():
        for i, row in enumerate(ds):
            if total >= max_samples:
                break
            choices = row["choices"]
            prompt = (f"{row['question'].strip()}\n"
                      f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:")
            enc = tokenizer(prompt, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits, dim=-1)
            answer_ids = tokenizer("Answer:", add_special_tokens=False)["input_ids"]
            input_ids = ids[0]
            answer_start = None
            for pos in range(len(input_ids) - len(answer_ids) + 1):
                if all(input_ids[pos + j].item() == answer_ids[j] for j in range(len(answer_ids))):
                    answer_start = pos + len(answer_ids)
                    break
            if answer_start is None or answer_start >= logits.size(1):
                continue
            choice_tokens = []
            for letter in ["A", "B", "C", "D"]:
                t = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
                choice_tokens.append(t[0] if t else tokenizer.eos_token_id)
            lp_choices = [lp[0, answer_start, tid].item() for tid in choice_tokens]
            student_probs = F.softmax(torch.tensor(lp_choices, device=device), dim=0)
            pred = int(torch.argmax(student_probs).item())
            if pred == int(row["answer"]):
                correct += 1
            ans_tid = choice_tokens[int(row["answer"])]
            ce_sum += -lp[0, answer_start, ans_tid].item()
            ce_tokens += 1
            # KLD vs original if available
            if orig_probs is not None and total < len(orig_probs):
                op = orig_probs[total]
                sp = student_probs
                kld = torch.sum(op * torch.log(torch.clamp(op, 1e-10) / torch.clamp(sp, 1e-10))).item()
                kld_sum += kld
            total += 1
    ppl = math.exp(ce_sum / ce_tokens) if ce_tokens > 0 else float("inf")
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "ppl": ppl,
        "mean_kld": (kld_sum / total) if total else 0.0,
    }


def apply_quantization(model, mapping, bits, symmetric, granularity, group_size, layer_mode):
    originals = []
    # per layer: compute shared params across all quantizable weights in each layer
    if layer_mode:
        layers = {}
        for name, module in model.named_modules():
            if not hasattr(module, "weight"):
                continue
            gguf = mapping.get(name)
            if not gguf or any(p in name.lower() for p in SKIP_PATTERNS) or any(p in gguf.lower() for p in SKIP_PATTERNS):
                continue
            layer_key = name.split(".")[2] if "layers." in name else "other"
            layers.setdefault(layer_key, []).append((name, module))
        for lk, items in layers.items():
            weights = [m.weight.data for _, m in items]
            scale, zp = layer_params(weights, bits, symmetric)
            for name, module in items:
                originals.append((module, module.weight.data.clone()))
                q = quantize_with_params(module.weight.data.float(), bits, symmetric, scale, zp)
                module.weight.data = q.to(module.weight.dtype).to(module.weight.device)
    else:
        for name, module in model.named_modules():
            if not hasattr(module, "weight"):
                continue
            gguf = mapping.get(name)
            if not gguf or any(p in name.lower() for p in SKIP_PATTERNS) or any(p in gguf.lower() for p in SKIP_PATTERNS):
                continue
            originals.append((module, module.weight.data.clone()))
            q = quantize_tensor(module.weight.data, bits, symmetric, granularity, group_size)
            module.weight.data = q.to(module.weight.dtype).to(module.weight.device)
    return originals


def restore(originals):
    for module, data in originals:
        module.weight.data = data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmlu-samples", type=int, default=500)
    parser.add_argument("--text-chunks", type=int, default=500)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--output", default="basic_results.json")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    hf_token = args.hf_token
    for mounted in (
        "/kaggle/input/hf-token-store/hf_token.txt",
        "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt",
    ):
        if os.path.exists(mounted):
            with open(mounted) as f:
                hf_token = f.read().strip()
            print("HF token loaded, length:", len(hf_token) if hf_token else 0)
            break
    if not hf_token:
        print("WARNING: no HF token found")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, token=hf_token, trust_remote_code=True)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    mapping = {}
    if args.mapping and os.path.exists(args.mapping):
        with open(args.mapping) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if "=" in line:
                    left, right = line.split("=", 1)
                    mapping[left.strip()] = right.strip()

    # text datasets
    text_sets = {
        "fineweb-edu": ("HuggingFaceFW/fineweb-edu", None, "train"),
        "finemath-4plus": ("HuggingFaceTB/finemath", "finemath-4plus", "train"),
        "the-stack-python": ("bigcode/the-stack", None, "train"),
    }
    text_samples = {}
    for name, (repo, cfg, split) in text_sets.items():
        text_samples[name] = build_text_samples(tokenizer, repo, cfg, split, args.seq_length, args.max_docs, args.text_chunks, hf_token)

    def cache_text(samples):
        cached = []
        with torch.inference_mode():
            for s in samples[:5]:
                ids = s.unsqueeze(0).to(device)
                mask = (ids != tokenizer.pad_token_id).long()
                logits = model(input_ids=ids, attention_mask=mask).logits
                lp = F.log_softmax(logits, dim=-1)
                k = min(TEACHER_TOPK, lp.size(-1))
                for i in range(lp.size(1)):
                    vals, inds = torch.topk(lp[0, i], k=k)
                    cached.append((vals.detach().cpu(), inds.detach().to(torch.int32).cpu()))
        # reshape to per-chunk list of tensors
        out = []
        for i in range(5):
            if i * 512 >= len(samples) * 512:
                break
            chunk_vals = []
            chunk_inds = []
            for j in range(512):
                pos = i * 512 + j
                if pos < len(cached):
                    chunk_vals.append(cached[pos][0])
                    chunk_inds.append(cached[pos][1])
            if chunk_vals:
                out.append((torch.stack(chunk_vals), torch.stack(chunk_inds)))
        return out

    text_cached = {name: cache_text(samples) for name, samples in text_samples.items()}

    # baseline
    results = {"model": args.model, "baseline": {}, "variants": []}
    print("Starting baseline text evaluation...", flush=True)
    base_text = {}
    for name, samples in text_samples.items():
        print(f"  Baseline text dataset: {name}", flush=True)
        base_text[name] = eval_text(model, samples, tokenizer, device, text_cached[name])
    print("Baseline text done.", flush=True)
    # cache original mmlu probs
    orig_mmlu_probs = []
    with torch.inference_mode():
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test", token=hf_token)
        cnt = 0
        for row in ds:
            if cnt >= args.mmlu_samples:
                break
            choices = row["choices"]
            prompt = (f"{row['question'].strip()}\n"
                      f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:")
            enc = tokenizer(prompt, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits, dim=-1)
            answer_ids = tokenizer("Answer:", add_special_tokens=False)["input_ids"]
            input_ids = ids[0]
            answer_start = None
            for pos in range(len(input_ids) - len(answer_ids) + 1):
                if all(input_ids[pos + j].item() == answer_ids[j] for j in range(len(answer_ids))):
                    answer_start = pos + len(answer_ids)
                    break
            if answer_start is None or answer_start >= logits.size(1):
                cnt += 1
                continue
            choice_tokens = []
            for letter in ["A", "B", "C", "D"]:
                t = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
                choice_tokens.append(t[0] if t else tokenizer.eos_token_id)
            lp_choices = [lp[0, answer_start, tid].item() for tid in choice_tokens]
            orig_mmlu_probs.append(F.softmax(torch.tensor(lp_choices, device=device), dim=0).detach().cpu())
            cnt += 1

    print("Starting baseline MMLU evaluation...", flush=True)
    base_mmlu = eval_mmlu(model, tokenizer, device, args.mmlu_samples, orig_mmlu_probs, hf_token)
    print("Baseline MMLU done.", flush=True)
    results["baseline"] = {"text": base_text, "mmlu": base_mmlu}

    variants = []
    for symmetric in (True, False):
        for bits in (3, 4, 6, 8):
            variants.append((symmetric, "per_tensor", None, bits))
            variants.append((symmetric, "per_channel", None, bits))
            variants.append((symmetric, "per_layer", None, bits))
            for gs in (32, 64, 128):
                variants.append((symmetric, "per_group", gs, bits))

    total_variants = len(variants)
    for vi, (symmetric, granularity, gs, bits) in enumerate(variants, 1):
        print(f"\n=== Variant {vi}/{total_variants}: {granularity} | symmetric={symmetric} | group={gs} | bits={bits} ===", flush=True)
        layer_mode = (granularity == "per_layer")
        print("  Quantizing...", flush=True)
        originals = apply_quantization(model, mapping, bits, symmetric, granularity, gs, layer_mode)
        text_res = {}
        for name, samples in text_samples.items():
            print(f"  Evaluating text: {name}", flush=True)
            text_res[name] = eval_text(model, samples, tokenizer, device, text_cached[name])
        print("  Evaluating MMLU...", flush=True)
        mmlu_res = eval_mmlu(model, tokenizer, device, args.mmlu_samples, orig_mmlu_probs, hf_token)
        restore(originals)
        gc.collect()
        print(f"  Variant {vi}/{total_variants} done.", flush=True)
        results["variants"].append({
            "symmetric": symmetric,
            "granularity": granularity,
            "group_size": gs,
            "bits": bits,
            "text": text_res,
            "mmlu": mmlu_res,
        })
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Upload results to Hugging Face
    hf_token_path = "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt"
    if os.path.exists(hf_token_path):
        try:
            with open(hf_token_path) as f:
                hf_token = f.read().strip()
            from huggingface_hub import HfApi, create_repo
            repo_id = "deepresearch1a10/basic-quantization-results"
            create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=args.output,
                path_in_repo=os.path.basename(args.output),
                repo_id=repo_id,
                repo_type="model",
                token=hf_token,
            )
            print("Uploaded results to", repo_id)
        except Exception as e:
            print("HF upload failed:", e)
    else:
        print("HF token file not found; skipping upload.")

    print("\nDone. Results saved to", args.output)


if __name__ == "__main__":
    main()
