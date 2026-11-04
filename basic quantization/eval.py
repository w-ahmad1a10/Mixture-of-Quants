"""Script 2: download each quantized model from Hugging Face, evaluate, upload results."""
import argparse, gc, json, math, os, random, shutil, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moq.moq_core import SKIP_PATTERNS

TEACHER_TOPK_KLD = 1000
TEACHER_TOPK_EAR = 10


def variant_name(symmetric, granularity, gs, bits):
    s = "sym" if symmetric else "asym"
    g = granularity.replace("per_", "")
    if granularity == "per_group":
        g = f"group{gs}"
    return f"{s}_{g}_{bits}bit"


VARIANTS = []
for symmetric in (True, False):
    for bits in (3, 4, 6, 8):
        VARIANTS.append((symmetric, "per_tensor", None, bits))
        VARIANTS.append((symmetric, "per_channel", None, bits))
        VARIANTS.append((symmetric, "per_layer", None, bits))
        for gs in (32, 64, 128):
            VARIANTS.append((symmetric, "per_group", gs, bits))


def build_text_samples(tokenizer, repo, config, split, seq_length, max_docs, num_chunks, hf_token):
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


def cache_kld(samples, model, device, k):
    cached = []
    with torch.inference_mode():
        for s in samples[:5]:
            ids = s.unsqueeze(0).to(device)
            mask = (ids != 0).long()
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits, dim=-1)
            for i in range(lp.size(1)):
                vals, inds = torch.topk(lp[0, i], k=min(k, lp.size(-1)))
                cached.append((vals.detach().cpu(), inds.detach().to(torch.int32).cpu()))
    out = []
    for i in range(5):
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


def cache_ear(samples, model, device, k):
    cached = []
    with torch.inference_mode():
        for s in samples:
            ids = s.unsqueeze(0).to(device)
            mask = (ids != 0).long()
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits, dim=-1)
            for i in range(lp.size(1)):
                vals, inds = torch.topk(lp[0, i], k=min(k, lp.size(-1)))
                cached.append((vals.detach().cpu(), inds.detach().to(torch.int32).cpu()))
    out = []
    for i in range(len(samples)):
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


def eval_text(model, samples, tokenizer, device, cached_kld, cached_ear):
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
            # KLD on first 5 chunks using top-1000 cache
            if cached_kld is not None and idx < len(cached_kld):
                tvals, tinds = cached_kld[idx]
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
            # EAR on all chunks using top-10 cache
            if cached_ear is not None and idx < len(cached_ear):
                tvals, tinds = cached_ear[idx]
                tvals = tvals.to(device)
                tinds = tinds.to(device).long()
                if tinds.dim() == 2:
                    for i in range(tinds.size(0)):
                        s_lp = lp[0, i].gather(0, tinds[i])
                        t_p = torch.exp(tvals[i])
                        mask_t = tvals[i] > -16.0
                        t_p = t_p * mask_t
                        s_p = torch.exp(s_lp)
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


def eval_mmlu(model, tokenizer, device, max_samples, orig_probs, batch_size=4):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test", token=None)
    rows = []
    for r in ds:
        if len(rows) >= max_samples:
            break
        rows.append(r)
    correct = 0
    total = 0
    kld_sum = 0.0
    ce_sum = 0.0
    ce_tokens = 0
    answer_ids = tokenizer("Answer:", add_special_tokens=False)["input_ids"]
    choice_tokens_fixed = []
    for letter in ["A", "B", "C", "D"]:
        t = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
        choice_tokens_fixed.append(t[0] if t else tokenizer.eos_token_id)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = []
            for row in batch:
                choices = row["choices"]
                prompts.append(f"{row['question'].strip()}\n"
                               f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:")
            enc = tokenizer(prompts, return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits, dim=-1)
            for bi, row in enumerate(batch):
                input_ids = ids[bi]
                answer_start = None
                for pos in range(len(input_ids) - len(answer_ids) + 1):
                    if all(input_ids[pos + j].item() == answer_ids[j] for j in range(len(answer_ids))):
                        answer_start = pos + len(answer_ids)
                        break
                if answer_start is None or answer_start >= logits.size(1):
                    continue
                lp_choices = [lp[bi, answer_start, tid].item() for tid in choice_tokens_fixed]
                student_probs = F.softmax(torch.tensor(lp_choices, device=device), dim=0)
                pred = int(torch.argmax(student_probs).item())
                if pred == int(row["answer"]):
                    correct += 1
                ans_tid = choice_tokens_fixed[int(row["answer"])]
                ce_sum += -lp[bi, answer_start, ans_tid].item()
                ce_tokens += 1
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--hf-repo", default="w-ahmad/basic-quantization-models")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmlu-samples", type=int, default=100)
    parser.add_argument("--mmlu-batch-size", type=int, default=4)
    parser.add_argument("--text-chunks", type=int, default=500)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    hf_token = args.hf_token
    for tp in ("/kaggle/input/hf-token-store/hf_token.txt",
               "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt"):
        if os.path.exists(tp):
            with open(tp) as f:
                hf_token = f.read().strip()
            break

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # original model
    orig_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, token=hf_token, trust_remote_code=True)
    orig_model.to(device)
    orig_model.eval()
    orig_model.config.use_cache = False

    text_sets = {
        "fineweb-edu": ("HuggingFaceFW/fineweb-edu", None, "train"),
        "finemath-4plus": ("HuggingFaceTB/finemath", "finemath-4plus", "train"),
    }
    text_samples = {}
    for name, (repo, cfg, split) in text_sets.items():
        text_samples[name] = build_text_samples(tokenizer, repo, cfg, split, args.seq_length, args.max_docs, args.text_chunks, hf_token)

    # cache original logits
    text_cached_kld = {}
    text_cached_ear = {}
    for name, samples in text_samples.items():
        text_cached_kld[name] = cache_kld(samples, orig_model, device, TEACHER_TOPK_KLD)
        text_cached_ear[name] = cache_ear(samples, orig_model, device, TEACHER_TOPK_EAR)

    # cache original mmlu probs
    orig_mmlu_probs = []
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test", token=hf_token)
    cnt = 0
    with torch.inference_mode():
        for row in ds:
            if cnt >= args.mmlu_samples:
                break
            choices = row["choices"]
            prompt = (f"{row['question'].strip()}\n"
                      f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:")
            enc = tokenizer(prompt, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            logits = orig_model(input_ids=ids, attention_mask=mask).logits
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

    # baseline
    print("Baseline text eval...", flush=True)
    base_text = {}
    for name, samples in text_samples.items():
        base_text[name] = eval_text(orig_model, samples, tokenizer, device, text_cached_kld[name], text_cached_ear[name])
    print("Baseline MMLU eval...", flush=True)
    base_mmlu = eval_mmlu(orig_model, tokenizer, device, args.mmlu_samples, orig_mmlu_probs, args.mmlu_batch_size)
    results = {"model": args.model, "baseline": {"text": base_text, "mmlu": base_mmlu}, "variants": []}

    api = None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
    except Exception:
        pass

    for vi, (symmetric, granularity, gs, bits) in enumerate(VARIANTS, 1):
        name = variant_name(symmetric, granularity, gs, bits)
        print(f"\n=== [{vi}/{len(VARIANTS)}] Evaluating {name} ===", flush=True)
        local_dir = os.path.join("/kaggle/working", "models", name)
        if not os.path.isdir(local_dir):
            print(f"  Downloading {name} from {args.hf_repo}...", flush=True)
            snapshot_download(repo_id=args.hf_repo, allow_patterns=f"{name}/*", local_dir="/kaggle/working/models", token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype=dtype, token=hf_token, trust_remote_code=True)
        model.to(device)
        model.eval()
        model.config.use_cache = False
        text_res = {}
        for dname, samples in text_samples.items():
            print(f"  Text: {dname}", flush=True)
            text_res[dname] = eval_text(model, samples, tokenizer, device, text_cached_kld[dname], text_cached_ear[dname])
        print("  MMLU...", flush=True)
        mmlu_res = eval_mmlu(model, tokenizer, device, args.mmlu_samples, orig_mmlu_probs, args.mmlu_batch_size)
        results["variants"].append({"name": name, "text": text_res, "mmlu": mmlu_res})
        # save
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        # upload results
        if api is not None:
            try:
                results_repo = "w-ahmad/basic-quantization-results"
                try:
                    create_repo(results_repo, repo_type="model", exist_ok=True, token=hf_token)
                except Exception:
                    pass
                api.upload_file(path_or_fileobj=args.output, path_in_repo=f"results_{name}.json",
                                repo_id=results_repo, repo_type="model", token=hf_token)
                print(f"  Uploaded results for {name}", flush=True)
            except Exception as e:
                print(f"  Upload failed: {e}", flush=True)
        # cleanup
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        shutil.rmtree(local_dir, ignore_errors=True)
        print(f"  [{vi}/{len(VARIANTS)}] done.", flush=True)

    print("\nALL EVAL DONE")


if __name__ == "__main__":
    main()
