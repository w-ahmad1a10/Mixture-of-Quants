"""Merged quantize + eval for Qwen3.5 4B on Kaggle GPU. No HF upload/download of models."""
import argparse, gc, json, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

SKIP_PATTERNS = (
    'norm', 'ln', 'rotary', 'bias', 'input_layernorm', 'ssm_conv1d', 'conv1d',
    'post_attention_layernorm', 'norm.weight', 'ln_f', 'ln_1', 'ln_2', 'conv.conv'
)

TEACHER_TOPK_KLD = 1000
TEACHER_TOPK_EAR = 10


def is_quantizable(name, module):
    if not hasattr(module, "weight"):
        return False
    if module.weight.dim() < 2:
        return False
    nl = name.lower()
    if any(p in nl for p in SKIP_PATTERNS):
        return False
    return True


def quantize_tensor(w, bits, symmetric, granularity, group_size=None):
    w = w.detach().clone().float()
    if w.dim() < 2:
        return w
    rows, cols = w.shape[0], w.shape[1]
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
    else:
        qmax = (1 << bits) - 1
    if granularity in ("per_tensor", "per_layer"):
        if symmetric:
            maxabs = w.abs().max()
            scale = (maxabs / qmax) if (maxabs > 0 and qmax > 0) else torch.tensor(1.0, device=w.device)
            zp = torch.tensor(0, device=w.device)
        else:
            wmin = w.min(); wmax = w.max()
            scale = ((wmax - wmin) / qmax) if (wmax > wmin) else torch.tensor(1.0, device=w.device)
            zp = torch.round(-wmin / scale).clamp(0, qmax)
        # quantize
        if symmetric:
            q = (w / scale).round().clamp(-qmax, qmax)
            return q * scale
        else:
            q = (w / scale + zp).round().clamp(0, qmax)
            return (q - zp) * scale
    if granularity == "per_channel":
        out = torch.empty_like(w)
        if symmetric:
            maxabs = w.abs().amax(dim=1, keepdim=True)
            scale = torch.where(maxabs > 0, maxabs / qmax, torch.ones_like(maxabs))
            zp = torch.zeros_like(scale)
        else:
            wmin = w.amin(dim=1, keepdim=True); wmax = w.amax(dim=1, keepdim=True)
            scale = torch.where(wmax > wmin, (wmax - wmin) / qmax, torch.ones_like(wmin))
            zp = torch.round(-wmin / scale).clamp(0, qmax)
        if symmetric:
            q = (w / scale).round().clamp(-qmax, qmax)
            return q * scale
        else:
            q = (w / scale + zp).round().clamp(0, qmax)
            return (q - zp) * scale
    if granularity == "per_group":
        gs = group_size or 32
        pad = (gs - (cols % gs)) % gs
        if pad:
            w = torch.nn.functional.pad(w, (0, pad))
        new_cols = w.shape[1]
        ng = new_cols // gs
        wg = w.view(rows, ng, gs)
        if symmetric:
            maxabs = wg.abs().amax(dim=2, keepdim=True)
            scale = torch.where(maxabs > 0, maxabs / qmax, torch.ones_like(maxabs))
            zp = torch.zeros_like(scale)
        else:
            wmin = wg.amin(dim=2, keepdim=True); wmax = wg.amax(dim=2, keepdim=True)
            scale = torch.where(wmax > wmin, (wmax - wmin) / qmax, torch.ones_like(wmin))
            zp = torch.round(-wmin / scale).clamp(0, qmax)
        if symmetric:
            q = (wg / scale).round().clamp(-qmax, qmax)
            dq = q * scale
        else:
            q = (wg / scale + zp).round().clamp(0, qmax)
            dq = (q - zp) * scale
        dq = dq.view(rows, new_cols)
        if pad:
            dq = dq[:, :cols]
        return dq
    return w


def layer_params(weights, bits, symmetric):
    flat = torch.cat([x.reshape(-1).float() for x in weights])
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        maxabs = flat.abs().max()
        scale = (maxabs / qmax) if (maxabs > 0 and qmax > 0) else 1.0
        return scale, 0
    qmax = (1 << bits) - 1
    wmin = flat.min().item(); wmax = flat.max().item()
    if wmax - wmin <= 0:
        return 1.0, 0
    scale = (wmax - wmin) / qmax
    zp = int(round(-wmin / scale))
    zp = max(0, min(qmax, zp))
    return scale, zp


def apply_quantization(model, bits, symmetric, granularity, group_size, layer_mode, clip_pct=None):
    originals = []
    def is_q(name, module):
        if not is_quantizable(name, module):
            return False
        return True
    if layer_mode:
        layers = {}
        for name, module in model.named_modules():
            if not is_q(name, module):
                continue
            layer_key = name.split(".")[2] if "layers." in name else "other"
            layers.setdefault(layer_key, []).append((name, module))
        for lk, items in layers.items():
            weights = [m.weight.data for _, m in items]
            scale, zp = layer_params(weights, bits, symmetric)
            for name, module in items:
                w = module.weight.data
                if clip_pct is not None:
                    thresh = torch.quantile(w.abs().reshape(-1).float(), clip_pct).item()
                    w = w.clamp(-thresh, thresh)
                originals.append((module, module.weight.data.clone()))
                q = quantize_with_params(w.float(), bits, symmetric, scale, zp)
                module.weight.data = q.to(module.weight.dtype).to(module.weight.device)
    else:
        for name, module in model.named_modules():
            if not is_q(name, module):
                continue
            w = module.weight.data
            if clip_pct is not None:
                thresh = torch.quantile(w.abs().reshape(-1).float(), clip_pct).item()
                w = w.clamp(-thresh, thresh)
            originals.append((module, module.weight.data.clone()))
            q = quantize_tensor(w, bits, symmetric, granularity, group_size)
            module.weight.data = q.to(module.weight.dtype).to(module.weight.device)
    return originals


def quantize_with_params(w, bits, symmetric, scale, zp):
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        q = (w / scale).round().clamp(-qmax, qmax)
        return q * scale
    qmax = (1 << bits) - 1
    q = (w / scale + zp).round().clamp(0, qmax)
    return (q - zp) * scale


def restore(originals):
    for module, data in originals:
        module.weight.data = data


def build_text_samples(tokenizer, repo, config, split, seq_length, max_docs, num_chunks, hf_token, text_column="text"):
    ds = load_dataset(repo, config, split=split, streaming=True, token=hf_token) if config else load_dataset(repo, split=split, streaming=True, token=hf_token)
    all_ids = []
    count = 0
    for row in ds:
        if count >= max_docs:
            break
        text = row.get(text_column, "")
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


def download_wiki(path, hf_token):
    if os.path.exists(path):
        return
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="w-ahmad/moq-lfm25-230m-results", filename="wiki.test (2).raw",
                        local_dir=os.path.dirname(path), token=hf_token, repo_type="dataset")
        # rename if needed
        src = os.path.join(os.path.dirname(path), "wiki.test (2).raw")
        if os.path.exists(src) and src != path:
            os.rename(src, path)
    except Exception as e:
        print("Wiki download failed:", e)


def cache_kld(samples, model, device, k):
    cached = []
    with torch.inference_mode():
        for s in samples[:5]:
            ids = s.unsqueeze(0).to(device)
            mask = (ids != 0).long() if False else (ids != -1).long()
            mask = (ids != 0).long()
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits, dim=-1)
            for i in range(lp.size(1)):
                vals, inds = torch.topk(lp[0, i], k=min(k, lp.size(-1)))
                cached.append((vals.detach().cpu(), inds.detach().to(torch.int32).cpu()))
    out = []
    for i in range(5):
        chunk_vals = []; chunk_inds = []
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
        chunk_vals = []; chunk_inds = []
        for j in range(512):
            pos = i * 512 + j
            if pos < len(cached):
                chunk_vals.append(cached[pos][0])
                chunk_inds.append(cached[pos][1])
        if chunk_vals:
            out.append((torch.stack(chunk_vals), torch.stack(chunk_inds)))
    return out


def eval_text(model, samples, tokenizer, device, cached_kld, cached_ear, batch_size=4):
    model.eval()
    ce_sum = 0.0; ce_tokens = 0
    kld_vals = []; ear_vals = []; mass_vals = []
    n = len(samples)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch = samples[start:start + batch_size]
            bs = len(batch)
            max_len = max(s.size(0) for s in batch)
            ids = torch.full((bs, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
            for bi, s in enumerate(batch):
                L = s.size(0)
                ids[bi, :L] = s.to(device)
                mask[bi, :L] = 1
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits, dim=-1)
            for bi in range(bs):
                sl = logits[bi, :-1, :].contiguous()
                sy = ids[bi, 1:].contiguous()
                sm = mask[bi, 1:].contiguous().bool()
                if sm.any():
                    ce = F.cross_entropy(sl[sm], sy[sm], reduction="sum")
                    ce_sum += ce.item()
                    ce_tokens += int(sm.sum().item())
            for bi in range(bs):
                idx = start + bi
                if cached_kld is not None and idx < len(cached_kld):
                    tvals, tinds = cached_kld[idx]
                    tvals = tvals.to(device); tinds = tinds.to(device).long()
                    if tinds.dim() == 2:
                        for i in range(tinds.size(0)):
                            s_lp = lp[bi, i].gather(0, tinds[i])
                            t_p = torch.exp(tvals[i]); mt = tvals[i] > -16.0
                            t_p = t_p * mt
                            kld = (t_p * (tvals[i] - s_lp)).sum()
                            kld = torch.clamp(kld, min=0.0)
                            kld_vals.append(kld.item())
                if cached_ear is not None and idx < len(cached_ear):
                    tvals, tinds = cached_ear[idx]
                    tvals = tvals.to(device); tinds = tinds.to(device).long()
                    if tinds.dim() == 2:
                        for i in range(tinds.size(0)):
                            s_lp = lp[bi, i].gather(0, tinds[i])
                            t_p = torch.exp(tvals[i]); mt = tvals[i] > -16.0
                            t_p = t_p * mt
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-chunks", type=int, default=500)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--output", default="results_4b.json")
    parser.add_argument("--hf-token", default=None)
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

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, token=hf_token, trust_remote_code=True, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    # datasets
    wiki_path = "/kaggle/working/wiki.test (2).raw"
    download_wiki(wiki_path, hf_token)
    text_sets = {}
    text_sets["fineweb-edu"] = ("HuggingFaceFW/fineweb-edu", None, "train", "text")
    text_sets["finemath-4plus"] = ("HuggingFaceTB/finemath", "finemath-4plus", "train", "text")
    text_sets["the-stack-python"] = ("bigcode/the-stack", None, "train", "content")
    if os.path.exists(wiki_path):
        text_sets["wiki"] = ("__wiki__", None, None, "text")

    text_samples = {}
    for name, (repo, cfg, split, tcol) in text_sets.items():
        if repo == "__wiki__":
            with open(wiki_path) as f:
                txts = [t for t in f.read().split("\n") if t.strip()]
            all_ids = []
            for t in txts:
                all_ids.extend(tokenizer.encode(t, add_special_tokens=False))
            samples = []
            for i in range(0, len(all_ids), args.seq_length):
                chunk = all_ids[i:i + args.seq_length]
                if len(chunk) == args.seq_length:
                    samples.append(torch.tensor(chunk, dtype=torch.long))
                elif len(chunk) > args.seq_length // 2:
                    chunk = chunk + [tokenizer.pad_token_id] * (args.seq_length - len(chunk))
                    samples.append(torch.tensor(chunk, dtype=torch.long))
                if len(samples) >= args.text_chunks:
                    break
            text_samples[name] = samples
        else:
            text_samples[name] = build_text_samples(tokenizer, repo, cfg, split, args.seq_length, args.max_docs, args.text_chunks, hf_token, tcol)

    # cache original logits per dataset
    print("Caching original logits...", flush=True)
    text_cached_kld = {}
    text_cached_ear = {}
    for name, samples in text_samples.items():
        text_cached_kld[name] = cache_kld(samples, model, device, TEACHER_TOPK_KLD)
        text_cached_ear[name] = cache_ear(samples, model, device, TEACHER_TOPK_EAR)

    # baseline
    print("Baseline eval...", flush=True)
    base_text = {}
    for name, samples in text_samples.items():
        base_text[name] = eval_text(model, samples, tokenizer, device, text_cached_kld[name], text_cached_ear[name])
    results = {"model": args.model, "baseline": {"text": base_text}, "variants": [], "clipping": []}

    # variants
    variants = []
    for symmetric in (True, False):
        for granularity in ("per_tensor", "per_channel", "per_group"):
            for bits in (4, 8):
                gs = 32 if granularity == "per_group" else None
                variants.append((symmetric, granularity, gs, bits))

    for vi, (symmetric, granularity, gs, bits) in enumerate(variants, 1):
        name = f"{'sym' if symmetric else 'asym'}_{granularity.replace('per_', '')}{'_group' + str(gs) if granularity == 'per_group' else ''}_{bits}bit"
        print(f"\n=== [{vi}/{len(variants)}] {name} ===", flush=True)
        layer_mode = (granularity == "per_layer")
        originals = apply_quantization(model, bits, symmetric, granularity, gs, layer_mode, None)
        text_res = {}
        for dname, samples in text_samples.items():
            print(f"  {dname}", flush=True)
            text_res[dname] = eval_text(model, samples, tokenizer, device, text_cached_kld[dname], text_cached_ear[dname])
        restore(originals)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results["variants"].append({"name": name, "symmetric": symmetric, "granularity": granularity,
                                    "group_size": gs, "bits": bits, "text": text_res})
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  done.", flush=True)

    # clipping comparison for 4-bit per_group
    print("\n=== Clipping comparison (4-bit per_group) ===", flush=True)
    for symmetric in (True, False):
        for clip_pct, clip_label in ((None, "no_clip"), (0.999, "clip99.9"), (0.99, "clip99")):
            print(f"  {clip_label} sym={symmetric}", flush=True)
            originals = apply_quantization(model, 4, symmetric, "per_group", 32, False, clip_pct)
            text_res = {}
            for dname, samples in text_samples.items():
                text_res[dname] = eval_text(model, samples, tokenizer, device, text_cached_kld[dname], text_cached_ear[dname])
            restore(originals)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results["clipping"].append({"clip": clip_label, "symmetric": symmetric, "bits": 4,
                                       "group_size": 32, "text": text_res})
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    # Upload results to Hugging Face
    hf_token = None
    for tp in ("/kaggle/input/hf-token-store/hf_token.txt",
               "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt"):
        if os.path.exists(tp):
            with open(tp) as f:
                hf_token = f.read().strip()
            break
    if hf_token:
        try:
            from huggingface_hub import HfApi, create_repo
            repo_id = "w-ahmad/basic-quantization-results-4b"
            create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
            api = HfApi(token=hf_token)
            api.upload_file(path_or_fileobj=args.output, path_in_repo=os.path.basename(args.output),
                            repo_id=repo_id, repo_type="model", token=hf_token)
            print("Uploaded results to", repo_id, flush=True)
        except Exception as e:
            print("HF upload failed:", e, flush=True)
    else:
        print("HF token not found; skipping upload.", flush=True)

    print("\nALL DONE. Results in", args.output, flush=True)


if __name__ == "__main__":
    main()
