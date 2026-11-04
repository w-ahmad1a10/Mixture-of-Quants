"""Script 1: quantize Qwen3 0.6B in all variants and upload each to Hugging Face."""
import argparse, gc, json, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moq.moq_core import SKIP_PATTERNS

VARIANTS = []
for symmetric in (True, False):
    for bits in (3, 4, 6, 8):
        VARIANTS.append((symmetric, "per_tensor", None, bits))
        VARIANTS.append((symmetric, "per_channel", None, bits))
        VARIANTS.append((symmetric, "per_layer", None, bits))
        for gs in (32, 64, 128):
            VARIANTS.append((symmetric, "per_group", gs, bits))


def is_quantizable(name, module):
    if not hasattr(module, "weight"):
        return False
    if module.weight.dim() < 2:
        return False
    nl = name.lower()
    if any(p in nl for p in SKIP_PATTERNS):
        return False
    return True


def layer_params(weights, bits, symmetric):
    flat = torch.cat([x.reshape(-1).float() for x in weights]) if weights else torch.tensor([0.0])
    if symmetric:
        maxabs = flat.abs().max().item() if flat.numel() else 0.0
        qmax = (1 << (bits - 1)) - 1
        scale = (maxabs / qmax) if (maxabs > 0 and qmax > 0) else 1.0
        return scale, 0
    qmax = (1 << bits) - 1
    wmin = flat.min().item() if flat.numel() else 0.0
    wmax = flat.max().item() if flat.numel() else 0.0
    if wmax - wmin <= 0:
        return 1.0, 0
    scale = (wmax - wmin) / qmax
    zp = int(round(-wmin / scale))
    zp = max(0, min(qmax, zp))
    return scale, zp


def quantize_with_params(w, bits, symmetric, scale, zp):
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        q = (w / scale).round().clamp(-qmax, qmax)
        return q * scale
    qmax = (1 << bits) - 1
    q = (w / scale + zp).round().clamp(0, qmax)
    return (q - zp) * scale


def quantize_tensor(w, bits, symmetric, granularity, group_size=None):
    w = w.detach().clone().float()
    if w.dim() < 2:
        return w
    rows, cols = w.shape[0], w.shape[1]
    out = torch.empty_like(w)
    if granularity == "per_tensor":
        # compute params on flattened
        maxabs = w.abs().max().item() if symmetric else None
        if symmetric:
            qmax = (1 << (bits - 1)) - 1
            scale = (maxabs / qmax) if (maxabs and qmax) else 1.0
            zp = 0
        else:
            qmax = (1 << bits) - 1
            wmin = w.min().item(); wmax = w.max().item()
            scale = (wmax - wmin) / qmax if wmax > wmin else 1.0
            zp = int(round(-wmin / scale)) if scale else 0
            zp = max(0, min(qmax, zp))
        return quantize_with_params(w, bits, symmetric, scale, zp)
    if granularity == "per_channel":
        for r in range(rows):
            if symmetric:
                ma = w[r].abs().max().item()
                qmax = (1 << (bits - 1)) - 1
                scale = (ma / qmax) if (ma and qmax) else 1.0
                zp = 0
            else:
                qmax = (1 << bits) - 1
                wmin = w[r].min().item(); wmax = w[r].max().item()
                scale = (wmax - wmin) / qmax if wmax > wmin else 1.0
                zp = int(round(-wmin / scale)) if scale else 0
                zp = max(0, min(qmax, zp))
            out[r] = quantize_with_params(w[r], bits, symmetric, scale, zp)
        return out
    if granularity == "per_group":
        gs = group_size or 32
        for r in range(rows):
            for start in range(0, cols, gs):
                end = min(start + gs, cols)
                block = w[r, start:end]
                if symmetric:
                    ma = block.abs().max().item()
                    qmax = (1 << (bits - 1)) - 1
                    scale = (ma / qmax) if (ma and qmax) else 1.0
                    zp = 0
                else:
                    qmax = (1 << bits) - 1
                    wmin = block.min().item(); wmax = block.max().item()
                    scale = (wmax - wmin) / qmax if wmax > wmin else 1.0
                    zp = int(round(-wmin / scale)) if scale else 0
                    zp = max(0, min(qmax, zp))
                out[r, start:end] = quantize_with_params(block, bits, symmetric, scale, zp)
        return out
    return w


def apply_quantization(model, bits, symmetric, granularity, group_size, layer_mode):
    originals = []
    if layer_mode:
        layers = {}
        for name, module in model.named_modules():
            if not is_quantizable(name, module):
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
            if not is_quantizable(name, module):
                continue
            originals.append((module, module.weight.data.clone()))
            q = quantize_tensor(module.weight.data, bits, symmetric, granularity, group_size)
            module.weight.data = q.to(module.weight.dtype).to(module.weight.device)
    return originals


def restore(originals):
    for module, data in originals:
        module.weight.data = data


def variant_name(symmetric, granularity, gs, bits):
    s = "sym" if symmetric else "asym"
    g = granularity.replace("per_", "")
    if granularity == "per_group":
        g = f"group{gs}"
    return f"{s}_{g}_{bits}bit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-repo", default="deepresearch1a10/basic-quantization-models")
    parser.add_argument("--output-dir", default="/kaggle/working/models")
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

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, token=hf_token, trust_remote_code=True)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    from huggingface_hub import HfApi, create_repo
    create_repo(args.hf_repo, repo_type="model", exist_ok=True, token=hf_token)
    api = HfApi(token=hf_token)

    os.makedirs(args.output_dir, exist_ok=True)
    total = len(VARIANTS)
    for vi, (symmetric, granularity, gs, bits) in enumerate(VARIANTS, 1):
        name = variant_name(symmetric, granularity, gs, bits)
        print(f"\n=== [{vi}/{total}] Quantizing {name} ===", flush=True)
        layer_mode = (granularity == "per_layer")
        originals = apply_quantization(model, bits, symmetric, granularity, gs, layer_mode)
        out_dir = os.path.join(args.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"  Saved to {out_dir}", flush=True)
        # upload folder
        api.upload_folder(folder_path=out_dir, path_in_repo=name, repo_id=args.hf_repo, repo_type="model", token=hf_token)
        print(f"  Uploaded {name} to {args.hf_repo}", flush=True)
        restore(originals)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{vi}/{total}] done.", flush=True)

    print("\nALL QUANTIZATION DONE")


if __name__ == "__main__":
    main()
