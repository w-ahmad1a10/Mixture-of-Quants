"""Extract one real matrix from Qwen3.5 0.8B, quantize it in 48 ways, report MSE, X, SQNR."""
import os, sys, gc, math, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantize import quantize_tensor, VARIANTS, variant_name

# --- Step 1: extract one matrix, then delete the model ---
import torch
hf_token = None
for tp in ("/kaggle/input/hf-token-store/hf_token.txt",
           "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt"):
    if os.path.exists(tp):
        with open(tp) as f:
            hf_token = f.read().strip()
        break
# local env fallback
if hf_token is None:
    hf_token = os.environ.get("HF_TOKEN")

from transformers import AutoModelForCausalLM
model_name = "Qwen/Qwen3.5-0.8B"
print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, token=hf_token, trust_remote_code=True, low_cpu_mem_usage=True
)
# pick middle layer query projection
target = None
target_name = None
for name, module in model.named_modules():
    if "layers.12" in name and name.endswith("q_proj") and hasattr(module, "weight"):
        target = module.weight.data.detach().clone().float()
        target_name = name
        break
if target is None:
    # fallback: first 2D weight
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.dim() >= 2:
            target = module.weight.data.detach().clone().float()
            target_name = name
            break
print("Extracted:", target_name, tuple(target.shape), flush=True)
# delete model completely
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Model deleted from memory.", flush=True)

W = target
x = torch.randn(W.shape[1])
y = W @ x

# --- Step 2: run all 48 variants ---
rows = []
for sym, gran, gs, bits in VARIANTS:
    Wq = quantize_tensor(W.clone(), bits, sym, gran, gs)
    yq = Wq @ x
    mse = F.mse_loss(yq, y).item()
    wvar = W.var().item()
    wnoise = ((W - Wq) ** 2).mean().item()
    sqnr = 10.0 * (wvar / wnoise).log10().item() if (wnoise > 0 and wvar > 0) else float("inf")
    rows.append({"name": variant_name(sym, gran, gs, bits), "bits": bits,
                 "mse": mse, "sqnr": sqnr, "granularity": gran, "symmetric": sym, "gs": gs})

# X per bit
by_bit = {}
for r in rows:
    by_bit.setdefault(r["bits"], []).append(r)
for b, lst in by_bit.items():
    mn = min(r["mse"] for r in lst)
    for r in lst:
        r["x"] = r["mse"] / mn

rows.sort(key=lambda r: r["mse"])
print("\n=== REAL MATRIX RESULTS ===")
print("variant, mse, x, sqnr_db")
for r in rows:
    print(f"{r['name']}, {r['mse']:.6e}, {r['x']:.2f}x, {r['sqnr']:.2f}")

# --- Step 3: compare ranking with random matrix ---
torch.manual_seed(42)
Wr = torch.randn(512, 512) * 0.02
xr = torch.randn(512)
yr = Wr @ xr
rand_rows = []
for sym, gran, gs, bits in VARIANTS:
    Wq = quantize_tensor(Wr.clone(), bits, sym, gran, gs)
    yq = Wq @ xr
    mse = F.mse_loss(yq, yr).item()
    rand_rows.append((variant_name(sym, gran, gs, bits), bits, mse))
# rank by mse
rand_rank = {name: i for i, (name, bits, mse) in enumerate(sorted(rand_rows, key=lambda r: r[2]))}
real_rank = {r["name"]: i for i, r in enumerate(rows)}
print("\n=== RANKING COMPARISON (top 10) ===")
print("rank | random matrix | real matrix")
for i in range(10):
    rname = sorted(rand_rows, key=lambda r: r[2])[i][0]
    realname = rows[i]["name"]
    print(f"{i+1:4d} | {rname:22s} | {realname}")
# spearman-ish: how many top-5 overlap
rand_top5 = set(sorted(rand_rows, key=lambda r: r[2])[i][0] for i in range(5))
real_top5 = set(rows[i]["name"] for i in range(5))
print("\nTop-5 overlap between random and real:", len(rand_top5 & real_top5), "of 5")
print("Overlap names:", sorted(rand_top5 & real_top5))

# save json
with open("real_matrix_results.json", "w") as f:
    json.dump({"matrix": target_name, "shape": list(W.shape), "results": rows}, f, indent=2)
print("\nSaved real_matrix_results.json")
print("Done. No model kept on disk.")
