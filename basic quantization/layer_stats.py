import os, gc, torch
from transformers import AutoModelForCausalLM
hf_token = os.environ.get("HF_TOKEN")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float16, token=hf_token, trust_remote_code=True, low_cpu_mem_usage=True)
print("Layer 12 matrices:\n")
for name, module in model.named_modules():
    if "layers.12" in name and hasattr(module, "weight") and module.weight.dim() >= 2:
        w = module.weight.data.detach().float()
        print(f"{name}: shape={tuple(w.shape)}")
        print(f"  mean={w.mean().item():.6f}  std={w.std().item():.6f}  min={w.min().item():.6f}  max={w.max().item():.6f}")
        print(f"  abs_mean={w.abs().mean().item():.6f}  var={w.var().item():.6e}")
        print()
del model
gc.collect()
print("Model deleted from memory. Disk cache kept.")
