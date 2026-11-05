import os, gc, torch
from transformers import AutoModelForCausalLM
hf_token = os.environ.get("HF_TOKEN")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float16, token=hf_token, trust_remote_code=True, low_cpu_mem_usage=True)
target = None
for name, module in model.named_modules():
    if name == "model.layers.12.linear_attn.out_proj":
        target = module.weight.data.detach().clone().float()
        break
del model
gc.collect()
W = target
print("shape:", tuple(W.shape))
print("mean:", W.mean().item())
print("std:", W.std().item())
print("variance:", W.var().item())
print("min:", W.min().item())
print("max:", W.max().item())
print("abs mean:", W.abs().mean().item())
del W, target
gc.collect()
print("done, no model kept")
