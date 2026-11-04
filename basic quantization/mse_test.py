"""Local MSE test: one fixed tensor, one fixed input, all 48 quantization variants."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantize import quantize_tensor, VARIANTS, variant_name

torch.manual_seed(42)
W = torch.randn(512, 512) * 0.02
x = torch.randn(512)
y = W @ x

rows = []
for symmetric, granularity, gs, bits in VARIANTS:
    Wq = quantize_tensor(W.clone(), bits, symmetric, granularity, gs)
    yq = Wq @ x
    mse = F.mse_loss(yq, y).item()
    rows.append((variant_name(symmetric, granularity, gs, bits), mse, bits, granularity, symmetric, gs))

rows.sort(key=lambda r: r[1])
print("variant, mse")
for name, mse, bits, g, sym, gs in rows:
    print(f"{name}, {mse:.6e}")
print("\nBest 5:")
for r in rows[:5]:
    print(r[0], r[1])
print("\nWorst 5:")
for r in rows[-5:]:
    print(r[0], r[1])
