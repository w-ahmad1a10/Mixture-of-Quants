"""Optimize a matrix so its 4-bit symmetric quantized output matches the original output."""
import os, sys, math, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
SIZE = 256
W0 = torch.randn(SIZE, SIZE) * 0.02
x = torch.randn(SIZE)
y0 = W0 @ x

BITS = 4
QMAX = (1 << (BITS - 1)) - 1


def quantize_ste(W):
    with torch.no_grad():
        scale = W.abs().max() / QMAX
        if scale.item() == 0:
            scale = torch.tensor(1.0)
        Wq = (W / scale).round().clamp(-QMAX, QMAX) * scale
    return W + (Wq - W).detach()


def mse(a, b):
    return F.mse_loss(a, b).item()


# before
W = W0.clone().detach().requires_grad_(True)
Wq = quantize_ste(W)
yq = Wq @ x
mse_before = mse(yq, y0)
weight_change_before = 0.0

# optimize
steps = 300
lr = 1e-2
history = []
for step in range(steps):
    Wq = quantize_ste(W)
    yq = Wq @ x
    loss = F.mse_loss(yq, y0)
    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad
        W.grad.zero_()
    if step % 30 == 0 or step == steps - 1:
        with torch.no_grad():
            Wq2 = quantize_ste(W)
            yq2 = Wq2 @ x
            cur = F.mse_loss(yq2, y0).item()
        history.append((step, cur))
        print(f"step {step:3d}: mse {cur:.6e}", flush=True)

with torch.no_grad():
    Wq_final = quantize_ste(W)
    yq_final = Wq_final @ x
    mse_after = mse(yq_final, y0)
    weight_mse = mse(W, W0)

print("\n=== RESULTS ===")
print(f"MSE before optimization: {mse_before:.6e}")
print(f"MSE after optimization:  {mse_after:.6e}")
print(f"Weight change (MSE W vs W0): {weight_mse:.6e}")
if mse_before > 0:
    print(f"Improvement factor: {mse_before / mse_after:.2f}x")

with open("optimize_output_results.json", "w") as f:
    json.dump({
        "mse_before": mse_before,
        "mse_after": mse_after,
        "weight_mse": weight_mse,
        "improvement": mse_before / mse_after if mse_after > 0 else float("inf"),
        "steps": steps,
        "lr": lr,
        "size": SIZE,
        "history": history,
    }, f, indent=2)
print("Saved optimize_output_results.json")
print("Done")
