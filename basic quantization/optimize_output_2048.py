"""Optimize a 2048x2048 matrix so its 4-bit symmetric quantized output matches original."""
import os, sys, math, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
SIZE = 2048
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


W = W0.clone().detach().requires_grad_(True)
Wq = quantize_ste(W)
yq = Wq @ x
mse_before = mse(yq, y0)

steps = 200
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
    with torch.no_grad():
        Wq2 = quantize_ste(W)
        yq2 = Wq2 @ x
        cur = F.mse_loss(yq2, y0).item()
    history.append({"step": step, "mse": cur})
    if step % 10 == 0 or step == steps - 1:
        print(f"step {step:3d}: mse {cur:.6e}", flush=True)

with torch.no_grad():
    Wq_final = quantize_ste(W)
    yq_final = Wq_final @ x
    mse_after = mse(yq_final, y0)
    weight_mse = mse(W, W0)

print("\n=== RESULTS ===")
print(f"MSE before: {mse_before:.6e}")
print(f"MSE after:  {mse_after:.6e}")
print(f"Weight change: {weight_mse:.6e}")
print(f"Improvement: {mse_before / mse_after:.2f}x")

# save json
with open("optimize_output_2048_results.json", "w") as f:
    json.dump({
        "size": SIZE,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "weight_mse": weight_mse,
        "improvement": mse_before / mse_after if mse_after > 0 else float("inf"),
        "steps": steps,
        "lr": lr,
        "history": history,
    }, f, indent=2)

# generate html
html = []
html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Output Optimization 2048</title>")
html.append("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;} .card{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);} table{border-collapse:collapse;width:100%;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#3498db;color:#fff;} tr:nth-child(even){background:#f9f9f9;} h1,h2{color:#333;}</style>")
html.append("</head><body><h1>Output-Level Optimization: 2048x2048</h1>")
html.append(f"<div class='card'><h2>Summary</h2>")
html.append(f"<p>Matrix size: {SIZE} by {SIZE}</p>")
html.append(f"<p>Quantization: 4-bit symmetric per tensor</p>")
html.append(f"<p>Steps: {steps}, learning rate: {lr}</p>")
html.append(f"<p>MSE before: {mse_before:.6e}</p>")
html.append(f"<p>MSE after: {mse_after:.6e}</p>")
html.append(f"<p>Improvement: {mse_before/mse_after:.2f}x</p>")
html.append(f"<p>Weight change MSE: {weight_mse:.6e}</p>")
html.append("</div>")
html.append("<div class='card'><h2>Loss per step</h2>")
html.append("<table><thead><tr><th>Step</th><th>MSE</th></tr></thead><tbody>")
for h in history:
    html.append(f"<tr><td>{h['step']}</td><td>{h['mse']:.6e}</td></tr>")
html.append("</tbody></table></div>")
html.append("</body></html>")
with open("optimize_output_2048.html", "w") as f:
    f.write("\n".join(html))
print("Saved optimize_output_2048.html")
print("Done")
