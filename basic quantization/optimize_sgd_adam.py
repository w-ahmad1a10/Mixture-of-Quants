"""Train on 100 inputs with SGD and Adam. 4-bit symmetric quantization, 2048x2048."""
import os, sys, math, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
SIZE = 2048
W0 = torch.randn(SIZE, SIZE) * 0.02
torch.manual_seed(7)
X_train = [torch.randn(SIZE) for _ in range(100)]
Y0_train = [W0 @ x for x in X_train]

BITS = 3
QMAX = (1 << (BITS - 1)) - 1


def quantize_ste(W):
    with torch.no_grad():
        scale = W.abs().max() / QMAX
        if scale.item() == 0:
            scale = torch.tensor(1.0)
        Wq = (W / scale).round().clamp(-QMAX, QMAX) * scale
    return W + (Wq - W).detach()


def train_mse(W):
    with torch.no_grad():
        Wq = quantize_ste(W)
        errs = [F.mse_loss(Wq @ x, y0).item() for x, y0 in zip(X_train, Y0_train)]
    return sum(errs) / len(errs)


# baseline before
with torch.no_grad():
    Wq0 = quantize_ste(W0)
    mse_before = sum(F.mse_loss(Wq0 @ x, y0).item() for x, y0 in zip(X_train, Y0_train)) / len(X_train)

STEPS = 100

# --- SGD ---
print("=== SGD ===", flush=True)
W_sgd = W0.clone().detach().requires_grad_(True)
opt_sgd = torch.optim.SGD([W_sgd], lr=1e-2)
hist_sgd = []
for step in range(STEPS):
    opt_sgd.zero_grad()
    Wq = quantize_ste(W_sgd)
    xi = X_train[step % len(X_train)]
    yi = Y0_train[step % len(X_train)]
    loss = F.mse_loss(Wq @ xi, yi)
    loss.backward()
    opt_sgd.step()
    if step % 10 == 0 or step == STEPS - 1:
        cur = train_mse(W_sgd.detach())
        hist_sgd.append({"step": step, "mse": cur})
        print(f"  step {step:3d}: train mse {cur:.4e}", flush=True)
with torch.no_grad():
    mse_sgd_after = train_mse(W_sgd.detach())
    weight_change_sgd = F.mse_loss(W_sgd.detach(), W0).item()

# --- Adam ---
print("=== Adam ===", flush=True)
W_adam = W0.clone().detach().requires_grad_(True)
opt_adam = torch.optim.Adam([W_adam], lr=1e-3, betas=(0.9, 0.95))
hist_adam = []
for step in range(STEPS):
    opt_adam.zero_grad()
    Wq = quantize_ste(W_adam)
    xi = X_train[step % len(X_train)]
    yi = Y0_train[step % len(X_train)]
    loss = F.mse_loss(Wq @ xi, yi)
    loss.backward()
    opt_adam.step()
    if step % 10 == 0 or step == STEPS - 1:
        cur = train_mse(W_adam.detach())
        hist_adam.append({"step": step, "mse": cur})
        print(f"  step {step:3d}: train mse {cur:.4e}", flush=True)
with torch.no_grad():
    mse_adam_after = train_mse(W_adam.detach())
    weight_change_adam = F.mse_loss(W_adam.detach(), W0).item()

print("\n=== RESULTS ===")
print(f"Train MSE before:        {mse_before:.6e}")
print(f"SGD train MSE after:     {mse_sgd_after:.6e}  (weight change {weight_change_sgd:.6e})")
print(f"Adam train MSE after:    {mse_adam_after:.6e}  (weight change {weight_change_adam:.6e})")
print(f"SGD improvement:         {mse_before/mse_sgd_after:.2f}x")
print(f"Adam improvement:        {mse_before/mse_adam_after:.2f}x")

with open("optimize_sgd_adam_results.json", "w") as f:
    json.dump({
        "mse_before": mse_before,
        "sgd": {"mse_after": mse_sgd_after, "weight_change": weight_change_sgd, "history": hist_sgd},
        "adam": {"mse_after": mse_adam_after, "weight_change": weight_change_adam, "history": hist_adam},
        "steps": STEPS,
        "size": SIZE,
        "n_train": len(X_train),
    }, f, indent=2)

# html
html = []
html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>SGD vs Adam</title>")
html.append("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;} .card{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);} table{border-collapse:collapse;width:100%;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#3498db;color:#fff;} tr:nth-child(even){background:#f9f9f9;} h1,h2{color:#333;}</style>")
html.append("</head><body><h1>SGD vs Adam: 3-bit Quantization Optimization (batch size 1)</h1>")
html.append(f"<div class='card'><h2>Summary</h2>")
html.append(f"<p>Matrix: {SIZE}x{SIZE}, 4-bit symmetric per tensor, {len(X_train)} train inputs</p>")
html.append(f"<p>Train MSE before: {mse_before:.6e}</p>")
html.append(f"<p>SGD after: {mse_sgd_after:.6e} (weight change {weight_change_sgd:.6e}, improvement {mse_before/mse_sgd_after:.2f}x)</p>")
html.append(f"<p>Adam after: {mse_adam_after:.6e} (weight change {weight_change_adam:.6e}, improvement {mse_before/mse_adam_after:.2f}x)</p>")
html.append("</div>")
html.append("<div class='card'><h2>Loss per step</h2>")
html.append("<table><thead><tr><th>Step</th><th>SGD train MSE</th><th>Adam train MSE</th></tr></thead><tbody>")
# align steps
sgd_map = {h["step"]: h["mse"] for h in hist_sgd}
adam_map = {h["step"]: h["mse"] for h in hist_adam}
for step in range(0, STEPS, 10):
    html.append(f"<tr><td>{step}</td><td>{sgd_map.get(step,0):.6e}</td><td>{adam_map.get(step,0):.6e}</td></tr>")
html.append(f"<tr><td>{STEPS-1}</td><td>{hist_sgd[-1]['mse']:.6e}</td><td>{hist_adam[-1]['mse']:.6e}</td></tr>")
html.append("</tbody></table></div>")
html.append("</body></html>")
with open("optimize_sgd_adam.html", "w") as f:
    f.write("\n".join(html))
print("Saved optimize_sgd_adam.html")
print("Done")
