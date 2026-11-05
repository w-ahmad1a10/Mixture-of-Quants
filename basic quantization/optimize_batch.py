"""Train with batch size 10 on 3-bit and 4-bit symmetric quantization."""
import os, sys, math, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
SIZE = 2048
W0 = torch.randn(SIZE, SIZE) * 0.02
torch.manual_seed(7)
X_train = [torch.randn(SIZE) for _ in range(100)]
Y0_train = [W0 @ x for x in X_train]

BATCH = 10
STEPS = 100

def quantize_ste(W, bits):
    QMAX = (1 << (bits - 1)) - 1
    with torch.no_grad():
        scale = W.abs().max() / QMAX
        if scale.item() == 0:
            scale = torch.tensor(1.0)
        Wq = (W / scale).round().clamp(-QMAX, QMAX) * scale
    return W + (Wq - W).detach()

def full_train_mse(W, bits):
    with torch.no_grad():
        Wq = quantize_ste(W, bits)
        errs = [F.mse_loss(Wq @ x, y0).item() for x, y0 in zip(X_train, Y0_train)]
    return sum(errs) / len(errs)

all_results = {}

for bits in (3, 4):
    print(f"\n========== {bits}-BIT ==========", flush=True)
    QMAX = (1 << (bits - 1)) - 1
    # baseline
    with torch.no_grad():
        Wq0 = quantize_ste(W0.clone(), bits)
        base_mse = sum(F.mse_loss(Wq0 @ x, y0).item() for x, y0 in zip(X_train, Y0_train)) / len(X_train)
    print(f"Baseline train MSE: {base_mse:.6e}", flush=True)

    # SGD
    print("--- SGD ---", flush=True)
    W_sgd = W0.clone().detach().requires_grad_(True)
    opt_sgd = torch.optim.SGD([W_sgd], lr=1e-1)
    hist_sgd = []
    for step in range(STEPS):
        # mini-batch
        start = (step * BATCH) % len(X_train)
        batch_x = X_train[start:start + BATCH]
        batch_y = Y0_train[start:start + BATCH]
        opt_sgd.zero_grad()
        Wq = quantize_ste(W_sgd, bits)
        loss = sum(F.mse_loss(Wq @ x, y0) for x, y0 in zip(batch_x, batch_y)) / len(batch_x)
        loss.backward()
        opt_sgd.step()
        if step % 10 == 0 or step == STEPS - 1:
            cur = full_train_mse(W_sgd.detach(), bits)
            hist_sgd.append({"step": step, "mse": cur})
            print(f"  step {step:3d}: train mse {cur:.4e}", flush=True)
    final_sgd = full_train_mse(W_sgd.detach(), bits)
    wchg_sgd = F.mse_loss(W_sgd.detach(), W0).item()

    # Adam
    print("--- Adam ---", flush=True)
    W_adam = W0.clone().detach().requires_grad_(True)
    opt_adam = torch.optim.Adam([W_adam], lr=1e-4, betas=(0.9, 0.95))
    hist_adam = []
    for step in range(STEPS):
        start = (step * BATCH) % len(X_train)
        batch_x = X_train[start:start + BATCH]
        batch_y = Y0_train[start:start + BATCH]
        opt_adam.zero_grad()
        Wq = quantize_ste(W_adam, bits)
        loss = sum(F.mse_loss(Wq @ x, y0) for x, y0 in zip(batch_x, batch_y)) / len(batch_x)
        loss.backward()
        opt_adam.step()
        if step % 10 == 0 or step == STEPS - 1:
            cur = full_train_mse(W_adam.detach(), bits)
            hist_adam.append({"step": step, "mse": cur})
            print(f"  step {step:3d}: train mse {cur:.4e}", flush=True)
    final_adam = full_train_mse(W_adam.detach(), bits)
    wchg_adam = F.mse_loss(W_adam.detach(), W0).item()

    all_results[bits] = {
        "baseline": base_mse,
        "sgd": {"final": final_sgd, "weight_change": wchg_sgd, "history": hist_sgd,
                "improvement": base_mse / final_sgd if final_sgd > 0 else float("inf")},
        "adam": {"final": final_adam, "weight_change": wchg_adam, "history": hist_adam,
                 "improvement": base_mse / final_adam if final_adam > 0 else float("inf")},
    }
    print(f"\n{bits}-bit: baseline {base_mse:.4e} | SGD {final_sgd:.4e} ({base_mse/final_sgd:.2f}x) | Adam {final_adam:.4e} ({base_mse/final_adam:.2f}x)")

with open("optimize_batch_results.json", "w") as f:
    json.dump({"size": SIZE, "batch": BATCH, "steps": STEPS, "n_train": len(X_train), "results": all_results}, f, indent=2)

# html
html = []
html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Batch Size 10 Optimization</title>")
html.append("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;} .card{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);} table{border-collapse:collapse;width:100%;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#3498db;color:#fff;} tr:nth-child(even){background:#f9f9f9;} h1,h2{color:#333;}</style>")
html.append("</head><body><h1>Batch Size 10: 3-bit and 4-bit</h1>")
html.append(f"<div class='card'><h2>Summary</h2>")
html.append(f"<p>Matrix: {SIZE}x{SIZE}, batch size {BATCH}, {STEPS} steps, 100 train inputs</p>")
for bits in (3, 4):
    r = all_results[bits]
    html.append(f"<h3>{bits}-bit</h3>")
    html.append(f"<p>Baseline train MSE: {r['baseline']:.6e}</p>")
    html.append(f"<p>SGD final: {r['sgd']['final']:.6e} (improvement {r['sgd']['improvement']:.2f}x, weight change {r['sgd']['weight_change']:.6e})</p>")
    html.append(f"<p>Adam final: {r['adam']['final']:.6e} (improvement {r['adam']['improvement']:.2f}x, weight change {r['adam']['weight_change']:.6e})</p>")
html.append("</div>")
for bits in (3, 4):
    r = all_results[bits]
    html.append(f"<div class='card'><h2>{bits}-bit loss per step</h2>")
    html.append("<table><thead><tr><th>Step</th><th>SGD train MSE</th><th>Adam train MSE</th></tr></thead><tbody>")
    sgd_map = {h["step"]: h["mse"] for h in r["sgd"]["history"]}
    adam_map = {h["step"]: h["mse"] for h in r["adam"]["history"]}
    for step in range(0, STEPS, 10):
        html.append(f"<tr><td>{step}</td><td>{sgd_map.get(step,0):.6e}</td><td>{adam_map.get(step,0):.6e}</td></tr>")
    html.append(f"<tr><td>{STEPS-1}</td><td>{r['sgd']['history'][-1]['mse']:.6e}</td><td>{r['adam']['history'][-1]['mse']:.6e}</td></tr>")
    html.append("</tbody></table></div>")
html.append("</body></html>")
with open("optimize_batch.html", "w") as f:
    f.write("\n".join(html))
print("Saved optimize_batch.html")
print("Done")
