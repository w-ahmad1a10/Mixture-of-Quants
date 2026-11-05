"""Train on 10 inputs, test on 10 unseen inputs. 4-bit symmetric quantization."""
import os, sys, math, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
SIZE = 2048
W0 = torch.randn(SIZE, SIZE) * 0.02
# 10 train inputs, 10 test inputs
torch.manual_seed(7)
X_train = [torch.randn(SIZE) for _ in range(10)]
X_test = [torch.randn(SIZE) for _ in range(10)]
# original outputs
Y0_train = [W0 @ x for x in X_train]
Y0_test = [W0 @ x for x in X_test]

BITS = 4
QMAX = (1 << (BITS - 1)) - 1


def quantize_ste(W):
    with torch.no_grad():
        scale = W.abs().max() / QMAX
        if scale.item() == 0:
            scale = torch.tensor(1.0)
        Wq = (W / scale).round().clamp(-QMAX, QMAX) * scale
    return W + (Wq - W).detach()


def mse_list(Wq, Xs, Y0s):
    errs = []
    for x, y0 in zip(Xs, Y0s):
        yq = Wq @ x
        errs.append(F.mse_loss(yq, y0).item())
    return sum(errs) / len(errs)

# before
W = W0.clone().detach().requires_grad_(True)
with torch.no_grad():
    Wq0 = quantize_ste(W)
train_mse_before = mse_list(Wq0, X_train, Y0_train)
test_mse_before = mse_list(Wq0, X_test, Y0_test)

steps = 200
lr = 1e-2
history = []
for step in range(steps):
    Wq = quantize_ste(W)
    loss = 0.0
    for x, y0 in zip(X_train, Y0_train):
        yq = Wq @ x
        loss = loss + F.mse_loss(yq, y0)
    loss = loss / len(X_train)
    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad
        W.grad.zero_()
    if step % 10 == 0 or step == steps - 1:
        with torch.no_grad():
            Wq2 = quantize_ste(W)
            cur_train = mse_list(Wq2, X_train, Y0_train)
            cur_test = mse_list(Wq2, X_test, Y0_test)
        history.append({"step": step, "train_mse": cur_train, "test_mse": cur_test})
        print(f"step {step:3d}: train {cur_train:.4e}  test {cur_test:.4e}", flush=True)

with torch.no_grad():
    Wq_final = quantize_ste(W)
    train_mse_after = mse_list(Wq_final, X_train, Y0_train)
    test_mse_after = mse_list(Wq_final, X_test, Y0_test)
    weight_mse = F.mse_loss(W, W0).item()

print("\n=== RESULTS ===")
print(f"Train MSE before: {train_mse_before:.6e}")
print(f"Train MSE after:  {train_mse_after:.6e}")
print(f"Test MSE before:  {test_mse_before:.6e}")
print(f"Test MSE after:   {test_mse_after:.6e}")
print(f"Weight change:    {weight_mse:.6e}")

with open("optimize_train_test_results.json", "w") as f:
    json.dump({
        "train_mse_before": train_mse_before,
        "train_mse_after": train_mse_after,
        "test_mse_before": test_mse_before,
        "test_mse_after": test_mse_after,
        "weight_mse": weight_mse,
        "steps": steps,
        "lr": lr,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "size": SIZE,
        "history": history,
    }, f, indent=2)

# html
html = []
html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Train/Test Optimization</title>")
html.append("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;} .card{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);} table{border-collapse:collapse;width:100%;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#3498db;color:#fff;} tr:nth-child(even){background:#f9f9f9;} h1,h2{color:#333;}</style>")
html.append("</head><body><h1>Train/Test Output Optimization</h1>")
html.append(f"<div class='card'><h2>Summary</h2>")
html.append(f"<p>Matrix size: {SIZE} by {SIZE}</p>")
html.append(f"<p>Train inputs: {len(X_train)}, Test inputs: {len(X_test)}</p>")
html.append(f"<p>Steps: {steps}, learning rate: {lr}</p>")
html.append(f"<p>Train MSE before: {train_mse_before:.6e}</p>")
html.append(f"<p>Train MSE after:  {train_mse_after:.6e}</p>")
html.append(f"<p>Test MSE before:  {test_mse_before:.6e}</p>")
html.append(f"<p>Test MSE after:   {test_mse_after:.6e}</p>")
html.append(f"<p>Weight change MSE: {weight_mse:.6e}</p>")
html.append("</div>")
html.append("<div class='card'><h2>Loss per step</h2>")
html.append("<table><thead><tr><th>Step</th><th>Train MSE</th><th>Test MSE</th></tr></thead><tbody>")
for h in history:
    html.append(f"<tr><td>{h['step']}</td><td>{h['train_mse']:.6e}</td><td>{h['test_mse']:.6e}</td></tr>")
html.append("</tbody></table></div>")
html.append("</body></html>")
with open("optimize_train_test.html", "w") as f:
    f.write("\n".join(html))
print("Saved optimize_train_test.html")
print("Done")
