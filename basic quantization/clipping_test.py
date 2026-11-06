"""Outlier clipping test: 99th and 99.9th percentile, 4-bit symmetric per tensor."""
import os, sys, math, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
SIZE = 2048
W = torch.randn(SIZE, SIZE) * 0.02
torch.manual_seed(7)
X = [torch.randn(SIZE) for _ in range(100)]
Y = [W @ x for x in X]

BITS = 4
QMAX = (1 << (BITS - 1)) - 1


def quantize(W, bits):
    scale = W.abs().max() / QMAX
    if scale.item() == 0:
        scale = torch.tensor(1.0)
    Wq = (W / scale).round().clamp(-QMAX, QMAX) * scale
    return Wq, scale.item()


def eval_variant(Wq, X, Y):
    mses = []
    sqnrs = []
    for x, y in zip(X, Y):
        yq = Wq @ x
        mse = F.mse_loss(yq, y).item()
        mses.append(mse)
        sig = (y ** 2).sum().item()
        noise = ((y - yq) ** 2).sum().item()
        if noise > 0 and sig > 0:
            sqnrs.append(10.0 * math.log10(sig / noise))
        else:
            sqnrs.append(float("inf"))
    return sum(mses) / len(mses), sum(sqnrs) / len(sqnrs)


# baseline: no clipping
Wq_base, scale_base = quantize(W, BITS)
mse_base, sqnr_base = eval_variant(Wq_base, X, Y)
clipped_base = 0
print(f"no clip: MSE {mse_base:.6e}, SQNR {sqnr_base:.2f} dB, scale {scale_base:.6f}, clipped 0", flush=True)

results = [{"name": "no clip", "mse": mse_base, "sqnr": sqnr_base, "scale": scale_base, "clipped": 0, "pct": 0.0}]

for pct in (0.99, 0.999):
    thresh = torch.quantile(W.abs().reshape(-1), pct).item()
    Wc = W.clamp(-thresh, thresh)
    n_clipped = int((W.abs() > thresh).sum().item())
    pct_clipped = n_clipped / W.numel() * 100
    Wq, scale = quantize(Wc, BITS)
    mse, sqnr = eval_variant(Wq, X, Y)
    label = f"clip {pct*100:.1f}%ile"
    print(f"{label}: MSE {mse:.6e}, SQNR {sqnr:.2f} dB, scale {scale:.6f}, clipped {n_clipped} ({pct_clipped:.3f}%)", flush=True)
    results.append({"name": label, "mse": mse, "sqnr": sqnr, "scale": scale,
                    "clipped": n_clipped, "pct_clipped": pct_clipped, "threshold": thresh})

with open("clipping_test_results.json", "w") as f:
    json.dump({"results": results,
               "training": {"baseline_mse": 0.03618, "adam_mse": 0.00830, "sgd_mse": 0.02216}}, f, indent=2)

# html
html = []
html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Clipping Test</title>")
html.append("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;} .card{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);} table{border-collapse:collapse;width:100%;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#3498db;color:#fff;} tr:nth-child(even){background:#f9f9f9;} h1,h2{color:#333;}</style>")
html.append("</head><body><h1>Outlier Clipping Test</h1>")
html.append("<div class='card'><h2>Setup</h2>")
html.append("<p>Matrix: 2048 by 2048, standard normal times 0.02, seed 42.</p>")
html.append("<p>Quantization: 4-bit symmetric per tensor.</p>")
html.append("<p>Inputs: 100 vectors of size 2048, seed 7.</p>")
html.append("</div>")
html.append("<div class='card'><h2>Results</h2>")
html.append("<table><thead><tr><th>Variant</th><th>MSE</th><th>SQNR dB</th><th>Scale</th><th>Clipped weights</th><th>% clipped</th></tr></thead><tbody>")
for r in results:
    html.append(f"<tr><td>{r['name']}</td><td>{r['mse']:.6e}</td><td>{r['sqnr']:.2f}</td><td>{r['scale']:.6f}</td><td>{r['clipped']}</td><td>{r.get('pct_clipped',0):.3f}%</td></tr>")
html.append("</tbody></table></div>")
html.append("<div class='card'><h2>Comparison with training</h2>")
html.append("<p>Training baseline MSE: 0.03618.</p>")
html.append("<p>Training Adam MSE: 0.00830, improvement 4.36x.</p>")
html.append("<p>Training SGD MSE: 0.02216, improvement 1.63x.</p>")
html.append("</div>")
html.append("</body></html>")
with open("clipping_test.html", "w") as f:
    f.write("\n".join(html))
print("Saved clipping_test.html")
print("Done")
