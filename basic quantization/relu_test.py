import os, sys, json, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import quantize_tensor, VARIANTS, variant_name

SEEDS = list(range(10))
SIZE = 2048
CONDITIONS = ["no_relu", "relu"]

# results[condition][variant] = list of {mse, x}
results = {c: {} for c in CONDITIONS}
for c in CONDITIONS:
    for sym, gran, gs, bits in VARIANTS:
        results[c][variant_name(sym, gran, gs, bits)] = []

for seed in SEEDS:
    for cond in CONDITIONS:
        torch.manual_seed(seed)
        W = torch.randn(SIZE, SIZE) * 0.02
        if cond == "relu":
            W = torch.relu(W)
        x = torch.randn(SIZE)
        y = W @ x
        mses_this = {}
        sqnrs_this = {}
        for sym, gran, gs, bits in VARIANTS:
            Wq = quantize_tensor(W.clone(), bits, sym, gran, gs)
            yq = Wq @ x
            mse = F.mse_loss(yq, y).item()
            mses_this[variant_name(sym, gran, gs, bits)] = mse
            # SQNR on weights
            wvar = W.var().item()
            wnoise = ((W - Wq) ** 2).mean().item()
            if wnoise > 0 and wvar > 0:
                sqnrs_this[variant_name(sym, gran, gs, bits)] = 10.0 * math.log10(wvar / wnoise)
            else:
                sqnrs_this[variant_name(sym, gran, gs, bits)] = float("inf")
        mn = min(mses_this.values())
        for name, mse in mses_this.items():
            results[cond][name].append({"mse": mse, "x": mse / mn, "sqnr": sqnrs_this[name]})

# aggregate
agg = {c: {} for c in CONDITIONS}
for cond in CONDITIONS:
    for name in results[cond]:
        ms = [r["mse"] for r in results[cond][name]]
        xs = [r["x"] for r in results[cond][name]]
        sqs = [r["sqnr"] for r in results[cond][name]]
        mean_mse = sum(ms) / len(ms)
        mean_x = sum(xs) / len(xs)
        mean_sqnr = sum(sqs) / len(sqs)
        var_mse = sum((m - mean_mse) ** 2 for m in ms) / len(ms)
        std_mse = var_mse ** 0.5
        cv_mse = (std_mse / mean_mse) if mean_mse > 0 else 0.0
        agg[cond][name] = {
            "mean_mse": mean_mse,
            "std_mse": std_mse,
            "cv_mse": cv_mse,
            "mean_x": mean_x,
            "mean_sqnr": mean_sqnr,
            "min_mse": min(ms),
            "max_mse": max(ms),
        }

# ratio table
ratio_rows = []
for sym, gran, gs, bits in VARIANTS:
    name = variant_name(sym, gran, gs, bits)
    a = agg["no_relu"][name]["mean_mse"]
    b = agg["relu"][name]["mean_mse"]
    sa = agg["no_relu"][name]["mean_sqnr"]
    sb = agg["relu"][name]["mean_sqnr"]
    ratio = b / a if a > 0 else float("inf")
    sqnr_ratio = sb / sa if sa != 0 else float("inf")
    ratio_rows.append((name, bits, a, b, ratio, sa, sb, sqnr_ratio))

with open("relu_results.json", "w") as f:
    json.dump({"conditions": CONDITIONS, "seeds": SEEDS, "size": SIZE, "aggregate": agg, "ratios": ratio_rows}, f, indent=2)

# html
html = []
html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>ReLU vs No-ReLU Test</title>")
html.append("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;} table{border-collapse:collapse;width:100%;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#3498db;color:#fff;} tr:nth-child(even){background:#f9f9f9;} h2{color:#333;} .card{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);}</style>")
html.append("</head><body><h1>ReLU vs No-ReLU Stability Test</h1>")
html.append(f"<p>Matrix size {SIZE}x{SIZE}, 10 seeds, 48 variants, no ReLU vs ReLU applied to weights.</p>")

for cond in CONDITIONS:
    label = "No ReLU" if cond == "no_relu" else "With ReLU"
    html.append(f"<div class='card'><h2>{label}</h2>")
    html.append("<table><thead><tr><th>Variant</th><th>Mean MSE</th><th>Std MSE</th><th>CV of MSE</th><th>Mean X</th><th>Mean SQNR dB</th></tr></thead><tbody>")
    for name in sorted(agg[cond], key=lambda n: agg[cond][n]["mean_mse"]):
        a = agg[cond][name]
        html.append(f"<tr><td>{name}</td><td>{a['mean_mse']:.4e}</td><td>{a['std_mse']:.4e}</td><td>{a['cv_mse']:.4f}</td><td>{a['mean_x']:.2f}x</td><td>{a['mean_sqnr']:.2f}</td></tr>")
    html.append("</tbody></table></div>")

html.append("<div class='card'><h2>Ratio: MSE with ReLU / MSE without ReLU</h2>")
html.append("<p>Below 1.0 means ReLU reduced error. Above 1.0 means ReLU increased error.</p>")
html.append("<table><thead><tr><th>Variant</th><th>MSE no ReLU</th><th>MSE with ReLU</th><th>MSE Ratio</th><th>SQNR no ReLU</th><th>SQNR with ReLU</th><th>SQNR Ratio</th></tr></thead><tbody>")
for name, bits, a, b, ratio, sa, sb, sqnr_ratio in sorted(ratio_rows, key=lambda r: r[4]):
    html.append(f"<tr><td>{name}</td><td>{a:.4e}</td><td>{b:.4e}</td><td>{ratio:.4f}</td><td>{sa:.2f}</td><td>{sb:.2f}</td><td>{sqnr_ratio:.4f}</td></tr>")
html.append("</tbody></table></div>")

html.append("</body></html>")
with open("relu_test.html", "w") as f:
    f.write("\n".join(html))
print("Saved relu_test.html")
print("Done")
