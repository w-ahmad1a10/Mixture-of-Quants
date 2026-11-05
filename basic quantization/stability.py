import os, sys, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import quantize_tensor, VARIANTS, variant_name

SIZES = [256, 512, 1024]
SEEDS = list(range(10))

all_results = {}  # size -> variant -> list of {mse, x}

for size in SIZES:
    all_results[size] = {}
    for sym, gran, gs, bits in VARIANTS:
        all_results[size][variant_name(sym, gran, gs, bits)] = []

for seed in SEEDS:
    for size in SIZES:
        torch.manual_seed(seed)
        W = torch.randn(size, size) * 0.02
        x = torch.randn(size)
        y = W @ x
        # per size min mse
        mses_this = {}
        for sym, gran, gs, bits in VARIANTS:
            Wq = quantize_tensor(W.clone(), bits, sym, gran, gs)
            yq = Wq @ x
            mse = F.mse_loss(yq, y).item()
            mses_this[variant_name(sym, gran, gs, bits)] = mse
        mn = min(mses_this.values())
        for name, mse in mses_this.items():
            all_results[size][name].append({"mse": mse, "x": mse / mn})

# aggregate
agg = {}
for size in SIZES:
    agg[size] = {}
    for name in all_results[size]:
        ms = [r["mse"] for r in all_results[size][name]]
        xs = [r["x"] for r in all_results[size][name]]
        mean_mse = sum(ms) / len(ms)
        mean_x = sum(xs) / len(xs)
        var_mse = sum((m - mean_mse) ** 2 for m in ms) / len(ms)
        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        # best count
        best_count = 0
        for seed in SEEDS:
            # recompute min for that seed/size? approximate: count how many times this variant had x==1
            pass
        # count best by checking stored x==1.0 (approx)
        bc = sum(1 for r in all_results[size][name] if r["x"] <= 1.0 + 1e-9)
        agg[size][name] = {
            "mean_mse": mean_mse,
            "std_mse": var_mse ** 0.5,
            "mean_x": mean_x,
            "std_x": var_x ** 0.5,
            "best_count": bc,
            "min_mse": min(ms),
            "max_mse": max(ms),
        }

# save json
with open("stability_results.json", "w") as f:
    json.dump({"sizes": SIZES, "seeds": SEEDS, "aggregate": agg}, f, indent=2)

# generate html
html = []
html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Stability Test</title>")
html.append("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;} table{border-collapse:collapse;width:100%;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#3498db;color:#fff;} tr:nth-child(even){background:#f9f9f9;} h2{color:#333;} .card{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);}</style>")
html.append("</head><body><h1>Seed and Size Stability Test</h1>")
html.append(f"<p>10 seeds (0-9), 3 matrix sizes, 48 variants. Mean and std of MSE and X across seeds.</p>")

for size in SIZES:
    html.append(f"<div class='card'><h2>Matrix size {size}x{size}</h2>")
    html.append("<table><thead><tr><th>Variant</th><th>Mean MSE</th><th>Std MSE</th><th>Mean X</th><th>Std X</th><th>Best count</th></tr></thead><tbody>")
    for name in sorted(agg[size], key=lambda n: agg[size][n]["mean_mse"]):
        a = agg[size][name]
        html.append(f"<tr><td>{name}</td><td>{a['mean_mse']:.4e}</td><td>{a['std_mse']:.4e}</td><td>{a['mean_x']:.2f}x</td><td>{a['std_x']:.2f}</td><td>{a['best_count']}/10</td></tr>")
    html.append("</tbody></table></div>")

# stability summary: top 3 most stable (lowest std_x) per size
html.append("<div class='card'><h2>Most stable variants (lowest std of X)</h2>")
for size in SIZES:
    ranked = sorted(agg[size].items(), key=lambda kv: kv[1]["std_x"])
    html.append(f"<p><b>{size}x{size}:</b> " + ", ".join(f"{n} (std {a['std_x']:.3f})" for n, a in ranked[:5]) + "</p>")
html.append("</div>")

# best variant per size
html.append("<div class='card'><h2>Best variant per size (by mean MSE)</h2>")
for size in SIZES:
    best = min(agg[size].items(), key=lambda kv: kv[1]["mean_mse"])
    html.append(f"<p><b>{size}x{size}:</b> {best[0]} (mean MSE {best[1]['mean_mse']:.4e}, best in {best[1]['best_count']}/10 seeds)</p>")
html.append("</div>")

html.append("</body></html>")
with open("stability.html", "w") as f:
    f.write("\n".join(html))
print("Saved stability.html")
print("Done")
