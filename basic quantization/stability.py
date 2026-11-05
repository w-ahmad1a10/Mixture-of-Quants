import os, sys, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import quantize_tensor, VARIANTS, variant_name

SIZES = [256, 512, 1024, 2048]
SEEDS = list(range(10))

all_results = {}
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
        mses_this = {}
        for sym, gran, gs, bits in VARIANTS:
            Wq = quantize_tensor(W.clone(), bits, sym, gran, gs)
            yq = Wq @ x
            mse = F.mse_loss(yq, y).item()
            mses_this[variant_name(sym, gran, gs, bits)] = mse
        mn = min(mses_this.values())
        for name, mse in mses_this.items():
            all_results[size][name].append({"mse": mse, "x": mse / mn})

agg = {}
for size in SIZES:
    agg[size] = {}
    for name in all_results[size]:
        ms = [r["mse"] for r in all_results[size][name]]
        xs = [r["x"] for r in all_results[size][name]]
        mean_mse = sum(ms) / len(ms)
        mean_x = sum(xs) / len(xs)
        var_mse = sum((m - mean_mse) ** 2 for m in ms) / len(ms)
        std_mse = var_mse ** 0.5
        cv_mse = (std_mse / mean_mse) if mean_mse > 0 else 0.0
        # overhead and total bits for this matrix size
        # parse variant
        parts = name.split("_")
        sym = parts[0] == "sym"
        bits = int(parts[-1].replace("bit", ""))
        if "group" in name:
            gs = int(parts[-2].replace("group", ""))
            groups = size * (size // gs)
        elif "channel" in name:
            groups = size
        else:
            groups = 1
        zp_overhead = 16 if not sym else 0
        overhead_bits = groups * (16 + zp_overhead)
        weight_bits = size * size * bits
        total_bits = weight_bits + overhead_bits
        kb = total_bits / 8 / 1024
        oh_pct = overhead_bits / weight_bits * 100
        agg[size][name] = {
            "mean_mse": mean_mse,
            "std_mse": std_mse,
            "cv_mse": cv_mse,
            "mean_x": mean_x,
            "min_mse": min(ms),
            "max_mse": max(ms),
            "bits": bits,
            "symmetric": sym,
            "groups": groups,
            "overhead_bits": overhead_bits,
            "total_bits": total_bits,
            "kb": kb,
            "oh_pct": oh_pct,
        }

with open("stability_results.json", "w") as f:
    json.dump({"sizes": SIZES, "seeds": SEEDS, "aggregate": agg}, f, indent=2)

# Build JS data per size
def js_data(size):
    items = []
    for name in sorted(agg[size], key=lambda n: agg[size][n]["mean_mse"]):
        a = agg[size][name]
        items.append(f'{{name:"{name}", bits:{a["bits"]}, mse:{a["mean_mse"]:.6e}, x:{a["mean_x"]:.2f}, oh:{a["oh_pct"]:.2f}, total:{a["total_bits"]}, kb:{a["kb"]:.2f}}}')
    return "[" + ",".join(items) + "]"

html = []
html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Stability Test with Calculator</title>")
html.append("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;} table{border-collapse:collapse;width:100%;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#3498db;color:#fff;cursor:pointer;} tr:nth-child(even){background:#f9f9f9;} h2{color:#333;} h3{color:#555;} .card{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);} .result{font-size:14px;color:#2c3e50;margin:6px 0;} .big{font-size:18px;font-weight:bold;color:#e74c3c;} select{padding:6px;font-size:13px;min-width:200px;} .row{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:8px 0;} button{padding:6px 12px;background:#3498db;color:#fff;border:none;border-radius:4px;cursor:pointer;}</style>")
html.append("</head><body><h1>Seed and Size Stability Test with Calculator</h1>")
html.append(f"<p>10 seeds (0-9), 4 matrix sizes, 48 variants. Mean MSE, std MSE, CV of MSE, and mean X across seeds.</p>")

for size in SIZES:
    html.append(f"<div class='card'><h2>Matrix size {size}x{size}</h2>")
    # table
    html.append("<table><thead><tr><th>Variant</th><th>Mean MSE</th><th>Std MSE</th><th>CV of MSE</th><th>Mean X</th><th>Total bits</th><th>Size KB</th></tr></thead><tbody>")
    for name in sorted(agg[size], key=lambda n: agg[size][n]["mean_mse"]):
        a = agg[size][name]
        html.append(f"<tr><td>{name}</td><td>{a['mean_mse']:.4e}</td><td>{a['std_mse']:.4e}</td><td>{a['cv_mse']:.4f}</td><td>{a['mean_x']:.2f}x</td><td>{a['total_bits']}</td><td>{a['kb']:.2f}</td></tr>")
    html.append("</tbody></table>")
    # best
    best = min(agg[size].items(), key=lambda kv: kv[1]["mean_mse"])
    html.append(f"<p><b>Best:</b> {best[0]} (mean MSE {best[1]['mean_mse']:.4e})</p>")
    # most stable
    ranked = sorted(agg[size].items(), key=lambda kv: kv[1]["cv_mse"])[:5]
    html.append(f"<p><b>Most stable:</b> " + ", ".join(f"{n} (CV {a['cv_mse']:.3f})" for n, a in ranked) + "</p>")
    html.append("</div>")

    # calculator for this size
    html.append(f"<div class='card'><h3>Calculator: {size}x{size}</h3>")
    html.append(f"<div class='row'><label>Model A:</label><select id='selA_{size}' onchange='updateCompare({size})'></select>")
    html.append(f"<label>Model B:</label><select id='selB_{size}' onchange='updateCompare({size})'></select></div>")
    html.append(f"<div class='result' id='resA_{size}'></div>")
    html.append(f"<div class='result' id='resB_{size}'></div>")
    html.append(f"<div class='result big' id='ratio_{size}'></div>")
    html.append(f"<div class='result' id='trade_{size}'></div>")
    html.append("</div>")

# JS data
html.append("<script>")
for size in SIZES:
    html.append(f"const data{size} = {js_data(size)};")
html.append("""
function populate(size) {
  const selA = document.getElementById('selA_' + size);
  const selB = document.getElementById('selB_' + size);
  const data = window['data' + size];
  selA.innerHTML = ''; selB.innerHTML = '';
  data.forEach((d, i) => {
    selA.add(new Option(d.name, i));
    selB.add(new Option(d.name, i));
  });
  if (selB.options.length > 1) selB.selectedIndex = 1;
}
function updateCompare(size) {
  const data = window['data' + size];
  const a = data[document.getElementById('selA_' + size).selectedIndex];
  const b = data[document.getElementById('selB_' + size).selectedIndex];
  if (!a || !b) return;
  document.getElementById('resA_' + size).innerHTML =
    "<b>" + a.name + "</b>: Mean MSE = " + a.mse.toExponential(4) +
    ", Mean X = " + a.x.toFixed(2) + "x, Overhead = " + a.oh.toFixed(2) + "%, Size = " + a.kb.toFixed(2) + " KB";
  document.getElementById('resB_' + size).innerHTML =
    "<b>" + b.name + "</b>: Mean MSE = " + b.mse.toExponential(4) +
    ", Mean X = " + b.x.toFixed(2) + "x, Overhead = " + b.oh.toFixed(2) + "%, Size = " + b.kb.toFixed(2) + " KB";
  const ratio = b.mse / a.mse;
  document.getElementById('ratio_' + size).innerText =
    "Model B has " + ratio.toFixed(2) + "x the error of Model A.";
  const sizeDiff = Math.abs(a.kb - b.kb);
  const smallerMse = Math.min(a.mse, b.mse);
  const largerMse = Math.max(a.mse, b.mse);
  const errorPctMore = smallerMse > 0 ? ((largerMse - smallerMse) / smallerMse * 100) : 0;
  const bigger = a.mse > b.mse ? a.name : b.name;
  const smaller = a.mse > b.mse ? b.name : a.name;
  document.getElementById('trade_' + size).innerText =
    "Size difference: " + sizeDiff.toFixed(2) + " KB. " + bigger + " has " + errorPctMore.toFixed(1) + "% more error than " + smaller + ".";
}
""")
for size in SIZES:
    html.append(f"populate({size}); updateCompare({size});")
html.append("</script>")
html.append("</body></html>")

with open("stability.html", "w") as f:
    f.write("\n".join(html))
print("Saved stability.html")
print("Done")
