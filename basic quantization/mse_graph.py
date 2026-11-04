import os, sys, torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantize import quantize_tensor, VARIANTS, variant_name

torch.manual_seed(42)
W = torch.randn(512, 512) * 0.02
x = torch.randn(512)
y = W @ x
N = W.numel()
ROWS, COLS = W.shape

def num_groups(granularity, gs):
    if granularity in ("per_tensor", "per_layer"):
        return 1
    if granularity == "per_channel":
        return ROWS
    if granularity == "per_group":
        g = gs or 32
        return ROWS * (COLS // g)
    return 1

data = []
for symmetric, granularity, gs, bits in VARIANTS:
    Wq = quantize_tensor(W.clone(), bits, symmetric, granularity, gs)
    yq = Wq @ x
    mse = F.mse_loss(yq, y).item()
    ng = num_groups(granularity, gs)
    zp_overhead = 16 if not symmetric else 0
    total_bits = N * bits + ng * (16 + zp_overhead)
    data.append({
        "name": variant_name(symmetric, granularity, gs, bits),
        "bits": bits,
        "symmetric": symmetric,
        "granularity": granularity,
        "gs": gs,
        "mse": mse,
        "total_bits": total_bits,
        "ng": ng,
    })

# drop per_layer because it equals per_tensor for a single tensor
data = [d for d in data if d["granularity"] != "per_layer"]

# print table
print("name, total_bits, mse")
for d in sorted(data, key=lambda r: r["total_bits"]):
    print(f"{d['name']}, {d['total_bits']}, {d['mse']:.6e}")

# plot: 2x2 grid by bit width, color by sym/asym, marker by granularity, annotate group size
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
bit_list = [3, 4, 6, 8]
for ax, b in zip(axes.flat, bit_list):
    sub = [d for d in data if d["bits"] == b]
    for d in sub:
        color = "tab:blue" if d["symmetric"] else "tab:red"
        marker = {"per_tensor": "o", "per_channel": "s", "per_layer": "^", "per_group": "D"}[d["granularity"]]
        size = 90 if d["granularity"] == "per_group" else 60
        ax.scatter(d["total_bits"], d["mse"], c=color, marker=marker, s=size,
                   alpha=0.85, edgecolors="black", linewidths=0.5)
        if d["granularity"] == "per_group":
            ax.annotate(f"g{d['gs']}", (d["total_bits"], d["mse"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_title(f"{b}-bit", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total storage (bits)")
    ax.grid(True, which="both", alpha=0.3)
axes[0, 0].set_ylabel("MSE (log scale)")
axes[1, 0].set_ylabel("MSE (log scale)")

from matplotlib.lines import Line2D
sym_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", label="symmetric", markersize=8),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", label="asymmetric", markersize=8)]
gran_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="gray", label=g.replace("per_", ""), markersize=8)
                for g, m in (("per_tensor", "o"), ("per_channel", "s"), ("per_group", "D"))]
fig.legend(handles=sym_handles + gran_handles, loc="lower center", ncol=4, fontsize=9)
fig.suptitle("Quantization tradeoff: storage size vs MSE (log scale)\n"
             "One fixed 512x512 tensor, one fixed input, 48 variants", fontsize=13)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mse_graph.png")
plt.savefig(out, dpi=150)
print("\nSaved graph to", out)
plt.close()
