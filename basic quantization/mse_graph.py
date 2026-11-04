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

# print table
print("name, total_bits, mse")
for d in sorted(data, key=lambda r: r["total_bits"]):
    print(f"{d['name']}, {d['total_bits']}, {d['mse']:.6e}")

# plot
fig, ax = plt.subplots(figsize=(10, 7))
colors = {3: "red", 4: "orange", 6: "green", 8: "blue"}
markers = {"per_tensor": "o", "per_channel": "s", "per_layer": "^", "per_group": "D"}
for d in data:
    ax.scatter(d["total_bits"], d["mse"],
               c=colors[d["bits"]], marker=markers[d["granularity"]],
               s=60, alpha=0.8, edgecolors="black", linewidths=0.5)
# annotate a few
for d in data:
    if d["granularity"] == "per_group" and d["gs"] == 32 and d["bits"] in (3, 8):
        ax.annotate(d["name"], (d["total_bits"], d["mse"]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

ax.set_yscale("log")
ax.set_xlabel("Total storage size (bits): weights + scales + zero points")
ax.set_ylabel("MSE (log scale)")
ax.set_title("Quantization tradeoff: storage size vs reconstruction error\n"
             "One fixed 512x512 tensor, one fixed input, 48 variants")
ax.grid(True, which="both", alpha=0.3)

# legend for bits
from matplotlib.lines import Line2D
bit_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[b],
                      label=f"{b}-bit", markersize=8) for b in (3, 4, 6, 8)]
gran_handles = [Line2D([0], [0], marker=markers[g], color="w", markerfacecolor="gray",
                       label=g.replace("per_", ""), markersize=8)
                for g in ("per_tensor", "per_channel", "per_layer", "per_group")]
leg1 = ax.legend(handles=bit_handles, title="Bits", loc="upper right")
ax.add_artist(leg1)
ax.legend(handles=gran_handles, title="Granularity", loc="lower left")

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mse_graph.png")
plt.savefig(out, dpi=150)
print("\nSaved graph to", out)
plt.close()
