import os, sys, json, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import quantize_tensor

torch.manual_seed(42)
SIZE = 2048
W_uniform = torch.rand(SIZE, SIZE) * 0.04 - 0.02
W_normal = torch.randn(SIZE, SIZE) * 0.02
x = torch.randn(SIZE)
y_uniform = W_uniform @ x
y_normal = W_normal @ x

def rescale(W, target_std=0.02):
    s = W.std()
    if s > 0:
        W = W / s * target_std
    return W

def sqnr_db(W, Wq):
    wvar = W.var().item()
    wnoise = ((W - Wq) ** 2).mean().item()
    if wnoise > 0 and wvar > 0:
        return 10.0 * math.log10(wvar / wnoise)
    return float("inf")

def eval_matrix(W, x):
    y = W @ x
    Wq = quantize_tensor(W.clone(), 4, True, "per_tensor", None)
    yq = Wq @ x
    mse = F.mse_loss(yq, y).item()
    sq = sqnr_db(W, Wq)
    return mse, sq

results = []

# baseline: uniform
mse, sq = eval_matrix(W_uniform, x)
results.append(("uniform (baseline)", W_uniform.clone(), mse, sq))

# baseline: normal
mse, sq = eval_matrix(W_normal, x)
results.append(("normal (baseline)", W_normal.clone(), mse, sq))

# candidate transforms on uniform
candidates = []
# tanh soft clip
for t in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]:
    Wt = torch.tanh(W_uniform / t) * t
    Wt = rescale(Wt)
    candidates.append((f"tanh_clip t={t}", Wt))
# power transform
for p in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
    Wt = torch.sign(W_uniform) * (W_uniform.abs() ** p)
    Wt = rescale(Wt)
    candidates.append((f"power p={p}", Wt))
# hard clip
for c in [0.02, 0.03, 0.04, 0.05]:
    Wt = W_uniform.clamp(-c, c)
    Wt = rescale(Wt)
    candidates.append((f"hard_clip c={c}", Wt))
# sigmoid squashing
for t in [0.01, 0.02, 0.05]:
    Wt = torch.sigmoid(W_uniform / t) * 2 * t - t  # approx
    Wt = rescale(Wt)
    candidates.append((f"sigmoid t={t}", Wt))

for name, Wt in candidates:
    mse, sq = eval_matrix(Wt, x)
    results.append((name, Wt, mse, sq))

# find best
best = max(results, key=lambda r: r[3])
print("=== RESULTS (4-bit symmetric per tensor) ===")
print(f"{'name':28s} {'MSE':>12s} {'SQNR_dB':>10s}")
for name, Wt, mse, sq in sorted(results, key=lambda r: -r[3]):
    marker = "  <- BEST" if name == best[0] else ""
    print(f"{name:28s} {mse:12.6e} {sq:10.2f}{marker}")

print(f"\nBest transform: {best[0]}")
print(f"Best SQNR: {best[3]:.2f} dB")
print(f"Best MSE: {best[2]:.6e}")
print(f"Uniform baseline SQNR: {results[0][3]:.2f} dB")
print(f"Normal baseline SQNR: {results[1][3]:.2f} dB")
print(f"Improvement over uniform: {best[3]-results[0][3]:.2f} dB")
print(f"Improvement over normal: {best[3]-results[1][3]:.2f} dB")

# save best matrix stats
Wbest = best[1]
print(f"\nBest matrix stats: mean={Wbest.mean().item():.6f} std={Wbest.std().item():.6f} min={Wbest.min().item():.6f} max={Wbest.max().item():.6f}")

with open("optimize_matrix_results.json", "w") as f:
    json.dump({
        "best": best[0],
        "best_sqnr": best[3],
        "best_mse": best[2],
        "uniform_sqnr": results[0][3],
        "normal_sqnr": results[1][3],
        "all": [{"name": r[0], "mse": r[2], "sqnr": r[3]} for r in results],
    }, f, indent=2)
print("Saved optimize_matrix_results.json")
print("Done")
