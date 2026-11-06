"""Clipping test across 3 seeds, no HTML."""
import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIZE = 2048
BITS = 4
QMAX = (1 << (BITS - 1)) - 1

def quantize(W):
    scale = W.abs().max() / QMAX
    if scale.item() == 0:
        scale = torch.tensor(1.0)
    return (W / scale).round().clamp(-QMAX, QMAX) * scale, scale.item()

def eval_variant(Wq, X, Y):
    mses = []; sqnrs = []
    for x, y in zip(X, Y):
        yq = Wq @ x
        mses.append(F.mse_loss(yq, y).item())
        sig = (y ** 2).sum().item(); noise = ((y - yq) ** 2).sum().item()
        sqnrs.append(10.0 * math.log10(sig / noise) if (noise > 0 and sig > 0) else float("inf"))
    return sum(mses)/len(mses), sum(sqnrs)/len(sqnrs)

seeds = [0, 1, 2]
agg = {"no clip": [], "clip99": [], "clip999": []}
for seed in seeds:
    torch.manual_seed(seed)
    W = torch.randn(SIZE, SIZE) * 0.02
    torch.manual_seed(seed + 100)
    X = [torch.randn(SIZE) for _ in range(100)]
    Y = [W @ x for x in X]
    # no clip
    Wq, sc = quantize(W)
    mse, sqnr = eval_variant(Wq, X, Y)
    agg["no clip"].append((mse, sqnr, sc, 0))
    # clip 99
    thresh = torch.quantile(W.abs().reshape(-1), 0.99).item()
    Wc = W.clamp(-thresh, thresh)
    nclip = int((W.abs() > thresh).sum().item())
    Wq, sc = quantize(Wc)
    mse, sqnr = eval_variant(Wq, X, Y)
    agg["clip99"].append((mse, sqnr, sc, nclip))
    # clip 99.9
    thresh = torch.quantile(W.abs().reshape(-1), 0.999).item()
    Wc = W.clamp(-thresh, thresh)
    nclip = int((W.abs() > thresh).sum().item())
    Wq, sc = quantize(Wc)
    mse, sqnr = eval_variant(Wq, X, Y)
    agg["clip999"].append((mse, sqnr, sc, nclip))

for name in ["no clip", "clip99", "clip999"]:
    ms = [r[0] for r in agg[name]]
    ss = [r[1] for r in agg[name]]
    print(f"{name}: mean MSE {sum(ms)/len(ms):.6e}, mean SQNR {sum(ss)/len(ss):.2f} dB, clipped weights {[r[3] for r in agg[name]]}")
print("Done")
