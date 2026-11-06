"""Hadamard rotation test on the same 2048x2048 random matrix and 100 inputs."""
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


def hadamard(n):
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (H.shape[0] ** 0.5)


def quantize_tensor(W, bits):
    scale = W.abs().max() / QMAX
    if scale.item() == 0:
        scale = torch.tensor(1.0)
    Wq = (W / scale).round().clamp(-QMAX, QMAX) * scale
    return Wq, scale.item()


def rotate(W, b):
    # row-wise Hadamard: H_block @ W
    N, M = W.shape
    H = hadamard(b)
    W = W.view(N // b, b, M)
    W = torch.matmul(H.unsqueeze(0), W)
    W = W.view(N, M)
    return W


def de_rotate(Wq, b):
    N, M = Wq.shape
    H = hadamard(b)
    Wq = Wq.view(N // b, b, M)
    Wq = torch.matmul(H.T.unsqueeze(0), Wq)
    Wq = Wq.view(N, M)
    return Wq


def eval_variant(Wq, X, Y):
    mses = []
    sqnrs = []
    for x, y in zip(X, Y):
        yq = Wq @ x
        mse = F.mse_loss(yq, y).item()
        mses.append(mse)
        # output-level sqnr
        sig = (y ** 2).sum().item()
        noise = ((y - yq) ** 2).sum().item()
        if noise > 0 and sig > 0:
            sqnrs.append(10.0 * math.log10(sig / noise))
        else:
            sqnrs.append(float("inf"))
    return sum(mses) / len(mses), sum(sqnrs) / len(sqnrs)


# baseline (no rotation)
Wq_base, scale_base = quantize_tensor(W, BITS)
mse_base, sqnr_base = eval_variant(Wq_base, X, Y)
print(f"simple (no rotation): MSE {mse_base:.6e}, SQNR {sqnr_base:.2f} dB, scale {scale_base:.6f}", flush=True)

results = [{"name": "simple (no rotation)", "mse": mse_base, "sqnr": sqnr_base, "scale": scale_base}]

for b in (64, 256, 512, 2048):
    Wrot = rotate(W, b)
    Wqrot, scale_rot = quantize_tensor(Wrot, BITS)
    # de-rotate weights
    Wq = de_rotate(Wqrot, b)
    mse, sqnr = eval_variant(Wq, X, Y)
    print(f"hadamard b={b}: MSE {mse:.6e}, SQNR {sqnr:.2f} dB, scale {scale_rot:.6f}", flush=True)
    results.append({"name": f"hadamard b={b}", "mse": mse, "sqnr": sqnr, "scale": scale_rot, "block": b})

# training values for comparison
print("\n=== COMPARISON WITH TRAINING TEST ===")
print("Training baseline MSE: 0.03618")
print("Training Adam MSE:     0.00830 (4.36x)")
print("Training SGD MSE:      0.02216 (1.63x)")
print(f"Hadamard best MSE:     {min(r['mse'] for r in results):.6e}")

with open("hadamard_test_results.json", "w") as f:
    json.dump({"matrix": "random 2048x2048 seed 42", "inputs": 100, "bits": BITS,
               "results": results,
               "training": {"baseline_mse": 0.03618, "adam_mse": 0.00830, "sgd_mse": 0.02216}}, f, indent=2)
print("\nSaved hadamard_test_results.json")
print("Done")
