import os, sys, math, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
SIZE = 2048
W0 = torch.randn(SIZE, SIZE) * 0.02
torch.manual_seed(7)
X_train = [torch.randn(SIZE) for _ in range(100)]
Y0_train = [W0 @ x for x in X_train]

BITS = 4
QMAX = (1 << (BITS - 1)) - 1


def quantize_ste(W):
    with torch.no_grad():
        scale = W.abs().max() / QMAX
        if scale.item() == 0:
            scale = torch.tensor(1.0)
        Wq = (W / scale).round().clamp(-QMAX, QMAX) * scale
    return W + (Wq - W).detach()


def full_mse(W):
    with torch.no_grad():
        Wq = quantize_ste(W)
        errs = [F.mse_loss(Wq @ x, y0).item() for x, y0 in zip(X_train, Y0_train)]
    return sum(errs) / len(errs)


STEPS = 100
baseline = full_mse(W0.clone())

# Variant A: cosine, no warmup
print("=== Adam cosine, no warmup ===", flush=True)
W = W0.clone().detach().requires_grad_(True)
opt = torch.optim.Adam([W], lr=1e-4, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
for step in range(STEPS):
    opt.zero_grad()
    Wq = quantize_ste(W)
    loss = sum(F.mse_loss(Wq @ x, y0) for x, y0 in zip(X_train, Y0_train)) / len(X_train)
    loss.backward()
    opt.step()
    sched.step()
    if step % 20 == 0 or step == STEPS - 1:
        print(f"  step {step:3d}: {full_mse(W.detach()):.4e}", flush=True)
mse_A = full_mse(W.detach())
wc_A = F.mse_loss(W.detach(), W0).item()

# Variant B: cosine with 20-step warmup
print("=== Adam cosine, warmup 20 ===", flush=True)
W = W0.clone().detach().requires_grad_(True)
opt = torch.optim.Adam([W], lr=1e-4, betas=(0.9, 0.95))
# warmup: linear 0->1e-4 over 20 steps, then cosine over 80
def lr_lambda(step):
    if step < 20:
        return (step + 1) / 20.0
    progress = (step - 20) / 80.0
    return 0.5 * (1 + math.cos(math.pi * progress))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
for step in range(STEPS):
    opt.zero_grad()
    Wq = quantize_ste(W)
    loss = sum(F.mse_loss(Wq @ x, y0) for x, y0 in zip(X_train, Y0_train)) / len(X_train)
    loss.backward()
    opt.step()
    sched.step()
    if step % 20 == 0 or step == STEPS - 1:
        print(f"  step {step:3d}: {full_mse(W.detach()):.4e}", flush=True)
mse_B = full_mse(W.detach())
wc_B = F.mse_loss(W.detach(), W0).item()

print("\n=== RESULTS ===")
print(f"Baseline MSE: {full_mse(W0):.6e}")
print(f"No warmup:  MSE {mse_A:.6e}, improvement {full_mse(W0)/mse_A:.2f}x, weight change {wc_A:.3e}")
print(f"Warmup 20:  MSE {mse_B:.6e}, improvement {full_mse(W0)/mse_B:.2f}x, weight change {wc_B:.3e}")
if mse_B < mse_A:
    print("Warmup helped.")
else:
    print("Warmup did not help.")
print("Done")
