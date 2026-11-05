"""Compare optimizers and tricks for 4-bit symmetric quantization on 2048x2048."""
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


def run_config(name, make_opt, steps, batch=10, clip=None, scheduler=None):
    print(f"\n=== {name} ===", flush=True)
    W = W0.clone().detach().requires_grad_(True)
    opt = make_opt(W)
    sched = scheduler(opt) if scheduler else None
    before = full_mse(W.detach())
    for step in range(steps):
        start = (step * batch) % len(X_train)
        bx = X_train[start:start + batch]
        by = Y0_train[start:start + batch]
        opt.zero_grad()
        Wq = quantize_ste(W)
        loss = sum(F.mse_loss(Wq @ x, y0) for x, y0 in zip(bx, by)) / len(bx)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_([W], clip)
        opt.step()
        if sched is not None:
            sched.step()
        if step % 20 == 0 or step == steps - 1:
            cur = full_mse(W.detach())
            print(f"  step {step:3d}: train mse {cur:.4e}", flush=True)
    after = full_mse(W.detach())
    wc = F.mse_loss(W.detach(), W0).item()
    print(f"  RESULT: before {before:.4e} after {after:.4e} improvement {before/after:.2f}x weight_change {wc:.3e}", flush=True)
    return {"name": name, "before": before, "after": after, "improvement": before / after if after > 0 else float("inf"),
            "weight_change": wc}


results = []

# 1. Adam lr=1e-4 baseline
results.append(run_config("Adam lr=1e-4",
    lambda W: torch.optim.Adam([W], lr=1e-4, betas=(0.9, 0.95)), 100))

# 2. Adam lr=1e-3 with grad clip
results.append(run_config("Adam lr=1e-3 clip=1.0",
    lambda W: torch.optim.Adam([W], lr=1e-3, betas=(0.9, 0.95)), 100, clip=1.0))

# 3. Adam lr=1e-4 cosine schedule
def cos_sched(opt):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
results.append(run_config("Adam lr=1e-4 cosine",
    lambda W: torch.optim.Adam([W], lr=1e-4, betas=(0.9, 0.95)), 100, scheduler=cos_sched))

# 4. SGD momentum 0.9 lr=1e-2
results.append(run_config("SGD mom=0.9 lr=1e-2",
    lambda W: torch.optim.SGD([W], lr=1e-2, momentum=0.9), 100))

# 5. Adam lr=1e-4 more steps
results.append(run_config("Adam lr=1e-4 300 steps",
    lambda W: torch.optim.Adam([W], lr=1e-4, betas=(0.9, 0.95)), 300))

# 6. RMSprop
results.append(run_config("RMSprop lr=1e-4",
    lambda W: torch.optim.RMSprop([W], lr=1e-4), 100))

print("\n=== SUMMARY ===")
for r in results:
    print(f"{r['name']:32s} after={r['after']:.4e}  improvement={r['improvement']:.2f}x  weight_change={r['weight_change']:.3e}")

with open("optimize_experiments_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved optimize_experiments_results.json")
print("Done")
