import os, sys, json, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import quantize_tensor, VARIANTS, variant_name

# random matrix results (seed 42, 512x512)
torch.manual_seed(42)
Wr = torch.randn(512, 512) * 0.02
xr = torch.randn(512)
yr = Wr @ xr
rand_mse = {}
for sym, gran, gs, bits in VARIANTS:
    Wq = quantize_tensor(Wr.clone(), bits, sym, gran, gs)
    yq = Wq @ xr
    rand_mse[variant_name(sym, gran, gs, bits)] = F.mse_loss(yq, yr).item()

# real matrix results
real = json.load(open("real_matrix_results.json"))
real_mse = {r["name"]: r["mse"] for r in real["results"]}

# align
names = [n for n in rand_mse if n in real_mse]
x = [rand_mse[n] for n in names]
y = [real_mse[n] for n in names]

def pearson(a, b):
    n = len(a)
    ma = sum(a)/n; mb = sum(b)/n
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da = (sum((v-ma)**2 for v in a))**0.5
    db = (sum((v-mb)**2 for v in b))**0.5
    return num/(da*db)

def spearman(a, b):
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0]*len(vals)
        for rank, idx in enumerate(order):
            r[idx] = rank + 1
        return r
    return pearson(ranks(a), ranks(b))

print("N variants:", len(names))
print("Pearson correlation (MSE):", pearson(x, y))
print("Spearman correlation (MSE):", spearman(x, y))

# also on log10 mse
lx = [math.log10(v) for v in x]
ly = [math.log10(v) for v in y]
print("Pearson correlation (log10 MSE):", pearson(lx, ly))
print("Spearman correlation (log10 MSE):", spearman(lx, ly))

# which random: describe
print("\nRandom matrix: seed 42, shape 512x512, standard normal times 0.02.")
print("Real matrix:", real["matrix"], real["shape"])
