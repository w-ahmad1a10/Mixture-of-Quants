import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mse_graph.py")).read().split("# print table")[0])
N = 512 * 512
# compute normalized mse per bit
by_bit = {}
for d in data:
    by_bit.setdefault(d["bits"], []).append(d)
for b, lst in by_bit.items():
    mses = [d["mse"] for d in lst]
    mn, mx = min(mses), max(mses)
    for d in lst:
        d["norm_mse"] = (d["mse"] - mn) / (mx - mn) if mx > mn else 0.0
# overhead pct
for d in data:
    wbits = N * d["bits"]
    d["oh_pct"] = (d["total_bits"] - wbits) / wbits * 100
# print grouped
for b in (3, 4, 6, 8):
    print(f"\n=== {b}-bit ===")
    print(f"{'variant':22s} {'mse':>12s} {'overhead%':>10s} {'norm_mse':>10s}")
    for d in sorted([x for x in data if x["bits"] == b], key=lambda r: r["mse"]):
        print(f"{d['name']:22s} {d['mse']:12.6e} {d['oh_pct']:10.4f} {d['norm_mse']:10.4f}")
