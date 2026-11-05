import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mse_graph.py")).read().split("# print table")[0])
N = 512 * 512
by_bit = {}
for d in data:
    by_bit.setdefault(d["bits"], []).append(d)
for b, lst in by_bit.items():
    mses = [d["mse"] for d in lst]
    mn = min(mses)
    for d in lst:
        d["xmult"] = d["mse"] / mn
        wbits = N * d["bits"]
        d["oh_pct"] = (d["total_bits"] - wbits) / wbits * 100
for b in (3, 4, 6, 8):
    print(f"\n=== {b}-bit (min MSE = {min(d['mse'] for d in by_bit[b]):.6e}) ===")
    print(f"{'variant':22s} {'mse':>12s} {'x_times':>10s} {'overhead%':>10s}")
    for d in sorted([x for x in data if x["bits"] == b], key=lambda r: r["mse"]):
        print(f"{d['name']:22s} {d['mse']:12.6e} {d['xmult']:10.2f}x {d['oh_pct']:10.4f}")
