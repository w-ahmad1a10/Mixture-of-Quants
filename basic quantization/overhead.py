import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mse_graph.py")).read().split("# print table")[0])
N = 512 * 512
for d in sorted(data, key=lambda r: (r["bits"], r["granularity"], r.get("gs") or 0)):
    wbits = N * d["bits"]
    oh = d["total_bits"] - wbits
    pct = oh / wbits * 100
    print(f"{d['name']}: weight_bits={wbits}, overhead_bits={oh}, total={d['total_bits']}, overhead_pct={pct:.4f}%")
