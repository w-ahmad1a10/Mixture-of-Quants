import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mse_graph.py")).read().split("# print table")[0])
by_bit = {}
for d in data:
    by_bit.setdefault(d["bits"], []).append(d)
for b, lst in by_bit.items():
    mn = min(d["mse"] for d in lst)
    for d in lst:
        d["x"] = d["mse"] / mn
        wbits = (512 * 512) * d["bits"]
        d["oh"] = (d["total_bits"] - wbits) / wbits * 100
N = 512 * 512
lines = ["const data = ["]
for d in data:
    kb = d["total_bits"] / 8 / 1024
    lines.append(f"  {{name:\"{d['name']}\", bits:{d['bits']}, mse:{d['mse']:.6e}, x:{d['x']:.2f}, oh:{d['oh']:.2f}, total:{d['total_bits']}, kb:{kb:.2f}}},")
lines.append("];")
js = "\n".join(lines)
html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calculator.html")
html = open(html_path).read()
html = re.sub(r"const data = \[.*?\];", js, html, count=1, flags=re.DOTALL)
html = html.replace(
"""      <tr>
        <th onclick="sortTable(0)">Variant</th>
        <th onclick="sortTable(1)">Bits</th>
        <th onclick="sortTable(2)">MSE</th>
        <th onclick="sortTable(3)">X-times</th>
        <th onclick="sortTable(4)">Overhead %</th>
      </tr>""",
"""      <tr>
        <th onclick="sortTable(0)">Variant</th>
        <th onclick="sortTable(1)">Bits</th>
        <th onclick="sortTable(2)">MSE</th>
        <th onclick="sortTable(3)">X-times</th>
        <th onclick="sortTable(4)">Overhead %</th>
        <th onclick="sortTable(5)">Total bits</th>
        <th onclick="sortTable(6)">Size KB</th>
      </tr>""")
html = html.replace(
'''    tr.innerHTML = "<td>" + d.name + "</td><td>" + d.bits + "</td><td>" +
      d.mse.toExponential(4) + "</td><td>" + d.x.toFixed(2) + "x</td><td>" +
      d.oh.toFixed(2) + "%</td>";''',
'''    tr.innerHTML = "<td>" + d.name + "</td><td>" + d.bits + "</td><td>" +
      d.mse.toExponential(4) + "</td><td>" + d.x.toFixed(2) + "x</td><td>" +
      d.oh.toFixed(2) + "%</td><td>" + d.total + "</td><td>" + d.kb.toFixed(2) + "</td>";''')
html = html.replace('const key = ["name","bits","mse","x","oh"][col];',
                    'const key = ["name","bits","mse","x","oh","total","kb"][col];')
html = html.replace(
'''  document.getElementById("resultA").innerHTML =
    "<b>" + a.name + " (" + a.bits + "bit)</b>: MSE = " + a.mse.toExponential(4) +
    ", X-times = " + a.x.toFixed(2) + "x, Overhead = " + a.oh.toFixed(2) + "%";''',
'''  document.getElementById("resultA").innerHTML =
    "<b>" + a.name + " (" + a.bits + "bit)</b>: MSE = " + a.mse.toExponential(4) +
    ", X-times = " + a.x.toFixed(2) + "x, Overhead = " + a.oh.toFixed(2) + "%, Size = " + a.kb.toFixed(2) + " KB";''')
html = html.replace(
'''  document.getElementById("resultB").innerHTML =
    "<b>" + b.name + " (" + b.bits + "bit)</b>: MSE = " + b.mse.toExponential(4) +
    ", X-times = " + b.x.toFixed(2) + "x, Overhead = " + b.oh.toFixed(2) + "%";''',
'''  document.getElementById("resultB").innerHTML =
    "<b>" + b.name + " (" + b.bits + "bit)</b>: MSE = " + b.mse.toExponential(4) +
    ", X-times = " + b.x.toFixed(2) + "x, Overhead = " + b.oh.toFixed(2) + "%, Size = " + b.kb.toFixed(2) + " KB";''')
open(html_path, "w").write(html)
print("patched calculator.html")
