"""Data check for whichever cases exist (or the ones named): Kirchhoff + trend plots.
usage (project root): python3 scripts/check_data.py [pmos_on ...]"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ALL = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
cases = [c for c in (sys.argv[1:] or ALL) if os.path.exists("data/%s.csv" % c)]
print("cases found:", cases)
os.makedirs("results", exist_ok=True)
D = {c: pd.read_csv("data/%s.csv" % c) for c in cases}
rows = []
for c, d in D.items():
    I = d[["id", "ig", "is", "ib"]]
    resid = I.sum(axis=1).abs(); rel = resid / I.abs().max(axis=1)
    rows.append(dict(case=c, n=len(d), nan=int(d.isna().sum().sum()), max_abs_sum_A=resid.max(),
                     median_rel=rel.median(), p99_rel=rel.quantile(.99), max_rel=rel.max()))
pd.DataFrame(rows).to_csv("results/kirchhoff.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: "%.3e" % x))

def panel(cs, x, y, xl, yl, fname, title):
    f, a = plt.subplots(1, len(cs), figsize=(5 * len(cs), 4), squeeze=False)
    for ax, c in zip(a[0], cs):
        ax.scatter(D[c][x], D[c][y].abs(), s=2, alpha=.4); ax.set_yscale("log")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title("%s: %s" % (c, title)); ax.grid(alpha=.3)
    f.tight_layout(); f.savefig("results/" + fname, dpi=150)
    for c in cs:
        print("spearman(%s, log|%s|) %s = %+.2f" % (x, y, c, D[c][x].corr(np.log10(D[c][y].abs()), method="spearman")))
off = [c for c in cases if c.endswith("_off")]; on = [c for c in cases if c.endswith("_on")]
if off:  # required plots 1 & 2: OFF current vs temperature / vth0
    panel(off, "temp", "id", "T (C)", "|Id| (A)", "trend_off_vs_temp.png", "OFF |Id| vs T")
    panel(off, "vth0", "id", "vth0 (V)", "|Id| (A)", "trend_off_vs_vth0.png", "OFF |Id| vs vth0")
if on:   # ON-state analogues (no OFF data for this case)
    panel(on, "temp", "id", "T (C)", "|Id| (A)", "trend_on_id_vs_temp.png", "ON |Id| vs T")
    panel(on, "vth0", "id", "vth0 (V)", "|Id| (A)", "trend_on_id_vs_vth0.png", "ON |Id| vs vth0")
if cases:  # required plot 3: gate current vs toxe
    panel(cases, "toxe", "ig", "toxe (m)", "|Ig| (A)", "trend_ig_vs_toxe.png", "|Ig| vs toxe")
