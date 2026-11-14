"""
03_data_check.py - deliverable 3: Kirchhoff check for each case + the three trend plots.

Run (from anywhere):  python dataset/03_data_check.py
Output:               dataset/data_check/
    kcl_check.csv                    KCL summary per case
    trend_summary.csv                Spearman correlation behind each trend plot
    trend_off_current_vs_temp.png    OFF current vs temperature   (NMOS OFF, PMOS OFF)
    trend_off_current_vs_vth0.png    OFF current vs |vth0|        (NMOS OFF, PMOS OFF)
    trend_gate_current_vs_toxe.png   gate current vs toxe         (all four cases)

Definitions
  KCL residual        |Id + Ig + Is + Ib|, relative to the largest of the four currents.
  OFF current         subthreshold channel current per width, |I| / W, measured at the source-side
                      terminal (the one nearest the gate voltage, where gate tunnelling ~ 0) on rows
                      at the textbook OFF bias: source-side terminal within 0.1 V of its rail
                      (Vgs ~ 0) and |Vds| >= 0.6 V.
  Gate current        |Ig| / W.
  Each plot shows every sample as a faint dot and the median in 12 equal-width bins as a line.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_check"
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
CURRENTS = ["id", "ig", "is", "ib"]
# Fixed categorical order (one colour per case, same in every figure)
COLOR = {"NMOS_ON": "#2a78d6", "NMOS_OFF": "#eb6834", "PMOS_ON": "#1baf7a", "PMOS_OFF": "#eda100"}
LABEL = {"NMOS_ON": "NMOS ON", "NMOS_OFF": "NMOS OFF", "PMOS_ON": "PMOS ON", "PMOS_OFF": "PMOS OFF"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(case):
    d = pd.read_csv(HERE / case / f"{case.lower()}.csv")
    d.columns = d.columns.str.strip()
    return d


def kcl_row(case, d):
    s = d[CURRENTS].sum(axis=1).abs()
    big = d[CURRENTS].abs().max(axis=1)
    rel = s / big
    return dict(case=LABEL[case], rows=len(d), max_abs_residual_A=s.max(), mean_abs_residual_A=s.mean(),
                median_rel_residual=rel.median(), max_rel_residual=rel.max(),
                rows_passing=int(((rel < 1e-6) | (s < 1e-14)).sum()))


def binned_median(x, y, n=12):
    edges = np.linspace(x.min(), x.max(), n + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    idx = np.clip(np.digitize(x, edges) - 1, 0, n - 1)
    med = np.array([np.median(y[idx == i]) if np.any(idx == i) else np.nan for i in range(n)])
    return mid, med


def panel(ax, case, x, y, xlabel):
    ax.scatter(x, y, s=4, alpha=0.18, color=COLOR[case], linewidths=0, rasterized=True)
    mx, my = binned_median(np.asarray(x), np.asarray(y))
    ax.plot(mx, my, color=COLOR[case], lw=2, label=f"{LABEL[case]} (bin median)")
    ax.set_yscale("log"); ax.set_xlabel(xlabel)


def off_rows(case, d, tol=0.1):
    """Rows at the textbook OFF bias, with the channel OFF current per width in column 'ioff'."""
    hi, lo = np.maximum(d.vd, d.vs), np.minimum(d.vd, d.vs)
    if case == "NMOS_OFF":      # gate at 0: source side = lower-voltage terminal
        src_v, src_is_s = lo, d.vs < d.vd
    else:                       # PMOS, gate at VDD: source side = higher-voltage terminal
        src_v, src_is_s = d.vdd - hi, d.vs > d.vd
    m = (src_v < tol) & (hi - lo >= 0.6)
    out = d[m].copy()
    out["ioff"] = np.where(src_is_s[m], out["is"].abs(), out.id.abs()) / (out.w * 1e6)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    data = {c: load(c) for c in CASES}

    kcl = pd.DataFrame([kcl_row(c, d) for c, d in data.items()])
    kcl.to_csv(OUT / "kcl_check.csv", index=False)
    print("Kirchhoff check (|Id+Ig+Is+Ib|):")
    print(kcl.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    trends = []
    off = {c: off_rows(c, data[c]) for c in ("NMOS_OFF", "PMOS_OFF")}

    # 1. OFF current vs temperature (one panel, both OFF cases)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for c, d in off.items():
        y = d.ioff
        panel(ax, c, d.temp, y, "Temperature (°C)")
        trends.append(dict(plot="OFF current vs temperature", case=LABEL[c], rows=len(d),
                           spearman=spearmanr(d.temp, y).statistic))
    ax.set_ylabel("OFF current / W (A/µm)")
    ax.set_title("OFF current vs temperature (Vgs ≈ 0, |Vds| ≥ 0.6 V)", color=INK, loc="left")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "trend_off_current_vs_temp.png"); plt.close(fig)

    # 2. OFF current vs |vth0| (two panels: NMOS and PMOS vth0 ranges do not overlap).
    #    Temperature dominates the spread, so one median line per temperature band (sequential ramp).
    bands = [(-40, 15, "#86b6ef"), (15, 70, "#2a78d6"), (70, 125.01, "#104281")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, (c, d) in zip(axes, off.items()):
        ax.scatter(d.vth0.abs(), d.ioff, s=4, alpha=0.15, color=INK2, linewidths=0, rasterized=True)
        for lo, hi, col in bands:
            b = d[(d.temp >= lo) & (d.temp < hi)]
            mx, my = binned_median(b.vth0.abs().to_numpy(), b.ioff.to_numpy(), n=8)
            ax.plot(mx, my, color=col, lw=2, marker="o", ms=4, label=f"{lo:.0f} to {min(hi, 125):.0f} °C")
            trends.append(dict(plot=f"OFF current vs |vth0| ({lo:.0f} to {min(hi, 125):.0f} C)", case=LABEL[c],
                               rows=len(b), spearman=spearmanr(b.vth0.abs(), b.ioff).statistic))
        ax.set_yscale("log"); ax.set_xlabel("|vth0| (V)"); ax.set_title(LABEL[c], color=INK, loc="left")
        trends.append(dict(plot="OFF current vs |vth0| (all temperatures)", case=LABEL[c], rows=len(d),
                           spearman=spearmanr(d.vth0.abs(), d.ioff).statistic))
    axes[0].set_ylabel("OFF current / W (A/µm)")
    axes[1].legend(title="bin median, temperature", loc="lower left")
    fig.suptitle("OFF current vs threshold voltage (Vgs ≈ 0, |Vds| ≥ 0.6 V)", color=INK, x=0.01, ha="left")
    fig.savefig(OUT / "trend_off_current_vs_vth0.png"); plt.close(fig)

    # 3. Gate current vs toxe (one panel, all four cases)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for c, d in data.items():
        y = d.ig.abs() / (d.w * 1e6)
        panel(ax, c, d.toxe * 1e9, y, "toxe (nm)")
        trends.append(dict(plot="gate current vs toxe", case=LABEL[c], rows=len(d),
                           spearman=spearmanr(d.toxe, y).statistic))
    ax.set_ylabel("Gate current |Ig| / W (A/µm)")
    ax.set_title("Gate current vs oxide thickness", color=INK, loc="left")
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(OUT / "trend_gate_current_vs_toxe.png"); plt.close(fig)

    tr = pd.DataFrame(trends)
    tr.to_csv(OUT / "trend_summary.csv", index=False)
    print("\nTrend check (Spearman rank correlation):")
    print(tr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
