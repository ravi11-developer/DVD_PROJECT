"""
STEP 0 - Data check for NMOS OFF  (deliverable 3 of the brief)
    * Kirchhoff check: Id + Ig + Is + Ib should be ~0 for every SPICE row
    * Three trend plots: OFF current vs temperature, OFF current vs vth0, gate current vs toxe

Run:  python step0_data_check.py
Output (in OUT_DIR/step0_data_check/):
    kcl_check.json, kcl_check.png, trend_ioff_vs_temp.png, trend_ioff_vs_vth0.png, trend_ig_vs_toxe.png
"""
import numpy as np
import matplotlib.pyplot as plt
from nmos_off_common import load_split, out_path, save_json, CURRENTS

df, tr, va, te = load_split()
print(f"Loaded {len(df)} rows")

# ---------------------------------------------------------------------------
# 1. Kirchhoff check
#    residual = |Id + Ig + Is + Ib| / (largest |terminal current| in that row)
# ---------------------------------------------------------------------------
abs_sum = df[CURRENTS].sum(axis=1).abs()
rel = abs_sum / df[CURRENTS].abs().max(axis=1)
kcl = {"rows": int(len(df)),
       "absolute_residual_A": {"median": float(abs_sum.median()), "max": float(abs_sum.max())},
       "relative_residual": {"median": float(rel.median()), "p95": float(rel.quantile(0.95)),
                             "max": float(rel.max())},
       "rows_with_relative_residual_below_1e-6": f"{(rel < 1e-6).mean() * 100:.2f}%",
       "verdict": "PASS - currents obey KCL to simulator precision" if rel.max() < 1e-6 else "CHECK",
       "fixed_signs": {c: ("always +" if (df[c] > 0).all() else "always -" if (df[c] < 0).all() else "changes")
                       for c in CURRENTS},
       "note": "itotal is a sum of the four currents (~0) and is never used as a model input"}
save_json(kcl, out_path("step0_data_check", "kcl_check.json"))
print(f"KCL relative residual: median {rel.median():.1e}, max {rel.max():.1e}  -> {kcl['verdict']}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(np.log10(rel.clip(lower=1e-16)), bins=60, color="#1F77B4")
ax.set_xlabel("log10( |Id+Ig+Is+Ib| / max|I| )"); ax.set_ylabel("rows")
ax.set_title(f"Kirchhoff check, NMOS OFF ({len(df)} rows): max residual {rel.max():.1e}")
fig.tight_layout(); fig.savefig(out_path("step0_data_check", "kcl_check.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Trend plots. Every row is a random PVT corner, so the cloud is wide;
#    the black line is the median in each bin and shows the trend clearly.
#    OFF current = the channel (drain-to-source) leakage, |Id - Is| / 2,
#    taken on rows with |Vds| > 0.3 V so the channel current dominates.
# ---------------------------------------------------------------------------
def trend(x, y, xlabel, ylabel, title, path, logy=True, bins=15):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x, y, s=4, alpha=0.25, color="#1F77B4", label="SPICE rows")
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    mid = 0.5 * (edges[1:] + edges[:-1])
    med = [np.median(y[(x >= a) & (x <= b)]) for a, b in zip(edges[:-1], edges[1:])]
    ax.plot(mid, med, "k-o", ms=4, lw=2, label="bin median")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


chan = df[(df.vd - df.vs).abs() > 0.3]
ioff = ((chan.id - chan["is"]).abs() / 2).values
trend(chan.temp.values, ioff, "Temperature (°C)", "OFF current |Id − Is|/2 (A)",
      f"OFF current vs temperature (|Vds| > 0.3 V, {len(chan)} rows)",
      out_path("step0_data_check", "trend_ioff_vs_temp.png"))
trend(chan.vth0.values, ioff, "vth0 (V)", "OFF current |Id − Is|/2 (A)",
      f"OFF current vs vth0 (|Vds| > 0.3 V, {len(chan)} rows)",
      out_path("step0_data_check", "trend_ioff_vs_vth0.png"))
trend(df.toxe.values * 1e9, df.ig.abs().values, "toxe (nm)", "Gate current |Ig| (A)",
      f"Gate current vs toxe (all {len(df)} rows)",
      out_path("step0_data_check", "trend_ig_vs_toxe.png"))
print("Saved data-check files in", out_path("step0_data_check", ""))
