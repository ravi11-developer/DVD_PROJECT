"""
STEP 0 - Data Quality & Physical Sanity Check for NMOS ON
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)
Case: NMOS ON (Vg = VDD, Vb = 0)

Performs:
    1. Kirchhoff's Current Law (KCL) Check: Validates Id + Ig + Is + Ib ≈ 0 for all rows.
    2. Physical Sanity Trend Plots:
       - ON Conduction Current vs. Temperature (mobility degradation check)
       - ON Conduction Current vs. |Vds| (linear-to-saturation curve)
       - Gate Tunneling Current vs. Oxide Thickness toxe (quantum tunneling check)
       - Substrate Current |Ib| vs. |Vds| (impact ionization onset check)

Run:
    python step0_data_check.py
Outputs (in results/step0_data_check/):
    kcl_check.json, kcl_check.png, trend_ion_vs_temp.png, trend_ion_vs_vds.png,
    trend_ig_vs_toxe.png, trend_ib_vs_vds.png
"""

import numpy as np
import matplotlib.pyplot as plt
from nmos_on_common import load_split, out_path, save_json, CURRENTS

df, tr, va, te = load_split()
print(f"Loaded NMOS ON dataset: {len(df)} rows across {len(df.columns)} columns.")

# ---------------------------------------------------------------------------
# 1. Kirchhoff's Current Law (KCL) Check
# ---------------------------------------------------------------------------
abs_sum = df[CURRENTS].sum(axis=1).abs()
rel = abs_sum / np.maximum(df[CURRENTS].abs().max(axis=1), 1e-15)

kcl = {
    "rows": int(len(df)),
    "absolute_residual_A": {
        "median": float(abs_sum.median()),
        "p95": float(abs_sum.quantile(0.95)),
        "max": float(abs_sum.max())
    },
    "relative_residual": {
        "median": float(rel.median()),
        "p95": float(rel.quantile(0.95)),
        "max": float(rel.max())
    },
    "rows_with_relative_residual_below_1e-6": f"{(rel < 1e-6).mean() * 100:.2f}%",
    "verdict": "PASS - currents strictly obey KCL to SPICE numerical tolerance" if rel.max() < 1e-6 else "CHECK",
    "terminal_polarities": {
        c: ("strictly positive (+)" if (df[c] > 0).all() else
            "strictly negative (-)" if (df[c] < 0).all() else "polarity changes with Vd vs Vs")
        for c in CURRENTS
    },
    "note": "itotal was an auxiliary sum column from simulation and is excluded from ML modeling."
}

save_json(kcl, out_path("step0_data_check", "kcl_check.json"))
print(f"KCL Relative Residual: Median {rel.median():.2e}, Max {rel.max():.2e} -> {kcl['verdict']}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(np.log10(np.maximum(rel, 1e-16)), bins=60, color="#1F77B4", edgecolor="black", alpha=0.8)
ax.set_xlabel(r"$\log_{10}(|I_d + I_g + I_s + I_b| \,/\, \max|I|)$")
ax.set_ylabel("Row Count")
ax.set_title(f"Kirchhoff Check: NMOS ON ({len(df)} rows, Max Residual: {rel.max():.2e})")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_path("step0_data_check", "kcl_check.png"), dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Physical Trend Plots
# ---------------------------------------------------------------------------
def plot_trend(x, y, xlabel, ylabel, title, path, logy=True, bins=15):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x, y, s=5, alpha=0.25, color="#1F77B4", label="SPICE Raw Rows")
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    mid = 0.5 * (edges[1:] + edges[:-1])
    med = [np.median(y[(x >= a) & (x <= b)]) for a, b in zip(edges[:-1], edges[1:])]
    ax.plot(mid, med, "k-o", ms=4, lw=2.2, label="Bin Median Trend")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


vds = (df["vd"] - df["vs"]).abs()
I_on = np.abs(df["id"] - df["is"]) / 2.0

# 1. Conduction Current vs Temperature
conduction_mask = vds > 0.2
plot_trend(
    df.loc[conduction_mask, "temp"], I_on[conduction_mask],
    "Temperature (°C)", "ON Channel Current $|I_{on}|$ (A)",
    "NMOS ON: Conduction Current vs. Temperature (|Vds| > 0.2V)",
    out_path("step0_data_check", "trend_ion_vs_temp.png"), logy=True
)

# 2. Conduction Current vs |Vds|
plot_trend(
    vds, I_on,
    "Drain-to-Source Voltage $|V_{ds}|$ (V)", "ON Channel Current $|I_{on}|$ (A)",
    "NMOS ON: Conduction Current vs. $|V_{ds}|$ (Linear to Saturation)",
    out_path("step0_data_check", "trend_ion_vs_vds.png"), logy=True
)

# 3. Gate Current vs Oxide Thickness toxe
plot_trend(
    df["toxe"] * 1e9, df["ig"],
    "Oxide Thickness $t_{oxe}$ (nm)", "Gate Tunneling Current $I_g$ (A)",
    "NMOS ON: Gate Tunneling vs. Oxide Thickness ($V_g = V_{dd}$)",
    out_path("step0_data_check", "trend_ig_vs_toxe.png"), logy=True
)

# 4. Substrate Current |Ib| vs |Vds| (Impact Ionization)
plot_trend(
    vds, np.abs(df["ib"]),
    "Drain-to-Source Voltage $|V_{ds}|$ (V)", "Substrate Current $|I_b|$ (A)",
    "NMOS ON: Substrate Current vs. $|V_{ds}|$ (Impact Ionization Onset)",
    out_path("step0_data_check", "trend_ib_vs_vds.png"), logy=True
)

print(f"Completed Step 0! All checks and trend plots saved in {out_path('step0_data_check', '')}")
