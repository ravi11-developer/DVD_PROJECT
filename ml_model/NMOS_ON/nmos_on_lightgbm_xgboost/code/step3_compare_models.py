"""
STEP 3 - Compare LightGBM and XGBoost for NMOS ON (Run after Step 1 & Step 2)
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)
Case: NMOS ON (Vg = VDD, Vb = 0)

Outputs:
    1. Model comparison table on the VALIDATION set: CSV & formatted PNG graphic.
    2. Model comparison table on the TEST set: CSV & formatted PNG graphic.
    3. Side-by-side performance comparison bar charts (lightgbm_vs_xgboost_test.png).
    4. One combined JSON report matching team specification (nmos_on_lightgbm_xgboost_report.json).

Run:
    python step3_compare_models.py
Outputs (in results/comparison/):
    model_comparison_validation.csv, model_comparison_validation.png,
    chosen_model_results_test.csv, chosen_model_results_test.png,
    lightgbm_vs_xgboost_test.png, nmos_on_lightgbm_xgboost_report.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nmos_on_common import out_path, save_json, CURRENTS

MODELS = {"LightGBM": "lightgbm", "XGBoost": "xgboost"}

# Load reports from Step 1 and Step 2
rep = {}
for name, key in MODELS.items():
    json_path = out_path(key, f"{key}_report.json")
    assert os.path.exists(json_path), f"Missing report: {json_path}. Please run step1_lightgbm.py and step2_xgboost.py first."
    with open(json_path, "r", encoding="utf-8") as f:
        rep[name] = json.load(f)


def num(s):
    return float(str(s).rstrip("%"))


# ---------------------------------------------------------------------------
# 1. Summary Tables (Validation & Test)
# ---------------------------------------------------------------------------
def build_table(split):
    rows = []
    for name, key in MODELS.items():
        for m in rep[name][f"{key}_{split}"]["metrics"]:
            rows.append({
                "Model": name,
                "Current": m["Terminal"],
                "R2 (log)": m["R2 (Log)"],
                "R2 (linear)": m["R2 (Linear)"],
                "Median error (%)": num(m["Median Error (%)"]),
                "95th-pct error (%)": num(m["95th-Pct Error (%)"]),
                "Within 1% (%)": num(m["Within 1% Tol (%)"]),
                "Within 5% (%)": num(m["Within 5% Tol (%)"]),
                "Mean accuracy (%)": num(m["Mean Accuracy (%)"]),
                "Median accuracy (%)": num(m["Median Accuracy (%)"]),
                "Sign correct (%)": num(m["Sign Correct (%)"])
            })
    return pd.DataFrame(rows)


val_df = build_table("validation")
test_df = build_table("test")

val_csv_path = out_path("comparison", "model_comparison_validation.csv")
test_csv_path = out_path("comparison", "chosen_model_results_test.csv")
val_df.to_csv(val_csv_path, index=False, encoding="utf-8-sig")
test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")


def draw_table_png(df, title, path):
    cols = ["Model", "Current", "R2 (log)", "R2 (linear)", "Median error (%)",
            "95th-pct error (%)", "Within 1% (%)", "Mean accuracy (%)", "Median accuracy (%)", "Sign correct (%)"]
    cell_text = []
    for row in df[cols].values:
        r_text = []
        for c, v in zip(cols, row):
            if c.startswith("R2"):
                r_text.append(f"{v:.4f}")
            elif "error" in c:
                r_text.append(f"{v:.3f}")
            elif isinstance(v, float):
                r_text.append(f"{v:.2f}")
            else:
                r_text.append(str(v))
        cell_text.append(r_text)

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    tab = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.05, 1.4)

    # Style header and rows
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F77B4")
            cell.set_text_props(color="white", weight="bold")
        else:
            m_idx = (r - 1) // 4
            cell.set_facecolor("#F8F9FA" if m_idx == 0 else "#FFFFFF")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


draw_table_png(val_df, "NMOS ON — Model Comparison (Validation Set)",
               out_path("comparison", "model_comparison_validation.png"))
draw_table_png(test_df, "NMOS ON — Model Evaluation (Test Set)",
               out_path("comparison", "chosen_model_results_test.png"))


# ---------------------------------------------------------------------------
# 2. Side-by-Side Test Bar Chart
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(CURRENTS))
w = 0.35

for ax, metric, title, logy in [
    (axes[0], "Median error (%)", "Median Error (%) on Test Set", True),
    (axes[1], "Mean accuracy (%)", "Mean Accuracy (%) on Test Set", False)
]:
    for i, name in enumerate(MODELS.keys()):
        sub = test_df[test_df["Model"] == name]
        vals = [sub[sub["Current"] == c.upper()][metric].values[0] for c in CURRENTS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=name, color="#1F77B4" if name == "LightGBM" else "#FF7F0E")
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in CURRENTS])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(80, 101)

fig.suptitle("NMOS ON: LightGBM vs. XGBoost Performance Comparison", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(out_path("comparison", "lightgbm_vs_xgboost_test.png"), dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Combined JSON Report
# ---------------------------------------------------------------------------
better = {}
for c in CURRENTS:
    lgb_err = val_df[(val_df["Model"] == "LightGBM") & (val_df["Current"] == c.upper())]["Median error (%)"].values[0]
    xgb_err = val_df[(val_df["Model"] == "XGBoost") & (val_df["Current"] == c.upper())]["Median error (%)"].values[0]
    better[c.upper()] = "LightGBM" if lgb_err <= xgb_err else "XGBoost"

combined_report = {
    "case": "NMOS ON (Vg = VDD, Vb = 0), 22nm_MGK, VDD = 0.9 V",
    "description": "Physics-guided single residual model + sign classifier comparing LightGBM and XGBoost",
    "dataset": rep["LightGBM"]["dataset"],
    "features_count": rep["LightGBM"]["features_count"],
    "lightgbm_hyperparameters": rep["LightGBM"]["hyperparameters"],
    "lightgbm_trees_used": rep["LightGBM"]["trees_used"],
    "lightgbm_train": rep["LightGBM"]["lightgbm_train"],
    "lightgbm_validation": rep["LightGBM"]["lightgbm_validation"],
    "lightgbm_test": rep["LightGBM"]["lightgbm_test"],
    "lightgbm_overfit_check": rep["LightGBM"]["overfit_check"],
    "xgboost_hyperparameters": rep["XGBoost"]["hyperparameters"],
    "xgboost_trees_used": rep["XGBoost"]["trees_used"],
    "xgboost_train": rep["XGBoost"]["xgboost_train"],
    "xgboost_validation": rep["XGBoost"]["xgboost_validation"],
    "xgboost_test": rep["XGBoost"]["xgboost_test"],
    "xgboost_overfit_check": rep["XGBoost"]["overfit_check"],
    "better_model_per_current_on_validation": better
}

save_json(combined_report, out_path("comparison", "nmos_on_lightgbm_xgboost_report.json"))

print("\n" + "=" * 80)
print("  STEP 3 COMPLETE: MODEL COMPARISON SUMMARY")
print("=" * 80)
print(f"Validation Table CSV : {val_csv_path}")
print(f"Test Table CSV       : {test_csv_path}")
print(f"Comparison Chart PNG : {out_path('comparison', 'lightgbm_vs_xgboost_test.png')}")
print(f"Combined Report JSON : {out_path('comparison', 'nmos_on_lightgbm_xgboost_report.json')}")
print(f"Winning Model per Terminal (Validation): {better}")
