"""
STEP 3 - Compare LightGBM and XGBoost (run after step1 and step2)

    * Model comparison table on the VALIDATION set: R², median error, 95th-pct error
      (deliverable 4 of the brief, NMOS OFF column)
    * Chosen-model results on the TEST set (deliverable 5)
    * One combined JSON in the team format (lightgbm_* and xgboost_* blocks)
    * Tuning history: how the validation results changed over the Optuna trials

Run:  python step3_compare_models.py
Output folder: OUT_DIR/comparison/
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nmos_off_common import out_path, save_json, CURRENTS

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"LightGBM": "lightgbm", "XGBoost": "xgboost"}
rep = {name: json.load(open(out_path(key, f"{key}_report.json"), encoding="utf-8")) for name, key in MODELS.items()}


def num(s):
    return float(str(s).rstrip("%"))


# ---------------------------------------------------------------------------
# 1. Tables (validation = model comparison, test = chosen-model results)
# ---------------------------------------------------------------------------
def table(split):
    rows = []
    for name, key in MODELS.items():
        for m in rep[name][f"{key}_{split}"]["metrics"]:
            rows.append({"Model": name, "Current": m["Terminal"], "R2 (log)": m["R2 (Log)"],
                         "R2 (linear)": m["R2 (Linear)"], "Median error (%)": num(m["Median Error (%)"]),
                         "95th-pct error (%)": num(m["95th-Pct Error (%)"]),
                         "Within 1% (%)": num(m["Within 1% Tol (%)"]),
                         "Mean accuracy (%)": num(m["Mean Accuracy (%)"]),
                         "Median accuracy (%)": num(m["Median Accuracy (%)"]),
                         "Sign correct (%)": num(m["Sign Correct (%)"])})
    return pd.DataFrame(rows)


val_t, test_t = table("validation"), table("test")
val_t.to_csv(out_path("comparison", "model_comparison_validation.csv"), index=False, encoding="utf-8-sig")
test_t.to_csv(out_path("comparison", "chosen_model_results_test.csv"), index=False, encoding="utf-8-sig")


def table_png(t, title, path):
    """Draw the table as an image; green = meets target, red = misses it."""
    cols = ["Model", "Current", "R2 (log)", "R2 (linear)", "Median error (%)", "95th-pct error (%)",
            "Within 1% (%)", "Mean accuracy (%)", "Median accuracy (%)", "Sign correct (%)"]
    cell = [[f"{v:.4f}" if c.startswith("R2") else f"{v:.3f}" if "error" in c else
             f"{v:.2f}" if isinstance(v, float) else str(v) for c, v in zip(cols, r)] for r in t[cols].values]
    colours = []
    for r in t[cols].values:
        row = []
        for c, v in zip(cols, r):
            if c.startswith("R2"):
                row.append("#C6EFCE" if v > 0.99 else "#FFC7CE")
            elif "error" in c:
                row.append("#C6EFCE" if v < 1 else "#FFC7CE")
            else:
                row.append("white")
        colours.append(row)
    fig, ax = plt.subplots(figsize=(17, 0.32 * len(t) + 0.9)); ax.axis("off")
    tb = ax.table(cellText=cell, colLabels=cols, cellColours=colours, bbox=[0, 0, 1, 1], cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
    for (r, c), cl in tb.get_celld().items():
        if r == 0:
            cl.set_facecolor("#1F3864"); cl.get_text().set_color("white"); cl.get_text().set_weight("bold")
    ax.set_title(title + "\n(green = meets target R² > 0.99 / error < 1 %, red = misses)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


table_png(val_t, "NMOS OFF - model comparison on the VALIDATION set (1500 rows)",
          out_path("comparison", "model_comparison_validation.png"))
table_png(test_t, "NMOS OFF - results on the TEST set (1500 rows, used once)",
          out_path("comparison", "chosen_model_results_test.png"))

# ---------------------------------------------------------------------------
# 2. Combined JSON in the team format
# ---------------------------------------------------------------------------
combined = {"pipeline": "NMOS OFF - physics-guided single model (Id, Ig, Is, Ib) + sign classifier; "
                        "LightGBM and XGBoost, Optuna-tuned",
            "case": rep["LightGBM"]["case"], "dataset": rep["LightGBM"]["dataset"],
            "features_count": rep["LightGBM"]["features_count"],
            "physics_baseline": rep["LightGBM"]["physics_baseline"],
            "accuracy_definition": rep["LightGBM"]["accuracy_definition"]}
for name, key in MODELS.items():
    combined[f"{key}_hyperparameters"] = rep[name]["hyperparameters"]
    combined[f"{key}_trees_used"] = rep[name]["trees_used"]
    for sp in ("train", "validation", "test"):
        combined[f"{key}_{sp}"] = rep[name][f"{key}_{sp}"]
    combined[f"{key}_overfit_check"] = rep[name]["overfit_check"]
best = {}
for c in CURRENTS:
    v = val_t[val_t.Current == c.upper()].set_index("Model")
    score = v["Median error (%)"] + 0.2 * v["95th-pct error (%)"]
    best[c.upper()] = score.idxmin()
combined["better_model_per_current_on_validation"] = best
save_json(combined, out_path("comparison", "nmos_off_lightgbm_xgboost_report.json"))

# ---------------------------------------------------------------------------
# 3. Side-by-side bar chart (validation and test)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colour = {"LightGBM": "#1F77B4", "XGBoost": "#FF7F0E"}
x = np.arange(4); w = 0.38
for ax, (col, ttl, logy, tgt) in zip(axes.flat, [
        ("Median error (%)", "Median error (%) - test", True, 1),
        ("95th-pct error (%)", "95th-percentile error (%) - test", True, 1),
        ("R2 (log)", "R² (log current) - test", False, 0.99),
        ("Within 1% (%)", "Rows within 1 % error (%) - test", False, None)]):
    for i, name in enumerate(MODELS):
        v = test_t[test_t.Model == name].set_index("Current").loc[[c.upper() for c in CURRENTS], col].values
        b = ax.bar(x + (i - 0.5) * w, v, w, color=colour[name], label=name)
        ax.bar_label(b, fmt="%.4g", fontsize=7, padding=1)
    if tgt is not None:
        ax.axhline(tgt, color="r", ls="--", lw=1, label="target")
    if logy:
        ax.set_yscale("log")
    if col == "R2 (log)":
        ax.set_ylim(0.97, 1.003)
    ax.set_xticks(x); ax.set_xticklabels(["$I_d$", "$I_g$", "$I_s$", "$I_b$"]); ax.set_title(ttl)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("NMOS OFF - LightGBM vs XGBoost (physics-guided, tuned), test set")
fig.tight_layout(); fig.savefig(out_path("comparison", "lightgbm_vs_xgboost_test.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Tuning history from the Optuna logs (validation set)
# ---------------------------------------------------------------------------
pat = re.compile(r"#(\d+) score ([\d.]+) id:([\d.]+)/([\d.]+) ig:([\d.]+)/([\d.]+) "
                 r"is:([\d.]+)/([\d.]+) ib:([\d.]+)/([\d.]+) trees (\d+)")
logs = {"LightGBM": os.path.join(HERE, "tuning_logs", "optuna_lightgbm_80trials.log"),
        "XGBoost": os.path.join(HERE, "tuning_logs", "optuna_xgboost_50trials.log")}
hist = {}
for name, f in logs.items():
    if os.path.exists(f):
        rows = [m.groups() for m in map(pat.match, open(f)) if m]
        h = pd.DataFrame([[float(v) for v in r] for r in rows],
                         columns=["trial", "score", "id_med", "id_p95", "ig_med", "ig_p95",
                                  "is_med", "is_p95", "ib_med", "ib_p95", "trees"])
        h["best_so_far"] = h.score.cummin(); hist[name] = h
        h.to_csv(out_path("comparison", f"tuning_history_{MODELS[name]}.csv"), index=False)

if hist:
    fig, axes = plt.subplots(len(hist), 3, figsize=(17, 4.6 * len(hist)), squeeze=False)
    cc = {"id": "#1F77B4", "ig": "#2CA02C", "is": "#FF7F0E", "ib": "#9467BD"}
    for i, (name, h) in enumerate(hist.items()):
        ax = axes[i, 0]
        ax.scatter(h.trial, h.score, s=14, color="#999999", label="each trial")
        ax.plot(h.trial, h.best_so_far, color=colour[name], lw=2, label="best so far")
        ax.set_yscale("log"); ax.set_xlabel("Optuna trial"); ax.set_ylabel("score (lower = better)")
        ax.set_title(f"{name}: tuning score per trial\nscore = mean over currents of (median + 0.2 x p95) % error")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        bi = h.score.cummin().values
        best_rows = h.loc[[int(np.argmin(h.score.values[:k + 1])) for k in range(len(h))]]
        for j, (suf, lbl) in enumerate((("med", "median % error"), ("p95", "95th-pct % error"))):
            ax = axes[i, 1 + j]
            for c in CURRENTS:
                ax.scatter(h.trial, h[f"{c}_{suf}"], s=8, alpha=0.35, color=cc[c])
                ax.plot(h.trial, best_rows[f"{c}_{suf}"].values, color=cc[c], lw=2, label=f"$I_{c[1]}$ (best trial so far)")
            ax.axhline(1, color="r", ls="--", lw=1, label="1% target")
            ax.set_yscale("log"); ax.set_xlabel("Optuna trial"); ax.set_ylabel(lbl)
            ax.set_title(f"{name}: validation {lbl} per trial"); ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_path("comparison", "tuning_history.png"), dpi=200); plt.close(fig)

print("Better model per current (validation):", best)
print(val_t.to_string(index=False))
print("Saved in", out_path("comparison", ""))
