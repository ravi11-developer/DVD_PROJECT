"""
STEP 4 - Hyperparameter Sensitivity Sweep for NMOS ON (LightGBM & XGBoost)
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)
Case: NMOS ON (Vg = VDD, Vb = 0)

Evaluates how validation and test errors move as key hyperparameters vary one at a time:
    * num_leaves, learning_rate, min_child_samples, linear_lambda for LightGBM
    * max_depth, learning_rate, reg_lambda, min_child_weight for XGBoost

Run:
    python step4_parameter_sweep.py lightgbm     # Run sweep for LightGBM (~15-30 min)
    python step4_parameter_sweep.py xgboost      # Run sweep for XGBoost (~30-60 min)
    python step4_parameter_sweep.py plot         # Redraw plots from existing CSVs
Outputs (in results/parameter_sweep/):
    sweep_lightgbm.csv, sweep_lightgbm.png, sweep_xgboost.csv, sweep_xgboost.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nmos_on_common import (
    load_split, fit_physics, long_table, feature_list, sign_rows, sign_target,
    to_currents, pct_err, out_path, CURRENTS, SEED
)

LGBM_BEST = dict(
    learning_rate=0.015,
    num_leaves=63,
    min_child_samples=10,
    linear_lambda=0.1,
    subsample=0.80,
    colsample_bytree=0.90,
    reg_lambda=0.01,
    max_bin=255
)

LGBM_SWEEP = {
    "num_leaves": [16, 31, 63, 127],
    "min_child_samples": [5, 10, 25, 50],
    "linear_lambda": [0.01, 0.1, 1.0, 10.0],
    "learning_rate": [0.005, 0.015, 0.03, 0.06]
}

XGB_BEST = dict(
    learning_rate=0.008,
    max_depth=8,
    min_child_weight=0.1,
    subsample=0.85,
    colsample_bytree=0.85,
    colsample_bylevel=0.60,
    reg_lambda=1.0,
    reg_alpha=1e-5,
    max_bin=512
)

XGB_SWEEP = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.004, 0.008, 0.016, 0.032],
    "min_child_weight": [0.01, 0.1, 1.0, 5.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0]
}


def run_sweep(model):
    import lightgbm as lgb
    import xgboost as xgb

    df, tr, va, te = load_split()
    phys = fit_physics(tr)
    splits = {"train": tr, "validation": va, "test": te}
    tables = {sp: long_table(d, phys) for sp, d in splits.items()}
    FEAT = feature_list(tables["train"])
    LT, LV = tables["train"], tables["validation"]

    if model == "lightgbm":
        clf = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=31,
                                 random_state=SEED, verbose=-1, n_jobs=2)
        clf.fit(LT.loc[sign_rows(LT), FEAT], sign_target(tr),
                eval_set=[(LV.loc[sign_rows(LV), FEAT], sign_target(va))],
                callbacks=[lgb.early_stopping(200, verbose=False)])
        best_p, sweep = LGBM_BEST, LGBM_SWEEP
    else:
        clf = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=6, tree_method="hist",
                                random_state=SEED, n_jobs=2, early_stopping_rounds=200, eval_metric="logloss")
        clf.fit(LT.loc[sign_rows(LT), FEAT], sign_target(tr),
                eval_set=[(LV.loc[sign_rows(LV), FEAT], sign_target(va))], verbose=False)
        best_p, sweep = XGB_BEST, XGB_SWEEP

    signs = {sp: clf.predict(tables[sp].loc[sign_rows(tables[sp]), FEAT]) for sp in splits}
    csv = out_path("parameter_sweep", f"sweep_{model}.csv")
    done = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    best_cache = None

    def evaluate_params(p):
        t0 = time.time()
        if model == "lightgbm":
            reg = lgb.LGBMRegressor(n_estimators=30000, linear_tree=True, subsample_freq=1,
                                    random_state=SEED, verbose=-1, n_jobs=2, **p)
            reg.fit(LT[FEAT], LT.y - LT.phys, eval_set=[(LV[FEAT], LV.y - LV.phys)],
                    categorical_feature=["cur"], callbacks=[lgb.early_stopping(300, verbose=False)])
            b = reg.best_iteration_

            def pred(L):
                return reg.predict(L[FEAT], num_iteration=b) + L.phys.values
        else:
            reg = xgb.XGBRegressor(n_estimators=20000, tree_method="hist", random_state=SEED,
                                   n_jobs=2, early_stopping_rounds=300, eval_metric="rmse", **p)
            reg.fit(LT[FEAT], LT.y - LT.phys, eval_set=[(LV[FEAT], LV.y - LV.phys)], verbose=False)
            b = reg.best_iteration + 1

            def pred(L):
                return reg.predict(L[FEAT]) + L.phys.values

        res = {"best_iteration": b, "fit_seconds": round(time.time() - t0, 1)}
        for sp, d in splits.items():
            P = to_currents(pred(tables[sp]), signs[sp], d)
            for c in CURRENTS:
                e = pct_err(d[c].values, P[c].values, c)
                res[f"{sp}_{c}_med"] = float(np.median(e))
                res[f"{sp}_{c}_p95"] = float(np.percentile(e, 95))
            res[f"{sp}_mean_med"] = float(np.mean([res[f"{sp}_{c}_med"] for c in CURRENTS]))
        return res

    results = done.to_dict("records") if len(done) else []

    for param, vals in sweep.items():
        for v in vals:
            if len(done) and ((done.parameter == param) & (done.value == str(v))).any():
                print(f"[{model}] Skipping already computed: {param} = {v}")
                continue
            p = dict(best_p); p[param] = v
            if p == best_p and best_cache is not None:
                r = dict(best_cache)
            else:
                print(f"[{model}] Running fit: {param} = {v} ...")
                r = evaluate_params(p)
                if p == best_p:
                    best_cache = r
            r["parameter"] = param
            r["value"] = str(v)
            results.append(r)
            pd.DataFrame(results).to_csv(csv, index=False)

    plot_sweep(model)


def plot_sweep(model):
    csv = out_path("parameter_sweep", f"sweep_{model}.csv")
    if not os.path.exists(csv):
        print(f"No sweep data found for {model} at {csv}")
        return
    df = pd.read_csv(csv)
    params = list(df.parameter.unique())
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 4.5))
    if len(params) == 1:
        axes = [axes]

    for ax, p in zip(axes, params):
        sub = df[df.parameter == p].copy()
        try:
            sub["v_num"] = sub.value.astype(float)
            sub = sub.sort_values("v_num")
            x_vals = sub.v_num.values
            ax.set_xscale("log" if (x_vals.max() / max(x_vals.min(), 1e-9) > 10) else "linear")
        except ValueError:
            sub = sub.sort_values("value")
            x_vals = np.arange(len(sub))
            ax.set_xticks(x_vals)
            ax.set_xticklabels(sub.value.values)

        for sp, col in [("train", "#7F7F7F"), ("validation", "#1F77B4"), ("test", "#D62728")]:
            ax.plot(x_vals, sub[f"{sp}_mean_med"], "-o", ms=4, label=sp.capitalize(), color=col)

        ax.axhline(1.0, color="g", ls="--", lw=1.2, label="1% Target")
        ax.set_xlabel(p)
        ax.set_ylabel("Median Error (%) [Mean of 4 currents]")
        ax.set_title(p, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle(f"NMOS ON: Parameter Sweep — {model.upper()} Sensitivity Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig_path = out_path("parameter_sweep", f"sweep_{model}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved parameter sweep plot: {fig_path}")


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "lightgbm"
    if arg in ("lightgbm", "xgboost"):
        run_sweep(arg)
    elif arg == "plot":
        plot_sweep("lightgbm")
        plot_sweep("xgboost")
    else:
        print("Usage: python step4_parameter_sweep.py [lightgbm | xgboost | plot]")
