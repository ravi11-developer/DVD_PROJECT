"""
STEP 4 - How the results change when ONE parameter changes (both models)

For each key parameter, all other parameters stay at their tuned value and only that
one is varied. Every fit records the error on train, validation and test.
    * validation/test curve  -> how accuracy on unseen data depends on the parameter
    * gap between train and validation -> how much the model memorises (overfitting)

Run:  python step4_parameter_sweep.py lightgbm     (about 30-40 minutes)
      python step4_parameter_sweep.py xgboost      (about 1-2 hours)
      python step4_parameter_sweep.py plot         (only redraw the graphs)
Output folder: OUT_DIR/parameter_sweep/
    sweep_lightgbm.csv, sweep_xgboost.csv           one row per fit (results are saved after every fit)
    sweep_lightgbm.png, sweep_xgboost.png           the graphs
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nmos_off_common import (load_split, fit_physics, long_table, feature_list, sign_rows, sign_target,
                             to_currents, pct_err, out_path, CURRENTS, SEED)

LGBM_BEST = dict(learning_rate=0.01705529871209399, num_leaves=49, min_child_samples=9,
                 linear_lambda=0.13410391049749207, subsample=0.7164413456497525,
                 colsample_bytree=0.9404714695259851, reg_lambda=0.004203980085981698, max_bin=255)
LGBM_SWEEP = {"num_leaves": [4, 8, 16, 32, 49, 96],
              "min_child_samples": [3, 9, 30, 100],
              "linear_lambda": [0.01, 0.13410391049749207, 1.0, 10.0],
              "learning_rate": [0.0085, 0.01705529871209399, 0.035, 0.07],
              "subsample": [0.5, 0.7164413456497525, 1.0]}

XGB_BEST = dict(learning_rate=0.006953214142421666, max_depth=8, min_child_weight=0.07712226024758602,
                subsample=0.8258492751181986, colsample_bytree=0.831837533285526,
                colsample_bylevel=0.5128653421137334, reg_lambda=1.2126276671013703,
                reg_alpha=1.4341037883227981e-05, max_bin=1024)
XGB_SWEEP = {"max_depth": [4, 6, 8, 10],
             "min_child_weight": [0.07712226024758602, 1.0, 10.0],
             "reg_lambda": [0.1, 1.2126276671013703, 10.0],
             "learning_rate": [0.014, 0.006953214142421666, 0.028]}


def run(model):
    import lightgbm as lgb
    import xgboost as xgb
    df, tr, va, te = load_split()
    phys = fit_physics(tr)
    splits = {"train": tr, "validation": va, "test": te}
    tables = {sp: long_table(d, phys) for sp, d in splits.items()}
    FEAT = feature_list(tables["train"]); LT, LV = tables["train"], tables["validation"]

    if model == "lightgbm":
        clf = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=31, random_state=SEED,
                                 verbose=-1, n_jobs=2)
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
    for pname, values in sweep.items():
        for v in values:
            is_best = np.isclose(v, best_p[pname])
            if len(done) and ((done.param == pname) & np.isclose(done.value, v)).any():
                continue                                  # already done (resume after a stop)
            if is_best and best_cache is not None:        # same fit as the tuned model: reuse it
                row = dict(best_cache, param=pname, value=v); done = pd.concat([done, pd.DataFrame([row])])
                done.to_csv(csv, index=False); continue
            p = dict(best_p); p[pname] = v
            t0 = time.time()
            if model == "lightgbm":
                reg = lgb.LGBMRegressor(n_estimators=30000, linear_tree=True, subsample_freq=1, random_state=SEED,
                                        verbose=-1, n_jobs=2, **p)
                reg.fit(LT[FEAT], LT.y - LT.phys, eval_set=[(LV[FEAT], LV.y - LV.phys)],
                        categorical_feature=["cur"], callbacks=[lgb.early_stopping(300, verbose=False)])
                trees = reg.best_iteration_
                pl = lambda L: reg.predict(L[FEAT], num_iteration=trees) + L.phys.values
            else:
                reg = xgb.XGBRegressor(n_estimators=20000, tree_method="hist", random_state=SEED, n_jobs=2,
                                       early_stopping_rounds=300, eval_metric="rmse", **p)
                reg.fit(LT[FEAT], LT.y - LT.phys, eval_set=[(LV[FEAT], LV.y - LV.phys)], verbose=False)
                trees = reg.best_iteration + 1
                pl = lambda L: reg.predict(L[FEAT], iteration_range=(0, trees)) + L.phys.values
            row = {"param": pname, "value": v, "is_tuned_value": bool(is_best), "trees": int(trees),
                   "fit_seconds": round(time.time() - t0, 1)}
            for sp, d in splits.items():
                P = to_currents(pl(tables[sp]), signs[sp], d)
                meds, p95s = [], []
                for c in CURRENTS:
                    e = pct_err(d[c].values, P[c].values)
                    row[f"{sp}_{c}_median"] = float(np.median(e)); row[f"{sp}_{c}_p95"] = float(np.percentile(e, 95))
                    meds.append(np.median(e)); p95s.append(np.percentile(e, 95))
                row[f"{sp}_median"] = float(np.mean(meds)); row[f"{sp}_p95"] = float(np.mean(p95s))
            if is_best:
                best_cache = row
            done = pd.concat([done, pd.DataFrame([row])]); done.to_csv(csv, index=False)
            print(f"{model} {pname}={v:.4g}: trees {trees}, val median {row['validation_median']:.3f}% "
                  f"p95 {row['validation_p95']:.2f}% | train median {row['train_median']:.3f}% "
                  f"({row['fit_seconds']:.0f}s)", flush=True)


def plot(model):
    csv = out_path("parameter_sweep", f"sweep_{model}.csv")
    if not os.path.exists(csv):
        return
    d = pd.read_csv(csv); params = list(dict.fromkeys(d.param))
    fig, axes = plt.subplots(2, len(params), figsize=(4.2 * len(params), 8), squeeze=False)
    col = {"train": "#7F7F7F", "validation": "#1F77B4", "test": "#D62728"}
    for j, pname in enumerate(params):
        s = d[d.param == pname].sort_values("value")
        for i, (key, name) in enumerate((("median", "median % error"), ("p95", "95th-pct % error"))):
            ax = axes[i, j]
            for sp in ("train", "validation", "test"):
                ax.plot(s.value, s[f"{sp}_{key}"], "-o", ms=4, color=col[sp], label=sp)
            tv = s[s.is_tuned_value]
            if len(tv):
                ax.scatter(tv.value, tv[f"validation_{key}"], s=140, facecolors="none", edgecolors="k",
                           lw=1.5, zorder=5, label="tuned value")
            ax.axhline(1, color="g", ls=":", lw=1)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
            ax.set_xlabel(pname)
            if j == 0:
                ax.set_ylabel(f"{name}\n(mean of Id, Ig, Is, Ib)")
            if i == 0:
                ax.set_title(pname)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"{'LightGBM' if model == 'lightgbm' else 'XGBoost'}: effect of each parameter "
                 "(others fixed at tuned values). Gap train-vs-validation = memorising", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path("parameter_sweep", f"sweep_{model}.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if what in ("lightgbm", "xgboost"):
        run(what)
        plot(what)
    else:
        plot("lightgbm"); plot("xgboost")
