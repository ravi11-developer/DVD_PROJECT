"""
STEP 1 - LightGBM, physics-guided, ONE model for Id, Ig, Is, Ib  (NMOS OFF)

How it works
    * A small physics model (gate tunnelling, subthreshold channel, junction current)
      is fitted on the training rows and gives a first guess of log10|I| for each current.
    * ONE LightGBM model learns the correction (log10|I| minus physics log10|I|) for all
      four currents. The input "cur" tells it which current (0=Id, 1=Ig, 2=Is, 3=Ib).
      Id and Is are predicted directly - no sum or difference.
    * A LightGBM classifier gives the sign of Id and Is. Ig and Ib are always negative.
    * Hyperparameters below were tuned with Optuna (80 trials, validation set only).

Run:  python step1_lightgbm.py      (about 2-5 minutes)
Output folder: OUT_DIR/lightgbm/
    lightgbm_report.json            team JSON format, with train / validation / test blocks
    lightgbm_metrics.csv            same numbers as a table
    lightgbm_overfit_check.csv      validation/train error ratios + verdict
    overfit_learning_curve.png      error vs number of trees (train, validation, test)
    overfit_train_val_test_bars.png median and p95 error per split
    overfit_error_cdf.png           error distribution per split
    parity_test.png                 predicted vs SPICE (deliverable 5)
    error_vs_current_test.png       where the large % errors are
    feature_importance.png
    lightgbm_model.joblib           saved model (deliverable 5)
"""
import time
import joblib
import numpy as np
import lightgbm as lgb
from nmos_off_common import (load_split, fit_physics, long_table, feature_list, sign_rows,
                             sign_target, report_model, out_path, SEED)

# ===========================================================================
# TUNED SETTINGS (Optuna, 80 trials, chosen on the validation set)
# ===========================================================================
PARAMS = dict(
    learning_rate=0.01705529871209399,     # step size per tree
    num_leaves=49,              # complexity of one tree
    min_child_samples=9,        # fewest rows in one leaf
    linear_lambda=0.13410391049749207,     # holds back the straight line inside each leaf
    subsample=0.7164413456497525,         # fraction of rows per tree
    colsample_bytree=0.9404714695259851,  # fraction of inputs per tree
    reg_lambda=0.004203980085981698,        # penalty on large leaf values
    max_bin=255,                # how finely each input is split
)
FIXED = dict(n_estimators=30000, linear_tree=True, subsample_freq=1, random_state=SEED, verbose=-1,
             n_jobs=2)   # n_jobs fixed so results repeat exactly on any PC
EARLY_STOP = 300
SIGN_PARAMS = dict(n_estimators=5000, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=2)

# ===========================================================================
df, tr, va, te = load_split()
phys = fit_physics(tr)
splits = {"train": tr, "validation": va, "test": te}
tables = {sp: long_table(d, phys) for sp, d in splits.items()}
FEAT = feature_list(tables["train"])
LT, LV = tables["train"], tables["validation"]
print(f"Train {len(tr)} | Validation {len(va)} | Test {len(te)} | {len(FEAT)} inputs")

t0 = time.time()
clf = lgb.LGBMClassifier(**SIGN_PARAMS)
clf.fit(LT.loc[sign_rows(LT), FEAT], sign_target(tr),
        eval_set=[(LV.loc[sign_rows(LV), FEAT], sign_target(va))],
        callbacks=[lgb.early_stopping(200, verbose=False)])

reg = lgb.LGBMRegressor(**FIXED, **PARAMS)
reg.fit(LT[FEAT], LT.y - LT.phys, eval_set=[(LV[FEAT], LV.y - LV.phys)],
        categorical_feature=["cur"], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
best = reg.best_iteration_
print(f"Trained in {time.time() - t0:.0f} s: {best} trees, sign model {clf.best_iteration_} trees")


def predict_log(L, n_trees=None):
    return reg.predict(L[FEAT], num_iteration=n_trees or best) + L.phys.values


def predict_sign(Ls):
    return clf.predict(Ls[FEAT])


settings = {"pipeline": "NMOS OFF - LightGBM, physics-guided, single model for Id/Ig/Is/Ib + sign classifier",
            "features_count": len(FEAT),
            "params": {**PARAMS, **{k: v for k, v in FIXED.items() if k != "verbose"},
                       "early_stopping_rounds": EARLY_STOP},
            "tuning": "Optuna TPE sampler, 80 trials, objective = mean over currents of (median error + 0.2 x p95 error) on the validation set"}
report_model("LightGBM", "lightgbm", splits, tables, predict_log, predict_sign, best, settings,
             importance=(FEAT, reg.booster_.feature_importance(importance_type="gain")), phys=phys)

joblib.dump({"model": reg, "sign_model": clf, "features": FEAT, "physics": phys},
            out_path("lightgbm", "lightgbm_model.joblib"))
print("Saved everything in", out_path("lightgbm", ""))
