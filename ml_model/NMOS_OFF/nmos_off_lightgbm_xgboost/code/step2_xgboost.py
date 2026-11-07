"""
STEP 2 - XGBoost, physics-guided, ONE model for Id, Ig, Is, Ib  (NMOS OFF)

Same idea and same inputs as step1_lightgbm.py, only the learning algorithm changes:
    * physics model (fitted on training rows) gives a first guess of log10|I|
    * ONE XGBoost model learns the correction for all four currents ("cur" = which current)
    * Id and Is predicted directly - no sum or difference
    * an XGBoost classifier gives the sign of Id and Is; Ig and Ib are always negative
    * hyperparameters tuned with Optuna (50 trials, validation set only; stopped when
      the last 10 trials gave no improvement)

Run:  python step2_xgboost.py      (about 5-15 minutes)
Output folder: OUT_DIR/xgboost/   (same files as the LightGBM folder)
"""
import time
import joblib
import numpy as np
import xgboost as xgb
from nmos_off_common import (load_split, fit_physics, long_table, feature_list, sign_rows,
                             sign_target, report_model, out_path, SEED)

# ===========================================================================
# TUNED SETTINGS (Optuna, 50 trials, chosen on the validation set)
# ===========================================================================
PARAMS = dict(
    learning_rate=0.006953214142421666,    # step size per tree
    max_depth=8,                           # how deep one tree may grow
    min_child_weight=0.07712226024758602,  # smallest "amount of data" in a leaf
    subsample=0.8258492751181986,          # fraction of rows per tree
    colsample_bytree=0.831837533285526,    # fraction of inputs per tree
    colsample_bylevel=0.5128653421137334,  # fraction of inputs per tree level
    reg_lambda=1.2126276671013703,         # L2 penalty on leaf values
    reg_alpha=1.4341037883227981e-05,      # L1 penalty on leaf values
    max_bin=1024,                          # how finely each input is split
)
FIXED = dict(n_estimators=20000, tree_method="hist", random_state=SEED, n_jobs=2,
             early_stopping_rounds=300, eval_metric="rmse")
SIGN_PARAMS = dict(n_estimators=5000, learning_rate=0.05, max_depth=6, tree_method="hist",
                   random_state=SEED, n_jobs=2, early_stopping_rounds=200, eval_metric="logloss")

# ===========================================================================
df, tr, va, te = load_split()
phys = fit_physics(tr)
splits = {"train": tr, "validation": va, "test": te}
tables = {sp: long_table(d, phys) for sp, d in splits.items()}
FEAT = feature_list(tables["train"])
LT, LV = tables["train"], tables["validation"]
print(f"Train {len(tr)} | Validation {len(va)} | Test {len(te)} | {len(FEAT)} inputs")

t0 = time.time()
clf = xgb.XGBClassifier(**SIGN_PARAMS)
clf.fit(LT.loc[sign_rows(LT), FEAT], sign_target(tr),
        eval_set=[(LV.loc[sign_rows(LV), FEAT], sign_target(va))], verbose=False)

reg = xgb.XGBRegressor(**FIXED, **PARAMS)
reg.fit(LT[FEAT], LT.y - LT.phys, eval_set=[(LV[FEAT], LV.y - LV.phys)], verbose=False)
best = reg.best_iteration + 1          # XGBoost counts from 0
print(f"Trained in {time.time() - t0:.0f} s: {best} trees, sign model {clf.best_iteration + 1} trees")


def predict_log(L, n_trees=None):
    return reg.predict(L[FEAT], iteration_range=(0, n_trees or best)) + L.phys.values


def predict_sign(Ls):
    return clf.predict(Ls[FEAT])


settings = {"pipeline": "NMOS OFF - XGBoost, physics-guided, single model for Id/Ig/Is/Ib + sign classifier",
            "features_count": len(FEAT),
            "params": {**PARAMS, **{k: v for k, v in FIXED.items() if k != "eval_metric"}},
            "tuning": "Optuna TPE sampler, 50 trials (stopped: no gain in the last 10), objective = mean over "
                      "currents of (median error + 0.2 x p95 error) on the validation set"}
imp = reg.get_booster().get_score(importance_type="gain")
report_model("XGBoost", "xgboost", splits, tables, predict_log, predict_sign, best, settings,
             importance=(FEAT, [imp.get(f, 0.0) for f in FEAT]), phys=phys)

joblib.dump({"model": reg, "sign_model": clf, "features": FEAT, "physics": phys},
            out_path("xgboost", "xgboost_model.joblib"))
print("Saved everything in", out_path("xgboost", ""))
