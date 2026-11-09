"""
STEP 2 - XGBoost, Physics-Guided, ONE Model for Id, Ig, Is, Ib (NMOS ON)
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)
Case: NMOS ON (Vg = VDD, Vb = 0)

Architecture:
    * Same physics baseline and identical long-format feature inputs as Step 1.
    * Evaluates high-performance histogram-based XGBoost (tree_method='hist').
    * ONE XGBoost model learns the residual correction Delta = log10|I| - physics_log10|I|.
    * An XGBoost binary classifier predicts the sign of Id and Is.
    * Generates parity plots, overfit checks, learning curves, and performance tables.

Run:
    python step2_xgboost.py
Outputs (in results/xgboost/):
    xgboost_report.json, xgboost_metrics.csv, xgboost_overfit_check.csv,
    overfit_learning_curve.png, overfit_train_val_test_bars.png, overfit_error_cdf.png,
    parity_test.png, error_vs_current_test.png, feature_importance.png,
    xgboost_model.joblib
"""

import time
import joblib
import numpy as np
import xgboost as xgb
from nmos_on_common import (
    load_split, fit_physics, long_table, feature_list, sign_rows,
    sign_target, report_model, out_path, SEED
)

# ===========================================================================
# HYPERPARAMETERS (Histogram-Based XGBoost)
# ===========================================================================
PARAMS = dict(
    learning_rate=0.008,                # Conservative learning rate for generalization
    max_depth=8,                        # Tree depth to capture multi-parameter PVT interactions
    min_child_weight=0.1,               # Minimum Hessian sum in leaf
    subsample=0.85,                     # Row subsampling ratio
    colsample_bytree=0.85,              # Subsampling ratio of columns per tree
    colsample_bylevel=0.60,             # Subsampling ratio of columns per level
    reg_lambda=1.0,                     # L2 regularization
    reg_alpha=1e-5,                     # L1 regularization
    max_bin=512,                        # High-resolution histogram binning
)

FIXED = dict(
    n_estimators=20000,
    tree_method="hist",
    random_state=SEED,
    n_jobs=2,
    early_stopping_rounds=300,
    eval_metric="rmse"
)

SIGN_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=6,
    tree_method="hist",
    random_state=SEED,
    n_jobs=2,
    early_stopping_rounds=200,
    eval_metric="logloss"
)

# ===========================================================================
# EXECUTION
# ===========================================================================
print("=" * 80)
print("  STEP 2: XGBOOST PHYSICS-GUIDED MODEL FOR NMOS ON")
print("=" * 80)

df, tr, va, te = load_split()
print("Fitting semiconductor physics baseline on training rows...")
phys = fit_physics(tr)

splits = {"train": tr, "validation": va, "test": te}
print("Constructing long tables (4 terminals per sample)...")
tables = {sp: long_table(d, phys) for sp, d in splits.items()}
FEAT = feature_list(tables["train"])
LT, LV = tables["train"], tables["validation"]
print(f"Data ready: Train {len(tr)} | Validation {len(va)} | Test {len(te)} | {len(FEAT)} features.")

t0 = time.time()
print("\n>>> Training XGBoost Sign Classifier for Id and Is...")
clf = xgb.XGBClassifier(**SIGN_PARAMS)
clf.fit(
    LT.loc[sign_rows(LT), FEAT], sign_target(tr),
    eval_set=[(LV.loc[sign_rows(LV), FEAT], sign_target(va))],
    verbose=False
)

print("\n>>> Training Physics-Guided XGBoost Regressor (hist)...")
reg = xgb.XGBRegressor(**FIXED, **PARAMS)
reg.fit(
    LT[FEAT], LT.y - LT.phys,
    eval_set=[(LV[FEAT], LV.y - LV.phys)],
    verbose=False
)
best = reg.best_iteration + 1
elapsed = time.time() - t0
print(f"✓ Training complete in {elapsed:.1f} s! Best trees: {best}, Sign model trees: {clf.best_iteration + 1}")


def predict_log(L, n_trees=None):
    if n_trees is not None:
        return reg.predict(L[FEAT], iteration_range=(0, n_trees)) + L.phys.values
    return reg.predict(L[FEAT]) + L.phys.values


def predict_sign(Ls):
    return clf.predict(Ls[FEAT])


settings = {
    "pipeline": "NMOS ON - XGBoost (hist), physics-guided, single model for Id/Ig/Is/Ib + sign classifier",
    "features_count": len(FEAT),
    "params": {**PARAMS, **{k: v for k, v in FIXED.items() if k != "eval_metric"}},
    "tuning": "Histogram gradient boosting with early stopping on validation RMSE"
}

# Feature importance from booster
score_dict = reg.get_booster().get_score(importance_type="gain")
vals = [score_dict.get(f, 0.0) for f in FEAT]

print("\n>>> Generating Full Deliverables, Metrics Tables, and Diagnostic Plots...")
report_model(
    "XGBoost", "xgboost", splits, tables, predict_log, predict_sign, best, settings,
    importance=(FEAT, vals),
    phys=phys
)

# Save artifact models
model_payload = {
    "model": reg,
    "sign_model": clf,
    "features": FEAT,
    "physics": phys,
    "best_iteration": best
}
joblib.dump(model_payload, out_path("xgboost", "xgboost_model.joblib"))
print(f"✓ Model payload saved to: {out_path('xgboost', 'xgboost_model.joblib')}")
print(f"✓ All outputs successfully generated in: {out_path('xgboost', '')}")
