"""
STEP 1 - LightGBM, Physics-Guided, ONE Model for Id, Ig, Is, Ib (NMOS ON)
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)
Case: NMOS ON (Vg = VDD, Vb = 0)

Architecture:
    * Compact semiconductor physics baseline predicts first estimate of log10|I|
      for each terminal current (conduction, tunneling, bulk).
    * ONE LightGBM model with Linear Trees (linear_tree=True) learns the correction
      Delta = log10|I_SPICE| - log10|I_physics| across all 4 terminal currents.
      Input "cur" (0=Id, 1=Ig, 2=Is, 3=Ib) tells the model which terminal to predict.
    * Linear trees place a linear model in each leaf, completely eliminating staircase
      approximation errors across the continuous Vd ≈ Vs boundary.
    * A LightGBM binary classifier predicts the sign of Id and Is.
      Ig is strictly positive and Ib is strictly negative for NMOS ON.
    * Produces full deliverable metrics, JSON reports, overfit checks, and diagnostic plots.

Run:
    python step1_lightgbm.py
Outputs (in results/lightgbm/):
    lightgbm_report.json, lightgbm_metrics.csv, lightgbm_overfit_check.csv,
    overfit_learning_curve.png, overfit_train_val_test_bars.png, overfit_error_cdf.png,
    parity_test.png, error_vs_current_test.png, feature_importance.png,
    lightgbm_model.joblib
"""

import time
import joblib
import numpy as np
import lightgbm as lgb
from nmos_on_common import (
    load_split, fit_physics, long_table, feature_list, sign_rows,
    sign_target, report_model, out_path, SEED
)

# ===========================================================================
# HYPERPARAMETERS (Optimized for High Accuracy with Linear Trees)
# ===========================================================================
PARAMS = dict(
    learning_rate=0.015,                # Fine gradient descent step
    num_leaves=63,                      # High capacity for PVT cross-terms
    min_child_samples=10,               # Minimum samples in leaf
    linear_lambda=0.1,                  # L2 regularizer on straight lines inside leaves
    subsample=0.80,                     # Row bagging fraction
    colsample_bytree=0.90,              # Feature bagging fraction
    reg_lambda=0.01,                    # L2 leaf penalty
    max_bin=255,                        # Quantization resolution
)

FIXED = dict(
    n_estimators=30000,
    linear_tree=True,                   # Linear regression inside each leaf
    subsample_freq=1,
    random_state=SEED,
    verbose=-1,
    n_jobs=2                            # Deterministic results across any system
)
EARLY_STOP = 300

SIGN_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=SEED,
    verbose=-1,
    n_jobs=2
)

# ===========================================================================
# EXECUTION
# ===========================================================================
print("=" * 80)
print("  STEP 1: LIGHTGBM PHYSICS-GUIDED MODEL FOR NMOS ON")
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
print("\n>>> Training Sign Classifier for Id and Is...")
clf = lgb.LGBMClassifier(**SIGN_PARAMS)
clf.fit(
    LT.loc[sign_rows(LT), FEAT], sign_target(tr),
    eval_set=[(LV.loc[sign_rows(LV), FEAT], sign_target(va))],
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

print("\n>>> Training Physics-Guided Linear Tree LightGBM Regressor...")
reg = lgb.LGBMRegressor(**FIXED, **PARAMS)
reg.fit(
    LT[FEAT], LT.y - LT.phys,
    eval_set=[(LV[FEAT], LV.y - LV.phys)],
    categorical_feature=["cur"],
    callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
)
best = reg.best_iteration_
elapsed = time.time() - t0
print(f"✓ Training complete in {elapsed:.1f} s! Best trees: {best}, Sign model trees: {clf.best_iteration_}")


def predict_log(L, n_trees=None):
    return reg.predict(L[FEAT], num_iteration=n_trees or best) + L.phys.values


def predict_sign(Ls):
    return clf.predict(Ls[FEAT])


settings = {
    "pipeline": "NMOS ON - LightGBM Linear Tree, physics-guided, single model for Id/Ig/Is/Ib + sign classifier",
    "features_count": len(FEAT),
    "params": {**PARAMS, **{k: v for k, v in FIXED.items() if k != "verbose"}, "early_stopping_rounds": EARLY_STOP},
    "tuning": "Linear Tree formulation (linear_tree=True) with early stopping on validation median error"
}

print("\n>>> Generating Full Deliverables, Metrics Tables, and Diagnostic Plots...")
report_model(
    "LightGBM", "lightgbm", splits, tables, predict_log, predict_sign, best, settings,
    importance=(FEAT, reg.booster_.feature_importance(importance_type="gain")),
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
joblib.dump(model_payload, out_path("lightgbm", "lightgbm_model.joblib"))
print(f"✓ Model payload saved to: {out_path('lightgbm', 'lightgbm_model.joblib')}")
print(f"✓ All outputs successfully generated in: {out_path('lightgbm', '')}")
