"""
STEP 2 - XGBoost Physics-Informed Polynomial Hybrid Model for NMOS ON
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)
Case: NMOS ON (Vg = VDD, Vb = 0)

Architecture:
    * Same semiconductor physics priors and Degree-2 Polynomial RidgeCV baseline as Step 1.
    * Decoupled histogram-based XGBoost (tree_method='hist') regressors trained on residuals.
    * Objective="reg:absoluteerror" (MAE) for alignment with percentage error.
    * Generates complete parity plots, overfit checks, learning curves, and performance tables.

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
import pandas as pd
import xgboost as xgb
from nmos_on_common import (
    load_split, fit_physics, get_phys_preds, make_features, make_targets,
    fit_poly_base, reconstruct_currents, report_hybrid_model, out_path,
    CURRENTS, SEED
)

# Hyperparameters
PARAMS = dict(
    objective="reg:absoluteerror",
    learning_rate=0.02,
    max_depth=7,
    min_child_weight=1.0,
    reg_lambda=1.0,
    subsample=0.85,
    colsample_bytree=0.85,
)

FIXED = dict(
    n_estimators=6000,
    tree_method="hist",
    random_state=SEED,
    n_jobs=4,
    early_stopping_rounds=200,
    eval_metric="mae"
)

SIGN_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    tree_method="hist",
    random_state=SEED,
    n_jobs=2,
    early_stopping_rounds=50,
    eval_metric="logloss"
)

print("=" * 80)
print("  STEP 2: XGBOOST PHYSICS-INFORMED HYBRID MODEL FOR NMOS ON")
print("=" * 80)

df, tr, va, te = load_split()
print("Fitting semiconductor physics baseline on training rows...")
phys = fit_physics(tr)

splits = {"train": tr, "validation": va, "test": te}
phys_feats = {sp: get_phys_preds(d, phys) for sp, d in splits.items()}
X_splits = {sp: make_features(d, phys_feats[sp]) for sp, d in splits.items()}
y_splits = {sp: make_targets(d) for sp, d in splits.items()}

FEAT = list(X_splits["train"].columns)
print(f"Data ready: Train {len(tr)} | Validation {len(va)} | Test {len(te)} | {len(FEAT)} features.")

t0 = time.time()
print("\n>>> Training XGBoost Sign Classifier for Id near zero-bias...")
sign_clf = xgb.XGBClassifier(**SIGN_PARAMS)
sign_clf.fit(
    X_splits["train"], tr.id.values > 0,
    eval_set=[(X_splits["validation"], va.id.values > 0)],
    verbose=False
)

print("\n>>> Fitting Degree-2 Polynomial RidgeCV Baseline...")
poly_base = fit_poly_base(X_splits["train"], y_splits["train"])
poly_preds = {sp: poly_base.predict(X_splits[sp]) for sp in splits}
resid = {sp: y_splits[sp] - poly_preds[sp] for sp in splits}

print("\n>>> Training Decoupled XGBoost Regressors on Residuals...")
models = {}
trees_used = {}
for j, c in enumerate(CURRENTS):
    print(f"  Training XGBoost on {c.upper()} residual...")
    reg = xgb.XGBRegressor(**PARAMS, **FIXED)
    reg.fit(
        X_splits["train"], resid["train"][:, j],
        eval_set=[(X_splits["validation"], resid["validation"][:, j])],
        verbose=False
    )
    models[c] = reg
    trees_used[c] = reg.best_iteration + 1
    print(f"  ✓ {c.upper()} best iteration: {reg.best_iteration + 1}")

elapsed = time.time() - t0
avg_trees = int(np.mean(list(trees_used.values())))
print(f"✓ Training complete in {elapsed:.1f} s! Average trees: {avg_trees}")

# Reconstruct predictions across splits
pred_currents = {}
for sp in splits:
    y_pred_log = np.zeros_like(poly_preds[sp])
    for j, c in enumerate(CURRENTS):
        y_pred_log[:, j] = poly_preds[sp][:, j] + models[c].predict(X_splits[sp])
    pred_currents[sp] = reconstruct_currents(y_pred_log, splits[sp], X_splits[sp], sign_clf)

# Feature importance (summed gain across all 4 terminal models)
importance_vals = np.zeros(len(FEAT))
for c in CURRENTS:
    score_dict = models[c].get_booster().get_score(importance_type="gain")
    for idx, f in enumerate(FEAT):
        importance_vals[idx] += score_dict.get(f, 0.0)

settings = {
    "pipeline": "NMOS ON - XGBoost (hist) Physics-Informed Polynomial Hybrid Model (Decoupled)",
    "features_count": len(FEAT),
    "params": {**PARAMS, **{k: v for k, v in FIXED.items() if k != 'eval_metric'}},
    "tuning": "Decoupled MAE regression on polynomial residuals with early stopping"
}

# Generate learning curve across iterations
curve_checkpoints = [50, 100, 200, 400, 800, 1500, 3000, avg_trees]
curve_checkpoints = sorted(list(set([k for k in curve_checkpoints if k <= avg_trees])))
if avg_trees not in curve_checkpoints:
    curve_checkpoints.append(avg_trees)

curve = {"iters": curve_checkpoints, "train": [], "validation": [], "test": []}
for k in curve_checkpoints:
    for sp in splits:
        y_k = np.zeros_like(poly_preds[sp])
        for j, c in enumerate(CURRENTS):
            num_it = min(k, models[c].best_iteration + 1)
            y_k[:, j] = poly_preds[sp][:, j] + models[c].predict(X_splits[sp], iteration_range=(0, num_it))
        P_k = reconstruct_currents(y_k, splits[sp], X_splits[sp], sign_clf)
        med_errs = [float(np.median(np.abs(P_k[c].values - splits[sp][c].values) / (np.abs(splits[sp][c].values) + 1e-10 if c in ["id","is"] else np.abs(splits[sp][c].values) + 1e-15) * 100.0)) for c in CURRENTS]
        curve[sp].append(float(np.mean(med_errs)))

print("\n>>> Generating Full Deliverables, Metrics Tables, and Diagnostic Plots...")
report_hybrid_model(
    "XGBoost", "xgboost", splits, pred_currents, avg_trees, settings,
    importance=(FEAT, importance_vals), phys=phys, curve=curve
)

# Save artifact models
model_payload = {
    "poly_base": poly_base,
    "models": models,
    "sign_model": sign_clf,
    "physics": phys,
    "features": FEAT,
    "trees_used": trees_used
}
joblib.dump(model_payload, out_path("xgboost", "xgboost_model.joblib"))
print(f"✓ Model payload saved to: {out_path('xgboost', 'xgboost_model.joblib')}")
print(f"✓ All outputs successfully generated in: {out_path('xgboost', '')}")
