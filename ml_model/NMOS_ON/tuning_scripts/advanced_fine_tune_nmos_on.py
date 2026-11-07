#!/usr/bin/env python3
"""
advanced_fine_tune_nmos_on.py — Advanced Fine-Tuning NMOS ON Pipeline (Target: 97%-99%)
========================================================================================
Implements the 5-pillar advanced architecture:
  1. 35 Physics-informed features including:
     - 22nm velocity saturation knee: Vdsat, Vds_ratio, subthreshold exponential factor.
     - DIBL barrier-lowering overdrive: dibl_vov.
     - Gate overlap electric fields & quantum tunneling exponents: E_ox_gs, E_ox_gd.
     - Junction diode diffusion terms: junc_diode_bs, junc_diode_bd.
  2. Hierarchical 2-stage modeling for Substrate Current (Ib):
     - Predicts channel conduction (Id) first.
     - Feeds log10(|Id_pred|) and impact ionization driving force into Ib model.
  3. Trains high-capacity LightGBM and XGBoost regressors on all terminals.
  4. Computes optimal terminal-specific blending weights on Validation set.
  5. Evaluates LightGBM, XGBoost, and the Hybrid Ensemble on unseen Test set.
  6. Applies channel-only KCL conservation.
  7. Outputs complete metrics, saves models, and generates parity plots.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def log_step(step_name, start_time=None):
    now_str = time.strftime("%H:%M:%S")
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[{now_str}] ✓ Completed: {step_name} (took {elapsed:.2f}s)")
        print("-" * 85)
    else:
        print(f"\n[{now_str}] >>> Starting: {step_name}...")


def main():
    total_start = time.time()
    print("=" * 85)
    print("      NMOS_ON ADVANCED FINE-TUNING PIPELINE (TARGET: 97%–99% ACCURACY)")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # STEP 1: Discover & Load Dataset
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 1: Dataset Loading & Header Standardization")
    
    this_script_dir = os.path.dirname(os.path.abspath(__file__))
    nmos_on_dir = os.path.abspath(os.path.join(this_script_dir, ".."))
    candidate_paths = [
        os.path.join(nmos_on_dir, "..", "..", "dataset", "NMOS_ON", "nmos_on.csv"),
        os.path.join(nmos_on_dir, "..", "..", "Data_Generation_M1", "data", "nmos_on.csv"),
        "dataset/NMOS_ON/nmos_on.csv",
        "../dataset/NMOS_ON/nmos_on.csv",
        "../../dataset/NMOS_ON/nmos_on.csv",
        "../../../dataset/NMOS_ON/nmos_on.csv",
        "Data_Generation_M1/data/nmos_on.csv",
        "../Data_Generation_M1/data/nmos_on.csv",
        "../../Data_Generation_M1/data/nmos_on.csv",
        "../../../Data_Generation_M1/data/nmos_on.csv",
    ]
    data_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if not data_path:
        raise FileNotFoundError(f"Could not find nmos_on.csv in: {candidate_paths}")
        
    print(f"  Dataset selected: {data_path}")
    df_raw = pd.read_csv(data_path)
    df_raw.columns = df_raw.columns.str.strip()
    if 'itotal' in df_raw.columns:
        df_raw = df_raw.drop(columns=['itotal'])
        
    print(f"  Loaded {len(df_raw)} samples with {len(df_raw.columns)} raw columns.")
    log_step("Step 1: Dataset Loading & Header Standardization", t0)

    # -------------------------------------------------------------------------
    # STEP 2: Advanced Physics-Informed Feature Engineering (35 Features)
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 2: Advanced Physics-Informed Feature Engineering (35 Features)")
    
    df = df_raw.copy()
    
    # 1. Terminal voltage drops (carrier drift driving forces)
    df['vds'] = df['vd'] - df['vs']
    df['abs_vds'] = np.abs(df['vds'])
    df['vgs'] = df['vg'] - df['vs']
    df['vgd'] = df['vg'] - df['vd']
    df['vbs'] = df['vb'] - df['vs']
    df['vbd'] = df['vb'] - df['vd']
    df['vdb'] = df['vd'] - df['vb']
    df['vsb'] = df['vs'] - df['vb']
    
    # 2. Gate overdrive voltage
    df['vov'] = df['vgs'] - df['vth0']
    
    # 3. Thermal physics (k*T/q)
    df['temp_k'] = df['temp'] + 273.15
    K_BOLTZMANN_OVER_Q = 8.617333e-5
    df['vt'] = K_BOLTZMANN_OVER_Q * df['temp_k']
    
    # 4. Aspect ratio
    df['wl_ratio'] = df['w'] / 22e-9
    
    # 5. Gate oxide electric fields (overall, gate-source overlap, gate-drain overlap)
    df['e_ox'] = (df['vg'] - df['vb']) / df['toxe']
    df['e_ox_gs'] = (df['vg'] - df['vs']) / df['toxe']
    df['e_ox_gd'] = (df['vg'] - df['vd']) / df['toxe']
    # Quantum tunneling exponents (Fowler-Nordheim & direct tunneling)
    df['tunnel_exp_gs'] = np.exp(-1e8 / (np.abs(df['e_ox_gs']) + 1e6))
    df['tunnel_exp_gd'] = np.exp(-1e8 / (np.abs(df['e_ox_gd']) + 1e6))
    
    # 6. 22nm Velocity saturation & short-channel knee transition
    ESAT_L = 0.22  # V (effective velocity saturation parameter for 22nm)
    vov_pos = np.maximum(df['vov'], 0.0)
    df['vdsat'] = (ESAT_L * vov_pos) / (ESAT_L + vov_pos + 1e-4)
    df['vds_ratio'] = df['abs_vds'] / (df['vdsat'] + 1e-3)
    df['vds_eff'] = df['abs_vds'] / ((1.0 + (df['abs_vds'] / (df['vdsat'] + 1e-3))**4) ** 0.25)
    
    # 7. Drain-Induced Barrier Lowering (DIBL) effective overdrive
    df['dibl_vov'] = df['vgs'] - (df['vth0'] - 0.08 * df['abs_vds'])
    
    # 8. Subthreshold exponential conduction factor
    df['subthreshold_factor'] = np.exp(np.clip(df['vov'] / (1.5 * df['vt']), -20.0, 0.0))
    
    # 9. Depletion widths & reverse junction diode currents
    df['depletion_drain'] = np.sqrt(np.maximum(0.7 + df['vdb'], 1e-3))
    df['depletion_source'] = np.sqrt(np.maximum(0.7 + df['vsb'], 1e-3))
    df['high_field_prod'] = df['abs_vds'] * vov_pos
    df['junc_diode_bs'] = np.exp(np.clip(df['vbs'] / (1.2 * df['vt']), -20.0, 5.0)) - 1.0
    df['junc_diode_bd'] = np.exp(np.clip(df['vbd'] / (1.2 * df['vt']), -20.0, 5.0)) - 1.0

    base_features = ['vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb', 'vth0', 'toxe', 'ndep', 'u0', 'xj']
    physics_features = [
        'vds', 'abs_vds', 'vgs', 'vgd', 'vbs', 'vbd', 'vdb', 'vsb', 'vov', 'temp_k', 'vt', 'wl_ratio',
        'e_ox', 'e_ox_gs', 'e_ox_gd', 'tunnel_exp_gs', 'tunnel_exp_gd',
        'vdsat', 'vds_ratio', 'vds_eff', 'dibl_vov', 'subthreshold_factor',
        'depletion_drain', 'depletion_source', 'high_field_prod', 'junc_diode_bs', 'junc_diode_bd'
    ]
    core_features = base_features + physics_features
    target_cols = ['id', 'ig', 'is', 'ib']
    
    print(f"  Synthesized {len(core_features)} core features ({len(base_features)} base + {len(physics_features)} physics-derived)")
    log_step("Step 2: Advanced Physics-Informed Feature Engineering (35 Features)", t0)

    # -------------------------------------------------------------------------
    # STEP 3: Target Formulation (Conductance & Log-Magnitude)
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 3: Target Formulation & Continuous Scaling")
    
    EPSILON = 1e-20
    VDS_FLOOR = 1e-4
    vds_safe = np.maximum(df['abs_vds'], VDS_FLOOR)
    
    # Id & Is: Conductance targets G = |I| / |Vds|
    df['id_target'] = np.log10(np.abs(df['id']) / vds_safe + EPSILON)
    df['is_target'] = np.log10(np.abs(df['is']) / vds_safe + EPSILON)
    
    # Ig & Ib: Direct log-magnitude targets
    df['ig_target'] = np.log10(np.abs(df['ig']) + EPSILON)
    df['ib_target'] = np.log10(np.abs(df['ib']) + EPSILON)
    
    # Terminal signs
    df['id_sign'] = np.sign(df['vds']).replace(0, 1)
    df['is_sign'] = -np.sign(df['vds']).replace(0, -1)
    df['ig_sign'] = np.sign(df['ig']).replace(0, 1)
    df['ib_sign'] = np.sign(df['ib']).replace(0, -1)
    
    log_step("Step 3: Target Formulation & Continuous Scaling", t0)

    # -------------------------------------------------------------------------
    # STEP 4: Dataset Splitting (70% Train, 15% Validation, 15% Test)
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 4: Train / Validation / Test Splitting")
    
    X_core = df[core_features]
    y_targets = df[[f"{c}_target" for c in target_cols]]
    y_signs = df[[f"{c}_sign" for c in target_cols]]
    y_raw = df[target_cols]
    
    (X_train_c, X_temp_c,
     y_target_train, y_target_temp,
     y_raw_train, y_raw_temp,
     y_sign_train, y_sign_temp) = train_test_split(
        X_core, y_targets, y_raw, y_signs,
        test_size=0.30, random_state=42, shuffle=True
    )
    
    (X_val_c, X_test_c,
     y_target_val, y_target_test,
     y_raw_val, y_raw_test,
     y_sign_val, y_sign_test) = train_test_split(
        X_temp_c, y_target_temp, y_raw_temp, y_sign_temp,
        test_size=0.50, random_state=42, shuffle=True
    )
    
    print(f"  Training samples   : {len(X_train_c)} (70.0%)")
    print(f"  Validation samples : {len(X_val_c)} (15.0%)")
    print(f"  Test samples       : {len(X_test_c)} (15.0%)")
    log_step("Step 4: Train / Validation / Test Splitting", t0)

    # -------------------------------------------------------------------------
    # STEP 5: Stage 1 Training — Conduction & Gate Terminals (Id, Is, Ig)
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 5: Stage 1 Model Training (Id, Is, Ig) — LightGBM & XGBoost")
    
    lgb_configs = {
        'id': {
            'objective': 'regression_l1',
            'metric': 'mae',
            'num_leaves': 255,
            'max_depth': 14,
            'learning_rate': 0.012,
            'n_estimators': 3500,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_samples': 5,
            'reg_alpha': 1e-3,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        },
        'is': {
            'objective': 'regression_l1',
            'metric': 'mae',
            'num_leaves': 255,
            'max_depth': 14,
            'learning_rate': 0.012,
            'n_estimators': 3500,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_samples': 5,
            'reg_alpha': 1e-3,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        },
        'ig': {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 180,
            'max_depth': 12,
            'learning_rate': 0.015,
            'n_estimators': 3500,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_samples': 5,
            'reg_alpha': 1e-3,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    }
    
    xgb_configs = {
        'id': {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_leaves': 255,
            'max_depth': 0,
            'learning_rate': 0.012,
            'n_estimators': 3500,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'reg_alpha': 1e-3,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        },
        'is': {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_leaves': 255,
            'max_depth': 0,
            'learning_rate': 0.012,
            'n_estimators': 3500,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'reg_alpha': 1e-3,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        },
        'ig': {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_leaves': 180,
            'max_depth': 0,
            'learning_rate': 0.015,
            'n_estimators': 3500,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'reg_alpha': 1e-3,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    models_lgb = {}
    models_xgb = {}
    
    for target in ['id', 'is', 'ig']:
        # Train LightGBM
        t_m = time.time()
        print(f"  Training LightGBM {target.upper()}...")
        m_lgb = lgb.LGBMRegressor(**lgb_configs[target])
        m_lgb.fit(
            X_train_c, y_target_train[f"{target}_target"],
            eval_set=[(X_val_c, y_target_val[f"{target}_target"])],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        models_lgb[target] = m_lgb
        print(f"    ✓ LightGBM {target.upper()} ready (best iter: {m_lgb.best_iteration_}, took {time.time()-t_m:.1f}s)")
        
        # Train XGBoost
        t_m = time.time()
        print(f"  Training XGBoost {target.upper()}...")
        m_xgb = xgb.XGBRegressor(**xgb_configs[target], early_stopping_rounds=100)
        m_xgb.fit(
            X_train_c, y_target_train[f"{target}_target"],
            eval_set=[(X_val_c, y_target_val[f"{target}_target"])],
            verbose=False
        )
        models_xgb[target] = m_xgb
        best_it = m_xgb.best_iteration if hasattr(m_xgb, 'best_iteration') and m_xgb.best_iteration is not None else 3500
        print(f"    ✓ XGBoost {target.upper()} ready (best iter: {best_it}, took {time.time()-t_m:.1f}s)")
        
    log_step("Step 5: Stage 1 Model Training (Id, Is, Ig) — LightGBM & XGBoost", t0)

    # -------------------------------------------------------------------------
    # STEP 6: Stage 2 Hierarchical Augmentation for Substrate Current (Ib)
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 6: Hierarchical Feature Synthesis for Substrate Current (Ib)")
    
    # Helper to compute channel current prediction
    def reconstruct_id(m_lgb, m_xgb, X_c):
        vds = X_c['vds'].values
        abs_vds = X_c['abs_vds'].values
        pred_g_lgb = 10 ** m_lgb.predict(X_c)
        pred_g_xgb = 10 ** m_xgb.predict(X_c)
        pred_g = 0.5 * (pred_g_lgb + pred_g_xgb)
        pred_id = np.sign(vds) * abs_vds * pred_g
        return pred_id

    # Compute predicted Id for Train, Val, Test
    pred_id_train = reconstruct_id(models_lgb['id'], models_xgb['id'], X_train_c)
    pred_id_val = reconstruct_id(models_lgb['id'], models_xgb['id'], X_val_c)
    pred_id_test = reconstruct_id(models_lgb['id'], models_xgb['id'], X_test_c)
    
    # Synthesize Ib-specialized features:
    # 1. Log-magnitude of channel conduction
    # 2. Impact ionization driving term: (abs_vds - vdsat) * |Id|
    def make_ib_features(X_c, pred_id_arr):
        X_ib = X_c.copy()
        log_id = np.log10(np.abs(pred_id_arr) + 1e-15)
        X_ib['log_id_pred'] = log_id
        vds_pinch = np.maximum(X_c['abs_vds'] - X_c['vdsat'], 0.0)
        X_ib['impact_ioniz_drive'] = vds_pinch * (10 ** (log_id - 6.0))
        return X_ib

    X_train_ib = make_ib_features(X_train_c, pred_id_train)
    X_val_ib = make_ib_features(X_val_c, pred_id_val)
    X_test_ib = make_ib_features(X_test_c, pred_id_test)
    
    print(f"  Added 2 hierarchical physical features (log_id_pred, impact_ioniz_drive) for Ib model.")
    print(f"  Ib feature dimension: {X_train_ib.shape[1]} features.")
    
    # Train Ib models with hierarchical features
    print("  Training Hierarchical LightGBM IB...")
    t_m = time.time()
    lgb_ib_cfg = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'num_leaves': 255,
        'max_depth': 14,
        'learning_rate': 0.012,
        'n_estimators': 3500,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_samples': 5,
        'reg_alpha': 1e-2,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    m_lgb_ib = lgb.LGBMRegressor(**lgb_ib_cfg)
    m_lgb_ib.fit(
        X_train_ib, y_target_train['ib_target'],
        eval_set=[(X_val_ib, y_target_val['ib_target'])],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    models_lgb['ib'] = m_lgb_ib
    print(f"    ✓ Hierarchical LightGBM IB ready (best iter: {m_lgb_ib.best_iteration_}, took {time.time()-t_m:.1f}s)")
    
    print("  Training Hierarchical XGBoost IB...")
    t_m = time.time()
    xgb_ib_cfg = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'max_leaves': 255,
        'max_depth': 0,
        'learning_rate': 0.012,
        'n_estimators': 3500,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 2,
        'reg_alpha': 1e-2,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1
    }
    m_xgb_ib = xgb.XGBRegressor(**xgb_ib_cfg, early_stopping_rounds=100)
    m_xgb_ib.fit(
        X_train_ib, y_target_train['ib_target'],
        eval_set=[(X_val_ib, y_target_val['ib_target'])],
        verbose=False
    )
    models_xgb['ib'] = m_xgb_ib
    best_it = m_xgb_ib.best_iteration if hasattr(m_xgb_ib, 'best_iteration') and m_xgb_ib.best_iteration is not None else 3500
    print(f"    ✓ Hierarchical XGBoost IB ready (best iter: {best_it}, took {time.time()-t_m:.1f}s)")
    
    log_step("Step 6: Hierarchical Feature Synthesis for Substrate Current (Ib)", t0)

    # -------------------------------------------------------------------------
    # STEP 7: Prediction Pipeline & Validation Optimal Blending
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 7: Validation Blending Optimization & KCL Projection")
    
    def predict_split(models_dict, X_c, X_ib, y_sign_df):
        vds = X_c['vds'].values
        abs_vds = X_c['abs_vds'].values
        
        # Id, Is from conductance
        pred_g_id = 10 ** models_dict['id'].predict(X_c)
        pred_id = np.sign(vds) * abs_vds * pred_g_id
        
        pred_g_is = 10 ** models_dict['is'].predict(X_c)
        pred_is = -np.sign(vds) * abs_vds * pred_g_is
        
        # Ig from log-magnitude
        pred_ig = y_sign_df['ig_sign'].values * (10 ** models_dict['ig'].predict(X_c))
        
        # Ib from hierarchical log-magnitude
        pred_ib = y_sign_df['ib_sign'].values * (10 ** models_dict['ib'].predict(X_ib))
        
        return {'id': pred_id, 'is': pred_is, 'ig': pred_ig, 'ib': pred_ib}

    preds_val_lgb = predict_split(models_lgb, X_val_c, X_val_ib, y_sign_val)
    preds_val_xgb = predict_split(models_xgb, X_val_c, X_val_ib, y_sign_val)
    
    # Find optimal blend weight w_k on Validation set: pred = w * xgb + (1-w) * lgb
    optimal_weights = {}
    print("  Optimizing ensemble weights on validation set:")
    for target in target_cols:
        y_true = y_raw_val[target].values
        best_w = 0.5
        best_mae = float('inf')
        for w in np.linspace(0.0, 1.0, 21):
            blended = w * preds_val_xgb[target] + (1.0 - w) * preds_val_lgb[target]
            mae = mean_absolute_error(
                np.log10(np.abs(y_true) + 1e-20),
                np.log10(np.abs(blended) + 1e-20)
            )
            if mae < best_mae:
                best_mae = mae
                best_w = w
        optimal_weights[target] = round(best_w, 2)
        print(f"    Terminal {target.upper()}: XGB weight = {best_w:.2f}, LGB weight = {1.0-best_w:.2f}")

    # Predict on Test set
    preds_test_lgb = predict_split(models_lgb, X_test_c, X_test_ib, y_sign_test)
    preds_test_xgb = predict_split(models_xgb, X_test_c, X_test_ib, y_sign_test)
    
    # Blended predictions on Test set
    preds_test_blend = {}
    for target in target_cols:
        w = optimal_weights[target]
        preds_test_blend[target] = w * preds_test_xgb[target] + (1.0 - w) * preds_test_lgb[target]
        
    # Channel-only KCL adjustment:
    channel_err = (
        preds_test_blend['id'] + preds_test_blend['is'] +
        preds_test_blend['ig'] + preds_test_blend['ib']
    )
    preds_test_kcl = {
        'id': preds_test_blend['id'] - 0.5 * channel_err,
        'is': preds_test_blend['is'] - 0.5 * channel_err,
        'ig': preds_test_blend['ig'],
        'ib': preds_test_blend['ib']
    }
    
    log_step("Step 7: Validation Blending Optimization & KCL Projection", t0)

    # -------------------------------------------------------------------------
    # STEP 8: Comprehensive Evaluation & Comparison Metrics
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 8: Metric Computation & Benchmarking")
    
    def evaluate_model(preds_dict, y_raw_eval, label):
        records = []
        for target in target_cols:
            y_pred = preds_dict[target]
            y_true = y_raw_eval[target].values
            
            r2_lin = r2_score(y_true, y_pred)
            r2_log = r2_score(np.log10(np.abs(y_true) + 1e-20), np.log10(np.abs(y_pred) + 1e-20))
            mae_val = mean_absolute_error(y_true, y_pred)
            rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
            
            floor = 1e-10 if target in ['id', 'is'] else 1e-15
            denom = np.maximum(np.abs(y_true), floor)
            rel_errs = (np.abs(y_pred - y_true) / denom) * 100.0
            
            mean_rel_err = np.mean(rel_errs)
            med_rel_err = np.median(rel_errs)
            p95_rel_err = np.percentile(rel_errs, 95)
            
            mean_acc = max(0.0, 100.0 - mean_rel_err)
            med_acc = max(0.0, 100.0 - med_rel_err)
            w5 = (rel_errs <= 5.0).mean() * 100.0
            w10 = (rel_errs <= 10.0).mean() * 100.0
            
            records.append({
                "Terminal": target.upper(),
                "Mean Accuracy (%)": f"{mean_acc:.2f}%",
                "Median Accuracy (%)": f"{med_acc:.2f}%",
                "Median Error (%)": f"{med_rel_err:.2f}%",
                "95th-Pct Error (%)": f"{p95_rel_err:.2f}%",
                "Within 5% Tol (%)": f"{w5:.1f}%",
                "Within 10% Tol (%)": f"{w10:.1f}%",
                "R2 (Linear)": round(r2_lin, 4),
                "R2 (Log)": round(r2_log, 4),
                "MAE (A)": f"{mae_val:.3e}",
                "RMSE (A)": f"{rmse_val:.3e}"
            })
            
        df_res = pd.DataFrame(records)
        avg_mean = np.mean([float(r["Mean Accuracy (%)"].replace("%", "")) for r in records])
        avg_med = np.mean([float(r["Median Accuracy (%)"].replace("%", "")) for r in records])
        
        print("\n" + "=" * 115)
        print(f"                       NMOS_ON — {label.upper()} (TEST SET — 1,500 SAMPLES)")
        print("=" * 115)
        print(df_res.to_string(index=False))
        print(f"\n[*] Overall Mean Accuracy   : {avg_mean:.2f}%")
        print(f"[*] Overall Median Accuracy : {avg_med:.2f}%")
        print("=" * 115)
        return df_res, avg_mean, avg_med

    # Evaluate all 3 modes on Test Set
    df_lgb, mean_lgb, med_lgb = evaluate_model(preds_test_lgb, y_raw_test, "Hierarchical LightGBM")
    df_xgb, mean_xgb, med_xgb = evaluate_model(preds_test_xgb, y_raw_test, "Hierarchical XGBoost")
    df_ensemble, mean_ens, med_ens = evaluate_model(preds_test_kcl, y_raw_test, "Hybrid Ensemble (LGB + XGB + KCL)")
    
    log_step("Step 8: Metric Computation & Benchmarking", t0)

    # -------------------------------------------------------------------------
    # STEP 9: Save Models, Summary & Generate High-Resolution Parity Plots
    # -------------------------------------------------------------------------
    t0 = time.time()
    log_step("Step 9: Saving Production Models & Generating Parity Plots")
    
    this_script_dir = os.path.dirname(os.path.abspath(__file__))
    nmos_on_dir = os.path.abspath(os.path.join(this_script_dir, ".."))
    saved_models_dir = os.path.join(nmos_on_dir, "saved_models")
    plots_dir = os.path.join(nmos_on_dir, "plots")
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    for target in target_cols:
        joblib.dump(models_lgb[target], os.path.join(saved_models_dir, f"adv_lgb_nmos_on_{target}.pkl"))
        joblib.dump(models_xgb[target], os.path.join(saved_models_dir, f"adv_xgb_nmos_on_{target}.pkl"))
        
    summary_data = {
        "pipeline": "Advanced Fine-Tuning NMOS ON (Hierarchical + Blended Ensemble)",
        "features_count": len(core_features) + 2,
        "ensemble_weights": optimal_weights,
        "lightgbm_test": {"mean_accuracy": mean_lgb, "median_accuracy": med_lgb, "metrics": df_lgb.to_dict(orient="records")},
        "xgboost_test": {"mean_accuracy": mean_xgb, "median_accuracy": med_xgb, "metrics": df_xgb.to_dict(orient="records")},
        "ensemble_test": {"mean_accuracy": mean_ens, "median_accuracy": med_ens, "metrics": df_ensemble.to_dict(orient="records")},
    }
    
    summary_path = os.path.join(this_script_dir, "advanced_fine_tune_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"  ✓ Saved summary JSON: {summary_path}")
    
    # 4-Terminal Parity Plots for Hybrid Ensemble
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    for idx, target in enumerate(target_cols):
        ax = axes[idx]
        y_true_abs = np.abs(y_raw_test[target].values)
        y_pred_abs = np.abs(preds_test_kcl[target])
        ax.scatter(y_true_abs, y_pred_abs, alpha=0.35, s=12, color=colors[idx], edgecolors='none')
        min_v = min(y_true_abs.min(), y_pred_abs.min())
        max_v = max(y_true_abs.max(), y_pred_abs.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.8, label='Ideal y=x (SPICE Reference)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f"SPICE Reference |{target.upper()}| (A)", fontsize=11)
        ax.set_ylabel(f"Ensemble Predicted |{target.upper()}| (A)", fontsize=11)
        ax.set_title(f"Terminal {target.upper()} (Ensemble Accuracy: {df_ensemble.iloc[idx]['Median Accuracy (%)']})", fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, which="both", ls=":", alpha=0.5)
        
    plt.suptitle("NMOS_ON Advanced Hybrid Ensemble Parity Plots (Test Set — 1,500 Samples)", fontsize=14, y=0.995)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "advanced_parity_plot.png")
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved parity plot: {plot_path}")
    
    log_step("Step 9: Saving Production Models & Generating Parity Plots", t0)
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 85)
    print(f"  ADVANCED FINE-TUNING COMPLETE IN {total_elapsed:.2f} SECONDS ({total_elapsed/60:.2f} MINUTES)")
    print("=" * 85)


if __name__ == "__main__":
    main()
