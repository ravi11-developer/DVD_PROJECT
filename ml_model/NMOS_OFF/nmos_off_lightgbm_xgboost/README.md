# NMOS OFF – LightGBM and XGBoost leakage models (physics-guided, tuned)

Project: ML-Driven PVT Variation Aware Leakage Estimation · 22nm_MGK · VDD = 0.9 V · case NMOS OFF (V<sub>G</sub> = 0, V<sub>B</sub> = 0)

## How to run (in this order)

```
pip install lightgbm xgboost scikit-learn scipy pandas numpy matplotlib joblib
cd code
python step0_data_check.py          # ~10 s   KCL check + 3 trend plots
python step1_lightgbm.py            # ~3-6 min
python step2_xgboost.py             # ~6-12 min
python step3_compare_models.py      # ~10 s   tables, combined JSON, tuning history
python step4_parameter_sweep.py lightgbm   # ~40 min (optional, graphs already in results/)
python step4_parameter_sweep.py xgboost    # ~1.5 h  (optional)
```
Paths are set at the top of `code/nmos_off_common.py` (`DATA_CSV`, `OUT_DIR`). Seed = 42 everywhere, `n_jobs = 2` fixed, so the numbers repeat exactly.
`optional_optuna_tuning.py` re-does the hyperparameter search (1–3 h); you don't need it.

## What is in `results/`

| Folder / file | What it shows | Brief item |
|---|---|---|
| `step0_data_check/kcl_check.json`, `kcl_check.png` | Id+Ig+Is+Ib ≈ 0 for every row (max relative residual 8.7×10⁻⁹) | 3 |
| `step0_data_check/trend_*.png` | OFF current vs temperature, OFF current vs vth0, gate current vs toxe | 3 |
| `comparison/model_comparison_validation.png/.csv` | R², median error, 95th-pct error on the **validation** set | 4 |
| `comparison/chosen_model_results_test.png/.csv` | Same on the **test** set (used once) | 5 |
| `comparison/nmos_off_lightgbm_xgboost_report.json` | Both models, team JSON format, train/validation/test blocks | – |
| `comparison/lightgbm_vs_xgboost_test.png` | Bar chart of the test results | – |
| `comparison/tuning_history.png` | How validation error changed over the Optuna trials | – |
| `lightgbm/`, `xgboost/` | Per model: `*_report.json`, `*_metrics.csv`, `*_overfit_check.csv`, `parity_test.png`, `error_vs_current_test.png`, `feature_importance.png`, the 3 overfitting plots | 5 |
| `parameter_sweep/sweep_*.png/.csv` | Effect of changing one parameter at a time (train / validation / test) | – |

The saved model files (`*_model.joblib`) are created by step 1 and 2 on your PC; they are not in the zip because of their size.

## Method in one paragraph

A small physics model (gate tunnelling through each terminal, subthreshold channel current with body effect and DIBL, reverse-junction bulk current; 23 fitted numbers) is fitted on the training rows only and gives a first estimate of log10|I| for each current. **One** gradient-boosted model then predicts the correction for all four currents; an input `cur` tells it which current. Id and Is are predicted directly (no sum or difference). A classifier predicts the sign of Id and Is; Ig and Ib are always negative in NMOS OFF. Hyperparameters were tuned with Optuna on the validation set only (LightGBM 80 trials, XGBoost 50 trials).

## How to read the overfitting plots

* `overfit_learning_curve.png` – error vs number of trees. **Overfitting in the harmful sense = the validation/test curve turns upward.** If it keeps falling, the extra trees still help unseen data.
* `overfit_train_val_test_bars.png` and `overfit_error_cdf.png` – train vs validation vs test. Validation and test on top of each other = the reported test numbers are honest. Train far below them = the model has memorised the training rows.
* `*_overfit_check.csv` – the ratio validation / train of the median error, with a verdict (warning above 1.5).
* `parameter_sweep/sweep_*.png` – for each parameter, how train and validation error move. A setting where the train error keeps dropping while validation stays flat or rises is where memorising starts.
