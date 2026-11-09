# NMOS ON – LightGBM and XGBoost Conduction & Leakage Models (Physics-Guided)

Project: ML-Driven PVT Variation Aware Leakage Estimation · 22nm_MGK · VDD = 0.9 V · Case: NMOS ON (V<sub>G</sub> = VDD, V<sub>B</sub> = 0)

---

## 1. How to Run

Activate your project environment:
```bash
conda activate dvd
cd /home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/NMOS_ON/nmos_on_lightgbm_xgboost/code
```

### Option A: Run Everything with One Command (Recommended)
```bash
python run_all.py
```
This executes Step 0 through Step 3 sequentially in ~8–15 minutes and saves all metrics, tables, JSON reports, and plots.

### Option B: Run Step by Step
```bash
python step0_data_check.py          
python step1_lightgbm.py
python step2_xgboost.py             
python step3_compare_models.py      
python step4_parameter_sweep.py lightgbm   # ~15-30 min (optional sensitivity analysis)
python step4_parameter_sweep.py xgboost    # ~30-60 min (optional sensitivity analysis)
```

Paths are automatically discovered in `nmos_on_common.py` (checking `dataset/NMOS_ON/nmos_on.csv` and `Data_Generation_M1/data/nmos_on.csv`). The random seed is fixed (`seed = 42`) and `n_jobs = 2` is set for exact reproducibility.

---

## 2. Directory Structure & Outputs in `results/`

```text
results/
├── step0_data_check/
│   ├── kcl_check.json                  # Max residual, median residual, pass verdict
│   ├── kcl_check.png                   # Histogram of KCL relative residuals
│   ├── trend_ion_vs_temp.png           # ON conduction current vs. temperature
│   ├── trend_ion_vs_vds.png            # Conduction current vs. |Vds| (linear-to-saturation)
│   ├── trend_ig_vs_toxe.png            # Gate tunneling current vs. oxide thickness
│   └── trend_ib_vs_vds.png             # Substrate current vs. |Vds| (impact ionization)
│
├── lightgbm/
│   ├── lightgbm_report.json            # Full metrics (Train, Validation, Test)
│   ├── lightgbm_metrics.csv            # Tabulated R2, Median/Mean Accuracy, Tolerances
│   ├── lightgbm_overfit_check.csv      # Validation/Train error ratios + verdict
│   ├── parity_test.png                 # Log-log scatter plot (SPICE vs. Predicted)
│   ├── error_vs_current_test.png       # Relative error across current magnitude
│   ├── overfit_learning_curve.png      # Error vs. tree iterations
│   ├── overfit_train_val_test_bars.png # Train vs. Val vs. Test error comparisons
│   ├── overfit_error_cdf.png           # Error cumulative distribution function
│   ├── feature_importance.png          # Top 20 features ranked by gain
│   └── lightgbm_model.joblib           # Trained model payload & sign classifier
│
├── xgboost/
│   ├── (Identical diagnostic structure as lightgbm/ above)
│   └── xgboost_model.joblib
│
├── comparison/
│   ├── model_comparison_validation.csv # Deliverable: Model comparison table (Validation)
│   ├── model_comparison_validation.png # Formatted image of validation results
│   ├── chosen_model_results_test.csv   # Deliverable: Model evaluation table (Test)
│   ├── chosen_model_results_test.png   # Formatted image of test results
│   ├── lightgbm_vs_xgboost_test.png    # Side-by-side comparison bar chart
│   └── nmos_on_lightgbm_xgboost_report.json # Final unified JSON deliverable
│
└── parameter_sweep/
    ├── sweep_lightgbm.csv / .png       # Sensitivity across num_leaves, lr, min_child, linear_lambda
    └── sweep_xgboost.csv / .png        # Sensitivity across max_depth, lr, reg_lambda
```

---

## 3. Methodology Summary

1. **Analytical Semiconductor Physics Baseline**:
   - A compact physics model is fitted on the training split (7,000 samples) using `scipy.optimize.least_squares`.
   - Captures:
     - Velocity-saturated channel conduction: $I_{\text{chan}} \propto \frac{W}{L} V_{ov} \frac{|V_{ds}|}{1 + |V_{ds}| / E_{\text{sat}} L}$.
     - Multi-terminal gate tunneling: Overlaps $F_d, F_s$ and channel tunneling driven by $V_{dd} - V_d, V_{dd} - V_s, t_{\text{oxe}}$.
     - Substrate current: Reverse-biased junction leakage + hot-carrier impact ionization $I_{\text{ii}} \propto |I_{\text{chan}}| |V_{ds}| e^{-\beta / |V_{ds}|}$.
2. **Single Shared Regressor on Residuals**:
   - Each sample generates 4 long-format rows (one per terminal current: $0=I_d, 1=I_g, 2=I_s, 3=I_b$).
   - The ML model learns only the residual:
     $$\Delta = \log_{10}|I_{\text{SPICE}}| - \log_{10}|I_{\text{Physics}}|$$
   - LightGBM employs **Linear Trees** (`linear_tree=True`), fitting straight-line hyperplanes inside leaves to eliminate staircase errors along $V_d \approx V_s$.
3. **Sign Handling**:
   - $I_g > 0$ strictly (gate current enters gate).
   - $I_b < 0$ strictly (substrate current leaves body).
   - $I_d$ and $I_s$ signs are determined by a dedicated binary classifier.
4. **SPICE Operational Noise Floors**:
   - Conduction floor: $I_{\text{floor}} = 10^{-10}\text{ A}$ for $I_d, I_s$ (prevents division explosions when $V_d \approx V_s$).
   - Leakage floor: $I_{\text{floor}} = 10^{-15}\text{ A}$ for $I_g, I_b$.

---

## 4. How to Read the Diagnostic Plots

* `overfit_learning_curve.png`: Plots median % error vs. boosting iterations. Harmful overfitting occurs if the validation/test curve turns upward. Early stopping cuts off training at the minimum validation error.
* `overfit_train_val_test_bars.png` & `overfit_error_cdf.png`: Compares train, validation, and test error distributions side-by-side. If the validation and test distributions overlay closely, the model generalizes properly without data memorization.
* `*_overfit_check.csv`: Records the validation/train median error ratio (flagged as WARNING if $> 1.5$).
* `parity_test.png`: Parity plot of predicted vs. SPICE reference. Points should lie tightly along the dashed $y = x$ diagonal across all current decades.
