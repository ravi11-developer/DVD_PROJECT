# ML-Driven PVT Variation Aware Leakage Estimation
## Fine-Tuned Transistor Models (`ml_model/`)

This directory is organized into dedicated subfolders for each of the four single-transistor operating states:

1. **[`NMOS_ON/`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/NMOS_ON/)**: NMOS Transistor in Conduction / ON state ($V_g = V_{dd}, V_b = 0$).
2. **[`NMOS_OFF/`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/NMOS_OFF/)**: NMOS Transistor in Subthreshold / OFF state ($V_g = 0, V_b = 0$).
3. **[`PMOS_ON/`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/PMOS_ON/)**: PMOS Transistor in Conduction / ON state ($V_g = 0, V_b = V_{dd}$).
4. **[`PMOS_OFF/`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/PMOS_OFF/)**: PMOS Transistor in Subthreshold / OFF state ($V_g = V_{dd}, V_b = V_{dd}$).

> 📖 **Accuracy Diagnostics & Improvement Details:**  
> See **[`ACCURACY_IMPROVEMENT_GUIDE.md`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/ACCURACY_IMPROVEMENT_GUIDE.md)** for a short breakdown of previous vs. new accuracy, root cause diagnosis, and output verification steps.

---

## 1. Directory Structure

```text
ml_model/
├── NMOS_ON/
│   ├── 01_nmos_on_lightgbm.ipynb       # Fine-tuned notebook for NMOS ON case
│   ├── plots/                          # Parity plots and feature importance charts
│   ├── saved_models/                   # Trained models (.pkl) & JSON metadata
│   └── tuning_scripts/                 # Standalone LightGBM/XGBoost fine-tuning scripts
│
├── NMOS_OFF/
│   ├── 02_nmos_off_lightgbm.ipynb      # Fine-tuned notebook for NMOS OFF case
│   └── nmos_off_lightgbm_xgboost/      # Comparative LightGBM vs. XGBoost pipeline
│       ├── code/                       # Step0-Step4 execution scripts
│       └── results/                    # Validation, test, parity, and sweep plots
│
├── PMOS_ON/
│   ├── 03_pmos_on_lightgbm.ipynb       # Fine-tuned notebook for PMOS ON case
│   ├── plots/                          # Evaluation plots
│   ├── saved_models/                   # Trained models (.pkl) & metadata
│   └── lightgbm_conductance/           # LightGBM hybrid (final) + MLP/XGBoost/RF/ET comparison, see its README
│
├── PMOS_OFF/
│   ├── 04_pmos_off_lightgbm.ipynb      # Fine-tuned notebook for PMOS OFF case
│   ├── plots/                          # Evaluation plots
│   └── saved_models/                   # Trained models (.pkl) & metadata
│
├── ACCURACY_IMPROVEMENT_GUIDE.md       # Diagnostic summary & improvement roadmap
└── README.md                           # Directory documentation
```

---

## 2. Key Enhancements Implemented in the Models

1. **Physics-Informed Feature Engineering (23+ Features)**:
   - Potential differences: $V_{ds} = V_d - V_s$, $|V_{ds}|$, $V_{gs}$, $V_{gd}$, $V_{bs}$, $V_{bd}$.
   - Overdrive & scaling: $V_{ov} = V_{gs} - V_{\text{th0}}$, thermal voltage $V_t$, aspect ratio $W/L$, and oxide field $E_{ox}$.
   - Eliminates tree staircase approximation errors along the continuous $V_d \approx V_s$ boundary.
2. **Deep LightGBM Tree Architecture**:
   - `num_leaves: 127` (captures high-order PVT interactions)
   - `max_depth: 12`
   - `learning_rate: 0.02`
   - `n_estimators: 2500` (with early stopping at 80 rounds)
3. **SPICE-Standard Error Formulation**:
   - Uses operational signal noise floors ($10^{-10}\text{ A}$ for conduction, $10^{-15}\text{ A}$ for leakage) to prevent artificial division-by-zero explosions when $V_d \approx V_s$.
4. **Robust Dataset Binding**:
   - Resolves automatically across `dataset/<CASE>/<case>.csv` and `Data_Generation_M1/data/<case>.csv` from any execution path.

---

## 3. How to Run

### Prerequisites
Activate your project conda environment:
```bash
conda activate dvd
```

### Option A: Interactively in VS Code (Recommended)
1. Open any notebook, e.g. [`ml_model/NMOS_ON/01_nmos_on_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/NMOS_ON/01_nmos_on_lightgbm.ipynb) in VS Code.
2. Select your Python kernel in the top-right corner (`dvd` conda environment).
3. Run the notebook cell by cell (or click **Run All**).

### Option B: Terminal Command (Headless)
To run in-place via terminal without a browser:
```bash
# NMOS ON
cd /home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/NMOS_ON
jupyter nbconvert --to notebook --execute --inplace 01_nmos_on_lightgbm.ipynb

# NMOS OFF
cd /home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/NMOS_OFF
jupyter nbconvert --to notebook --execute --inplace 02_nmos_off_lightgbm.ipynb

# PMOS ON
cd /home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/PMOS_ON
jupyter nbconvert --to notebook --execute --inplace 03_pmos_on_lightgbm.ipynb

# PMOS OFF
cd /home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/PMOS_OFF
jupyter nbconvert --to notebook --execute --inplace 04_pmos_off_lightgbm.ipynb
```

---

## 4. Expected Performance

- **$R^2$ Score**: $\ge 0.995 – 0.998$
- **Median Relative Error**: $\le 3\% – 5\%$
- **Median Accuracy**: $\ge 95\% – 97\%$
- **Overall Mean Accuracy**: $\ge 90\% – 96\%$
