# ML-Driven PVT Variation Aware Leakage Estimation
## Fine-Tuned LightGBM Transistor Models (`ml_model/`)

This directory contains the 4 Jupyter notebooks developed to train, evaluate, and save **fine-tuned LightGBM** models for each of the four single-transistor operating states:

1. **[`01_nmos_on_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/01_nmos_on_lightgbm.ipynb)**: NMOS Transistor in Conduction / ON state ($V_g = V_{dd}, V_b = 0$).
2. **[`02_nmos_off_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/02_nmos_off_lightgbm.ipynb)**: NMOS Transistor in Subthreshold / OFF state ($V_g = 0, V_b = 0$).
3. **[`03_pmos_on_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/03_pmos_on_lightgbm.ipynb)**: PMOS Transistor in Conduction / ON state ($V_g = 0, V_b = V_{dd}$).
4. **[`04_pmos_off_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/04_pmos_off_lightgbm.ipynb)**: PMOS Transistor in Subthreshold / OFF state ($V_g = V_{dd}, V_b = V_{dd}$).

> 📖 **Accuracy Diagnostics & Improvement Details:**  
> See **[`ACCURACY_IMPROVEMENT_GUIDE.md`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/ACCURACY_IMPROVEMENT_GUIDE.md)** for a short breakdown of previous vs. new accuracy, root cause diagnosis, and output verification steps.

---

## 1. Directory Structure

```text
ml_model/
├── 01_nmos_on_lightgbm.ipynb           # Fine-tuned notebook for NMOS ON case
├── 02_nmos_off_lightgbm.ipynb          # Fine-tuned notebook for NMOS OFF case
├── 03_pmos_on_lightgbm.ipynb           # Fine-tuned notebook for PMOS ON case
├── 04_pmos_off_lightgbm.ipynb          # Fine-tuned notebook for PMOS OFF case
├── ACCURACY_IMPROVEMENT_GUIDE.md       # Diagnostic summary & improvement roadmap
├── README.md                           # Quickstart execution guide
│
├── saved_models/                       # Trained LightGBM models (.pkl) & JSON metadata
│   ├── nmos_on/                        # Models for Id, Ig, Is, Ib & metadata
│   ├── nmos_off/                       # Models for Id, Ig, Is, Ib & metadata
│   ├── pmos_on/                        # Models for Id, Ig, Is, Ib & metadata
│   └── pmos_off/                       # Models for Id, Ig, Is, Ib & metadata
│
└── plots/                              # Parity plots and feature importance charts
    ├── nmos_on/
    ├── nmos_off/
    ├── pmos_on/
    └── pmos_off/
```

---

## 2. Key Enhancements Implemented in the Notebooks

1. **Physics-Informed Feature Engineering (23 Total Features)**:
   - Added potential differences: $V_{ds} = V_d - V_s$, $|V_{ds}|$, $V_{gs}$, $V_{gd}$, $V_{bs}$, $V_{bd}$.
   - Added overdrive & scaling: $V_{ov} = V_{gs} - V_{\text{th0}}$, thermal voltage $V_t$, aspect ratio $W/L$, and oxide field $E_{ox}$.
   - Eliminates tree staircase approximation errors along the continuous $V_d \approx V_s$ boundary.
2. **Deep LightGBM Tree Architecture**:
   - `num_leaves: 127` (captures high-order PVT interactions)
   - `max_depth: 12`
   - `learning_rate: 0.02`
   - `n_estimators: 2500` (with early stopping at 80 rounds)
3. **SPICE-Standard Error Formulation**:
   - Uses operational signal noise floors ($10^{-10}\text{ A}$ for conduction, $10^{-15}\text{ A}$ for leakage) to prevent artificial division-by-zero explosions when $V_d \approx V_s$.
4. **Clean Data Binding**:
   - Points automatically to `dataset/<CASE>/<case>.csv`, dropping auxiliary simulation columns (`itotal`).

---

## 3. How to Run the Notebooks

### Prerequisites
Activate your project conda environment:
```bash
conda activate dvd
```

### Option A: Interactively in VS Code (Recommended)
1. Open [`ml_model/01_nmos_on_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/01_nmos_on_lightgbm.ipynb) in VS Code.
2. Select your Python kernel in the top-right corner (`dvd` conda environment).
3. Run the notebook cell by cell (or click **Run All**).

### Option B: Terminal Command (Headless)
To run in-place via terminal without a browser:
```bash
cd /home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model

# Run NMOS_ON notebook
jupyter nbconvert --to notebook --execute --inplace 01_nmos_on_lightgbm.ipynb
```

---

## 4. Expected Performance

- **$R^2$ Score**: $\ge 0.995 – 0.998$
- **Median Relative Error**: $\le 3\% – 5\%$
- **Median Accuracy**: $\ge 95\% – 97\%$
- **Overall Mean Accuracy**: $\ge 90\% – 96\%$
