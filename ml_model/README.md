# ML-Driven PVT Variation Aware Leakage Estimation
## LightGBM Transistor Models (`ml_model/`)

This directory contains the 4 Jupyter notebooks developed to train, evaluate, and save **LightGBM (Light Gradient Boosting Machine)** models for each of the four single-transistor operating states:

1. **[`01_nmos_on_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/01_nmos_on_lightgbm.ipynb)**: NMOS Transistor in Conduction / ON state ($V_g = V_{dd}, V_b = 0$).
2. **[`02_nmos_off_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/02_nmos_off_lightgbm.ipynb)**: NMOS Transistor in Subthreshold / OFF state ($V_g = 0, V_b = 0$).
3. **[`03_pmos_on_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/03_pmos_on_lightgbm.ipynb)**: PMOS Transistor in Conduction / ON state ($V_g = 0, V_b = V_{dd}$).
4. **[`04_pmos_off_lightgbm.ipynb`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/ml_model/04_pmos_off_lightgbm.ipynb)**: PMOS Transistor in Subthreshold / OFF state ($V_g = V_{dd}, V_b = V_{dd}$).

---

## 1. Directory Structure

```text
ml_model/
├── 01_nmos_on_lightgbm.ipynb      # Notebook for NMOS ON case
├── 02_nmos_off_lightgbm.ipynb     # Notebook for NMOS OFF case
├── 03_pmos_on_lightgbm.ipynb      # Notebook for PMOS ON case
├── 04_pmos_off_lightgbm.ipynb     # Notebook for PMOS OFF case
├── README.md                      # Instructions & documentation
│
├── saved_models/                  # Trained LightGBM models (.pkl) & JSON metadata
│   ├── nmos_on/                   # Models for Id, Ig, Is, Ib & accuracy summary
│   ├── nmos_off/                  # Models for Id, Ig, Is, Ib & accuracy summary
│   ├── pmos_on/                   # Models for Id, Ig, Is, Ib & accuracy summary
│   └── pmos_off/                  # Models for Id, Ig, Is, Ib & accuracy summary
│
└── plots/                         # Parity plots and feature importance charts
    ├── nmos_on/
    ├── nmos_off/
    ├── pmos_on/
    └── pmos_off/
```

---

## 2. Block-by-Block Notebook Workflow

Every notebook follows an identical, rigorous machine learning and VLSI engineering workflow:

1. **Environment Setup & Imports**:
   - Imports `lightgbm`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`.
   - Sets global seed (`random_state=42`) for 100% reproducible results.

2. **Configuration & Data Paths**:
   - Automatically searches and binds to `Data_Generation_M1/data/<case>.csv`.
   - Defines the 12 input features: `[vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj]`.
   - Defines the 4 target currents: `[id, ig, is, ib]`.

3. **Data Inspection & Kirchhoff's Current Law (KCL)**:
   - Validates data completeness (no nulls/NaNs).
   - Computes KCL residual: $|I_d + I_g + I_s + I_b| \approx 0$ on the training data.

4. **Log-Transformation (First ML Step)**:
   - Currents span up to 14 orders of magnitude ($10^{-15}\text{ A}$ to $10^{-4}\text{ A}$).
   - Decomposes each current into **sign** ($\pm 1$) and **log-magnitude**:
     $$y_{\log} = \log_{10}(|I| + \epsilon), \quad \epsilon = 10^{-20}\text{ A}$$
   - Plots side-by-side distributions of raw currents vs. log-transformed currents.

5. **Train / Validation / Test Split**:
   - Strict adherence to course specifications:
     - **Training Set**: 70% (7,000 samples)
     - **Validation Set**: 15% (1,500 samples)
     - **Test Set**: 15% (1,500 samples)
   - Uses `random_state=42`.

6. **LightGBM Regressor Training**:
   - Trains 4 specialized gradient boosted decision tree regressors per case (one each for $I_d, I_g, I_s, I_b$).
   - Configures leaf-wise histogram tree growth with early stopping on the validation set.

7. **Accuracy Evaluation Metrics**:
   - Reconstructs physical currents: $\hat{I} = \text{sign}(I) \times 10^{\hat{y}_{\log}}$.
   - Computes:
     - $R^2$ Score (Log Domain)
     - $R^2$ Score (Linear Domain)
     - **Median Relative Percentage Error**: $\text{Median}\left(\left|\frac{I_{\text{pred}} - I_{\text{true}}}{I_{\text{true}}}\right|\right) \times 100\%$
     - **95th-Percentile Relative Error**: 95th percentile of percentage error (worst-case metric)
     - Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in Amperes.
   - Displays formatted comparison tables for both Validation and Test sets.

8. **Deliverable Visualizations**:
   - **Log-Log Parity Plots**: Scatter plot of $|I_{\text{SPICE}}|$ vs. $|I_{\text{Pred}}|$ with $y = x$ reference line.
   - **Feature Importance**: Bar charts identifying which physical parameters (Temperature, $V_{\text{th0}}$, $V_{ds}$, etc.) dictate each current.

9. **Model Persistence**:
   - Saves `lgbm_<case>_<target>.pkl` files via `joblib`.
   - Writes `<case>_metadata.json` containing complete metrics and configuration.

10. **Inference Pipeline Helper**:
    - Ready-to-use function demonstrating how to predict all 4 terminal currents for any new PVT corner vector.

---

## 3. How to Run the Notebooks

### Prerequisites
Make sure your Python environment has the necessary packages installed:

```bash
# If using conda:
conda activate dvd

# Install dependencies if not already installed:
pip install lightgbm scikit-learn pandas numpy matplotlib seaborn joblib jupyter
```

---

### Method A: Running Interactively in VS Code (Recommended)
1. Open the project folder in VS Code.
2. Navigate to the `ml_model/` folder.
3. Click on any notebook (e.g., `01_nmos_on_lightgbm.ipynb`).
4. Select your Python kernel in the top right (`dvd_leakage` conda environment or Python 3.10).
5. Click **Run All** (or run cells block by block with `Shift + Enter`).

---




## 4. Output Artifacts Produced After Execution

Once executed, each notebook will populate the following directories:

### Models & Metadata:
- `saved_models/nmos_on/lgbm_nmos_on_id.pkl`
- `saved_models/nmos_on/lgbm_nmos_on_ig.pkl`
- `saved_models/nmos_on/lgbm_nmos_on_is.pkl`
- `saved_models/nmos_on/lgbm_nmos_on_ib.pkl`
- `saved_models/nmos_on/nmos_on_metadata.json`
*(Similar structure for `nmos_off`, `pmos_on`, and `pmos_off`)*

### Figures for Reports:
- `plots/nmos_on/nmos_on_parity_plot.png`
- `plots/nmos_on/nmos_on_feature_importance.png`
*(Similar structure for `nmos_off`, `pmos_on`, and `pmos_off`)*
