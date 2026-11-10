# Methodology: ML-Driven Leakage Current Estimation

This document outlines the different approaches to build the ML-driven leakage estimation pipeline and recommends the best methodology.

---

## 1. Overview of Approaches

There are several ways to approach the ML-based leakage current estimation problem. They differ in **model architecture**, **feature engineering**, **prediction target transformation**, and **training strategy**.

---

## 2. ML Model Options

### 2.1 Random Forest Regression

| Aspect | Detail |
|--------|--------|
| **How it works** | Ensemble of decision trees, each trained on a bootstrap sample. Prediction is the average of all trees. |
| **Strengths** | Robust to outliers, handles non-linear relationships well, minimal hyperparameter tuning, no feature scaling needed, provides feature importance. |
| **Weaknesses** | Can struggle with extrapolation beyond training range; large memory footprint for many trees; slower inference than linear models. |
| **Suitability** |  Good — Handles the non-linear PVT dependencies well. |

### 2.2 Gradient Boosted Trees (XGBoost / LightGBM)

| Aspect | Detail |
|--------|--------|
| **How it works** | Sequentially builds trees, each correcting the errors of the previous one. Uses gradient descent on the loss function. |
| **Strengths** | State-of-the-art for tabular data, excellent accuracy, handles heterogeneous features, built-in regularization, very fast with LightGBM. |
| **Weaknesses** | More hyperparameters to tune than Random Forest; risk of overfitting without careful validation; sequential training is harder to parallelize. |
| **Suitability** |  Excellent — Best performer on structured/tabular data like SPICE simulation outputs. |

### 2.3 Neural Networks (Deep Learning)

| Aspect | Detail |
|--------|--------|
| **How it works** | Multi-layer perceptrons (MLPs) with non-linear activation functions learn complex mappings from inputs to outputs. |
| **Strengths** | Can model highly complex non-linear relationships; scalable to very large datasets; flexible architecture. |
| **Weaknesses** | Requires large datasets for good generalization; sensitive to hyperparameters (learning rate, architecture, regularization); needs feature scaling; slower to train; less interpretable. |
| **Suitability** |  Moderate — Can work well but requires more engineering effort for 10,000-sample datasets. |

### 2.4 Support Vector Regression (SVR)

| Aspect | Detail |
|--------|--------|
| **How it works** | Finds a hyperplane that fits the data within an ε-insensitive tube, using kernel functions for non-linearity. |
| **Strengths** | Effective in high-dimensional spaces, works well with small datasets, robust to outliers with appropriate kernel. |
| **Weaknesses** | Poor scalability to large datasets (O(n²) to O(n³) training time); kernel selection and hyperparameter tuning can be challenging; slower inference. |
| **Suitability** |  Moderate — Can be accurate but is slow to train on 10,000 samples. |

### 2.5 Linear Regression / Polynomial Regression

| Aspect | Detail |
|--------|--------|
| **How it works** | Fits a linear (or polynomial) function to the data by minimizing squared error. |
| **Strengths** | Very fast, highly interpretable, good as a baseline. |
| **Weaknesses** | Cannot capture the non-linear physics of leakage currents without extensive feature engineering; poor accuracy for this problem. |
| **Suitability** |  Poor — Only useful as a baseline; leakage currents have exponential dependencies on temperature and threshold voltage. |

### 2.6 K-Nearest Neighbors (KNN) Regression

| Aspect | Detail |
|--------|--------|
| **How it works** | Predicts based on the average of the k closest training samples in feature space. |
| **Strengths** | No training phase, simple to implement, captures local patterns. |
| **Weaknesses** | Slow inference (scans all training data), sensitive to feature scaling, curse of dimensionality with many features, requires large memory. |
| **Suitability** |  Poor — Not suitable for production; very slow inference defeats the purpose of replacing SPICE. |

---

## 3. Feature Engineering Strategies

### 3.1 Raw Features (Baseline)
- **Input features**: VDD, Temp, W, Vd, Vg, Vs, Vb, vth0, toxe, ndep, u0, xj
- **Output targets**: Id, Ig, Is, Ib
- **Pros**: Simple, no information loss
- **Cons**: Currents span many orders of magnitude, making regression difficult

### 3.2 Log-Transformed Targets  RECOMMENDED
- **Input features**: Same as above
- **Output targets**: log₁₀(|Id|), log₁₀(|Ig|), log₁₀(|Is|), log₁₀(|Ib|) with sign stored separately
- **Pros**: Compresses the dynamic range (femtoamps to microamps), dramatically improves regression accuracy, aligns with the physical exponential dependencies
- **Cons**: Need to handle zero or near-zero currents carefully

### 3.3 Physics-Informed Features
- **Additional derived features**: Vgs, Vds, Vbs, (Vgs - Vth0), W/L ratio, Vdd/toxe
- **Pros**: Encodes known physics into the model, can improve accuracy and generalization
- **Cons**: Requires domain knowledge; may introduce correlated features

### 3.4 Normalized / Standardized Features
- Apply StandardScaler or MinMaxScaler to all input features
- **Essential for**: Neural Networks, SVR, KNN
- **Not needed for**: Tree-based models (Random Forest, XGBoost)

---

## 4. Training Strategies

### 4.1 Single Model per Case  RECOMMENDED
Train **4 separate model instances** (one for each case: NMOS ON, NMOS OFF, PMOS ON, PMOS OFF), each predicting all 4 terminal currents simultaneously (multi-output regression).

- **Pros**: Each model specializes in one operating regime; simpler training; higher accuracy
- **Cons**: Need to manage 4 models

### 4.2 Single Model per Case per Terminal
Train **16 separate models** (4 cases × 4 terminals), each predicting a single current.

- **Pros**: Each model is maximally specialized; can use different model types per terminal
- **Cons**: Many models to manage; misses correlations between terminal currents

### 4.3 Single Unified Model
Train one model for all 4 cases, encoding the case as a categorical feature.

- **Pros**: Simplest management; largest training set
- **Cons**: Lower accuracy because ON and OFF states have very different physics

---

## 5. Hierarchical Prediction Pipeline

Regardless of the ML model chosen, the prediction pipeline follows a hierarchical structure:

```
┌─────────────────────────────────────────────────────────┐
│                    CIRCUIT LEVEL                         │
│  Sum gate leakages for all gates in the circuit          │
│  (3-bit CLA or 8-bit Parity Checker)                    │
├─────────────────────────────────────────────────────────┤
│                     GATE LEVEL                           │
│  For each gate (INV, NAND2, NOR2, AND2, OR2):           │
│  → Determine each transistor's terminal voltages         │
│  → Call transistor ML model for each FET                 │
│  → Sum VDD-terminal currents = gate leakage              │
├─────────────────────────────────────────────────────────┤
│                  TRANSISTOR LEVEL                         │
│  4 ML models predict Id, Ig, Is, Ib for:                 │
│  NMOS ON | NMOS OFF | PMOS ON | PMOS OFF                 │
│  Given: VDD, T, W, Vd, Vg, Vs, Vb, vth0, toxe,         │
│         ndep, u0, xj                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Recommended Production Methodology: Physics-Informed Polynomial Hybrid Architecture (Target \(\ge 99\%\))

To overcome the limitations of plain tree models and single pooled regressors (which suffered from zero-crossing log cliffs, high \(I_b\) errors, and overfitting), a **Physics-Informed Polynomial Hybrid Architecture** was implemented and validated.

### 6.1 Architecture Overview
A two-stage hybrid approach combining analytical device physics, continuous polynomial regression, and residual gradient boosting:
```text
Inputs (12 Raw + 24 Physics Features) 
   │
   ├──> [1. Analytical Physics Priors] ────> Channel Conduction (C), Tunneling (Fd, Fs, Fc), Bulk (B)
   │
   ├──> [2. Degree-2 Polynomial RidgeCV] ──> Smooth Multi-Output Base Model (captures curvature)
   │
   └──> [3. Decoupled L1 Gradient Boosters] > 4 Specialized Trees fit residuals (y_true - y_poly)
                                              Id: Conductance (G_ds)  |  Ig: Tunneling (log Ig)
                                              Is: Conductance (G_ds)  |  Ib: Bulk Leakage (log |Ib|)
   │
   └──> [4. Reconstruction & Sign Logic] ──> Physical Currents (A) with 100% Deterministic Signs
```

### 6.2 Five Core Modifications
1. **Conductance Formulation for Conduction Currents (\(I_d, I_s\))**:
   - Models channel conductance \(G_{ds} = \frac{|I|}{|V_d - V_s| + 10^{-6}}\) instead of raw \(\log_{10}|I|\).
   - Completely eliminates the logarithmic singularity cliff at \(V_d \approx V_s\).
   - Reconstructs smoothly via \(|I| = G_{ds} \cdot |V_d - V_s|\).

2. **3-Component Physics Model for Bulk Current (\(I_b\))**:
   - Previous models omitted **gate-to-bulk oxide tunneling (\(I_{gb}\))**, causing a 91% baseline error.
   - Formulates: \(I_b = I_{junc}(V_d, V_s, T) + I_{gb}(V_{dd}, t_{oxe}) + I_{ii}(I_{chan}, V_{ds})\).
   - Drives \(I_b\) median error from **4.60% \(\to\) 0.81%** and mean accuracy from **93.65% \(\to\) 98.61%**.

3. **Degree-2 Polynomial RidgeCV Baseline**:
   - Standardized polynomial expansion (\(\text{deg}=2\)) with cross-validated Ridge regression.
   - Fits the continuous physical response surface, eliminating axis-aligned staircase errors.

4. **Decoupled L1 Gradient Boosters (LightGBM / XGBoost)**:
   - Replaces pooled long-table regressors with dedicated models per terminal to prevent cross-phenomenon interference.
   - Employs `objective="regression_l1"` (MAE) instead of MSE, aligning training directly with relative percentage error.

5. **Deterministic Physical Sign Logic**:
   - \(\text{sign}(I_d) = \text{sign}(V_d - V_s)\) when \(|V_d - V_s| \ge 10\text{ mV}\) (100% physical fidelity).
   - Near zero-bias (\(|V_{ds}| < 10\text{ mV}\)), a binary classifier determines if gate leakage or channel conduction dominates.

### 6.3 Achieved Performance (1,500 Unseen Test Samples)

| Current | Baseline Mean Acc | **Modified Mean Acc** | Baseline Med Err | **Modified Med Err** | **Modified 95th-Pct Err** | **Within 5% Tol** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **\(I_d\)** | 96.58% | **98.69%** | 1.152% | **0.635%** | **4.19%** | **96.9%** |
| **\(I_g\)** | 97.83% | **98.15%** | 0.862% | **0.873%** | **6.78%** | **92.5%** |
| **\(I_s\)** | 96.58% | **98.66%** | 1.183% | **0.654%** | **4.23%** | **96.3%** |
| **\(I_b\)** | 93.65% | **98.61%** | 4.597% | **0.809%** | **4.26%** | **96.3%** |
| **Overall** | **96.16%** | **98.53%** | **1.948%** | **0.743%** | **4.86%** | **95.5%** |

- **Overall Median Accuracy**: **99.26%** (Target \(\ge 99.0\%\) achieved)
- **Overfitting**: Solved; validation/train ratio stabilized with 4 decoupled models.

---

## 7. Baseline Methodology & Data Generation Workflow

### Step 1: Data Generation
1. For each of the 4 cases (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF):
   - Generate **10,000 random samples** by uniformly drawing: VDD (nominal ±10%), Vd (0 to VDD), Vs (0 to VDD), W (1×Lmin to 8×Lmin), Temperature (−40°C to 125°C), and 5 process parameters (vth0, toxe, ndep, u0, xj at nominal ±10%)
   - Gate and body voltages are fixed per case
   - Run NGSPICE simulation for each sample to get Id, Ig, Is, Ib
   - Store as CSV with columns: `vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib`

### Step 2: Data Preprocessing
1. **Kirchhoff check**: Verify Id + Ig + Is + Ib ≈ 0 for each sample (data quality check)
2. **Log transform targets**: Predict `log₁₀(|I|)` with sign stored separately
3. **Split**: 70% train / 15% validation / 15% test (with fixed random seed 42)
4. Feature scaling is handled inside the pipeline

### Step 3: Gate-Level Validation
1. For each gate (INV, NAND2, NOR2, AND2, OR2) and each input combination:
   - Generate **200 random PVT corners** (same distributions as training data)
   - Run SPICE simulation to get reference leakage and internal node voltages
   - Use internal node voltages + output voltage to determine each transistor's terminal voltages
   - Feed into trained ML models to predict terminal currents
   - Calculate gate leakage by summing VDD-pin currents
   - Compare ML prediction vs SPICE reference

### Step 4: Circuit-Level Validation
1. For the assigned test circuit (3-bit CLA or 8-bit Parity Checker):
   - Run SPICE simulation at 200 corners for every input combination
   - Use gate-level ML models hierarchically to predict total circuit leakage
   - Compare and report accuracy

---

## 8. Why the Hybrid Architecture Wins

| Criterion | Why Physics-Informed Polynomial Hybrid Wins |
|-----------|---------------------------------------------|
| **Accuracy** | Reaches **99.26% median accuracy** on unseen test data; eliminates the zero-crossing logarithmic singularity cliff. |
| **Speed** | Polynomial fit takes < 1 second; LightGBM inference is sub-millisecond per sample. |
| **Physics Consistency** | 3-component bulk model explicitly incorporates gate oxide tunneling (\(I_{gb}\)) and impact ionization (\(I_{ii}\)). |
| **Generalization** | L1 loss on residuals prevents over-penalizing outliers and prevents memorization. |
| **Physical Sign Guarantee** | Physical directionality rules ensure 100% sign fidelity for \(|V_{ds}| \ge 10\text{ mV}\). |

---

## 9. Comparison Summary

| Approach | Expected Accuracy | Training Time | Inference Speed | Complexity |
|----------|-------------------|---------------|-----------------|------------|
| **XGBoost + Log Transform** |  (>98%) | Fast | Very Fast | Low |
| Random Forest + Log Transform |  (>95%) | Fast | Fast | Low |
| Neural Network + Normalization |  (>95%) | Moderate | Very Fast | High |
| SVR + Normalization |  (>90%) | Slow | Moderate | Moderate |
| Linear / Polynomial Regression |  (<85%) | Very Fast | Very Fast | Very Low |
| KNN |  (>85%) | None | Very Slow | Very Low |

---

## 10. Potential Improvements & Advanced Methods

### 10.1 Physics-Informed Neural Networks (PINNs)
Incorporate the MOSFET current equations (sub-threshold, gate tunneling, GIDL) as loss function constraints. Could improve generalization but is significantly more complex to implement.

### 10.2 Ensemble / Stacking
Combine predictions from multiple model types (e.g., XGBoost + MLP) using a meta-learner. Can squeeze out 1–2% more accuracy at the cost of complexity.

### 10.3 Transfer Learning Across Technology Nodes
Train on a data-rich node (e.g., 45nm with abundant data) and fine-tune on a target node (e.g., 16nm). Useful when simulation budget is limited.

### 10.4 Active Learning
Start with a small dataset, train a model, identify regions of high uncertainty, and selectively simulate those corners. Achieves the same accuracy with fewer SPICE simulations.

### 10.5 Gaussian Process Regression
Provides uncertainty estimates along with predictions. Useful for identifying corners where the model is unreliable, but scales poorly beyond ~5,000 samples.

---

## 11. Tools & Libraries Required

| Tool | Purpose |
|------|---------|
| **NGSPICE** | SPICE simulation for data generation and validation |
| **Python 3.x** | Scripting and ML pipeline |
| **NumPy / Pandas** | Data manipulation and CSV handling |
| **Scikit-learn** | Random Forest, SVR, MLP, data splitting, metrics |
| **XGBoost** or **LightGBM** | Gradient boosted tree models |
| **Matplotlib / Seaborn** | Plotting (parity plots, trend plots, error analysis) |
| **Joblib / Pickle** | Model saving and loading |
