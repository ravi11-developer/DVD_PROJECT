# PMOS ON: LightGBM (polynomial base + LightGBM residual) (`lightgbm_conductance/`)

Single-transistor ML model for **PMOS ON** (Vg = 0, Vb = VDD), 22 nm MGK, nominal VDD 0.9 V.
**Final model: LightGBM hybrid, degree 2** (`models/pmos_on_lightgbm_improved.joblib`). Chosen because the team uses tree-based models.
An MLP, XGBoost, Random Forest, Extra Trees and plain LightGBM were compared on the same 70/15/15 split (seed 42).

## Final model: test-set results (1,500 unseen samples)

| Current | R2 (linear) | Median err % | 95th-pct err % | Mean accuracy % |
|---|---|---|---|---|
| Id | 0.9969 | 3.12 | 13.50 | 95.1 |
| Ig | 0.9986 | 1.84 | 8.25 | 97.3 |
| Is | 0.9972 | 3.17 | 14.40 | 95.2 |
| Ib | 0.9863 | 3.33 | 13.54 | 95.3 |

The 99% mean-accuracy target is **not** reached (95 to 97%).

## Comparison (test set, median / 95th-pct relative error %)

| Model | Id | Ig | Is | Ib |
|---|---|---|---|---|
| LightGBM plain | 3.46 / 24.0 | 5.74 / 20.8 | 3.49 / 24.7 | 14.36 / 48.7 |
| LightGBM hybrid, degree 2 (final) | 3.12 / 13.5 | 1.84 / 8.3 | 3.17 / 14.4 | 3.33 / 13.5 |
| LightGBM hybrid, degree 3 | 3.30 / 12.8 | 3.72 / 15.5 | 3.27 / 13.1 | 5.24 / 19.4 |
| MLP (more accurate on Id/Is, not used) | 1.53 / 5.7 | 2.50 / 9.4 | 1.44 / 5.8 | 3.51 / 12.3 |

Mean over the four currents of the validation median error: MLP 2.31, hybrid deg 2 3.02, hybrid deg 3 4.02,
XGBoost 6.69, plain LightGBM 6.68, Extra Trees 30.2, Random Forest 31.1. Validation data was also used for early stopping,
so validation numbers are slightly optimistic; the test set was used once for reporting.

**Metric:** relative error is `|ML - SPICE| / |SPICE|` with no floor. Mean accuracy = 100 minus the mean relative error with each error capped at 100%.

## Method
* Inputs: 12 raw columns + vds, vsb, vgs, vdb, vgd, ln(W), 1/T (19 features).
* Targets: Id and Is as conductance log10(|I|/|Vds|) (rebuilt as sign x Vds x 10^prediction); Ig and Ib as log10|I|. Signs are constant per case and stored with the model.
* Base: StandardScaler -> degree-2 polynomial -> StandardScaler -> RidgeCV (alpha chosen per target). It captures the smooth part that trees fit poorly.
* Residual: one LightGBM per current on what the base left (lr 0.05, 63 leaves, depth 10, up to 600 trees, early stopping 80).
* Plain LightGBM (`train_lightgbm.py`): lr 0.05, 127 leaves, depth 12, up to 1,000 trees, early stopping 80.
* A LightGBM `linear_tree` variant was tried and dropped because it was too slow; its accuracy on the real data was not measured.

## Folder
```
lightgbm_conductance/
├── code/        ml_lib.py  train_models.py (RF/ET/XGBoost/MLP)  train_lightgbm.py (plain)  train_lightgbm_improved.py (final)
├── models/      pmos_on_lightgbm_improved.joblib    (final; force-added because *.joblib is git-ignored)
├── results/     CSV tables, plots, logs, pmos_on_lightgbm_error_analysis.csv
└── report/      PMOS_ON_report_section.docx
```
Data generation, the example netlist and the data check are in `../../../dataset/PMOS_ON/ashlin_run/` (its `data/pmos_on.csv` is the training data).
`results/pmos_on_pred_vs_spice.png` and `pmos_on_test_best.csv` are the **MLP** run; the LightGBM files are named `pmos_on_lightgbm_*`.

## Reproduce (from this folder; `pip install numpy pandas scikit-learn xgboost lightgbm matplotlib`)
```
cd ml_model/PMOS_ON/lightgbm_conductance
DATA=../../../dataset/PMOS_ON/ashlin_run/data/pmos_on.csv
python3 code/train_models.py pmos_on $DATA            # RF / ET / XGBoost / MLP comparison, about 3 min
python3 code/train_lightgbm.py pmos_on $DATA          # plain LightGBM
python3 code/train_lightgbm_improved.py pmos_on $DATA # final hybrid + error analysis
```

## Use in gate/circuit prediction
```python
import sys, joblib
sys.path.insert(0, "ml_model/PMOS_ON/lightgbm_conductance/code")   # ml_lib must be importable to load the model
from ml_lib import predict_currents
bundle = joblib.load("ml_model/PMOS_ON/lightgbm_conductance/models/pmos_on_lightgbm_improved.joblib")
I = predict_currents(bundle, df)   # df columns: vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj -> id,ig,is,ib (A, into the terminal)
```
