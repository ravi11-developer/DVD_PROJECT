# NMOS OFF — Physics-guided LightGBM leakage model (22nm_MGK)

One LightGBM model predicts all four NMOS OFF terminal currents (I_D, I_G, I_S, I_B) directly.
It learns a small correction on top of physics formulas that are fitted on the training set only.
No current is ever computed by adding or subtracting other measured currents.

## How to run

```
pip install -r requirements.txt
cd code
set NMOS_OFF_CSV=C:\path\to\nmos_off.csv        (Windows)
export NMOS_OFF_CSV=/path/to/nmos_off.csv       (Linux / Mac)
python train_final.py
```

It takes about 10 minutes on one CPU core. Everything goes into `results/`.
To get exactly the numbers below, keep `N_JOBS = 1` in `train_final.py`.

## Files

| File | What it does |
|---|---|
| `code/nmos_off_common.py` | Your original helper file: data loading, 70/15/15 split (seed 42), error function |
| `code/physics_baseline.py` | Physics formulas (gate tunnelling, subthreshold channel, body current) and their fitting |
| `code/train_final.py` | Builds the training table, trains 3 LightGBM models, reports all metrics, saves everything |
| `results/metrics.csv` | Mean accuracy, median, p95, max error, within-1 %, R² (log and linear), sign accuracy, for train/val/test, for "physics only" and for the final model |
| `results/test_predictions.csv` | SPICE vs predicted currents for every test row |
| `results/parity_test.png` | Predicted vs SPICE plot for the test set |
| `results/lgbm_seed*.joblib`, `physics_params.joblib`, `feature_list.joblib` | Saved model files |

## Results (test set, 1,500 unseen rows)

| Current | Mean accuracy | Median error | 95th-pct error |
|---|---|---|---|
| I_D | 99.69 % | 0.020 % | 1.003 % |
| I_G | 99.95 % | 0.017 % | 0.214 % |
| I_S | 99.61 % | 0.020 % | 0.981 % |
| I_B | 99.40 % | 0.035 % | 0.552 % |

Validation: I_D 0.95 %, I_G 0.22 %, I_S 1.00 %, I_B 0.51 % (95th percentile).

## Method in short

1. **Physics baseline** (`least_squares`, training rows only), fitted in stages:
   gate tunnelling on I_G → subthreshold channel on I_D and I_S → joint polish →
   smooth quadratic correction to the channel → body-current formula (robust `soft_l1` loss).
   The physics guess for I_D is its own parts (channel + drain-edge tunnelling), as in SPICE.
2. **Shared terminal model.** I_D and I_S rows are described from "this terminal / other terminal".
   With V_G = V_B = 0 the device is drain-source symmetric, so one function learns from 14,000 examples.
3. **Targets.** I_D, I_S: the current divided by a physics size scale (smooth through zero, handles the sign).
   I_G, I_B: the log-ratio to the physics value.
4. **LightGBM**, Huber loss (robust to rare extreme rows), linear trees, early stopping on validation,
   3 random seeds averaged.

## Notes for the report

- **Preprocessing must be described:** the physics baseline, the scaled target for I_D/I_S and the log target for I_G/I_B.
- **Ablation:** `metrics.csv` also has "physics only" rows, which show what the ML adds on top of the formulas.
- **Largest errors:** rows where I_D or I_S nearly cancel (channel current ≈ opposite tunnelling current).
  Example: SPICE I_D = −8.5e−15 A, so a tiny absolute miss becomes a huge percentage.
- **I_B noise floor:** I_B ≈ 1e−14 A. NGSpice precision (reltol = 1e−6) leaves noise of up to ~0.66 % of I_B
  (95th percentile, from the KCL residual), so I_B accuracy is close to what the data can support.
- **Junction geometry:** the netlists have no AD/AS/PD/PS, so BSIM4 uses its default junction saturation current
  and I_B stays near 1e−14 A.
- The test set was evaluated after the method was fixed on validation; it was not used for any tuning.
