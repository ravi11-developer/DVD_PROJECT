# PMOS ON — Physics-guided LightGBM model (22nm_MGK)

Predicts all four PMOS ON terminal currents (I_D, I_G, I_S, I_B) directly (Vg = 0, Vb = VDD).
LightGBM learns a small correction on top of physics formulas that are fitted on the training set only.
No current is ever computed by adding or subtracting other measured currents.

The final model combines two versions:

| Version | Physics | ML |
|---|---|---|
| A | `physics_baseline_a.py` | 3 LightGBM models (seeds 42, 7, 123) |
| B | `physics_baseline_b.py` — adds gate-to-body tunnelling | 3 LightGBM models (seeds 42, 7, 123) |
| **FINAL** | I_D, I_G, I_B = average of A and B (6 models); I_S = version A | |

This rule was chosen on the validation set so that no validation metric got worse than version A alone.

## How to run (Windows, PowerShell)

1. Put `pmos_on.csv` somewhere simple, for example `D:\DVD_ML_PROJECT\data\pmos_on.csv`.
2. Open this folder in File Explorer, click the address bar, type `powershell` and press Enter.
3. Install the libraries (once):
   ```
   python -m pip install -r requirements.txt
   ```
4. Tell the script where the data is:
   ```
   $env:PMOS_ON_CSV = "D:\DVD_ML_PROJECT\data\pmos_on.csv"
   ```
5. Choose one:
   - **Check the results with the saved models (no training, about 15 minutes):**
     ```
     $env:EVAL_ONLY = "1"
     cd code
     python train_final.py
     ```
   - **Retrain everything from scratch (about 30–80 minutes, 6 models):** first set a new results folder so the
     shipped models stay untouched, then run:
     ```
     $env:EVAL_ONLY = "0"
     $env:PMOS_ON_RESULTS = "D:\DVD_ML_PROJECT\models\pmos_on_physics_ml\my_run"
     cd code
     python train_final.py
     ```

PowerShell forgets `$env:` settings when the window closes, so type them again in a new window.
Keep `N_JOBS = 1` in `pipeline.py` for exactly reproducible numbers (`-1` is faster; results may differ very slightly).
On Linux/Mac use `export NAME=value` instead of `$env:NAME = "value"`.

## Files

| File | What it does |
|---|---|
| `code/pon_common.py` | Data loading, 70/15/15 split (seed 42), error function |
| `code/physics_baseline_a.py` | Physics version A: EKV channel, gate tunnelling, body current |
| `code/physics_baseline_b.py` | Physics version B: version A + gate-to-body tunnelling term |
| `code/pipeline.py` | Shared building blocks: training table, targets, metrics, parity plot |
| `code/train_final.py` | Trains (or loads) both versions, combines them, writes all results |
| `results/metrics.csv` | Mean accuracy, median, p95, max error, within-1 %, R² (log and linear), sign accuracy for train/val/test — physics only (A, B), version A, version B, and FINAL |
| `results/test_predictions.csv` | SPICE vs FINAL predicted currents for every test row |
| `results/parity_test.png` | FINAL predicted vs SPICE plot for the test set |
| `results/model_a/`, `results/model_b/` | Saved models: 3 LightGBM files, physics parameters and feature list each |

Open `metrics.csv` in Excel and filter `model = FINAL`. The `.joblib` files are only for Python (`joblib.load`).

## Results — FINAL model

| Current | Test: mean acc. | Test: median | Test: p95 | Val: mean acc. | Val: median | Val: p95 |
|---|---|---|---|---|---|---|
| I_D | 99.53 % | 0.212 % | 1.45 % | 99.52 % | 0.230 % | 1.36 % |
| I_G | 99.19 % | 0.355 % | 2.85 % | 99.18 % | 0.373 % | 2.54 % |
| I_S | 99.59 % | 0.210 % | 1.40 % | 99.53 % | 0.233 % | 1.33 % |
| I_B | 99.35 % | 0.317 % | 2.16 % | 99.35 % | 0.304 % | 2.35 % |

Targets met for all four currents: mean accuracy > 99 % and median error < 1 %. Not yet met: 95th percentile < 1 %.

Compared with version A alone (test): I_G p95 3.73 % → 2.85 %, I_B p95 2.98 % → 2.16 %, I_S unchanged,
I_D p95 1.42 % → 1.45 % (about 1–2 rows of 1,500, within noise).
Compared with the previous team model (test): about 3.1 % median, 12.6–14.3 % p95, 95–97 % mean accuracy.

## Method in short

1. **Physics baseline** (`least_squares`, training rows only), fitted in stages:
   - gate tunnelling at each edge + a gate-to-channel term split between drain and source, fitted on I_G
     (starting point from an exact linear fit, `numpy.linalg.lstsq`); version B adds gate-to-body tunnelling;
   - channel current with an EKV-style formula — one smooth equation from subthreshold to strong inversion —
     with body effect, DIBL and mobility/velocity-saturation degradation, fitted on I_D and I_S;
   - joint polish, then a smooth quadratic correction to the channel;
   - body current: junction floor + gate-to-body tunnelling through the inverted channel ends
     (BSIM4-style inversion charge), robust `soft_l1` loss.
2. **Shared terminal model:** I_D and I_S rows are described from "this terminal / other terminal". With Vg = 0 and
   Vb = VDD both ends behave alike, so one function learns from 14,000 examples.
3. **Targets:** I_D, I_S as the current divided by a physics size scale (smooth through zero, handles the sign);
   I_G, I_B as the log-ratio to the physics value.
4. **LightGBM:** Huber loss, linear trees, early stopping on validation, 3 seeds averaged per version.

## Notes for the report

- **Dataset range:** Vs is random from 0 to VDD with Vg = 0, so many rows are only weakly on or subthreshold.
  |I_D| spans about 1e-14 A to 1.5e-4 A (median about 7 nA). A formula covering both regimes is essential.
- **Error analysis:** before the gate-to-body term, the largest I_G and I_B errors were in rows where the transistor
  is barely on (both terminals below about 0.4 V). There, gate-to-body tunnelling (Vgb = −VDD, independent of Vd and
  Vs) dominates. I_B was also overpredicted near Vds ≈ 0. Adding the term cut the physics-only p95 error for I_G
  from 41.6 % to 18.5 % and for I_B from 28.8 % to 12.3 % (validation).
- **Overfitting:** train p95 about 0.02–0.04 %, validation and test 1.3–2.9 %. The model fits training rows tightly,
  but validation and test agree closely, so it generalises to unseen data.
- **Honesty about the test set:** all choices were made on validation. The test set was evaluated for the
  development versions and for the final model; no choice was made using test results.
