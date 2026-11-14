# NMOS ON — Physics-guided LightGBM model (22nm_MGK)

Predicts all four NMOS ON terminal currents (I_D, I_G, I_S, I_B) directly (Vg = VDD, Vb = 0).
LightGBM learns a small correction on top of physics formulas fitted on the training set only.
No current is ever computed by adding or subtracting other measured currents.

## Results — FINAL model (all four currents above 99 % mean accuracy)

| Current | Test: mean acc. | Test: median | Test: p95 | Val: mean acc. | Val: median | Val: p95 |
|---|---|---|---|---|---|---|
| I_D | 99.41 % | 0.271 % | 2.03 % | 99.40 % | 0.269 % | 1.94 % |
| I_G | 99.27 % | 0.254 % | 2.33 % | 99.22 % | 0.252 % | 2.48 % |
| I_S | 99.42 % | 0.279 % | 1.95 % | 99.44 % | 0.267 % | 1.85 % |
| I_B | 99.10 % | 0.429 % | 3.49 % | 99.12 % | 0.406 % | 3.07 % |

Targets met: mean accuracy > 99 % and median error < 1 % for all four currents. Not met: 95th percentile < 1 %.

## How the final model is built

| Version | Physics file | What is special |
|---|---|---|
| A | `physics_a.py` | No separate gate-to-body term. Best channel fit (physics-only I_D median 1.6 %). |
| B | `physics_b.py` | Adds gate-to-body tunnelling to I_G and I_B. |
| C | `physics_c.py` | Version B + a velocity-saturation term in the channel formula. |
| D | `physics_d.py` | Version B + gate-to-channel tunnelling proportional to the inversion charge (BSIM4-style). Best I_G (physics-only median 1.3 %). |

Each version = its physics formula + 3 LightGBM models (seeds 42, 7, 123). The FINAL prediction of each current is a
weighted average of the four versions, with weights in `code/blend_weights.json`:

| Current | A | B | C | D |
|---|---|---|---|---|
| I_D | 0.80 | 0.10 | 0.10 | 0 |
| I_G | 0.25 | 0 | 0.10 | 0.65 |
| I_S | 0.95 | 0.05 | 0 | 0 |
| I_B | 0.35 | 0.35 | 0.20 | 0.10 |

The weights were chosen on the **validation** set (highest mean accuracy per current). The test set was used once, for the final report.

**Mirror trick:** NMOS ON (gate at VDD, body at 0) is the mirror image of PMOS ON (gate at 0, body at VDD). Measuring every
voltage downward from VDD (V' = VDD − V) and flipping every current's sign gives exactly the PMOS ON structure, so the
PMOS ON physics + ML pipeline is reused (`non_common.py`). Relative errors are identical in both views, and all saved
predictions and plots are converted back to real NMOS ON signs.

## How to run (Windows, PowerShell)

1. The data is read automatically from `dataset/NMOS_ON/nmos_on.csv` in the project root (no setup needed).
2. Open this folder in File Explorer, click the address bar, type `powershell` and press Enter.
3. Install the libraries (once):
   ```
   python -m pip install -r ..\..\..\requirements.txt   (project-wide file in the project root)
   ```
4. (Optional) To use a different CSV, set `$env:NMOS_ON_CSV = "C:\path\to\nmos_on.csv"`.
5. Choose one:
   - **Check the results with the saved models (no training, about 30–60 minutes on one core):**
     ```
     $env:EVAL_ONLY = "1"
     cd code
     python train_final.py
     ```
   - **Retrain everything (12 LightGBM models, about 1–2 hours on one core):** set a new results folder first so the
     shipped models stay untouched:
     ```
     $env:EVAL_ONLY = "0"
     $env:NMOS_ON_RESULTS = "D:\DVD_ML_PROJECT\models\nmos_on_physics_ml\my_run"
     cd code
     python train_final.py
     ```

PowerShell forgets `$env:` settings when the window closes. For speed, set `N_JOBS = -1` in `pipeline.py`
(results may then differ very slightly from the table above).

## Files

| File | What it does |
|---|---|
| `code/non_common.py` | Data loading in the mirrored view, 70/15/15 split (seed 42), error function |
| `code/physics_a.py` … `physics_d.py` | The four physics baselines (EKV channel, gate tunnelling, body current) and their fitting |
| `code/pipeline.py` | Training table, targets, metrics, parity plot |
| `code/train_final.py` | Trains (or loads) all four versions, blends them, writes all results |
| `code/blend_weights.json` | Per-current blend weights (chosen on validation) |
| `results/metrics.csv` | Mean accuracy, median, p95, max error, within-1 %, R² (log, linear), sign accuracy — for every version and FINAL, on train/val/test |
| `results/test_predictions.csv` | SPICE vs FINAL predictions for every test row, in real NMOS ON signs |
| `results/parity_test.png` | FINAL predicted vs SPICE, test set |
| `results/model_a/` … `model_d/` | Saved models (3 LightGBM files, physics parameters, nominal values, feature list each) |

## Notes for the report

- **Dataset:** NMOS ON currents are large (|I_D| median 41 µA, up to 530 µA), and many rows are near threshold because
  Vs ranges from 0 to VDD with Vg = VDD.
- **Error analysis:** the largest errors are in subthreshold / near-threshold rows (gate overdrive below about 0.2 V) and at
  very small Vds. The I_G breakthrough came from making gate-to-channel tunnelling proportional to the inversion charge,
  which switches off below threshold: physics-only I_G error fell from 3.3 % to 1.3 % (median).
- **I_B noise floor:** NGSpice noise (reltol = 1e-6, judged from the KCL residual) is above 1 % of I_B in 20.7 % of rows,
  so an I_B 95th-percentile error below 1 % is not achievable with this data.
- **Comparison with the earlier team model:** its reported numbers (mean accuracy 98.05–98.74 %) came from a different
  dataset, so they are not a direct comparison.
- **Overfitting:** validation and test agree closely (mean accuracy within 0.05 points), so the model generalises to unseen data.
