# PMOS OFF — physics-guided LightGBM leakage model (22nm_MGK) — chosen model

PMOS OFF (Vg = Vb = VDD) is the mirror image of NMOS OFF (Vg = Vb = 0). Measuring every voltage
downward from VDD (V' = VDD − V), negating every current and using |vth0| gives exactly the NMOS OFF
structure, so the NMOS OFF physics + LightGBM pipeline is reused (`pmoff_common.mirror`). The only
change in `physics_baseline.py` is the normalisation around the PMOS nominal values from
`model/22nm_MGK.pm` (toxe 0.67 nm, |vth0| 0.25399 V, ndep 4.4e18, u0 0.0023, xj 7.2 nm).
The split is the same 70/15/15 (seed 42) used by the earlier MLP model (removed; still in git history).

## How to run

```
cd code
python train_final.py          # about 20 min on one core; data: dataset/PMOS_OFF/pmos_off.csv
```

Results go into `results/`: `metrics.csv`, `test_predictions.csv` (real PMOS OFF signs),
`parity_test.png`, and the model files `physics_params.joblib`, `feature_list.joblib`,
`lgbm_seed{42,7,123}.joblib`.

## Why this model replaced the MLP

Validation mean accuracy (%), same split (`../model_selection.csv`):

| Current | MLP ensemble (removed `mlp.ipynb`) | physics + LightGBM (this folder) |
|---|---|---|
| I_D | 98.98 | **99.79** |
| I_G | 99.67 | **99.94** |
| I_S | 99.01 | **99.79** |
| I_B | 99.60 | **99.68** |

Test set: I_D 99.69 %, I_G 99.94 %, I_S 99.75 %, I_B 99.48 %.

The MLP rebuilds Id = −(Ig+Ib)/2 + c, which assumes the gate current splits equally between drain
and source; near the zero crossing of Id this is a difference of two nearly equal terms. This model
predicts Id/(C + F_d) instead, which is smooth through zero.
