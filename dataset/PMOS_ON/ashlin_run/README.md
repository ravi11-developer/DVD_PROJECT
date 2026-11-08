# PMOS ON data generation and data check (`dataset/PMOS_ON/ashlin_run/`)

Ashlin K Sunil (2024102025). 10,000 single-PMOS ngspice samples (Vg = 0, Vb = VDD), 22 nm MGK, nominal VDD 0.9 V, seed 1002.
This is the dataset used by `ml_model/PMOS_ON/lightgbm_conductance/` (it is a different random draw from `../pmos_on.csv`).

```
code/      spice_lib.py  gen_dataset.py  smoke_test.py  check_data.py
data/      pmos_on.csv                  (generated; 10,000 rows)
netlists/  fet_pmos_on.cir              (example single-PMOS netlist)
results/   kirchhoff_pmos_on.csv  trend_ig_vs_toxe.png  trend_on_id_filtered.png
```
Run from this folder (needs ngspice 42, Python 3.9+, numpy pandas matplotlib):
```
python3 code/smoke_test.py
python3 code/gen_dataset.py 10000 data pmos_on     # about 5 min
python3 code/check_data.py pmos_on                 # Kirchhoff table + trend plots in results/
```
Worst Kirchhoff residual: 5.2e-6 of the largest terminal current (solver tolerance).
