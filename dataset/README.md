# Dataset — single-transistor SPICE data (22nm_MGK)

This is the **only** dataset used by the four transistor models in `ml_model/`.
Every CSV here was re-simulated row-by-row in ngspice and reproduces to ~1e-6 % (Id, Ig, Is, Ib).

## Layout

```text
dataset/
├── NMOS_ON/    Vg = VDD, Vb = 0
├── NMOS_OFF/   Vg = 0,   Vb = 0
├── PMOS_ON/    Vg = 0,   Vb = VDD
└── PMOS_OFF/   Vg = VDD, Vb = VDD
    └── each case folder contains:
        01_<case>_inputs.py     step 1: draws 10,000 random PVT samples (seed 42)
        02_<case>_simulate.py   step 2: runs ngspice on them in 10 batches of 1,000
        <case>_inputs.csv       output of step 1 (12 input columns)
        <case>.csv              output of step 2 — THE TRAINING DATA
        sim_<case>/batch_*.net  the exact netlists ngspice ran
```

## Which model uses which file

| Model (`ml_model/…`)                     | Training CSV                     |
|------------------------------------------|----------------------------------|
| `NMOS_ON/nmos_on_code`                   | `dataset/NMOS_ON/nmos_on.csv`    |
| `NMOS_OFF/nmos_off_physics_ml`           | `dataset/NMOS_OFF/nmos_off.csv`  |
| `PMOS_ON/pmos_on_physics_ml`             | `dataset/PMOS_ON/pmos_on.csv`    |
| `PMOS_OFF/mlp.ipynb`                     | `dataset/PMOS_OFF/pmos_off.csv`  |

The model code finds these paths on its own; an environment variable
(`NMOS_ON_CSV`, `NMOS_OFF_CSV`, `PMOS_ON_CSV`) can override them.

## Columns (`<case>.csv`)

| Column | Meaning | Range |
|---|---|---|
| `vdd` | supply voltage (V) | 0.81 – 0.99 (0.9 V ± 10 %) |
| `temp` | temperature (°C) | −40 – 125 |
| `w` | channel width (m), L = 22 nm fixed | 22 nm – 176 nm (1–8 × Lmin) |
| `vd`, `vs` | drain / source voltage (V) | 0 – vdd, independent uniform |
| `vg`, `vb` | gate / body voltage (V) | exactly 0 or vdd, per case (table above) |
| `vth0`, `toxe`, `ndep`, `u0`, `xj` | BSIM4 process parameters | nominal ± 10 % (NMOS / PMOS nominals from `model/22nm_MGK.pm`) |
| `id`, `ig`, `is`, `ib` | terminal currents (A) | **+ = current flowing into the terminal** |
| `itotal` | `id+ig+is+ib` (KCL check, ≈ 0) | not a model input |

SPICE options: `gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`. Values are stored with 9 significant digits,
and the input columns are read back from ngspice, so they are exactly what was simulated.

## Regenerating

Needs Python (numpy, pandas) and ngspice on `PATH` (or set `NGSPICE=/path/to/ngspice`).

```bash
cd dataset/PMOS_OFF          # any case
python 01_pmos_off_inputs.py # -> pmos_off_inputs.csv  (deterministic, seed 42)
python 02_pmos_off_simulate.py  # -> pmos_off.csv + sim_pmos_off/*.net  (~minutes)
```

Step 1 reproduces the committed `*_inputs.csv` bit-for-bit.
