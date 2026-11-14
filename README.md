# CURRENT LEAKAGE ESTIMATION BASED ON MACHINE LEARNING ALGORITHMS

ML-driven PVT-variation-aware leakage estimation (22nm_MGK, VDD = 0.9 V, test circuit: 3-bit CLA).
Four transistor models (NMOS/PMOS, ON/OFF) predict Id, Ig, Is, Ib; gate and circuit leakage are
built from them and checked against SPICE.


The slow SPICE stages (`data`, `spice`) only run when asked for; their outputs are already in the repo.

| Stage | What it does | Output |
|---|---|---|
| `data` | 10,000 SPICE samples per case | `dataset/<CASE>/<case>.csv` |
| `datacheck` | KCL check, 3 trend plots, example netlists | `dataset/data_check/`, `dataset/<CASE>/example_*.net` |
| `train` | trains the 4 models in parallel | model files in `ml_model/*/results/` |
| `check` | model summary; checks the gate code reproduces each model | `ml_model/chosen_models_*.csv`, `gate_circuit/results/model_check.csv` |
| `spice` | gate + CLA SPICE references | `gate_circuit/data/` |
| `gates` | ML gate + CLA leakage, tables, parity plots | `gate_circuit/results/` |
| `collect` | copies every deliverable, numbered like the project document | `deliverables/` |
| `report` | refreshes report tables, numbers and figures | `report/generated/`, `report/figures/` |

The report is [`report/report.tex`](./report/report.tex) (compile it inside `report/`).

## Layout

- [`dataset/`](./dataset/) — the four training datasets and the scripts that generate and check them
- [`ml_model/`](./ml_model/) — the four transistor models (chosen: `NMOS_ON/nmos_on_code`, `NMOS_OFF/nmos_off_physics_ml`, `PMOS_ON/pmos_on_physics_ml`, `PMOS_OFF/pmos_off_physics_ml`)
- [`gate_circuit/`](./gate_circuit/) — Part B (gates) and Part C (3-bit CLA); `leakage_models.py` connects them to the saved models
- [`report/`](./report/) — LaTeX report
- [`model/`](./model/) — the PTM model card

## Things to Know If You Are a Contributor

If you want to develop or modify the model algorithms, strictly work inside the [`ml_model/`](./ml_model/) directory.

Create your respective folders according to the model category:

- [`NMOS_ON/`](./ml_model/NMOS_ON/)
- [`NMOS_OFF/`](./ml_model/NMOS_OFF/)
- [`PMOS_ON/`](./ml_model/PMOS_ON/)
- [`PMOS_OFF/`](./ml_model/PMOS_OFF/)

Develop, experiment, and innovate within your respective directories.

**Important:** Do not modify files outside your assigned model directory without proper discussion.
