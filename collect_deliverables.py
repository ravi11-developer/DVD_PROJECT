"""
collect_deliverables.py - copy every deliverable of the project document (section 6) into
deliverables/, numbered like the document. Re-run it after any stage; it only copies.

    deliverables/01_datasets/        4 CSVs with exactly the 16 required columns
    deliverables/02_netlists/        1 netlist per transistor case, per gate, and the 3-bit CLA
    deliverables/03_data_check/      KCL table + 3 trend plots
    deliverables/05_chosen_models/   test/val metric tables, 4 parity plots, the saved model files
    deliverables/06_gate_results/    gate results table (18 cases)
    deliverables/07_gate_parity/     gate parity plot
    deliverables/08_test_circuit/    CLA results table + parity plot
    (04 model comparison: not part of this submission; 09 code = this repository; 10 report = report/)

Gate / CLA results are only collected when they come from the ML models (not the oracle test).
Run:  python collect_deliverables.py
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deliverables"
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]
MODEL_DIRS = {
    "NMOS_ON": ROOT / "ml_model/NMOS_ON/nmos_on_code",
    "NMOS_OFF": ROOT / "ml_model/NMOS_OFF/nmos_off_physics_ml",
    "PMOS_ON": ROOT / "ml_model/PMOS_ON/pmos_on_physics_ml",
    "PMOS_OFF": ROOT / "ml_model/PMOS_OFF/pmos_off_physics_ml",
}
GC = ROOT / "gate_circuit"
problems = []


def copy(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        problems.append(f"missing {src.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def ml_backend(csv):
    if not csv.exists():
        return None
    return pd.read_csv(csv, usecols=["backend"], nrows=1)["backend"].iloc[0]


def main():
    # 1. datasets with exactly the required columns
    for c in CASES:
        d = pd.read_csv(ROOT / "dataset" / c / f"{c.lower()}.csv")
        d.columns = d.columns.str.strip()
        (OUT / "01_datasets").mkdir(parents=True, exist_ok=True)
        d[COLUMNS].to_csv(OUT / "01_datasets" / f"{c.lower()}.csv", index=False)
        aug = ROOT / "dataset" / c / f"{c.lower()}_aug.csv"          # extra gate-region samples
        if aug.exists():
            a = pd.read_csv(aug); a.columns = a.columns.str.strip()
            (OUT / "01_datasets" / "extra_gate_region").mkdir(parents=True, exist_ok=True)
            a[COLUMNS].to_csv(OUT / "01_datasets" / "extra_gate_region" / f"{c.lower()}_aug.csv", index=False)

    # 2. netlists
    for c in CASES:
        copy(ROOT / "dataset" / c / f"example_{c.lower()}.net", OUT / "02_netlists/transistors" / f"{c.lower()}.net")
    for g in ("inv", "nand2", "nor2", "and2", "or2"):
        copy(GC / "netlists" / f"{g}.net", OUT / "02_netlists/gates" / f"{g}.net")
    copy(GC / "netlists/cla3.net", OUT / "02_netlists/test_circuit/cla3.net")

    # 3. data check
    for f in ("kcl_check.csv", "trend_summary.csv", "trend_off_current_vs_temp.png",
              "trend_off_current_vs_vth0.png", "trend_gate_current_vs_toxe.png"):
        copy(ROOT / "dataset/data_check" / f, OUT / "03_data_check" / f)

    # 5. chosen models: metric tables, parity plots, saved model files
    for split in ("test", "val"):
        copy(ROOT / f"ml_model/chosen_models_{split}.csv", OUT / f"05_chosen_models/chosen_models_{split}.csv")
    for c, d in MODEL_DIRS.items():
        copy(d / "results/parity_test.png", OUT / "05_chosen_models/parity" / f"{c.lower()}_parity_test.png")
        for f in sorted((d / "results").rglob("*.joblib")):
            copy(f, OUT / "05_chosen_models/saved_models" / c / f.relative_to(d / "results"))
        if c == "NMOS_ON":
            copy(d / "code/blend_weights.json", OUT / "05_chosen_models/saved_models/NMOS_ON/blend_weights.json")

    # 6-8. gate and test-circuit results (ML backend only)
    gb, cb = ml_backend(GC / "data/gate_ml.csv"), ml_backend(GC / "data/cla_ml.csv")
    if gb == "ml":
        for f in ("gate_results.csv", "gate_results.txt", "gate_summary_by_gate.csv"):
            copy(GC / "results/gate" / f, OUT / "06_gate_results" / f)
        copy(GC / "results/gate/gate_parity.png", OUT / "07_gate_parity/gate_parity.png")
        copy(GC / "results/gate/gate_parity_by_gate.png", OUT / "07_gate_parity/gate_parity_by_gate.png")
    else:
        problems.append(f"gate results not collected (backend={gb}); run gate_circuit/05_gate_ml.py with the ML models")
    if cb == "ml":
        for f in ("cla_results.csv", "cla_results.txt", "cla_vs_gate_level.csv", "cla_error_decomposition.csv", "cla_parity.png"):
            copy(GC / "results/cla" / f, OUT / "08_test_circuit" / f)
    else:
        problems.append(f"CLA results not collected (backend={cb}); run gate_circuit/07_cla_ml.py after the ML gate run")

    print(f"Deliverables collected in {OUT.relative_to(ROOT)}/")
    for p in problems:
        print("  WARNING:", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
