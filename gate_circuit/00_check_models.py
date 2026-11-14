"""
00_check_models.py - check that leakage_models.py reproduces each trained model.

For every case it takes the model's own TEST rows (same 70/15/15 split, seed 42), predicts them
through leakage_models.predict() exactly as the gate code will (real, un-mirrored inputs), and
compares the mean accuracy per current with the numbers in that model's metrics.csv.
A match proves the saved files, the preprocessing and the sign conventions are wired correctly.

Run:  cd gate_circuit && python 00_check_models.py
Writes results/model_check.csv. Exit code 1 if any case does not match.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import leakage_models
from config import FEATURES, REPO, RESULTS_DIR

CASES = {  # case -> (dataset csv, metrics.csv, model label used in metrics.csv)
    "nmos_on": ("NMOS_ON", leakage_models.RESULTS["nmos_on"] / "metrics.csv", "FINAL"),
    "nmos_off": ("NMOS_OFF", leakage_models.RESULTS["nmos_off"] / "metrics.csv", "physics + LightGBM (3-seed avg)"),
    "pmos_on": ("PMOS_ON", leakage_models.RESULTS["pmos_on"] / "metrics.csv", "FINAL"),
    "pmos_off": ("PMOS_OFF", leakage_models.RESULTS["pmos_off"] / "metrics.csv", "physics + LightGBM (3-seed avg)"),
}
CURRENTS = ["id", "ig", "is", "ib"]
TOL = 0.01   # percentage points of mean accuracy


def test_rows(case_dir):
    d = pd.read_csv(REPO / "dataset" / case_dir / f"{case_dir.lower()}.csv")
    d.columns = d.columns.str.strip()
    _, tmp = train_test_split(np.arange(len(d)), test_size=0.30, random_state=42)
    _, te = train_test_split(tmp, test_size=0.50, random_state=42)
    return d.iloc[te].reset_index(drop=True)


def main():
    rows, ok = [], True
    for case, (case_dir, metrics_csv, label) in CASES.items():
        te = test_rows(case_dir)
        try:
            pred = leakage_models.predict(case, te[FEATURES])
        except FileNotFoundError as e:
            print(f"{case:8s} NOT READY: {e}"); ok = False; continue
        m = pd.read_csv(metrics_csv, skipinitialspace=True)
        m.columns = m.columns.str.strip()
        for c in ("model", "split", "current"):
            m[c] = m[c].astype(str).str.strip()
        ref = m[(m.model == label) & (m.split == "test")].set_index("current")["mean_accuracy"]
        for j, c in enumerate(CURRENTS):
            e = np.abs(pred[:, j] - te[c]) / np.abs(te[c]) * 100
            acc = 100 - np.minimum(e, 100).mean()
            diff = acc - ref[c.upper()]
            rows.append(dict(case=case, current=c.upper(), mean_accuracy=acc, metrics_csv=ref[c.upper()],
                             difference=diff, median_err=np.median(e), p95_err=np.percentile(e, 95)))
            ok &= abs(diff) <= TOL
        r = pd.DataFrame(rows)
        r = r[r.case == case]
        status = "OK" if (r.difference.abs() <= TOL).all() else "MISMATCH"
        print(f"{case:8s} {status}: test mean accuracy " +
              "  ".join(f"{x.current} {x.mean_accuracy:.3f}%" for x in r.itertuples()))
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "model_check.csv", index=False)
    print("ALL MODELS WIRED CORRECTLY" if ok else "SOME MODELS ARE MISSING OR DO NOT MATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
