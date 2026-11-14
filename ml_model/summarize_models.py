"""
summarize_models.py - deliverable 5: one table for the four chosen transistor models.

Reads each chosen model's metrics.csv and writes
    ml_model/chosen_models_test.csv   test-set median / p95 error and mean accuracy per current
    ml_model/chosen_models_val.csv    the same on the validation set (used for model selection)
and checks that the saved model files exist.

Run (from anywhere):  python ml_model/summarize_models.py
Mean accuracy = 100 - mean(min(|err|, 100)) with err = |ML - SPICE| / |SPICE| in percent.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CHOSEN = {  # case -> (model folder, label in metrics.csv, short description, files that must exist)
    "NMOS ON": (HERE / "NMOS_ON/nmos_on_code/results", "FINAL",
                "physics + LightGBM, 4 versions x 3 seeds, blended per current",
                [f"model_{v}/lgbm_seed{s}.joblib" for v in "abcd" for s in (42, 7, 123)]),
    "NMOS OFF": (HERE / "NMOS_OFF/nmos_off_physics_ml/results", "physics + LightGBM (3-seed avg)",
                 "physics + LightGBM, 3 seeds", [f"lgbm_seed{s}.joblib" for s in (42, 7, 123)]),
    "PMOS ON": (HERE / "PMOS_ON/pmos_on_physics_ml/results", "FINAL",
                "physics + LightGBM, 2 versions x 3 seeds, averaged",
                [f"model_{v}/lgbm_seed{s}.joblib" for v in "ab" for s in (42, 7, 123)]),
    "PMOS OFF": (HERE / "PMOS_OFF/pmos_off_physics_ml/results", "physics + LightGBM (3-seed avg)",
                 "physics + LightGBM (mirrored NMOS OFF pipeline), 3 seeds",
                 [f"lgbm_seed{s}.joblib" for s in (42, 7, 123)]),
}


def load(path):
    m = pd.read_csv(path, skipinitialspace=True)
    m.columns = m.columns.str.strip()
    for c in ("model", "split", "current"):
        m[c] = m[c].astype(str).str.strip()
    return m


def main():
    out = {"test": [], "val": []}
    for case, (folder, label, desc, files) in CHOSEN.items():
        missing = [f for f in files if not (folder / f).exists()]
        m = load(folder / "metrics.csv")
        for split in out:
            r = m[(m.model == label) & (m.split == split)].set_index("current")
            for cur in ("ID", "IG", "IS", "IB"):
                out[split].append(dict(case=case, current=cur, model=desc,
                                       median_err_pct=r.loc[cur, "median_err"], p95_err_pct=r.loc[cur, "p95_err"],
                                       mean_accuracy_pct=r.loc[cur, "mean_accuracy"], r2_log=r.loc[cur, "r2_log"],
                                       model_files_ok=not missing))
        print(f"{case:9s} model files: {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
    for split, rows in out.items():
        df = pd.DataFrame(rows)
        df.to_csv(HERE / f"chosen_models_{split}.csv", index=False, float_format="%.5g")
        print(f"\n{split.upper()} set")
        print(df.drop(columns=["model", "model_files_ok"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        worst = df.mean_accuracy_pct.min()
        print(f"lowest mean accuracy on {split}: {worst:.3f}%  ({'all >= 99%' if worst >= 99 else 'below 99%'})")


if __name__ == "__main__":
    main()
