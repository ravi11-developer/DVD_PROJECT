"""ONE place that connects the gate/circuit code to the four trained transistor
models (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF).

Contract for every predictor
----------------------------
    input : pandas DataFrame with exactly the columns config.FEATURES
            ["vdd","temp","w","vd","vs","vg","vb","vth0","toxe","ndep","u0","xj"]
            in SI units / degC, same meaning and sign as dataset/<CASE>/<case>.csv
            (PMOS vth0 is negative, vg/vb are exactly 0 or VDD).
    output: numpy array, shape (n, 4) = [id, ig, is, ib] in amperes, with the
            dataset sign convention (+ = current flowing INTO the transistor
            terminal, i.e. the same sign as the id/ig/is/ib columns of the
            training CSVs).

Each team member's model has its own preprocessing (log targets, conductance,
physics baseline, scaler ...). That preprocessing must live INSIDE the
predictor below, so the gate code never needs to know about it.

TODO (team): fill in the four functions below once the final model for each
case is chosen and saved to disk. Until then, 05_gate_ml.py can be run with
`--backend oracle` to test the whole pipeline end-to-end on SPICE currents.
"""
import numpy as np
import pandas as pd

from config import FEATURES, REPO

MODELS_ROOT = REPO / "ml_model"


def _not_ready(case):
    raise NotImplementedError(
        f"No predictor wired for '{case}'. Edit gate_circuit/leakage_models.py "
        f"(function predict_{case}) to load the saved final model, or run with "
        f"--backend oracle to test the pipeline.")


def predict_nmos_on(X: pd.DataFrame) -> np.ndarray:
    # e.g.  model = joblib.load(MODELS_ROOT / "NMOS_ON/.../lightgbm_model.joblib")
    #       ... apply the same feature engineering as in training ...
    #       return np.column_stack([id, ig, is_, ib])
    _not_ready("nmos_on")


def predict_nmos_off(X: pd.DataFrame) -> np.ndarray:
    _not_ready("nmos_off")


def predict_pmos_on(X: pd.DataFrame) -> np.ndarray:
    _not_ready("pmos_on")


def predict_pmos_off(X: pd.DataFrame) -> np.ndarray:
    _not_ready("pmos_off")


PREDICTORS = {
    "nmos_on": predict_nmos_on,
    "nmos_off": predict_nmos_off,
    "pmos_on": predict_pmos_on,
    "pmos_off": predict_pmos_off,
}


def predict(case: str, X: pd.DataFrame) -> np.ndarray:
    """Predict [id, ig, is, ib] for transistors of one case."""
    X = X[FEATURES].reset_index(drop=True)
    y = np.asarray(PREDICTORS[case](X), dtype=float)
    if y.shape != (len(X), 4):
        raise ValueError(f"{case}: predictor returned shape {y.shape}, expected {(len(X), 4)}")
    if not np.isfinite(y).all():
        raise ValueError(f"{case}: predictor returned NaN/inf")
    return y
