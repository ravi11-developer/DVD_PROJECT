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

How it works
------------
Each predictor loads the SAVED model files written by that model's training script and
re-uses that model's own code (physics baseline, feature table, output transform), so the
gate pipeline cannot drift from training. Model folders (relative to the project root):

    NMOS ON  : ml_model/NMOS_ON/nmos_on_code/results/model_{a,b,c,d}/   (+ code/blend_weights.json)
    NMOS OFF : ml_model/NMOS_OFF/nmos_off_physics_ml/results/
    PMOS ON  : ml_model/PMOS_ON/pmos_on_physics_ml/results/model_{a,b}/
    PMOS OFF : ml_model/PMOS_OFF/pmos_off_physics_ml/results/

Each folder holds physics_params.joblib, feature_list.joblib and lgbm_seed{42,7,123}.joblib
(NMOS ON also nominal.joblib). Train them with ml_model/train_parallel.py or each
train_final.py. Check the wiring with 00_check_models.py before running 05_gate_ml.py.

The model code folders reuse module names (pipeline, physics_baseline, train_final), so every
case is imported in isolation (_load_modules) and keeps its own module objects.
"""
import importlib
import json
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import FEATURES, REPO

MODELS_ROOT = REPO / "ml_model"
SEEDS = [42, 7, 123]
CODE = {
    "nmos_on": MODELS_ROOT / "NMOS_ON" / "nmos_on_code" / "code",
    "nmos_off": MODELS_ROOT / "NMOS_OFF" / "nmos_off_physics_ml" / "code",
    "pmos_on": MODELS_ROOT / "PMOS_ON" / "pmos_on_physics_ml" / "code",
    "pmos_off": MODELS_ROOT / "PMOS_OFF" / "pmos_off_physics_ml" / "code",
}
RESULTS = {k: v.parent / "results" for k, v in CODE.items()}
N_THREADS = int(os.environ.get("PREDICT_THREADS", os.cpu_count() or 1))


@lru_cache(maxsize=None)
def _model(path: Path):
    """Load each saved LightGBM model once per process."""
    return joblib.load(_need(path))
_SHARED_NAMES = ("pipeline", "physics_baseline", "train_final")


# ---------------------------------------------------------------- helpers
def _load_modules(case, names):
    """Import `names` from the case's code folder without clashing with other cases."""
    code = str(CODE[case])
    for n in _SHARED_NAMES + tuple(names):
        sys.modules.pop(n, None)
    sys.path.insert(0, code)
    try:
        mods = {n: importlib.import_module(n) for n in names}
    finally:
        sys.path.remove(code)
        for n in _SHARED_NAMES + tuple(names):      # module objects stay alive via `mods`
            sys.modules.pop(n, None)
    return mods


def _need(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file {path.relative_to(REPO)}. Train the model first "
            f"(python ml_model/train_parallel.py <case>, or the case's train_final.py).")
    return path


def _with_dummy_currents(X):
    """The feature-table builders also compute training targets from id/ig/is/ib. Those targets
    are not used for prediction, so placeholder values are enough."""
    d = X[FEATURES].reset_index(drop=True).copy()
    for c, v in (("id", 1e-12), ("ig", -1e-12), ("is", -1e-12), ("ib", -1e-12)):
        d[c] = v
    return d


def _seed_average(pl_build, pl_to_currents, P, folder, d):
    """Average the 3 LightGBM seeds in the model's output space, then convert to currents.
    Prediction uses all CPU cores (each row is independent, so results do not change)."""
    F = joblib.load(_need(folder / "feature_list.joblib"))
    B = pl_build(d, P)
    outs = [_model(folder / f"lgbm_seed{s}.joblib").predict(B["L"][F], num_threads=N_THREADS) for s in SEEDS]
    return pl_to_currents(B, np.mean(outs, axis=0))


# OFF-state physics uses ln(V) of the drain/source voltage (gate-edge tunnelling), so a terminal
# exactly on the rail (source of an OFF transistor tied to GND / VDD inside a gate) gives ln(0).
# Training data never has V = 0 exactly (closest: 7-31 uV), so such a terminal is moved RAIL_EPS
# off the rail (both terminals shift together, so Vds is preserved). 30 uV was chosen on 400 fresh single-transistor SPICE samples per OFF case with the
# source exactly on the rail (best Id accuracy; see README "Rail clamp").
RAIL_EPS = 3e-5


def _clamp_rail(d):
    """In the OFF-state (Vg = Vb = 0) view: if a terminal is closer than RAIL_EPS to the rail, shift
    BOTH terminals up by the same amount, so the lower one sits at RAIL_EPS and Vds is unchanged."""
    d = d.copy()
    shift = np.maximum(RAIL_EPS - np.minimum(d.vd, d.vs), 0.0)
    d["vd"], d["vs"] = d.vd + shift, d.vs + shift
    return d


def _as_array(df):
    return df[["id", "ig", "is", "ib"]].to_numpy(dtype=float)


# ---------------------------------------------------------------- NMOS ON
@lru_cache(maxsize=None)
def _nmos_on():
    m = _load_modules("nmos_on", ["non_common", "pipeline", "physics_a", "physics_b", "physics_c", "physics_d"])
    versions = {"A": m["physics_a"], "B": m["physics_b"], "C": m["physics_c"], "D": m["physics_d"]}
    weights = json.load(open(_need(CODE["nmos_on"] / "blend_weights.json")))
    params = {}
    for tag, mod in versions.items():
        folder = RESULTS["nmos_on"] / f"model_{tag.lower()}"
        mod.NOM.update(joblib.load(_need(folder / "nominal.joblib")))   # training-set medians
        params[tag] = (mod, joblib.load(_need(folder / "physics_params.joblib")), folder)
    return m, params, weights


def predict_nmos_on(X: pd.DataFrame) -> np.ndarray:
    """4 physics-guided LightGBM versions (A-D), blended per current, in the mirrored view."""
    m, params, weights = _nmos_on()
    d = m["non_common"].mirror(_with_dummy_currents(X))       # NMOS ON -> PMOS-ON-like view
    PL = m["pipeline"]
    preds = {}
    for tag, (mod, P, folder) in params.items():
        PL.PB = mod
        preds[tag] = _seed_average(PL.build_table, PL.to_currents, P, folder, d)
    fin = pd.DataFrame({c: sum(weights[c][t] * preds[t][c] for t in preds) for c in ("id", "ig", "is", "ib")})
    return -_as_array(fin)                                     # back to real NMOS ON signs


# ---------------------------------------------------------------- NMOS OFF
@lru_cache(maxsize=None)
def _nmos_off():
    m = _load_modules("nmos_off", ["train_final"])
    return m["train_final"], joblib.load(_need(RESULTS["nmos_off"] / "physics_params.joblib"))


def predict_nmos_off(X: pd.DataFrame) -> np.ndarray:
    """Physics baseline + LightGBM correction, 3 seeds averaged."""
    tf, P = _nmos_off()
    d = _clamp_rail(_with_dummy_currents(X))
    return _as_array(_seed_average(tf.build_table, tf.to_currents, P, RESULTS["nmos_off"], d))


# ---------------------------------------------------------------- PMOS ON
@lru_cache(maxsize=None)
def _pmos_on():
    m = _load_modules("pmos_on", ["pipeline", "physics_baseline_a", "physics_baseline_b"])
    params = {}
    for tag, mod in (("A", m["physics_baseline_a"]), ("B", m["physics_baseline_b"])):
        folder = RESULTS["pmos_on"] / f"model_{tag.lower()}"
        params[tag] = (mod, joblib.load(_need(folder / "physics_params.joblib")), folder)
    return m["pipeline"], params


def predict_pmos_on(X: pd.DataFrame) -> np.ndarray:
    """Versions A and B averaged; Is taken from version A only (as in train_final.py)."""
    PL, params = _pmos_on()
    d = _with_dummy_currents(X)
    preds = {}
    for tag, (mod, P, folder) in params.items():
        PL.PB = mod
        preds[tag] = _seed_average(PL.build_table, PL.to_currents, P, folder, d)
    fin = (preds["A"] + preds["B"]) / 2
    fin["is"] = preds["A"]["is"]
    return _as_array(fin)


# ---------------------------------------------------------------- PMOS OFF
@lru_cache(maxsize=None)
def _pmos_off():
    m = _load_modules("pmos_off", ["pmoff_common", "train_final"])
    return m["pmoff_common"], m["train_final"], joblib.load(_need(RESULTS["pmos_off"] / "physics_params.joblib"))


def predict_pmos_off(X: pd.DataFrame) -> np.ndarray:
    """NMOS OFF pipeline on the mirrored PMOS OFF data (V' = VDD - V, |vth0|, currents negated)."""
    common, tf, P = _pmos_off()
    d = _clamp_rail(common.mirror(_with_dummy_currents(X)))      # clamp in the mirrored (NMOS-OFF-like) view
    return -_as_array(_seed_average(tf.build_table, tf.to_currents, P, RESULTS["pmos_off"], d))


PREDICTORS = {
    "nmos_on": predict_nmos_on,
    "nmos_off": predict_nmos_off,
    "pmos_on": predict_pmos_on,
    "pmos_off": predict_pmos_off,
}


def predict(case: str, X: pd.DataFrame) -> np.ndarray:
    """Predict [id, ig, is, ib] for transistors of one case."""
    X = X[FEATURES].reset_index(drop=True)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")            # placeholder targets give harmless log() warnings
        y = np.asarray(PREDICTORS[case](X), dtype=float)
    if y.shape != (len(X), 4):
        raise ValueError(f"{case}: predictor returned shape {y.shape}, expected {(len(X), 4)}")
    if not np.isfinite(y).all():
        raise ValueError(f"{case}: predictor returned NaN/inf")
    return y
