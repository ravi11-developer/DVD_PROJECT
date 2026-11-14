"""pmoff_common.py - PMOS OFF data loading (mirrored), 70/15/15 split (seed 42), error function.

PMOS OFF (Vg = Vb = VDD) is the mirror image of NMOS OFF (Vg = Vb = 0): measuring every voltage
downward from VDD (V' = VDD - V), flipping the sign of every current and using |vth0| gives exactly
the NMOS OFF structure, so the NMOS OFF physics + LightGBM pipeline is reused (same idea as
NMOS ON reusing the PMOS ON pipeline). Relative errors are identical in both views.
The split is the same one used by the earlier MLP model (train_test_split on the row index, seed 42).
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

K_Q = 8.617333e-5                     # Boltzmann constant / electron charge (V per K)
CURRENTS = ['id', 'ig', 'is', 'ib']
SEED = 42
_DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', '..', 'dataset', 'PMOS_OFF', 'pmos_off.csv'))
DATA_CSV = os.environ.get('PMOS_OFF_CSV', _DEFAULT_DATA)   # override with PMOS_OFF_CSV if needed

def mirror(df):
    """PMOS OFF -> NMOS-OFF-like view. Applying it twice gives back the original."""
    m = df.copy()
    for v in ('vd', 'vs', 'vg', 'vb'): m[v] = df.vdd - df[v]
    for c in CURRENTS: m[c] = -df[c]
    m['vth0'] = -df.vth0
    return m

AUG_CSV = os.environ.get('PMOS_OFF_AUG_CSV', DATA_CSV.replace('.csv', '_aug.csv'))


def _view(d):
    """PMOS OFF -> mirrored NMOS-OFF view."""
    return mirror(d)

# Extra training samples at gate operating points (dataset/05_augment_inputs.py): split 70/15/15 with the
# same seed; their train/val parts are ADDED to train/val, their test part is kept apart as "test_aug".
# The original 10,000-row split (and therefore the original test set) is unchanged.
AUG_INDEX_OFFSET = 1_000_000          # keeps original and extra row labels distinct
# The OFF physics uses ln(V) of the drain/source voltage. A terminal exactly on the rail (V = 0 in the
# NMOS-OFF view) is moved RAIL_EPS off it, and the other terminal by the same amount so Vds is
# unchanged - the same rule gate_circuit/leakage_models.py applies.
RAIL_EPS = 3e-5
# Extra samples with every terminal within microvolts of the rail have |Ig|, |Ib| at the simulator's
# noise floor (value and sign not reproducible). Those Ig / Ib values (and any with the physically
# wrong sign) are set to NaN and ignored in
# training and in the metrics; Id and Is of the same samples are kept.
NOISE_FLOOR = 1e-15


def _split3(n, seed):
    i_tr, i_tmp = train_test_split(np.arange(n), test_size=0.30, random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=seed)
    return i_tr, i_va, i_te


def _read(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _prepare(d, extra):
    """NMOS-OFF view: rail clamp for all rows, noise floor for the extra rows."""
    d = _view(d)
    shift = np.maximum(RAIL_EPS - np.minimum(d.vd, d.vs), 0.0)   # move BOTH terminals: Vds is preserved
    d["vd"], d["vs"] = d.vd + shift, d.vs + shift
    if extra:
        for c in ("ig", "ib"):          # in the OFF view Ig < 0 and Ib < 0; anything else is solver noise
            d.loc[(d[c].abs() < NOISE_FLOOR) | (d[c] >= 0), c] = np.nan
    return d


def _aug():
    if not os.path.exists(AUG_CSV):
        return None
    a = _read(AUG_CSV)
    a.index = a.index + AUG_INDEX_OFFSET
    return a


def load_split(seed=SEED):
    """train/val = original + extra samples; test = ORIGINAL test rows only (NMOS-OFF view)."""
    df = _read(DATA_CSV)
    assert df.isnull().sum().sum() == 0, "There are missing values in the CSV"
    i_tr, i_va, i_te = _split3(len(df), seed)
    tr, va, te = (_prepare(df.iloc[i], False) for i in (i_tr, i_va, i_te))
    full = _prepare(df, False)
    a = _aug()
    if a is not None:
        j_tr, j_va, _ = _split3(len(a), seed)
        tr = pd.concat([tr, _prepare(a.iloc[j_tr], True)])
        va = pd.concat([va, _prepare(a.iloc[j_va], True)])
        full = pd.concat([full, _prepare(a, True)])
    return full, tr, va, te


def load_test_aug(seed=SEED):
    """Test part of the extra samples (NMOS-OFF view), or None if there are none."""
    a = _aug()
    if a is None:
        return None
    return _prepare(a.iloc[_split3(len(a), seed)[2]], True)

def pct_err(t, p): return np.abs(p - t) / np.abs(t) * 100
