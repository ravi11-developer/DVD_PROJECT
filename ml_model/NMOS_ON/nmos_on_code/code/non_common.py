"""non_common.py - NMOS ON data loading (mirrored), 70/15/15 split (seed 42), error function.

NMOS ON (Vg = VDD, Vb = 0) is the mirror image of PMOS ON (Vg = 0, Vb = VDD): measuring every
voltage downward from VDD (V' = VDD - V) and flipping the sign of every current gives exactly the
PMOS ON structure, so the PMOS ON physics + ML pipeline is reused unchanged.
Relative errors are identical in both views (|(-p) - (-t)| / |-t| = |p - t| / |t|).
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
K_Q = 8.617333e-5; CURRENTS = ['id', 'ig', 'is', 'ib']
_DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', '..', 'dataset', 'NMOS_ON', 'nmos_on.csv'))
DATA_CSV = os.environ.get('NMOS_ON_CSV', _DEFAULT_DATA)   # override with NMOS_ON_CSV if needed
def mirror(df):
    """NMOS ON -> PMOS-ON-like view. Applying it twice gives back the original."""
    m = df.copy()
    for v in ('vd', 'vs', 'vg', 'vb'): m[v] = df.vdd - df[v]
    for c in CURRENTS: m[c] = -df[c]
    return m
AUG_CSV = os.environ.get('NMOS_ON_AUG_CSV', DATA_CSV.replace('.csv', '_aug.csv'))
AUG_INDEX_OFFSET = 1_000_000          # keeps original and extra row labels distinct

# Extra training samples at gate operating points (dataset/05_augment_inputs.py): split 70/15/15 with the
# same seed; their train/val parts are ADDED to train/val, their test part is kept apart as "test_aug".
# The original 10,000-row split (and therefore the original test set) is unchanged.

def _split3(n, seed):
    i_tr, i_tmp = train_test_split(np.arange(n), test_size=0.30, random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=seed)
    return i_tr, i_va, i_te

def _read(path):
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    return df

def _aug():
    if not os.path.exists(AUG_CSV):
        return None
    a = _read(AUG_CSV); a.index = a.index + AUG_INDEX_OFFSET
    return a

def load_split(seed=42):
    """Mirrored view. train/val = original + extra samples; test = ORIGINAL test rows only."""
    df = _read(DATA_CSV)
    i_tr, i_va, i_te = _split3(len(df), seed)
    tr, va, te = df.iloc[i_tr], df.iloc[i_va], df.iloc[i_te]
    a = _aug()
    if a is not None:
        j_tr, j_va, _ = _split3(len(a), seed)
        tr, va = pd.concat([tr, a.iloc[j_tr]]), pd.concat([va, a.iloc[j_va]])
        df = pd.concat([df, a])
    return mirror(df), mirror(tr), mirror(va), mirror(te)

def load_test_aug(seed=42):
    """Test part of the extra samples (mirrored view), or None if there are none."""
    a = _aug()
    if a is None:
        return None
    return mirror(a.iloc[_split3(len(a), seed)[2]])
def pct_err(t, p): return np.abs(p - t) / np.abs(t) * 100
