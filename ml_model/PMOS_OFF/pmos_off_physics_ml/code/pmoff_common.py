"""pmoff_common.py - PMOS OFF data loading (mirrored), 70/15/15 split (seed 42), error function.

PMOS OFF (Vg = Vb = VDD) is the mirror image of NMOS OFF (Vg = Vb = 0): measuring every voltage
downward from VDD (V' = VDD - V), flipping the sign of every current and using |vth0| gives exactly
the NMOS OFF structure, so the NMOS OFF physics + LightGBM pipeline is reused (same idea as
NMOS ON reusing the PMOS ON pipeline). Relative errors are identical in both views.
The split is the same one used by mlp.ipynb (train_test_split on the row index, seed 42).
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

def load_split(seed=SEED):
    df = pd.read_csv(DATA_CSV); df.columns = df.columns.str.strip()
    assert df.isnull().sum().sum() == 0, "There are missing values in the CSV"
    i_tr, i_tmp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=seed)
    m = mirror(df)
    return m, m.iloc[i_tr], m.iloc[i_va], m.iloc[i_te]

def pct_err(t, p): return np.abs(p - t) / np.abs(t) * 100
