"""non_common.py - NMOS ON data loading (mirrored), 70/15/15 split (seed 42), error function.

NMOS ON (Vg = VDD, Vb = 0) is the mirror image of PMOS ON (Vg = 0, Vb = VDD): measuring every
voltage downward from VDD (V' = VDD - V) and flipping the sign of every current gives exactly the
PMOS ON structure, so the PMOS ON physics + ML pipeline is reused unchanged.
Relative errors are identical in both views (|(-p) - (-t)| / |-t| = |p - t| / |t|).
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
K_Q = 8.617333e-5; CURRENTS = ['id', 'ig', 'is', 'ib']
DATA_CSV = os.environ.get('NMOS_ON_CSV', 'nmos_on.csv')
def mirror(df):
    """NMOS ON -> PMOS-ON-like view. Applying it twice gives back the original."""
    m = df.copy()
    for v in ('vd', 'vs', 'vg', 'vb'): m[v] = df.vdd - df[v]
    for c in CURRENTS: m[c] = -df[c]
    return m
def load_split(seed=42):
    df = pd.read_csv(DATA_CSV); df.columns = df.columns.str.strip()
    i_tr, i_tmp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=seed)
    m = mirror(df)
    return m, m.iloc[i_tr], m.iloc[i_va], m.iloc[i_te]
def pct_err(t, p): return np.abs(p - t) / np.abs(t) * 100
