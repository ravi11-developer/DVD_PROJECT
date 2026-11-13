"""pon_common.py - data loading, 70/15/15 split (seed 42) and the error function for PMOS ON."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

K_Q = 8.617333e-5                     # Boltzmann constant / electron charge (V per K)
CURRENTS = ['id', 'ig', 'is', 'ib']
DATA_CSV = os.environ.get('PMOS_ON_CSV', 'pmos_on.csv')   # set PMOS_ON_CSV to your file path
SEED = 42

def load_split(seed=SEED):
    """Read the CSV and split it 70/15/15 (train/validation/test) with a fixed seed."""
    df = pd.read_csv(DATA_CSV); df.columns = df.columns.str.strip()
    i_tr, i_tmp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=seed)
    return df, df.iloc[i_tr], df.iloc[i_va], df.iloc[i_te]

def pct_err(true, pred):
    """Relative error in percent: |pred - true| / |true| * 100."""
    return np.abs(pred - true) / np.abs(true) * 100
