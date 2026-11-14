"""pon_common.py - data loading, 70/15/15 split (seed 42) and the error function for PMOS ON."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

K_Q = 8.617333e-5                     # Boltzmann constant / electron charge (V per K)
CURRENTS = ['id', 'ig', 'is', 'ib']
_DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', '..', 'dataset', 'PMOS_ON', 'pmos_on.csv'))
DATA_CSV = os.environ.get('PMOS_ON_CSV', _DEFAULT_DATA)   # override with PMOS_ON_CSV if needed
SEED = 42

AUG_CSV = os.environ.get('PMOS_ON_AUG_CSV', DATA_CSV.replace('.csv', '_aug.csv'))
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

def load_split(seed=SEED):
    """train/val = original + extra samples (70/15/15 each, fixed seed); test = ORIGINAL test rows only."""
    df = _read(DATA_CSV)
    i_tr, i_va, i_te = _split3(len(df), seed)
    tr, va, te = df.iloc[i_tr], df.iloc[i_va], df.iloc[i_te]
    a = _aug()
    if a is not None:
        j_tr, j_va, _ = _split3(len(a), seed)
        tr, va = pd.concat([tr, a.iloc[j_tr]]), pd.concat([va, a.iloc[j_va]])
        df = pd.concat([df, a])
    return df, tr, va, te

def load_test_aug(seed=SEED):
    """Test part of the extra samples, or None if there are none."""
    a = _aug()
    if a is None:
        return None
    return a.iloc[_split3(len(a), seed)[2]]

def pct_err(true, pred):
    """Relative error in percent: |pred - true| / |true| * 100."""
    return np.abs(pred - true) / np.abs(true) * 100
