"""Shared ML helpers: features + physics-aware target transform. Reusable for any single-FET case.
Targets (all log10 so the model learns smooth functions):
  id -> log10(|id|/|vds|)  (conductance; sign of id = s_id*sign(vds))
  is -> log10(|is|/|vds|)  (same idea)
  ig -> log10(|ig|),  ib -> log10(|ib|)  (sign constant per case, stored in bundle['signs'])"""
import numpy as np, pandas as pd
RAW = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
TARGETS = ["id", "ig", "is", "ib"]
VDS_MIN = 1e-6   # rows with |vds| below this are left out of TRAINING (conductance undefined)
FLOOR = 1e-20

def feats(df):
    X = df[RAW].copy()
    X["vds"] = df.vd - df.vs; X["vsb"] = df.vs - df.vb; X["vgs"] = df.vg - df.vs
    X["vdb"] = df.vd - df.vb; X["vgd"] = df.vg - df.vd
    X["logw"] = np.log(df.w); X["invT"] = 1.0 / (df.temp + 273.15)
    return X

def fit_signs(df):
    """Majority sign per target (id,is: sign(I*vds); ig,ib: sign(I)) + how consistent it is."""
    vds = (df.vd - df.vs).values; s, frac = {}, {}
    for t in TARGETS:
        q = np.sign(df[t].values * (vds if t in ("id", "is") else 1.0)); q = q[q != 0]
        s[t] = 1.0 if (q > 0).mean() >= 0.5 else -1.0; frac[t] = float(max((q > 0).mean(), (q < 0).mean()))
    return s, frac

def forward(df):
    vds = np.maximum(np.abs((df.vd - df.vs).values), VDS_MIN)
    cols = []
    for t in TARGETS:
        a = np.maximum(np.abs(df[t].values), FLOOR)
        cols.append(np.log10(a / vds) if t in ("id", "is") else np.log10(a))
    return np.column_stack(cols)

def inverse(P, df, signs):
    vds = (df.vd - df.vs).values; out = []
    for j, t in enumerate(TARGETS):
        m = 10.0 ** P[:, j]
        out.append(signs[t] * vds * m if t in ("id", "is") else signs[t] * m)
    return np.column_stack(out)

def predict_currents(bundle, df):
    """bundle = joblib.load('models/<case>_best.joblib'); df needs columns RAW -> DataFrame id,ig,is,ib (A)."""
    P = bundle["model"].predict(feats(df))
    return pd.DataFrame(inverse(P, df, bundle["signs"]), columns=TARGETS, index=df.index)


class LGBMulti:
    """One LightGBM regressor per target column; predict() returns an (n, 4) array like the other models."""
    def __init__(self, models): self.models = models
    def predict(self, X): return np.column_stack([m.predict(X) for m in self.models])


FLOORS = {"id": 1e-10, "is": 1e-10, "ig": 1e-15, "ib": 1e-15}   # floors used by the repo's LightGBM notebooks

def evaluate(I_true, I_pred):
    """Per-current metrics. Strict relative error (no floor) and the floored version used in the repo notebooks."""
    from sklearn.metrics import r2_score
    out = {}
    for j, t in enumerate(TARGETS):
        a, p = I_true[:, j], I_pred[:, j]
        rel = np.abs(p - a) / np.maximum(np.abs(a), 1e-18) * 100
        relf = np.abs(p - a) / (np.abs(a) + FLOORS[t]) * 100
        out[t] = dict(R2_lin=r2_score(a, p), med_err_pct=np.median(rel), p95_err_pct=np.percentile(rel, 95),
                      mean_acc_pct=100 - np.mean(np.minimum(rel, 100)),
                      med_err_floor_pct=np.median(relf), p95_err_floor_pct=np.percentile(relf, 95),
                      within5_pct=100 * np.mean(rel <= 5), within10_pct=100 * np.mean(rel <= 10))
    return out


class HybridMulti:
    """Smooth base model (standardise -> polynomial -> ridge, multi-output) + one LightGBM per target on its residual."""
    def __init__(self, base, models): self.base, self.models = base, models
    def predict(self, X): return self.base.predict(X) + np.column_stack([m.predict(X) for m in self.models])
