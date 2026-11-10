"""Shared ML helpers for a single-FET case (PMOS ON etc.): features, target transform, metrics, model wrappers.
Targets (log10, smooth):  id,is -> log10(|I|/|vds|) ;  ig,ib -> log10(|I|).  Signs are constant per case (stored in bundle)."""
import numpy as np, pandas as pd

RAW = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
TARGETS = ["id", "ig", "is", "ib"]
VDS_MIN = 1e-6      # rows with |vds| below this are left out of TRAINING only (conductance undefined)
FLOOR = 1e-20
KB_Q = 8.617333e-5  # k/q in V/K


def feats(df):
    """12 raw columns + voltage differences + physics features (thermal-voltage scaled, logs of PVT params)."""
    X = df[RAW].copy()
    T = df.temp + 273.15
    vt = KB_Q * T
    X["vds"] = df.vd - df.vs; X["vsb"] = df.vs - df.vb; X["vgs"] = df.vg - df.vs
    X["vdb"] = df.vd - df.vb; X["vgd"] = df.vg - df.vd
    X["logw"] = np.log(df.w); X["invT"] = 1.0 / T
    vov = np.abs(X["vgs"]) - np.abs(df.vth0)           # overdrive magnitude (works for N and P)
    X["vov"] = vov; X["vov_vt"] = vov / vt; X["vds_vt"] = X["vds"] / vt
    X["vth0_vt"] = np.abs(df.vth0) / vt
    X["lnu0"] = np.log(df.u0); X["lntoxe"] = np.log(df.toxe); X["lnndep"] = np.log(df.ndep)
    X["eox"] = np.abs(df.vg - df.vb) / df.toxe * 1e-9  # oxide field proxy
    return X


def fit_signs(df):
    """Majority sign per target (id,is: sign(I*vds); ig,ib: sign(I)) and how consistent it is."""
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
    """bundle = joblib.load('models/<case>_best.joblib'); df needs columns RAW -> DataFrame id,ig,is,ib (A, into terminal)."""
    P = bundle["model"].predict(feats(df))
    return pd.DataFrame(inverse(P, df, bundle["signs"]), columns=TARGETS, index=df.index)


def evaluate(I_true, I_pred):
    """Per-current metrics. Relative error = |ML-SPICE|/|SPICE| (no floor), in %."""
    from sklearn.metrics import r2_score
    out = {}
    for j, t in enumerate(TARGETS):
        a, p = I_true[:, j], I_pred[:, j]
        rel = np.abs(p - a) / np.maximum(np.abs(a), 1e-18) * 100
        out[t] = dict(R2_lin=r2_score(a, p), R2_log=r2_score(np.log10(np.maximum(np.abs(a), 1e-20)),
                                                          np.log10(np.maximum(np.abs(p), 1e-20))),
                      med_err_pct=np.median(rel), p95_err_pct=np.percentile(rel, 95),
                      mean_acc_pct=100 - np.mean(np.minimum(rel, 100)))
    return out


class PerTarget:
    """One fitted regressor per target column; predict() returns (n, 4)."""
    def __init__(self, models): self.models = models
    def predict(self, X): return np.column_stack([m.predict(X) for m in self.models])


class HybridMulti:
    """Smooth base (standardise -> polynomial -> ridge, multi-output) + tree model(s) fitted on the base residual."""
    def __init__(self, base, resid): self.base, self.resid = base, resid
    def predict(self, X): return self.base.predict(X) + self.resid.predict(X)
