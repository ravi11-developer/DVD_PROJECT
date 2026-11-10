"""
nmos_on_common.py  -  Shared code for NMOS ON conduction & leakage models
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)
Case: NMOS ON (Vg = VDD, Vb = 0)

Used by step0 ... step4.
What lives here:
    1. Paths and fixed random seed (42).
    2. Loading the CSV and fixed 70 / 15 / 15 split.
    3. The analytical physics baseline model (fitted on TRAINING rows only).
    4. Input feature table generation and target conductance transformation.
    5. Smooth polynomial baseline (degree 2 + RidgeCV).
    6. Standard SPICE error metrics, noise floor handling, and team JSON reporting.
    7. Complete diagnostic plots (parity, overfitting checks, CDF, learning curve).
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ===========================================================================
# 1. PATHS AND SEED
# ===========================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.abspath(os.path.join(_THIS_DIR, "..", "results"))
_candidates = [
    os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "dataset", "NMOS_ON", "nmos_on.csv")),
    os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "Data_Generation_M1", "data", "nmos_on.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "dataset", "NMOS_ON", "nmos_on.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "Data_Generation_M1", "data", "nmos_on.csv")),
]
_DEFAULT_DATA = next((p for p in _candidates if os.path.exists(p)), _candidates[0])

DATA_CSV = os.environ.get("NMOS_ON_CSV", _DEFAULT_DATA)
OUT_DIR = os.environ.get("NMOS_ON_OUT", _DEFAULT_OUT)
SEED = 42
CURRENTS = ["id", "ig", "is", "ib"]
K_Q = 8.617333e-5                           # Boltzmann constant / electron charge (V/K)

# SPICE standard operational noise floors (prevents artificial division by zero when Vd ≈ Vs)
FLOOR = {
    "id": 1e-10,   # 100 pA for conduction current (0.0001% of full scale ~100 uA)
    "ig": 1e-15,   # 1 fA for gate leakage
    "is": 1e-10,   # 100 pA for conduction current
    "ib": 1e-15    # 1 fA for substrate leakage
}

os.makedirs(OUT_DIR, exist_ok=True)


def out_path(*parts):
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ===========================================================================
# 2. DATA LOADING & 70 / 15 / 15 SPLIT
# ===========================================================================
def load_split():
    """Read NMOS ON CSV and split 70 / 15 / 15 with fixed seed 42.
    itotal (Id+Ig+Is+Ib, ~0 by KCL) is never used as a model input."""
    assert os.path.exists(DATA_CSV), f"Dataset file not found at: {DATA_CSV}"
    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip()
    if "itotal" in df.columns:
        df = df.drop(columns=["itotal"])
    assert df.isnull().sum().sum() == 0, "Missing values found in dataset"
    tr, tmp = train_test_split(df, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    return df, tr, va, te


# ===========================================================================
# 3. SEMICONDUCTOR PHYSICS BASELINE (Fitted on TRAINING rows only)
# ===========================================================================
def cols(d):
    return dict(
        vd=d.vd.values, vs=d.vs.values, vg=d.vg.values, vb=d.vb.values,
        w=d.w.values, T=d.temp.values + 273.15,
        tox=d.toxe.values * 1e9, vth=d.vth0.values,
        nd=d.ndep.values / 1.2e19, u0=d.u0.values / 0.018,
        xj=d.xj.values / 7.2e-9, vdd=d.vdd.values
    )


def ln_gate_fn(th, V, c):
    c0, p, a_tox, k_t = th
    v_diff = np.maximum(c["vdd"] - V, 0.0) + 0.02
    return c0 + np.log(c["w"]) + p * np.log(v_diff) - a_tox * c["tox"] + k_t * (c["T"] - 300.0) / 100.0


def ln_gate_tot(th, c):
    c0, p, a_tox, k_t, c_c = th
    sub_th = [c0, p, a_tox, k_t]
    fd = np.exp(ln_gate_fn(sub_th, c["vd"], c))
    fs = np.exp(ln_gate_fn(sub_th, c["vs"], c))
    fc = np.exp(c0 + c_c + np.log(c["w"]) + p * np.log(c["vdd"]) - a_tox * c["tox"] + k_t * (c["T"] - 300.0) / 100.0)
    return np.log(np.maximum(fd + fs + fc, 1e-40))


def ln_chan_fn(th, c):
    c0, pw, pov, pvds, pt, esat = th
    vh = np.maximum(c["vd"], c["vs"])
    vl = np.minimum(c["vd"], c["vs"])
    avds = vh - vl
    vgs = c["vdd"] - vl
    vov = np.maximum(vgs - c["vth"], 0.01)
    esat_safe = np.maximum(esat, 0.05)
    vds_eff = avds / (1.0 + avds / esat_safe)
    return (c0 + pw * np.log(c["w"]) + pov * np.log(vov) + pvds * np.log(vds_eff + 1e-5)
            - pt * np.log(c["T"] / 300.0) + np.log(np.maximum(c["u0"], 1e-3)))


def ln_bulk_improved(th, c, pred_c, pred_g):
    """3-component bulk model: Reverse pn-junction + Gate-to-bulk oxide tunneling + Impact ionization."""
    c_junc, eg, c_gb, c_ii, beta = th
    vt = K_Q * c["T"]
    vh = np.maximum(c["vd"], c["vs"])
    vl = np.minimum(c["vd"], c["vs"])
    avds = vh - vl

    # 1. Reverse pn junction
    isat = np.exp(c_junc + np.log(c["w"] / 1e-7) + 3.0 * np.log(c["T"] / 300.0) - eg * (1.0 / vt - 1.0 / (K_Q * 300.0)))
    junc = isat * ((1.0 - np.exp(-c["vd"] / (1.5 * vt))) + (1.0 - np.exp(-c["vs"] / (1.5 * vt))))
    junc = np.maximum(junc, 0.0)

    # 2. Gate-to-bulk tunneling
    I_gb = np.exp(c_gb) * pred_g

    # 3. Impact ionization
    beta_safe = np.maximum(np.abs(beta), 0.05)
    ii = np.exp(c_ii) * pred_c * (avds / 0.9) * np.exp(-beta_safe / (avds + 0.05))
    ii = np.maximum(ii, 0.0)

    return np.log(np.maximum(junc + I_gb + ii, 1e-40))


def fit_physics(tr):
    """Fit semiconductor physics models on training rows only."""
    c_tr = cols(tr)

    # 1. Gate current
    y_g = np.log(np.maximum(tr.ig.values, 1e-20))
    th_g = least_squares(lambda th: ln_gate_tot(th, c_tr) - y_g, [-12.0, 1.5, 2.5, 0.1, 0.5], max_nfev=3000).x
    pred_g_tr = np.exp(ln_gate_tot(th_g, c_tr))

    # 2. Channel current
    I_chan_true = np.abs(tr.id.values - tr["is"].values) / 2.0
    y_c = np.log(np.maximum(I_chan_true, 1e-20))
    th_c = least_squares(lambda th: ln_chan_fn(th, c_tr) - y_c, [-2.0, 1.0, 1.5, 1.0, 1.5, 0.5], max_nfev=4000).x
    pred_c_tr = np.exp(ln_chan_fn(th_c, c_tr))

    # 3. Bulk current
    y_b = np.log(np.maximum(-tr.ib.values, 1e-25))
    th_b = least_squares(lambda th: ln_bulk_improved(th, c_tr, pred_c_tr, pred_g_tr) - y_b,
                         [-30.0, 0.5, -7.0, -11.0, 1.4], max_nfev=5000).x

    return dict(gate=th_g, channel=th_c, bulk=th_b)


def get_phys_preds(d, phys):
    c = cols(d)
    pg = np.exp(ln_gate_tot(phys["gate"], c))
    pc = np.exp(ln_chan_fn(phys["channel"], c))
    pb = np.exp(ln_bulk_improved(phys["bulk"], c, pc, pg))
    return pd.DataFrame({
        "log_phys_g": np.log10(np.maximum(pg, 1e-30)),
        "log_phys_c": np.log10(np.maximum(pc, 1e-30)),
        "log_phys_b": np.log10(np.maximum(pb, 1e-30)),
    }, index=d.index)


# ===========================================================================
# 4. FEATURE ENGINEERING & CONDUCTANCE TARGET TRANSFORMATION
# ===========================================================================
RAW = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]


def make_features(d, phys_feats):
    X = d[RAW].copy()
    vd, vs = d.vd.values, d.vs.values
    avds = np.abs(vd - vs)
    vds = vd - vs
    T_K = d.temp.values + 273.15
    vt = K_Q * T_K
    vh = np.maximum(vd, vs)
    vl = np.minimum(vd, vs)
    vgs = d.vdd.values - vl
    vgd = d.vdd.values - vh
    vov = np.maximum(vgs - d.vth0.values, 0.005)

    X["vds"] = vds
    X["abs_vds"] = avds
    X["v_high"] = vh
    X["v_low"] = vl
    X["vgs"] = vgs
    X["vgd"] = vgd
    X["vov"] = vov
    X["inv_vt"] = 1.0 / vt
    X["vth_vt"] = d.vth0.values / vt
    X["vds_vt"] = avds / vt
    X["log_w"] = np.log10(d.w.values)
    X["log_tox"] = np.log10(d.toxe.values)
    X["vh_tox"] = vh / d.toxe.values * 1e-9
    X["vl_tox"] = vl / d.toxe.values * 1e-9
    X["vds_tox"] = avds / d.toxe.values * 1e-9
    X["field_ox"] = d.vdd.values / d.toxe.values * 1e-9
    X["wl_ratio"] = d.w.values / 22e-9
    X["inv_avds"] = 1.0 / (avds + 0.02)
    X["vov_avds"] = vov * avds
    X["log_w_vov"] = np.log10(d.w.values) * vov
    X["exp_avds"] = np.exp(-1.0 / (avds + 0.05))

    # Physics predictions
    X["log_phys_g"] = phys_feats["log_phys_g"]
    X["log_phys_c"] = phys_feats["log_phys_c"]
    X["log_phys_b"] = phys_feats["log_phys_b"]
    return X


def make_targets(d):
    """Targets: Id and Is as channel conductance log10(|I| / (|Vds| + 1e-6));
    Ig as log10(Ig); Ib as log10(|Ib|)."""
    avds = np.abs(d.vd.values - d.vs.values) + 1e-6
    y = np.zeros((len(d), 4))
    y[:, 0] = np.log10(np.maximum(np.abs(d.id.values) / avds, 1e-12))
    y[:, 1] = np.log10(np.maximum(d.ig.values, 1e-25))
    y[:, 2] = np.log10(np.maximum(np.abs(d["is"].values) / avds, 1e-12))
    y[:, 3] = np.log10(np.maximum(np.abs(d.ib.values), 1e-25))
    return y


def fit_poly_base(X_tr, y_tr):
    """Fits Degree-2 Polynomial RidgeCV multi-output base model."""
    poly_base = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 7), alpha_per_target=True)
    )
    poly_base.fit(X_tr, y_tr)
    return poly_base


def reconstruct_currents(y_pred, d, X, sign_model=None):
    """Reconstruct physical currents in Amperes with correct signs."""
    avds = np.abs(d.vd.values - d.vs.values) + 1e-6
    vds = d.vd.values - d.vs.values

    if sign_model is not None:
        s_pred = sign_model.predict(X)
        s_id = np.where(np.abs(vds) > 0.01, np.where(vds >= 0, 1.0, -1.0), np.where(s_pred == 1, 1.0, -1.0))
    else:
        s_id = np.where(vds >= 0, 1.0, -1.0)
    s_is = -s_id

    pred = pd.DataFrame(index=d.index)
    pred["id"] = s_id * avds * (10.0 ** y_pred[:, 0])
    pred["ig"] = 10.0 ** y_pred[:, 1]
    pred["is"] = s_is * avds * (10.0 ** y_pred[:, 2])
    pred["ib"] = - (10.0 ** y_pred[:, 3])
    return pred


# ===========================================================================
# 5. JUDGING METRICS & ACCURACY EVALUATION (Team-Standard Format)
# ===========================================================================
def pct_err(t, p, c):
    return np.abs(p - t) / (np.abs(t) + FLOOR[c]) * 100.0


def metrics_block(true_df, pred_df):
    rows, ma, md = [], [], []
    for c in CURRENTS:
        t = true_df[c].values
        p = pred_df[c].values
        e = pct_err(t, p, c)
        acc = 100.0 - np.minimum(e, 100.0)
        med_err = np.median(e)
        med_acc = 100.0 - med_err

        r2_lin = float(r2_score(t, p))
        t_log = np.log10(np.maximum(np.abs(t), 1e-25))
        p_log = np.log10(np.maximum(np.abs(p), 1e-25))
        r2_log = float(r2_score(t_log, p_log))

        rows.append({
            "Terminal": c.upper(),
            "Mean Accuracy (%)": f"{acc.mean():.2f}%",
            "Median Accuracy (%)": f"{med_acc:.2f}%",
            "Median Error (%)": f"{med_err:.3f}%",
            "95th-Pct Error (%)": f"{np.percentile(e, 95):.2f}%",
            "Within 1% Tol (%)": f"{(e < 1.0).mean() * 100.0:.1f}%",
            "Within 5% Tol (%)": f"{(e < 5.0).mean() * 100.0:.1f}%",
            "Within 10% Tol (%)": f"{(e < 10.0).mean() * 100.0:.1f}%",
            "R2 (Linear)": round(r2_lin, 4),
            "R2 (Log)": round(r2_log, 4),
            "Sign Correct (%)": f"{(np.sign(p) == np.sign(t)).mean() * 100.0:.2f}%",
            "MAE (A)": f"{np.mean(np.abs(p - t)):.3e}",
            "RMSE (A)": f"{np.sqrt(np.mean((p - t) ** 2)):.3e}",
        })
        ma.append(acc.mean())
        md.append(med_acc)

    return {
        "mean_accuracy": round(float(np.mean(ma)), 4),
        "median_accuracy": round(float(np.mean(md)), 4),
        "metrics": rows
    }


def flat_errors(true_df, pred_df):
    out = {}
    for c in CURRENTS:
        t = true_df[c].values
        p = pred_df[c].values
        e = pct_err(t, p, c)
        t_log = np.log10(np.maximum(np.abs(t), 1e-25))
        p_log = np.log10(np.maximum(np.abs(p), 1e-25))
        out[c] = dict(
            median=float(np.median(e)),
            p95=float(np.percentile(e, 95)),
            r2_log=float(r2_score(t_log, p_log)),
            r2_lin=float(r2_score(t, p)),
            within1=float((e < 1.0).mean() * 100.0),
            within5=float((e < 5.0).mean() * 100.0),
            sign=float((np.sign(p) == np.sign(t)).mean() * 100.0)
        )
    return out


def overfit_check(err_by_split, limit=1.5):
    rows = []
    for c in CURRENTS:
        tr_, va_, te_ = (err_by_split[s][c] for s in ("train", "validation", "test"))
        r_med = va_["median"] / max(tr_["median"], 1e-6)
        r_p95 = va_["p95"] / max(tr_["p95"], 1e-6)
        rows.append({
            "Terminal": c.upper(),
            "Train median (%)": round(tr_["median"], 4),
            "Validation median (%)": round(va_["median"], 4),
            "Test median (%)": round(te_["median"], 4),
            "Val/Train median ratio": round(r_med, 2),
            "Train p95 (%)": round(tr_["p95"], 3),
            "Validation p95 (%)": round(va_["p95"], 3),
            "Test p95 (%)": round(te_["p95"], 3),
            "Val/Train p95 ratio": round(r_p95, 2),
            "Verdict": ("OK - generalising perfectly" if r_med <= limit
                        else f"WARNING - validation error is {r_med:.1f}x train error")
        })
    return rows


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# ===========================================================================
# 6. DIAGNOSTIC PLOTTING FUNCTIONS
# ===========================================================================
SPLIT_COL = {"train": "#7F7F7F", "validation": "#1F77B4", "test": "#D62728"}
LABEL = {"id": "$I_d$", "ig": "$I_g$", "is": "$I_s$", "ib": "$I_b$"}


def plot_parity(true_df, pred_df, title, path, extra=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, c in zip(axes.flat, CURRENTS):
        if extra is not None:
            ax.scatter(np.abs(extra[0][c]), np.abs(extra[1][c]), s=4, alpha=0.15,
                       color=SPLIT_COL["train"], label="Train")
        t = np.abs(true_df[c].values)
        p = np.abs(pred_df[c].values)
        ax.scatter(t, p, s=6, alpha=0.5, color=SPLIT_COL["test"], label="Test")
        lo = max(min(t.min(), p.min()), 1e-15)
        hi = max(t.max(), p.max()) * 1.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Ideal y=x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        e = pct_err(true_df[c].values, pred_df[c].values, c)
        t_log = np.log10(np.maximum(t, 1e-25))
        p_log = np.log10(np.maximum(p, 1e-25))
        r2 = r2_score(t_log, p_log)
        ax.set_title(f"{LABEL[c]}: $R^2$(log) = {r2:.4f}, Median Err = {np.median(e):.3f}%", fontsize=10)
        ax.set_xlabel(f"SPICE |{LABEL[c]}| (A)")
        ax.set_ylabel(f"Predicted |{LABEL[c]}| (A)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_error_vs_current(true_df, pred_df, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, c in zip(axes.flat, CURRENTS):
        t = true_df[c].values
        e = pct_err(t, pred_df[c].values, c)
        ax.scatter(np.abs(t), e, s=6, alpha=0.4, color=SPLIT_COL["test"])
        ax.axhline(1.0, color="g", ls="--", lw=1.2, label="1% Target")
        ax.axhline(5.0, color="orange", ls=":", lw=1.0, label="5% Bound")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"SPICE |{LABEL[c]}| (A)")
        ax.set_ylabel("Relative Error (%)")
        ax.set_title(LABEL[c], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_split_bars(err_by_split, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(CURRENTS))
    w = 0.26
    for ax, key, name in ((axes[0], "median", "Median Error (%)"), (axes[1], "p95", "95th-Percentile Error (%)")):
        for i, sp in enumerate(("train", "validation", "test")):
            v = [err_by_split[sp][c][key] for c in CURRENTS]
            b = ax.bar(x + (i - 1) * w, v, w, color=SPLIT_COL[sp], label=sp.capitalize())
            ax.bar_label(b, fmt="%.3g", fontsize=7, padding=1)
        ax.axhline(1.0, color="g", ls="--", lw=1, label="1% Target")
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[c] for c in CURRENTS])
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_error_cdf(splits, preds, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, c in zip(axes.flat, CURRENTS):
        for sp in ("train", "validation", "test"):
            e = np.sort(pct_err(splits[sp][c].values, preds[sp][c].values, c))
            ax.plot(e, np.arange(1, len(e) + 1) / len(e) * 100.0, color=SPLIT_COL[sp], label=sp.capitalize(), lw=1.5)
        ax.axvline(1.0, color="g", ls="--", lw=1.2, label="1% Target")
        ax.axvline(5.0, color="orange", ls=":", lw=1.0, label="5% Threshold")
        ax.axhline(95.0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e2)
        ax.set_ylim(0, 101)
        ax.set_xlabel("Relative Error (%)")
        ax.set_ylabel("% of samples with error < x")
        ax.set_title(LABEL[c], fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_learning_curve(curve, best_iter, title, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for sp in ("train", "validation", "test"):
        ax.plot(curve["iters"], curve[sp], "-o", ms=3, color=SPLIT_COL[sp], label=sp.capitalize())
    ax.axvline(best_iter, color="k", ls="--", lw=1.2, label=f"Optimal ({best_iter} trees)")
    ax.axhline(1.0, color="g", ls=":", lw=1.2, label="1% Target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Median Error (%) [Mean over 4 terminals]")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ===========================================================================
# 7. MODEL EVALUATION & REPORTING ENGINE
# ===========================================================================
def report_hybrid_model(name, key, splits, preds, best_iter, settings,
                        importance=None, phys=None, curve=None):
    folder = key
    errs = {sp: flat_errors(splits[sp], preds[sp]) for sp in splits}

    report = {
        "pipeline": settings["pipeline"],
        "case": "NMOS ON (Vg = VDD, Vb = 0), 22nm_MGK, VDD = 0.9 V",
        "dataset": (f"{sum(len(d) for d in splits.values())} rows, split 70/15/15 (seed 42): "
                    f"train {len(splits['train'])}, validation {len(splits['validation'])}, test {len(splits['test'])}"),
        "features_count": settings["features_count"],
        "physics_baseline": ("Physics-informed polynomial base (degree 2 + RidgeCV) + decoupled residual gradient boosters "
                             "with conductance target formulation for Id, Is and 3-component bulk model for Ib."),
        "accuracy_definition": ("per-row accuracy = 100 - |pred - SPICE| / (|SPICE| + Ifloor) * 100, floored at 0; "
                               "median accuracy = 100 - median error"),
        "noise_floors": FLOOR,
        "hyperparameters": settings["params"],
        "tuning": settings["tuning"],
        "trees_used": int(best_iter)
    }

    for sp in ("train", "validation", "test"):
        report[f"{key}_{sp}"] = metrics_block(splits[sp], preds[sp])
    report["overfit_check"] = overfit_check(errs)
    if phys is not None:
        report["physics_fit"] = {k: [float(x) for x in v] for k, v in phys.items()}
    save_json(report, out_path(folder, f"{folder}_report.json"))

    # CSV Metrics Table
    rows = []
    for sp in ("train", "validation", "test"):
        for m in report[f"{key}_{sp}"]["metrics"]:
            rows.append({"Split": sp.capitalize(), **m})
    pd.DataFrame(rows).to_csv(out_path(folder, f"{folder}_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(report["overfit_check"]).to_csv(out_path(folder, f"{folder}_overfit_check.csv"),
                                                 index=False, encoding="utf-8-sig")

    # Learning Curve
    if curve is not None:
        save_json(curve, out_path(folder, "learning_curve.json"))
        plot_learning_curve(curve, best_iter, f"{name} (NMOS ON): Learning Curve",
                            out_path(folder, "overfit_learning_curve.png"))

    # Diagnostic Plots
    plot_split_bars(errs, f"{name} (NMOS ON): Train vs Validation vs Test Error",
                    out_path(folder, "overfit_train_val_test_bars.png"))
    plot_error_cdf(splits, preds, f"{name} (NMOS ON): Cumulative Error Distribution",
                   out_path(folder, "overfit_error_cdf.png"))
    plot_parity(splits["test"], preds["test"], f"{name} (NMOS ON): Predicted vs. SPICE (Test Set)",
                out_path(folder, "parity_test.png"), extra=(splits["train"], preds["train"]))
    plot_error_vs_current(splits["test"], preds["test"], f"{name} (NMOS ON): Error vs. Current Magnitude",
                          out_path(folder, "error_vs_current_test.png"))

    if importance is not None:
        names, vals = importance
        order = np.argsort(vals)[-20:]
        fig, ax = plt.subplots(figsize=(8.5, 7))
        ax.barh(np.array(names)[order], np.array(vals)[order], color="#1F77B4", edgecolor="black")
        ax.set_title(f"{name} (NMOS ON): Top-20 Feature Importance (Gain)")
        fig.tight_layout()
        fig.savefig(out_path(folder, "feature_importance.png"), dpi=200)
        plt.close(fig)

    print(f"\n{name} — NMOS ON Error Summary Across Splits:")
    for sp in ("train", "validation", "test"):
        summary_str = " | ".join(
            f"{c.upper()} Med {errs[sp][c]['median']:.3f}%, P95 {errs[sp][c]['p95']:.2f}%, R2(lin) {errs[sp][c]['r2_lin']:.4f}"
            for c in CURRENTS
        )
        print(f"  [{sp.capitalize():10s}] {summary_str}")
    print("  Overfitting Check: " + "; ".join(f"{r['Terminal']} val/train ratio: {r['Val/Train median ratio']}"
                                            for r in report["overfit_check"]))
    return report, errs
