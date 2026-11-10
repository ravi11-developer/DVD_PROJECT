"""
nmos_off_common.py  -  shared code for the NMOS OFF leakage models
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)

Used by step0 ... step4. You normally do not run this file directly.

What lives here
    1. Paths and the fixed random seed
    2. Loading the CSV and the fixed 70 / 15 / 15 split
    3. The small physics model (fitted on TRAINING rows only)
    4. The input table for the single model (one model -> Id, Ig, Is, Ib)
    5. Error measures and the JSON report block (same layout as the team JSON)
    6. Plots shared by both models (parity, overfitting checks, ...)
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ===========================================================================
# 1. PATHS AND SEED  -  edit the two paths if your folders are different
# ===========================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.abspath(os.path.join(_THIS_DIR, "..", "results"))
_DEFAULT_DATA = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "dataset", "NMOS_OFF", "nmos_off.csv"))
if not os.path.exists(_DEFAULT_DATA):
    _alt = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "Data_Generation_M1", "data", "nmos_off.csv"))
    if os.path.exists(_alt):
        _DEFAULT_DATA = _alt

DATA_CSV = os.environ.get("NMOS_OFF_CSV", _DEFAULT_DATA)
OUT_DIR = os.environ.get("NMOS_OFF_OUT", _DEFAULT_OUT)
SEED = 42                                   # same split every run
CURRENTS = ["id", "ig", "is", "ib"]
K_Q = 8.617333e-5                           # Boltzmann constant / electron charge (V per K)

os.makedirs(OUT_DIR, exist_ok=True)


def out_path(*parts):
    """Path inside OUT_DIR; makes the sub-folder if needed."""
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ===========================================================================
# 2. DATA + SPLIT
# ===========================================================================
def load_split():
    """Read the CSV and split 70 / 15 / 15 with the fixed seed.
    itotal (Id+Ig+Is+Ib, ~0 by KCL) is never used as an input."""
    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip()
    assert df.isnull().sum().sum() == 0, "There are missing values in the CSV"
    tr, tmp = train_test_split(df, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    return df, tr, va, te


# ===========================================================================
# 3. SMALL PHYSICS MODEL  (fitted with least_squares on TRAINING rows only)
# ===========================================================================
def cols(d):
    """Raw columns in convenient units (T in kelvin, tox in nm, others normalised)."""
    return dict(vd=d.vd.values, vs=d.vs.values, w=d.w.values, T=d.temp.values + 273.15,
                tox=d.toxe.values * 1e9, vth=d.vth0.values, nd=d.ndep.values / 1.2e19,
                u0=d.u0.values / 0.018, xj=d.xj.values / 7.2e-9, vdd=d.vdd.values)


def ln_gate(th, V, c):
    """ln of gate-tunnelling current through ONE terminal at voltage V (gate at 0 V)."""
    c0, p, a, a2, t1, t2, k = th
    return (c0 + np.log(c["w"]) + p * np.log(V) + a * V + a2 * V ** 2
            + t1 * c["tox"] + t2 * V * c["tox"] + k * (c["T"] - 300) / 100)


def ln_channel(ph, c):
    """ln of subthreshold channel current (body effect via v_low, DIBL via v_high)."""
    c0, av, bv, eta, n, kT, kn = ph
    vh = np.maximum(c["vd"], c["vs"]); vl = np.minimum(c["vd"], c["vs"]); vt = K_Q * c["T"]
    return (c0 + np.log(c["w"]) + (-av * vl - bv * c["vth"] + eta * vh) / (n * vt)
            + np.log1p(-np.exp(-(vh - vl) / vt) + 1e-300)
            + kT * np.log(c["T"] / 300) + kn * np.log(c["nd"]))


def ln_bulk(th, c):
    """ln of bulk current: reverse-biased drain/source junctions + a field-driven term."""
    c0, eg, n, g0, ga, gb, gt, gw, cw = th
    vt = K_Q * c["T"]
    isat = np.exp(c0 + cw * np.log(c["w"] / 1e-7) + 3 * np.log(c["T"] / 300)
                  - eg * (1 / vt - 1 / (K_Q * 300)))
    junction = isat * ((1 - np.exp(-c["vd"] / (n * vt))) + (1 - np.exp(-c["vs"] / (n * vt))))
    field = (np.exp(g0 + gw * np.log(c["w"] / 1e-7) + gt * (c["T"] - 300) / 100)
             * (np.exp(ga * c["vd"] + gb * c["vd"] ** 2) + np.exp(ga * c["vs"] + gb * c["vs"] ** 2)))
    return np.log(junction + field)


def fit_physics(tr):
    """Fit the 3 small physics models (7 + 7 + 9 numbers) on training rows."""
    c = cols(tr)
    y_g = np.log(np.abs(tr.ig.values))
    th_g = least_squares(lambda th: np.log(np.exp(ln_gate(th, c["vd"], c))
                                           + np.exp(ln_gate(th, c["vs"], c))) - y_g,
                         [-10, 2, 1, 0, -2, 0, 0]).x
    dF = np.abs(np.exp(ln_gate(th_g, c["vd"], c)) - np.exp(ln_gate(th_g, c["vs"], c)))
    y_d = np.log(np.abs(tr.id.values - tr["is"].values))
    th_c = least_squares(lambda ph: np.log(2 * np.exp(ln_channel(ph, c)) + dF) - y_d,
                         [-5, 1, 1, 0.05, 1.3, 1, 0], max_nfev=5000).x
    y_b = np.log(-tr.ib.values)
    th_b = least_squares(lambda th: ln_bulk(th, c) - y_b,
                         [-33, 1.1, 1, -40, 5, 0, 0, 1, 0], max_nfev=20000).x
    return dict(gate=th_g, channel=th_c, bulk=th_b)


# ===========================================================================
# 4. INPUT TABLE FOR THE SINGLE MODEL
#    Every SPICE row appears 4 times (once per current). The column "cur"
#    (0=Id, 1=Ig, 2=Is, 3=Ib) tells the model which current to give.
#    Target = log10|I| minus the physics log10|I|  (the model learns the correction).
# ===========================================================================
def base_table(d, phys):
    vd, vs = d.vd.values, d.vs.values
    T = d.temp.values + 273.15; vt = K_Q * T
    vh, vl = np.maximum(vd, vs), np.minimum(vd, vs); avds = vh - vl
    c = cols(d)
    X = pd.DataFrame({
        "vdd": d.vdd, "temp": d.temp, "log_w": np.log10(d.w), "vth0": d.vth0, "toxe": d.toxe,
        "ndep": d.ndep, "u0": d.u0, "xj": d.xj,
        "v_high": vh, "v_low": vl, "abs_vds": avds,
        "inv_vt": 1 / vt, "vth_vt": d.vth0.values / vt, "vds_vt": avds / vt,
        "log1m": np.log10(1 - np.exp(-avds / vt) + 1e-30), "vh_vt": vh / vt, "vl_vt": vl / vt,
        "log_tox": np.log10(d.toxe), "vh2": vh ** 2, "vl2": vl ** 2,
        "vh_tox": vh / d.toxe.values * 1e-9, "vl_tox": vl / d.toxe.values * 1e-9,
        "log_T": np.log10(T), "vd": vd, "vs": vs, "vds": vd - vs,
    }, index=d.index)
    log_ib = ln_bulk(phys["bulk"], c) / np.log(10)
    X["log_ibphys"] = log_ib
    X["ed"] = 1 - np.exp(-vd / vt); X["es"] = 1 - np.exp(-vs / vt)
    X["vd2"] = vd ** 2; X["vs2"] = vs ** 2
    X["vd_tox"] = vd / c["tox"]; X["vs_tox"] = vs / c["tox"]
    X["log_vd"] = np.log10(vd); X["log_vs"] = np.log10(vs)
    Fd = np.exp(ln_gate(phys["gate"], vd, c)); Fs = np.exp(ln_gate(phys["gate"], vs, c))
    C = np.exp(ln_channel(phys["channel"], c)); sg = np.sign(vd - vs)
    X["log_Fd"] = np.log10(Fd); X["log_Fs"] = np.log10(Fs); X["log_C"] = np.log10(C)
    base = {"id": np.log10(np.abs(sg * C + Fd) + 1e-40),     # physics |Id|
            "ig": np.log10(Fd + Fs),                          # physics |Ig|
            "is": np.log10(np.abs(-sg * C + Fs) + 1e-40),    # physics |Is|
            "ib": log_ib}                                     # physics |Ib|
    return X, base


def long_table(d, phys):
    X, base = base_table(d, phys); parts = []
    for k, cur in enumerate(CURRENTS):
        P = X.copy(); P["cur"] = k
        for j, cc in enumerate(CURRENTS):
            P["is_" + cc] = float(j == k)
        P["phys"] = base[cur]
        P["y"] = np.log10(np.abs(d[cur].values))
        parts.append(P)
    return pd.concat(parts, ignore_index=True)


def feature_list(L):
    return [c for c in L.columns if c != "y"]


def sign_rows(L):
    """Rows of the long table that belong to Id or Is (the currents whose sign changes)."""
    return L.cur.isin([0, 2]).values


def sign_target(d):
    return np.concatenate([(d.id.values > 0), (d["is"].values > 0)]).astype(int)


def to_currents(log_mag, sign_pred, d):
    """Turn predicted log10|I| (long order) + predicted signs into Id, Ig, Is, Ib in amperes."""
    n = len(d); mag = 10 ** log_mag; out = {}
    s_id, s_is = sign_pred[:n], sign_pred[n:]
    for k, cur in enumerate(CURRENTS):
        m = mag[k * n:(k + 1) * n]
        if cur in ("ig", "ib"):
            out[cur] = -m                                   # always negative in NMOS OFF
        else:
            s = s_id if cur == "id" else s_is
            out[cur] = np.where(s == 1, 1.0, -1.0) * m
    return pd.DataFrame(out, index=d.index)


# ===========================================================================
# 5. ERROR MEASURES + JSON BLOCK (same layout as the team's JSON file)
#    per-row accuracy = 100 - |pred - SPICE| / |SPICE| * 100, floored at 0
# ===========================================================================
def pct_err(t, p):
    return np.abs(p - t) / np.abs(t) * 100


def metrics_block(true_df, pred_df):
    rows, ma, md = [], [], []
    for c in CURRENTS:
        t = true_df[c].values; p = pred_df[c].values; e = pct_err(t, p)
        acc = 100 - np.minimum(e, 100)
        rows.append({
            "Terminal": c.upper(),
            "Mean Accuracy (%)": f"{acc.mean():.2f}%",
            "Median Accuracy (%)": f"{100 - np.median(e):.2f}%",
            "Median Error (%)": f"{np.median(e):.3f}%",
            "95th-Pct Error (%)": f"{np.percentile(e, 95):.2f}%",
            "Within 1% Tol (%)": f"{(e < 1).mean() * 100:.1f}%",
            "Within 5% Tol (%)": f"{(e < 5).mean() * 100:.1f}%",
            "Within 10% Tol (%)": f"{(e < 10).mean() * 100:.1f}%",
            "R2 (Linear)": round(float(r2_score(t, p)), 4),
            "R2 (Log)": round(float(r2_score(np.log10(np.abs(t)), np.log10(np.abs(p)))), 4),
            "Sign Correct (%)": f"{(np.sign(p) == np.sign(t)).mean() * 100:.2f}%",
            "MAE (A)": f"{np.mean(np.abs(p - t)):.3e}",
            "RMSE (A)": f"{np.sqrt(np.mean((p - t) ** 2)):.3e}",
        })
        ma.append(acc.mean()); md.append(100 - np.median(e))
    return {"mean_accuracy": round(float(np.mean(ma)), 4),
            "median_accuracy": round(float(np.mean(md)), 4), "metrics": rows}


def flat_errors(true_df, pred_df):
    """Plain numbers per current (for tables and plots)."""
    out = {}
    for c in CURRENTS:
        t = true_df[c].values; p = pred_df[c].values; e = pct_err(t, p)
        out[c] = dict(median=float(np.median(e)), p95=float(np.percentile(e, 95)),
                      r2_log=float(r2_score(np.log10(np.abs(t)), np.log10(np.abs(p)))),
                      r2_lin=float(r2_score(t, p)), within1=float((e < 1).mean() * 100),
                      sign=float((np.sign(p) == np.sign(t)).mean() * 100))
    return out


def overfit_check(err_by_split, limit=1.5):
    """Ratio validation / train and test / train of the median and p95 error."""
    rows = []
    for c in CURRENTS:
        tr_, va_, te_ = (err_by_split[s][c] for s in ("train", "validation", "test"))
        r_med = va_["median"] / tr_["median"]; r_p95 = va_["p95"] / tr_["p95"]
        rows.append({"Terminal": c.upper(),
                     "Train median (%)": round(tr_["median"], 4), "Validation median (%)": round(va_["median"], 4),
                     "Test median (%)": round(te_["median"], 4),
                     "Val/Train median ratio": round(r_med, 2),
                     "Train p95 (%)": round(tr_["p95"], 3), "Validation p95 (%)": round(va_["p95"], 3),
                     "Test p95 (%)": round(te_["p95"], 3),
                     "Val/Train p95 ratio": round(r_p95, 2),
                     "Verdict": ("OK - no sign of overfitting" if r_med <= limit
                                 else f"WARNING - validation error is {r_med:.1f}x train error")})
    return rows


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# ===========================================================================
# 6. PLOTS
# ===========================================================================
SPLIT_COL = {"train": "#7F7F7F", "validation": "#1F77B4", "test": "#D62728"}
LABEL = {"id": "$I_d$", "ig": "$I_g$", "is": "$I_s$", "ib": "$I_b$"}


def plot_parity(true_df, pred_df, title, path, extra=None):
    """Predicted vs SPICE on log axes. `extra` = (true_train, pred_train) to overlay in grey."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, c in zip(axes.flat, CURRENTS):
        if extra is not None:
            ax.scatter(np.abs(extra[0][c]), np.abs(extra[1][c]), s=4, alpha=0.15,
                       color=SPLIT_COL["train"], label="train")
        t = np.abs(true_df[c].values); p = np.abs(pred_df[c].values)
        ax.scatter(t, p, s=6, alpha=0.5, color=SPLIT_COL["test"], label="test")
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        e = pct_err(true_df[c].values, pred_df[c].values)
        r2 = r2_score(np.log10(t), np.log10(p))
        ax.set_title(f"{LABEL[c]}:  R²(log) = {r2:.4f},  median err = {np.median(e):.3f}%")
        ax.set_xlabel(f"SPICE |{LABEL[c]}| (A)"); ax.set_ylabel(f"Predicted |{LABEL[c]}| (A)")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_error_vs_current(true_df, pred_df, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, c in zip(axes.flat, CURRENTS):
        t = true_df[c].values; e = pct_err(t, pred_df[c].values)
        ax.scatter(np.abs(t), e, s=6, alpha=0.4, color=SPLIT_COL["test"])
        ax.axhline(1, color="g", ls="--", lw=1, label="1% target")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"SPICE |{LABEL[c]}| (A)"); ax.set_ylabel("error (%)")
        ax.set_title(LABEL[c]); ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_split_bars(err_by_split, title, path):
    """Median and p95 error on train / validation / test, side by side.
    Similar bars = the model generalises; train much lower than val/test = overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(CURRENTS)); w = 0.26
    for ax, key, name in ((axes[0], "median", "Median error (%)"), (axes[1], "p95", "95th-percentile error (%)")):
        for i, sp in enumerate(("train", "validation", "test")):
            v = [err_by_split[sp][c][key] for c in CURRENTS]
            b = ax.bar(x + (i - 1) * w, v, w, color=SPLIT_COL[sp], label=sp)
            ax.bar_label(b, fmt="%.3g", fontsize=7, padding=1)
        ax.axhline(1, color="g", ls="--", lw=1, label="1% target")
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CURRENTS])
        ax.set_yscale("log"); ax.set_title(name); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_error_cdf(true_by_split, pred_by_split, title, path):
    """Share of rows below each error level, per split. Curves on top of each other = no overfitting."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, c in zip(axes.flat, CURRENTS):
        for sp in ("train", "validation", "test"):
            e = np.sort(pct_err(true_by_split[sp][c].values, pred_by_split[sp][c].values))
            ax.plot(e, np.arange(1, len(e) + 1) / len(e) * 100, color=SPLIT_COL[sp], label=sp, lw=1.5)
        ax.axvline(1, color="g", ls="--", lw=1, label="1% error")
        ax.axhline(95, color="k", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_xlim(1e-3, 1e3); ax.set_ylim(0, 101)
        ax.set_xlabel("absolute % error"); ax.set_ylabel("% of rows with error below x")
        ax.set_title(LABEL[c]); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_learning_curve(curve, best_iter, title, path):
    """curve = dict(iters=[...], train=[...], validation=[...], test=[...]) of mean median-% error
    over the four currents. Validation flattening while train keeps dropping = start of overfitting;
    early stopping cuts at the dashed line."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for sp in ("train", "validation", "test"):
        ax.plot(curve["iters"], curve[sp], "-o", ms=3, color=SPLIT_COL[sp], label=sp)
    ax.axvline(best_iter, color="k", ls="--", lw=1, label=f"early stopping: {best_iter} trees")
    ax.axhline(1, color="g", ls=":", lw=1, label="1% target")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of trees"); ax.set_ylabel("median % error (mean of Id, Ig, Is, Ib)")
    ax.set_title(title); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def checkpoints(best_iter, n=25):
    pts = np.unique(np.round(np.geomspace(10, max(best_iter, 11), n)).astype(int))
    return [int(p) for p in pts if p <= best_iter]


# ===========================================================================
# 7. ONE FUNCTION THAT WRITES EVERYTHING FOR A TRAINED MODEL
#    predict_log(L, n_trees=None) -> predicted log10|I| for a long table
#    predict_sign(L_signrows)     -> 1 / 0 for Id, Is rows
# ===========================================================================
def report_model(name, key, splits, tables, predict_log, predict_sign, best_iter,
                 settings, importance=None, phys=None):
    folder = name.lower().replace(" ", "_")
    preds = {}
    for sp, d in splits.items():
        L = tables[sp]
        preds[sp] = to_currents(predict_log(L), predict_sign(L[sign_rows(L)]), d)
    errs = {sp: flat_errors(splits[sp], preds[sp]) for sp in splits}

    # ---- JSON in the team format (+ train/validation blocks for the overfitting check)
    report = {"pipeline": settings["pipeline"],
              "case": "NMOS OFF (Vg = 0, Vb = 0), 22nm_MGK, VDD = 0.9 V",
              "dataset": f"{sum(len(d) for d in splits.values())} rows, split 70/15/15 (seed 42): "
                         f"train {len(splits['train'])}, validation {len(splits['validation'])}, test {len(splits['test'])}",
              "features_count": settings["features_count"],
              "physics_baseline": "compact leakage model (gate tunnelling per terminal, subthreshold channel "
                                  "current, junction/bulk current) fitted on training rows only; the ML model "
                                  "learns log10|I| minus the physics log10|I|",
              "accuracy_definition": "per-row accuracy = 100 - |pred - SPICE| / |SPICE| * 100, floored at 0; "
                                     "median accuracy = 100 - median error",
              "hyperparameters": settings["params"], "tuning": settings["tuning"],
              "trees_used": int(best_iter)}
    for sp in ("train", "validation", "test"):
        report[f"{key}_{sp}"] = metrics_block(splits[sp], preds[sp])
    report["overfit_check"] = overfit_check(errs)
    if phys is not None:
        report["physics_fit"] = {k: [float(x) for x in v] for k, v in phys.items()}
    save_json(report, out_path(folder, f"{folder}_report.json"))

    rows = []
    for sp in ("train", "validation", "test"):
        for m in report[f"{key}_{sp}"]["metrics"]:
            rows.append({"Split": sp, **m})
    pd.DataFrame(rows).to_csv(out_path(folder, f"{folder}_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(report["overfit_check"]).to_csv(out_path(folder, f"{folder}_overfit_check.csv"),
                                                 index=False, encoding="utf-8-sig")

    # ---- learning curve: error vs number of trees on all three splits
    its = checkpoints(best_iter)
    curve = {"iters": its, "train": [], "validation": [], "test": []}
    for k in its:
        for sp, d in splits.items():
            L = tables[sp]
            P = to_currents(predict_log(L, k), predict_sign(L[sign_rows(L)]), d)
            curve[sp].append(float(np.mean([np.median(pct_err(d[c].values, P[c].values)) for c in CURRENTS])))
    save_json(curve, out_path(folder, "learning_curve.json"))
    plot_learning_curve(curve, best_iter, f"{name}: learning curve (overfitting check)",
                        out_path(folder, "overfit_learning_curve.png"))

    # ---- other plots
    plot_split_bars(errs, f"{name}: train vs validation vs test error (overfitting check)",
                    out_path(folder, "overfit_train_val_test_bars.png"))
    plot_error_cdf(splits, preds, f"{name}: error distribution per split (overfitting check)",
                   out_path(folder, "overfit_error_cdf.png"))
    plot_parity(splits["test"], preds["test"], f"{name}: predicted vs SPICE (test set, train in grey)",
                out_path(folder, "parity_test.png"), extra=(splits["train"], preds["train"]))
    plot_error_vs_current(splits["test"], preds["test"], f"{name}: error vs current size (test set)",
                          out_path(folder, "error_vs_current_test.png"))
    if importance is not None:
        names, vals = importance
        order = np.argsort(vals)[-20:]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(np.array(names)[order], np.array(vals)[order], color="#1F77B4")
        ax.set_title(f"{name}: top-20 feature importance (gain)")
        fig.tight_layout(); fig.savefig(out_path(folder, "feature_importance.png"), dpi=200); plt.close(fig)

    # ---- console summary
    print(f"\n{name} - errors per split")
    for sp in ("train", "validation", "test"):
        print(f"  {sp:10s} " + " | ".join(
            f"{c.upper()} med {errs[sp][c]['median']:.3f}% p95 {errs[sp][c]['p95']:.2f}% R2log {errs[sp][c]['r2_log']:.4f}"
            for c in CURRENTS))
    print("  overfit check: " + "; ".join(f"{r['Terminal']} val/train {r['Val/Train median ratio']}"
                                          for r in report["overfit_check"]))
    return report, errs
