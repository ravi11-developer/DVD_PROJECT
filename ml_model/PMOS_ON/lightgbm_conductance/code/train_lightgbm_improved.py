"""Lower-error LightGBM variants for ONE case. usage (project root): python3 scripts/train_lightgbm_improved.py pmos_on [path/to/pmos_on.csv]
Variants: poly2_resid / poly3_resid = ridge on polynomial features + LightGBM on the residual (a linear_tree variant was tried and dropped: slow)."""
import sys, os, time, joblib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_lib import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

case = sys.argv[1] if len(sys.argv) > 1 else "pmos_on"
SEED = 42
for d in ["results", "models"]: os.makedirs(d, exist_ok=True)
df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "data/%s.csv" % case)
signs, _ = fit_signs(df)
idx_tr, idx_tmp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=SEED)
idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=SEED)          # same split as the other scripts
va_all, te = df.iloc[idx_va], df.iloc[idx_te]
keep = lambda d: d[np.abs(d.vd - d.vs) >= VDS_MIN]
tr, va = keep(df.iloc[idx_tr]), keep(va_all)
Xtr, ytr, Xva, yva = feats(tr), forward(tr), feats(va), forward(va)

P = dict(objective="regression", learning_rate=0.05, num_leaves=63, max_depth=10, min_child_samples=15, subsample=0.85,
         subsample_freq=1, colsample_bytree=0.85, reg_alpha=1e-3, reg_lambda=1e-2, n_estimators=600, random_state=SEED,
         n_jobs=-1, verbose=-1)

def fit_boost(X, y, Xv, yv, **kw):
    ms = []
    for j in range(y.shape[1]):
        m = lgb.LGBMRegressor(**{**P, **kw}); m.fit(X, y[:, j], eval_set=[(Xv, yv[:, j])], callbacks=[lgb.early_stopping(80, verbose=False)]); ms.append(m)
    return ms

def make_poly(deg):
    base = make_pipeline(StandardScaler(), PolynomialFeatures(deg, include_bias=False), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-3, 3, 7), alpha_per_target=True))
    return base.fit(Xtr, ytr)

variants = {}
for deg in (2, 3):
    t0 = time.time(); base = make_poly(deg)
    ms = fit_boost(Xtr, ytr - base.predict(Xtr), Xva, yva - base.predict(Xva))
    variants["poly%d_resid" % deg] = HybridMulti(base, ms); print("poly%d_resid done %.0fs" % (deg, time.time() - t0), flush=True)

rows = []
for name, m in variants.items():
    for split, d in [("validation", va_all), ("test", te)]:
        r = evaluate(d[TARGETS].values, inverse(m.predict(feats(d)), d, signs))
        for t in TARGETS: rows.append(dict(case=case, model="LightGBM-" + name, split=split, current=t, **r[t]))
res = pd.DataFrame(rows); res.to_csv("results/%s_lightgbm_improved_metrics.csv" % case, index=False)
pd.set_option("display.width", 200)
val = res[res.split == "validation"].groupby("model").med_err_pct.mean().sort_values()
best = val.index[0]; print("\nvalidation mean of median error (%) per variant:\n" + val.round(3).to_string() + "\nBEST on validation:", best)
for metric in ["med_err_pct", "p95_err_pct", "R2_lin", "mean_acc_pct"]:
    print("\nTEST", metric); print(res[res.split == "test"].pivot(index="model", columns="current", values=metric)[TARGETS].round(4).to_string())
for f, lab in [("results/%s_lightgbm_metrics.csv" % case, "LightGBM baseline (previous run)")]:
    if os.path.exists(f):
        b = pd.read_csv(f); b = b[b.split == "test"].set_index("current").loc[TARGETS]
        print("\n%s, TEST median / 95th pct err %%:" % lab, [(round(a, 2), round(c, 1)) for a, c in zip(b.med_err_pct, b.p95_err_pct)])
f = "results/%s_test_best.csv" % case
if os.path.exists(f):
    b = pd.read_csv(f).set_index("current").loc[TARGETS]; print("MLP, TEST median / 95th pct err %:", [(round(a, 2), round(c, 1)) for a, c in zip(b.med_err_pct, b.p95_err_pct)])

mname = best.replace("LightGBM-", "")
joblib.dump(dict(model=variants[mname], name=best, case=case, features=list(feats(df).columns), targets=TARGETS, signs=signs),
            "models/%s_lightgbm_improved.joblib" % case)
Pt = inverse(variants[mname].predict(feats(te)), te, signs); I = te[TARGETS].values
fig, a = plt.subplots(1, 4, figsize=(16, 4))
for j, (ax, t) in enumerate(zip(a, TARGETS)):
    x, y = np.abs(I[:, j]), np.maximum(np.abs(Pt[:, j]), 1e-20); lo, hi = max(min(x.min(), y.min()), 1e-20), max(x.max(), y.max())
    ax.loglog(x, y, ".", ms=2, alpha=.4); ax.plot([lo, hi], [lo, hi], "r-", lw=1)
    ax.set_xlabel("SPICE |%s| (A)" % t); ax.set_ylabel("ML |%s| (A)" % t); ax.set_title("%s %s test (%s)" % (case, t, best)); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/%s_lightgbm_improved_pred_vs_spice.png" % case, dpi=150)

# ---- error analysis of the chosen model on the test set (strict relative error, %) ----
rel = np.abs(Pt - I) / np.maximum(np.abs(I), 1e-18) * 100
groups = {"abs_vds": pd.cut(np.abs(te.vd - te.vs).values, [0, .01, .05, .1, .2, .4, 1.0], include_lowest=True)}
for c in ["temp", "vth0", "toxe", "w", "vdd"]: groups[c] = pd.qcut(te[c].values, 4, duplicates="drop")
rows = []
for g, cut in groups.items():
    for j, t_ in enumerate(TARGETS):
        s = pd.DataFrame({"rel": rel[:, j], "bin": cut}).groupby("bin", observed=True).rel.agg(
            n="size", median="median", p95=lambda x: np.percentile(x, 95)).reset_index()
        s["bin"] = s["bin"].astype(str); s.insert(0, "current", t_); s.insert(0, "variable", g); rows.append(s)
ea = pd.concat(rows); ea.to_csv("results/%s_lightgbm_error_analysis.csv" % case, index=False)
print("\nERROR ANALYSIS of %s on the test set: median / 95th-pct relative error (%%) per bin" % best)
print(ea[ea.current.isin(["id", "ib"])].round(2).to_string(index=False))
