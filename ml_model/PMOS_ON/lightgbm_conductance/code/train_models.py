"""Train + compare 4 tree models (RandomForest, ExtraTrees, XGBoost, LightGBM) for ONE single-FET case.
usage: python3 train_models.py pmos_on path/to/pmos_on.csv
Every model = smooth polynomial-ridge base (captures the smooth physics) + the tree model fitted on the base RESIDUAL.
Split 70/15/15, seed 42. Best model chosen on VALIDATION; test set is used only to report the chosen model.
Outputs: results/<case>_val_comparison.csv, results/<case>_test_all.csv, results/<case>_test_best.csv,
         results/<case>_pred_vs_spice.png, models/<case>_best.joblib   (needs ml_lib.py on the path to load)."""
import sys, os, time, joblib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_lib import *
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=SEED)
va_all, te = df.iloc[idx_va], df.iloc[idx_te]
keep = lambda d: d[np.abs(d.vd - d.vs) >= VDS_MIN]
tr, va = keep(df.iloc[idx_tr]), keep(va_all)
Xtr, ytr, Xva, yva = feats(tr), forward(tr), feats(va), forward(va)

# ---- shared smooth base: poly(2) + ridge (alpha per target) ----
t0 = time.time()
base = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                     RidgeCV(alphas=np.logspace(-3, 3, 7), alpha_per_target=True)).fit(Xtr, ytr)
rtr, rva = ytr - base.predict(Xtr), yva - base.predict(Xva)      # residuals the trees must learn
print("base done %.0fs; base-only median |resid| (log10) per target:" % (time.time() - t0),
      np.median(np.abs(rva), axis=0).round(4), flush=True)

# ---- tree models on the residual ----
def fit_rf(cls, **kw):
    return cls(n_estimators=400, min_samples_leaf=2, max_features=0.7, n_jobs=-1, random_state=SEED, **kw).fit(Xtr, rtr)

def fit_xgb():
    ms = []
    for j in range(rtr.shape[1]):
        m = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=7, min_child_weight=2, subsample=0.85,
                             colsample_bytree=0.85, reg_lambda=1.0, tree_method="hist", n_jobs=-1, random_state=SEED,
                             early_stopping_rounds=100)
        m.fit(Xtr, rtr[:, j], eval_set=[(Xva, rva[:, j])], verbose=False); ms.append(m)
    return PerTarget(ms)

def fit_lgb():
    ms = []
    for j in range(rtr.shape[1]):
        m = lgb.LGBMRegressor(objective="regression", learning_rate=0.03, num_leaves=63, max_depth=10, min_child_samples=10,
                              subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_alpha=1e-3, reg_lambda=1e-2,
                              n_estimators=3000, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr, rtr[:, j], eval_set=[(Xva, rva[:, j])], callbacks=[lgb.early_stopping(100, verbose=False)]); ms.append(m)
    return PerTarget(ms)

fitters = {"RandomForest": lambda: fit_rf(RandomForestRegressor), "ExtraTrees": lambda: fit_rf(ExtraTreesRegressor),
           "XGBoost": fit_xgb, "LightGBM": fit_lgb}
models = {}
for name, f in fitters.items():
    t0 = time.time(); models[name] = HybridMulti(base, f()); print("%-12s trained %.0fs" % (name, time.time() - t0), flush=True)

# ---- evaluate ----
def table(d, split):
    rows = []
    for name, m in models.items():
        r = evaluate(d[TARGETS].values, inverse(m.predict(feats(d)), d, signs))
        for t in TARGETS: rows.append(dict(case=case, model=name, split=split, current=t, **r[t]))
    return pd.DataFrame(rows)
val, tst = table(va_all, "validation"), table(te, "test")
val.to_csv("results/%s_val_comparison.csv" % case, index=False); tst.to_csv("results/%s_test_all.csv" % case, index=False)
pd.set_option("display.width", 200)
for metric in ["med_err_pct", "p95_err_pct", "R2_lin"]:
    print("\nVALIDATION", metric); print(val.pivot(index="model", columns="current", values=metric)[TARGETS].round(4).to_string())
score = val.groupby("model").med_err_pct.mean().sort_values()
print("\nvalidation mean of median error (%):\n" + score.round(3).to_string())
best = score.index[0]; print("BEST on validation:", best)
b = tst[tst.model == best].set_index("current").loc[TARGETS]
b.to_csv("results/%s_test_best.csv" % case)
print("\nTEST (%s):" % best); print(b[["R2_lin", "med_err_pct", "p95_err_pct", "mean_acc_pct"]].round(4).to_string())

joblib.dump(dict(model=models[best], name=best, case=case, features=list(feats(df).columns), targets=TARGETS, signs=signs),
            "models/%s_best.joblib" % case)
Pt = inverse(models[best].predict(feats(te)), te, signs); I = te[TARGETS].values
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for j, (a, t) in enumerate(zip(ax, TARGETS)):
    x, y = np.abs(I[:, j]), np.maximum(np.abs(Pt[:, j]), 1e-20); lo, hi = max(min(x.min(), y.min()), 1e-20), max(x.max(), y.max())
    a.loglog(x, y, ".", ms=2, alpha=.4); a.plot([lo, hi], [lo, hi], "r-", lw=1)
    a.set_xlabel("SPICE |%s| (A)" % t); a.set_ylabel("ML |%s| (A)" % t); a.set_title("%s %s test (%s)" % (case, t, best)); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/%s_pred_vs_spice.png" % case, dpi=150)
