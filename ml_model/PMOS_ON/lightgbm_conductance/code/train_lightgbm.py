"""LightGBM for ONE case (same features/targets/split as train_models.py).
usage (from the project root): python3 scripts/train_lightgbm.py pmos_on [path/to/pmos_on.csv]"""
import sys, os, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_lib import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

case = sys.argv[1] if len(sys.argv) > 1 else "pmos_on"
SEED = 42
for d in ["results", "models"]: os.makedirs(d, exist_ok=True)
df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "data/%s.csv" % case)
signs, frac = fit_signs(df)
idx_tr, idx_tmp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=SEED)
idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=SEED)       # same split as train_models.py
tr, va, te = df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]
tr, va = [d[np.abs(d.vd - d.vs) >= VDS_MIN] for d in (tr, va)]   # conductance undefined at Vds = 0
Xtr, ytr, Xva, yva = feats(tr), forward(tr), feats(va), forward(va)
print("train/val/test rows:", len(tr), len(va), len(te))

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, max_depth=12, min_child_samples=15,
              subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_alpha=1e-3, reg_lambda=1e-2,
              n_estimators=1000, random_state=SEED, n_jobs=-1, verbose=-1)
models = []
for j, t in enumerate(TARGETS):
    t0 = time.time()
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(Xtr, ytr[:, j], eval_set=[(Xva, yva[:, j])], callbacks=[lgb.early_stopping(80, verbose=False)])
    models.append(m); print("%s: %d trees, %.0fs" % (t, m.best_iteration_ or PARAMS["n_estimators"], time.time() - t0), flush=True)
model = LGBMulti(models)

rows = []
for name, d in [("validation", df.iloc[idx_va]), ("test", te)]:
    P = inverse(model.predict(feats(d)), d, signs); r = evaluate(d[TARGETS].values, P)
    for t in TARGETS: rows.append(dict(case=case, model="LightGBM", split=name, current=t, **r[t]))
res = pd.DataFrame(rows); res.to_csv("results/%s_lightgbm_metrics.csv" % case, index=False)
pd.set_option("display.width", 200)
for s in ["validation", "test"]:
    print("\n=== LightGBM %s (strict relative error, no floor; last two columns use the repo-notebook floors) ===" % s)
    print(res[res.split == s][["current", "R2_lin", "med_err_pct", "p95_err_pct", "mean_acc_pct", "within5_pct", "within10_pct",
                               "med_err_floor_pct", "p95_err_floor_pct"]].round(3).to_string(index=False))
f = "results/%s_test_best.csv" % case
if os.path.exists(f):
    print("\n=== earlier MLP, test set (for comparison) ==="); print(pd.read_csv(f)[["current", "R2_lin", "med_err_pct", "p95_err_pct", "mean_acc_pct"]].round(3).to_string(index=False))

joblib.dump(dict(model=model, name="LightGBM", case=case, features=list(feats(df).columns), targets=TARGETS, signs=signs),
            "models/%s_lightgbm.joblib" % case)
P = inverse(model.predict(feats(te)), te, signs); I = te[TARGETS].values
fig, a = plt.subplots(1, 4, figsize=(16, 4))
for j, (ax, t) in enumerate(zip(a, TARGETS)):
    x, y = np.abs(I[:, j]), np.maximum(np.abs(P[:, j]), 1e-20); lo, hi = max(min(x.min(), y.min()), 1e-20), max(x.max(), y.max())
    ax.loglog(x, y, ".", ms=2, alpha=.4); ax.plot([lo, hi], [lo, hi], "r-", lw=1)
    ax.set_xlabel("SPICE |%s| (A)" % t); ax.set_ylabel("ML |%s| (A)" % t); ax.set_title("%s %s test (LightGBM)" % (case, t)); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/%s_lightgbm_pred_vs_spice.png" % case, dpi=150)
# where do the errors sit? error vs |Vds| for Id (strict relative error)
rel = np.abs(P[:, 0] - I[:, 0]) / np.maximum(np.abs(I[:, 0]), 1e-18) * 100; av = np.abs(te.vd - te.vs).values
bins = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0]; cut = pd.cut(av, bins, include_lowest=True)
print("\nId median / 95th-pct relative error (%) by |Vds| bin:")
print(pd.DataFrame({"rel": rel, "bin": cut}).groupby("bin", observed=True).rel.agg(n="size", median="median", p95=lambda s: np.percentile(s, 95)).round(2).to_string())
