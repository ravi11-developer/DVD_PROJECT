"""Train + compare 4 model types for ONE case. usage: python3 scripts/train_models.py pmos_on [path/to/pmos_on.csv]"""
import sys, os, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_lib import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

case = sys.argv[1] if len(sys.argv) > 1 else "pmos_on"
SEED = 42
for d in ["results", "models"]: os.makedirs(d, exist_ok=True)
df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "data/%s.csv" % case)
signs, frac = fit_signs(df)
print("target signs:", signs, "| consistency:", {k: round(v, 5) for k, v in frac.items()})
idx_tr, idx_tmp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=SEED)
idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=SEED)
tr, va, te = df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]
tr_fit = tr[np.abs(tr.vd - tr.vs) >= VDS_MIN]
print("split train/val/test:", len(tr), len(va), len(te), "(train rows used for fit: %d)" % len(tr_fit))
Xtr, ytr = feats(tr_fit), forward(tr_fit)

models = {
 "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
 "ExtraTrees":   ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
 "XGBoost":      MultiOutputRegressor(XGBRegressor(n_estimators=1500, max_depth=7, learning_rate=0.03, subsample=0.9,
                                      colsample_bytree=0.9, n_jobs=2, random_state=SEED)),
 "MLP":          TransformedTargetRegressor(
                    regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256, 128), activation="tanh",
                                            alpha=1e-6, learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                                            validation_fraction=0.1, n_iter_no_change=60, random_state=SEED)),
                    transformer=StandardScaler()),
}

def metrics(m, d):
    I_true = d[TARGETS].values; I_pred = inverse(m.predict(feats(d)), d, signs); out = {}
    for j, t in enumerate(TARGETS):
        rel = np.abs(I_pred[:, j] - I_true[:, j]) / np.maximum(np.abs(I_true[:, j]), 1e-18) * 100
        out[t] = dict(R2_lin=r2_score(I_true[:, j], I_pred[:, j]),
                      R2_log=r2_score(np.log10(np.maximum(np.abs(I_true[:, j]), FLOOR)), np.log10(np.maximum(np.abs(I_pred[:, j]), FLOOR))),
                      med_err_pct=np.median(rel), p95_err_pct=np.percentile(rel, 95),
                      mean_acc_pct=100 - np.mean(np.minimum(rel, 100)))
    return out, I_pred

rows, fitted = [], {}
for name, m in models.items():
    t0 = time.time(); m.fit(Xtr, ytr); fitted[name] = m
    mv, _ = metrics(m, va)
    for t in TARGETS: rows.append(dict(case=case, model=name, current=t, **mv[t]))
    print("%-13s fit %4.0fs | val median err %% (id,ig,is,ib): %s" % (name, time.time() - t0,
          ["%.2f" % mv[t]["med_err_pct"] for t in TARGETS]), flush=True)
val = pd.DataFrame(rows); val.to_csv("results/%s_val_comparison.csv" % case, index=False)
score = val.groupby("model").med_err_pct.mean().sort_values(); best = score.index[0]
print("\nVALIDATION mean (over 4 currents) of median error %:\n" + score.round(3).to_string() + "\nBEST:", best)
for c in ["med_err_pct", "p95_err_pct", "R2_log"]:
    print("\nval", c); print(val.pivot(index="model", columns="current", values=c).round(4).to_string())

mt, P = metrics(fitted[best], te)
tab = pd.DataFrame([dict(case=case, model=best, current=t, **mt[t]) for t in TARGETS])
tab.to_csv("results/%s_test_best.csv" % case, index=False)
print("\nTEST (%s):" % best); print(tab.round(4).to_string(index=False))
joblib.dump(dict(model=fitted[best], name=best, case=case, features=list(feats(df).columns), targets=TARGETS, signs=signs),
            "models/%s_best.joblib" % case)

f, a = plt.subplots(1, 4, figsize=(16, 4)); I = te[TARGETS].values
for j, (ax, t) in enumerate(zip(a, TARGETS)):
    x, y = np.abs(I[:, j]), np.maximum(np.abs(P[:, j]), 1e-20); lo, hi = max(min(x.min(), y.min()), 1e-20), max(x.max(), y.max())
    ax.loglog(x, y, ".", ms=2, alpha=.4); ax.plot([lo, hi], [lo, hi], "r-", lw=1)
    ax.set_xlabel("SPICE |%s| (A)" % t); ax.set_ylabel("ML |%s| (A)" % t); ax.set_title("%s %s test (%s)" % (case, t, best)); ax.grid(alpha=.3)
f.tight_layout(); f.savefig("results/%s_pred_vs_spice.png" % case, dpi=150)
# where are the worst errors? (error analysis input for the report)
rel = np.abs(P[:, 0] - I[:, 0]) / np.maximum(np.abs(I[:, 0]), 1e-18) * 100
w = te.assign(rel_id=rel).sort_values("rel_id", ascending=False).head(5)
print("\nworst 5 test errors on id:"); print(w[["vdd", "temp", "w", "vd", "vs", "vth0", "id", "rel_id"]].to_string(index=False))
print("corr of |vds| with id rel-err (spearman): %.2f" % pd.Series(np.abs(te.vd - te.vs).values).corr(pd.Series(rel), method="spearman"))
