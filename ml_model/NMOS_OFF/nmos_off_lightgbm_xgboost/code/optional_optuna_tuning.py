"""
OPTIONAL - redo the hyperparameter search with Optuna (this is how the settings in
step1 / step2 were found). You do NOT need to run this; it takes 1-3 hours.

    pip install optuna
    python optional_optuna_tuning.py lightgbm 80
    python optional_optuna_tuning.py xgboost 50

Only the VALIDATION set is used to score a trial; the test set is never touched.
score = mean over Id, Ig, Is, Ib of (median % error + 0.2 x 95th-pct % error)
Each finished trial is printed and appended to OUT_DIR/tuning/optuna_<model>.log;
the best settings so far are kept in OUT_DIR/tuning/best_<model>.json.
"""
import sys
import json
import numpy as np
import optuna
from nmos_off_common import (load_split, fit_physics, long_table, feature_list, sign_rows, sign_target,
                             to_currents, pct_err, out_path, CURRENTS, SEED)

model = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"
n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 80
optuna.logging.set_verbosity(optuna.logging.WARNING)

df, tr, va, te = load_split()
phys = fit_physics(tr)
LT, LV = long_table(tr, phys), long_table(va, phys)
FEAT = feature_list(LT)

if model == "lightgbm":
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=31, random_state=SEED,
                             verbose=-1, n_jobs=2)
    clf.fit(LT.loc[sign_rows(LT), FEAT], sign_target(tr), eval_set=[(LV.loc[sign_rows(LV), FEAT], sign_target(va))],
            callbacks=[lgb.early_stopping(200, verbose=False)])
    start = dict(learning_rate=0.03, num_leaves=15, min_child_samples=20, linear_lambda=1.0, subsample=0.8,
                 colsample_bytree=1.0, reg_lambda=1e-3, max_bin=255)

    def space(t):
        return dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.12, log=True),
                    num_leaves=t.suggest_int("num_leaves", 4, 127, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 5, 100, log=True),
                    linear_lambda=t.suggest_float("linear_lambda", 0.01, 30, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
                    max_bin=t.suggest_categorical("max_bin", [255, 511, 1023]))

    def fit(p):
        reg = lgb.LGBMRegressor(n_estimators=30000, linear_tree=True, subsample_freq=1, random_state=SEED,
                                verbose=-1, n_jobs=2, **p)
        reg.fit(LT[FEAT], LT.y - LT.phys, eval_set=[(LV[FEAT], LV.y - LV.phys)], categorical_feature=["cur"],
                callbacks=[lgb.early_stopping(300, verbose=False)])
        return reg.predict(LV[FEAT], num_iteration=reg.best_iteration_) + LV.phys.values, reg.best_iteration_
else:
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=6, tree_method="hist",
                            random_state=SEED, n_jobs=2, early_stopping_rounds=200, eval_metric="logloss")
    clf.fit(LT.loc[sign_rows(LT), FEAT], sign_target(tr),
            eval_set=[(LV.loc[sign_rows(LV), FEAT], sign_target(va))], verbose=False)
    start = dict(learning_rate=0.03, max_depth=6, min_child_weight=1, subsample=0.8, colsample_bytree=1.0,
                 colsample_bylevel=1.0, reg_lambda=1.0, reg_alpha=1e-6, max_bin=256)

    def space(t):
        return dict(learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                    max_depth=t.suggest_int("max_depth", 3, 12),
                    min_child_weight=t.suggest_float("min_child_weight", 0.05, 20, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                    colsample_bylevel=t.suggest_float("colsample_bylevel", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 1e-6, 1, log=True),
                    max_bin=t.suggest_categorical("max_bin", [256, 512, 1024]))

    def fit(p):
        reg = xgb.XGBRegressor(n_estimators=20000, tree_method="hist", random_state=SEED, n_jobs=2,
                               early_stopping_rounds=300, eval_metric="rmse", **p)
        reg.fit(LT[FEAT], LT.y - LT.phys, eval_set=[(LV[FEAT], LV.y - LV.phys)], verbose=False)
        return reg.predict(LV[FEAT], iteration_range=(0, reg.best_iteration + 1)) + LV.phys.values, reg.best_iteration

SIGN_V = clf.predict(LV.loc[sign_rows(LV), FEAT])
LOG = out_path("tuning", f"optuna_{model}.log")


def objective(trial):
    log_pred, trees = fit(space(trial))
    P = to_currents(log_pred, SIGN_V, va)
    res = {c: (np.median(pct_err(va[c].values, P[c].values)), np.percentile(pct_err(va[c].values, P[c].values), 95))
           for c in CURRENTS}
    for c in CURRENTS:
        trial.set_user_attr(c, [round(float(res[c][0]), 4), round(float(res[c][1]), 3)])
    trial.set_user_attr("trees", int(trees))
    return float(np.mean([m + 0.2 * p for m, p in res.values()]))


def callback(study, t):
    a = t.user_attrs
    line = (f"#{t.number} score {t.value:.4f} " + " ".join(f"{c}:{a[c][0]}/{a[c][1]}" for c in CURRENTS)
            + f" trees {a['trees']} | best {study.best_value:.4f}")
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    json.dump({"best": study.best_params, "best_value": study.best_value, "attrs": study.best_trial.user_attrs},
              open(out_path("tuning", f"best_{model}.json"), "w"), indent=1)


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
study.enqueue_trial(start)                      # trial 0 = the untuned starting point
study.optimize(objective, n_trials=n_trials, callbacks=[callback])
print("Best settings:", study.best_params)
