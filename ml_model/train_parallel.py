"""
train_parallel.py - train the NMOS ON / PMOS ON physics-guided LightGBM models in parallel.

Each (version, seed) LightGBM model is trained in its own process with exactly the same data,
parameters, seed and n_jobs=1 as train_final.py, so the saved models are identical to a sequential
run. Afterwards train_final.py is run with EVAL_ONLY=1: it loads the saved models and writes
metrics.csv, test_predictions.csv and parity_test.png exactly as a full run would.

Usage (from anywhere):
    python ml_model/train_parallel.py nmos_on  [--workers 6]
    python ml_model/train_parallel.py pmos_on  [--workers 6]

Saved models: <case code dir>/../results/model_<v>/{physics_params,feature_list,lgbm_seed*}.joblib
"""
import argparse, os, subprocess, sys, time
# One BLAS/OpenMP thread per process: with several worker processes, numpy/scipy would otherwise each
# start one thread per CPU and the machine thrashes (the least_squares physics fits are BLAS-heavy).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    "nmos_on": dict(code=os.path.join(HERE, "NMOS_ON", "nmos_on_code", "code"), common="non_common",
                    versions={"a": "physics_a", "b": "physics_b", "c": "physics_c", "d": "physics_d"},
                    set_nominal=True),
    "pmos_on": dict(code=os.path.join(HERE, "PMOS_ON", "pmos_on_physics_ml", "code"), common="pon_common",
                    versions={"a": "physics_baseline_a", "b": "physics_baseline_b"},
                    set_nominal=False),
}
CFG = None


def _setup(case):
    """Import the case's modules inside a worker (same state train_final.py would have)."""
    global CFG
    CFG = CASES[case]
    if CFG["code"] not in sys.path:
        sys.path.insert(0, CFG["code"])


def _version(tag, tr):
    import importlib, numpy as np
    import pipeline as PL
    mod = importlib.import_module(CFG["versions"][tag])
    PL.PB = mod
    if CFG["set_nominal"]:   # identical to train_final.run_version (NMOS ON)
        mod.NOM.update(vth=np.median(np.abs(tr.vth0)), ndep=np.median(tr.ndep), toxe=np.median(tr.toxe),
                       u0=np.median(tr.u0), xj=np.median(tr.xj), vdd=np.median(tr.vdd))
    return PL, mod, os.path.join(PL.OUT, f"model_{tag}")


def fit_physics(case, tag):
    _setup(case)
    import importlib, joblib
    common = importlib.import_module(CFG["common"])
    _, tr, _, _ = common.load_split()
    PL, mod, vdir = _version(tag, tr)
    os.makedirs(vdir, exist_ok=True)
    t0 = time.time()
    P = mod.fit_physics(tr)
    joblib.dump(P, os.path.join(vdir, "physics_params.joblib"))
    if CFG["set_nominal"]:
        joblib.dump(dict(mod.NOM), os.path.join(vdir, "nominal.joblib"))
    return f"[{case} {tag}] physics fitted in {time.time() - t0:.0f}s"


def fit_lgbm(case, tag, seed):
    _setup(case)
    import importlib, joblib, lightgbm as lgb
    common = importlib.import_module(CFG["common"])
    _, tr, va, _ = common.load_split()
    PL, mod, vdir = _version(tag, tr)
    P = joblib.load(os.path.join(vdir, "physics_params.joblib"))
    BT, BV = PL.build_table(tr, P), PL.build_table(va, P)
    F = [c for c in BT["L"].columns if c != "y"]
    joblib.dump(F, os.path.join(vdir, "feature_list.joblib"))
    t0 = time.time()
    reg = lgb.LGBMRegressor(**PL.PARAMS, random_state=seed).fit(
        BT["L"][F], BT["t"], eval_set=[(BV["L"][F], BV["t"])], categorical_feature=["cur"],
        callbacks=[lgb.early_stopping(300, verbose=False)])
    joblib.dump(reg, os.path.join(vdir, f"lgbm_seed{seed}.joblib"), compress=3)
    cache = os.path.join(vdir, f"pred_cache_seed{seed}.npz")   # stale cache would bypass the new model
    if os.path.exists(cache):
        os.remove(cache)
    return f"[{case} {tag}] seed {seed}: {reg.best_iteration_} trees, {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=CASES)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    _setup(a.case)
    import pipeline as PL
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(fit_physics, [a.case] * len(CFG["versions"]), list(CFG["versions"])):
            print(msg, flush=True)
        jobs = [(t, s) for t in CFG["versions"] for s in PL.SEEDS]
        futs = [ex.submit(fit_lgbm, a.case, t, s) for t, s in jobs]
        for f in futs:
            print(f.result(), flush=True)
    print(f"[{a.case}] all models trained in {time.time() - t0:.0f}s; evaluating with train_final.py", flush=True)
    env = dict(os.environ, EVAL_ONLY="1")
    subprocess.run([sys.executable, "train_final.py"], cwd=CFG["code"], env=env, check=True)
    print(f"[{a.case}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
