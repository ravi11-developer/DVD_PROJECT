"""
train_parallel.py - train the four transistor models on all CPU cores.

Every unit of work runs in its own single-threaded process, scheduled as soon as its inputs exist:
    physics fit (one per physics version)  ->  LightGBM model (one per version x seed)
                                            ->  prediction cache (train/val/test/test_aug outputs)
Each job does exactly what the case's train_final.py would do (same data, parameters, seeds,
n_jobs=1), so the results are identical to a sequential run. Afterwards:
    NMOS ON: select_blend.py picks the per-current blend weights on the validation set;
    every case: train_final.py with EVAL_ONLY=1 reads the saved models / caches and writes
                metrics.csv, test_predictions.csv and parity_test.png.

Usage (from anywhere):
    python ml_model/train_parallel.py all            [--workers N]   (default: CPU count - 1)
    python ml_model/train_parallel.py nmos_on pmos_off ...
"""
import argparse, os, subprocess, sys, time
# One BLAS/OpenMP thread per process; parallelism comes from running many processes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = 1_000_000                       # row labels >= AUG are the extra (gate-region) samples
CASES = {
    # builder: module with build_table / PARAMS / SEEDS / OUT and a module-global PB (physics module)
    "nmos_on": dict(code=os.path.join(HERE, "NMOS_ON", "nmos_on_code", "code"), common="non_common",
                    builder="pipeline", versions={"a": "physics_a", "b": "physics_b", "c": "physics_c", "d": "physics_d"},
                    set_nominal=True, blend="select_blend.py"),
    "pmos_on": dict(code=os.path.join(HERE, "PMOS_ON", "pmos_on_physics_ml", "code"), common="pon_common",
                    builder="pipeline", versions={"a": "physics_baseline_a", "b": "physics_baseline_b"}),
    "nmos_off": dict(code=os.path.join(HERE, "NMOS_OFF", "nmos_off_physics_ml", "code"), common="nmos_off_common",
                     builder="train_final", versions={"": "physics_baseline"}),
    "pmos_off": dict(code=os.path.join(HERE, "PMOS_OFF", "pmos_off_physics_ml", "code"), common="pmoff_common",
                     builder="train_final", versions={"": "physics_baseline"}),
}


# ---------------------------------------------------------------- worker side
def _setup(case):
    """Import the case's modules in a clean state. Worker processes are reused for jobs of other
    cases, and the model folders reuse module names (pipeline, physics_baseline, train_final), so
    every module loaded from ANY model folder is dropped and the other folders leave sys.path."""
    cfg = CASES[case]
    dirs = {c["code"] for c in CASES.values()}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None) or ""
        if any(f.startswith(d + os.sep) for d in dirs):
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if p not in dirs]
    sys.path.insert(0, cfg["code"]); os.chdir(cfg["code"])
    import importlib, warnings
    warnings.simplefilter("ignore")
    importlib.invalidate_caches()
    common, builder = importlib.import_module(cfg["common"]), importlib.import_module(cfg["builder"])
    assert os.path.dirname(builder.__file__) == cfg["code"], f"wrong module for {case}: {builder.__file__}"
    return cfg, common, builder


def _version(cfg, builder, tag, tro):
    import importlib, numpy as np
    mod = importlib.import_module(cfg["versions"][tag])
    builder.PB = mod
    if cfg.get("set_nominal"):        # as train_final.run_version (NMOS ON): medians of the ORIGINAL training rows
        mod.NOM.update(vth=np.median(np.abs(tro.vth0)), ndep=np.median(tro.ndep), toxe=np.median(tro.toxe),
                       u0=np.median(tro.u0), xj=np.median(tro.xj), vdd=np.median(tro.vdd))
    vdir = os.path.join(builder.OUT, f"model_{tag}") if tag else builder.OUT
    os.makedirs(vdir, exist_ok=True)
    return mod, vdir


def fit_physics(case, tag):
    import joblib
    cfg, common, builder = _setup(case)
    _, tr, _, _ = common.load_split()
    tro = tr[tr.index < AUG]          # physics on the original training rows only
    mod, vdir = _version(cfg, builder, tag, tro)
    t0 = time.time()
    P = mod.fit_physics(tro)
    joblib.dump(P, os.path.join(vdir, "physics_params.joblib"))
    if cfg.get("set_nominal"):
        joblib.dump(dict(mod.NOM), os.path.join(vdir, "nominal.joblib"))
    return f"[{case} {tag or '-'}] physics fitted in {time.time() - t0:.0f}s"


def fit_lgbm(case, tag, seed):
    import joblib, numpy as np, lightgbm as lgb
    cfg, common, builder = _setup(case)
    _, tr, va, _ = common.load_split()
    mod, vdir = _version(cfg, builder, tag, tr[tr.index < AUG])
    P = joblib.load(os.path.join(vdir, "physics_params.joblib"))
    BT, BV = builder.build_table(tr, P), builder.build_table(va, P)
    F = [c for c in BT["L"].columns if c != "y"]
    joblib.dump(F, os.path.join(vdir, "feature_list.joblib"))
    okt, okv = np.isfinite(BT["t"]), np.isfinite(BV["t"])            # skip noise-floor targets
    t0 = time.time()
    reg = lgb.LGBMRegressor(**builder.PARAMS, random_state=seed).fit(
        BT["L"][F][okt], BT["t"][okt], eval_set=[(BV["L"][F][okv], BV["t"][okv])], categorical_feature=["cur"],
        callbacks=[lgb.early_stopping(300, verbose=False)])
    joblib.dump(reg, os.path.join(vdir, f"lgbm_seed{seed}.joblib"), compress=3)
    cache = os.path.join(vdir, f"pred_cache_seed{seed}.npz")
    if os.path.exists(cache):
        os.remove(cache)              # stale predictions of an older model
    return f"[{case} {tag or '-'}] seed {seed}: {reg.best_iteration_} trees, {time.time() - t0:.0f}s"


def predict_cache(case, tag, seed):
    import joblib, numpy as np
    cfg, common, builder = _setup(case)
    _, tr, va, te = common.load_split()
    ta = common.load_test_aug()
    splits = [("train", tr), ("val", va), ("test", te)] + ([("test_aug", ta)] if ta is not None else [])
    mod, vdir = _version(cfg, builder, tag, tr[tr.index < AUG])
    P = joblib.load(os.path.join(vdir, "physics_params.joblib"))
    F = joblib.load(os.path.join(vdir, "feature_list.joblib"))
    reg = joblib.load(os.path.join(vdir, f"lgbm_seed{seed}.joblib"))
    t0 = time.time()
    o = {k: reg.predict(builder.build_table(d, P)["L"][F], num_threads=1) for k, d in splits}
    cache = os.path.join(vdir, f"pred_cache_seed{seed}.npz")
    tmp = cache[:-4] + ".tmp.npz"
    np.savez(tmp, **o); os.replace(tmp, cache)                      # atomic: never a half-written cache
    return f"[{case} {tag or '-'}] seed {seed}: predictions cached in {time.time() - t0:.0f}s"


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", choices=list(CASES) + ["all"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    cases = list(CASES) if "all" in a.cases else a.cases
    seeds = [42, 7, 123]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        running = {ex.submit(fit_physics, c, v): ("physics", c, v, None) for c in cases for v in CASES[c]["versions"]}
        print(f"{len(running)} physics fits, {sum(len(CASES[c]['versions']) for c in cases) * len(seeds)} LightGBM "
              f"models on {a.workers} workers", flush=True)
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                kind, c, v, s = running.pop(f)
                print(f"{time.time() - t0:6.0f}s  {f.result()}", flush=True)       # re-raises worker errors
                if kind == "physics":
                    for sd in seeds:
                        running[ex.submit(fit_lgbm, c, v, sd)] = ("lgbm", c, v, sd)
                elif kind == "lgbm":
                    running[ex.submit(predict_cache, c, v, s)] = ("cache", c, v, s)
    print(f"{time.time() - t0:6.0f}s  all models trained and cached; evaluating", flush=True)
    env = dict(os.environ, EVAL_ONLY="1")
    for c in cases:                    # NMOS ON: blend weights from the validation set first
        if CASES[c].get("blend"):
            subprocess.run([sys.executable, CASES[c]["blend"]], cwd=CASES[c]["code"], check=True)
    procs = {c: subprocess.Popen([sys.executable, "train_final.py"], cwd=CASES[c]["code"], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for c in cases}
    failed = []
    for c, p in procs.items():
        out, _ = p.communicate()
        tail = [l for l in out.splitlines() if "Warning" not in l and "eval_set" not in l]
        print(f"\n===== {c} =====\n" + "\n".join(tail[-24:]), flush=True)
        if p.returncode:
            failed.append(c)
    print(f"\ndone in {time.time() - t0:.0f}s" + (f"; FAILED: {failed}" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
