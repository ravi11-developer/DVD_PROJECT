"""
train_final.py  -  final NMOS ON model: four physics-guided LightGBM versions, blended per current.

Run:   python train_final.py                    full training (12 LightGBM models, about 30-90 min on one CPU core)
       $env:EVAL_ONLY = "1"; python train_final.py
                                                no training: load the saved models and recompute every result
Data:  set NMOS_ON_CSV to your nmos_on.csv path.

  Versions A, B, C, D = physics_a.py ... physics_d.py, each + 3 LightGBM models (seeds 42, 7, 123),
  built with pipeline.py (shared terminal model for Id/Is, scaled target for Id/Is, log-ratio target
  for Ig/Ib, Huber loss, linear trees, early stopping on validation).
  FINAL: each current is a weighted average of the four versions, with weights in blend_weights.json.
  The weights were chosen on the VALIDATION set (highest mean accuracy per current); the test set was
  only used for the final report.
  NMOS ON is handled in a mirrored view (V' = VDD - V, currents sign-flipped), see non_common.py.
  All saved predictions and plots are converted back to real NMOS ON signs.
"""
import os, json, time, numpy as np, pandas as pd, joblib, lightgbm as lgb
import pipeline as PL
import physics_a, physics_b, physics_c, physics_d
from non_common import load_split, load_test_aug, CURRENTS

OUT = PL.OUT
EVAL_ONLY = os.environ.get('EVAL_ONLY', '0') == '1'
VERSIONS = [('a', physics_a), ('b', physics_b), ('c', physics_c), ('d', physics_d)]
WEIGHTS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blend_weights.json')))

def run_version(tag, mod, splits):
    tr = dict(splits)['train']
    tro = tr[tr.index < 1_000_000]                   # original training rows (physics + nominal values)
    vdir = os.path.join(OUT, f'model_{tag}'); os.makedirs(vdir, exist_ok=True)
    PL.PB = mod
    mod.NOM.update(vth=np.median(np.abs(tro.vth0)), ndep=np.median(tro.ndep), toxe=np.median(tro.toxe),
                   u0=np.median(tro.u0), xj=np.median(tro.xj), vdd=np.median(tro.vdd))
    if EVAL_ONLY:
        P = joblib.load(os.path.join(vdir, 'physics_params.joblib'))
    else:
        print(f'[{tag}] fitting physics on the training set ...', flush=True)
        P = mod.fit_physics(tr[tr.index < 1_000_000]); joblib.dump(P, os.path.join(vdir, 'physics_params.joblib'))  # physics on the original training rows only; LightGBM uses all rows
        joblib.dump(dict(mod.NOM), os.path.join(vdir, 'nominal.joblib'))
    B = {k: PL.build_table(d, P) for k, d in splits}
    F = [c for c in B['train']['L'].columns if c != 'y']; joblib.dump(F, os.path.join(vdir, 'feature_list.joblib'))
    okt, okv = np.isfinite(B['train']['t']), np.isfinite(B['val']['t'])   # skip any NaN targets
    outs = {k: [] for k in B}
    for s in PL.SEEDS:
        path = os.path.join(vdir, f'lgbm_seed{s}.joblib')
        if EVAL_ONLY and os.path.exists(os.path.join(vdir, f'pred_cache_seed{s}.npz')):
            reg = None
        elif EVAL_ONLY:
            reg = joblib.load(path)
        else:
            t0 = time.time()
            reg = lgb.LGBMRegressor(**PL.PARAMS, random_state=s).fit(
                B['train']['L'][F][okt], B['train']['t'][okt], eval_set=[(B['val']['L'][F][okv], B['val']['t'][okv])],
                categorical_feature=['cur'], callbacks=[lgb.early_stopping(300, verbose=False)])
            joblib.dump(reg, path, compress=3)
            print(f'[{tag}] seed {s}: {reg.best_iteration_} trees, {time.time()-t0:.0f}s', flush=True)
        cache = os.path.join(vdir, f'pred_cache_seed{s}.npz')     # lets an interrupted run resume
        if os.path.exists(cache) and set(np.load(cache).files) >= set(B):   # reuse only if it covers every split
            z = np.load(cache); o = {k: z[k] for k in B}
        else:
            o = {k: reg.predict(B[k]['L'][F]) for k in B}; np.savez(cache, **o)
        for k in B: outs[k].append(o[k])
        print(f'[{tag}] model seed {s} ready', flush=True)
    preds = {k: PL.to_currents(B[k], np.mean(outs[k], axis=0)) for k in B}
    phys = {}
    for k, d in splits:
        ph = mod.components(P, d)
        phys[k] = pd.DataFrame({'id': ph['Id'], 'ig': ph['Ig'], 'is': ph['Is'], 'ib': ph['Ib']}, index=d.index)
    return preds, phys

def main():
    t0 = time.time()
    df, tr, va, te = load_split()                       # mirrored view
    ta = load_test_aug()                              # test part of the extra (gate-region) samples
    splits = [('train', tr), ('val', va), ('test', te)] + ([('test_aug', ta)] if ta is not None else [])
    print(f'rows: train {len(tr)}, val {len(va)}, test {len(te)}, test_aug {0 if ta is None else len(ta)}', flush=True)
    res = {tag: run_version(tag, mod, splits) for tag, mod in VERSIONS}
    rows = []
    for k, d in splits:
        for tag, _ in VERSIONS:
            rows += PL.metrics(d, res[tag][1][k], k, f'physics only ({tag.upper()})')
            rows += PL.metrics(d, res[tag][0][k], k, f'version {tag.upper()} (3-seed avg)')
        fin = pd.DataFrame({c: sum(WEIGHTS[c][t.upper()] * res[t][0][k][c] for t, _ in VERSIONS) for c in CURRENTS})
        rows += PL.metrics(d, fin, k, 'FINAL')                # relative errors are the same in both sign views
        if k == 'test':
            real_spice = -te[CURRENTS]; real_pred = -fin       # back to real NMOS ON signs
            real_spice.add_prefix('spice_').join(real_pred.add_prefix('pred_')).to_csv(os.path.join(OUT, 'test_predictions.csv'))
            PL.plot_parity(real_spice, real_pred)
    m = pd.DataFrame(rows); m.to_csv(os.path.join(OUT, 'metrics.csv'), index=False, float_format='%.5g')
    print('\nFINAL MODEL')
    print(m[m.model == 'FINAL'][['split', 'current', 'mean_accuracy', 'median_err', 'p95_err', 'within_1pct', 'r2_log']]
          .to_string(index=False, float_format='%.4f'))
    print(f'\nDone in {time.time()-t0:.0f}s. Results in {os.path.abspath(OUT)}')

if __name__ == '__main__':
    main()
