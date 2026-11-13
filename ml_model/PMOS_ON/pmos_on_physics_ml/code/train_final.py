"""
train_final.py  -  final PMOS ON model: two physics-guided LightGBM versions, combined.

Run:   python train_final.py                 (full training, about 20-80 minutes on one CPU core)
       $env:EVAL_ONLY = "1"; python train_final.py
                                             (no training: load the saved models and recompute all results)
Data:  set PMOS_ON_CSV to your pmos_on.csv path.

  Version A = physics_baseline_a.py + 3 LightGBM models (seeds 42, 7, 123)
  Version B = physics_baseline_b.py (adds gate-to-body tunnelling) + 3 LightGBM models
  FINAL     : Id, Ig, Ib = average of A and B (6 models)
              Is         = version A only
  This rule was chosen on the VALIDATION set so that no validation metric got worse than version A.
  Each version follows pipeline.py: shared terminal model for Id/Is, scaled target for Id/Is,
  log-ratio target for Ig/Ib, Huber loss, linear trees, early stopping on validation.
"""
import os, time, numpy as np, pandas as pd, joblib, lightgbm as lgb
import pipeline as PL
import physics_baseline_a as PA, physics_baseline_b as PBB
from pon_common import load_split, CURRENTS

OUT = PL.OUT
EVAL_ONLY = os.environ.get('EVAL_ONLY', '0') == '1'
VERSIONS = [('a', PA), ('b', PBB)]

def run_version(tag, mod, tr, va, te):
    """Fit (or load) one physics version and its 3 LightGBM models; return predictions per split."""
    vdir = os.path.join(OUT, f'model_{tag}'); os.makedirs(vdir, exist_ok=True)
    PL.PB = mod                                   # pipeline uses this physics module
    if EVAL_ONLY:
        P = joblib.load(os.path.join(vdir, 'physics_params.joblib'))
    else:
        print(f'[{tag}] fitting physics on the training set ...', flush=True)
        P = mod.fit_physics(tr); joblib.dump(P, os.path.join(vdir, 'physics_params.joblib'))
    B = {k: PL.build_table(d, P) for k, d in (('train', tr), ('val', va), ('test', te))}
    F = [c for c in B['train']['L'].columns if c != 'y']; joblib.dump(F, os.path.join(vdir, 'feature_list.joblib'))
    outs = {k: [] for k in B}
    for s in PL.SEEDS:
        path = os.path.join(vdir, f'lgbm_seed{s}.joblib')
        if EVAL_ONLY:
            reg = joblib.load(path)
        else:
            t0 = time.time()
            reg = lgb.LGBMRegressor(**PL.PARAMS, random_state=s).fit(
                B['train']['L'][F], B['train']['t'], eval_set=[(B['val']['L'][F], B['val']['t'])],
                categorical_feature=['cur'], callbacks=[lgb.early_stopping(300, verbose=False)])
            joblib.dump(reg, path, compress=3)
            print(f'[{tag}] seed {s}: {reg.best_iteration_} trees, {time.time()-t0:.0f}s', flush=True)
        for k in B: outs[k].append(reg.predict(B[k]['L'][F]))
    preds = {k: PL.to_currents(B[k], np.mean(outs[k], axis=0)) for k in B}
    phys = {}
    for k, d in (('train', tr), ('val', va), ('test', te)):
        ph = mod.components(P, d)
        phys[k] = pd.DataFrame({'id': ph['Id'], 'ig': ph['Ig'], 'is': ph['Is'], 'ib': ph['Ib']}, index=d.index)
    return preds, phys

def main():
    t0 = time.time()
    df, tr, va, te = load_split()
    print(f'rows: train {len(tr)}, val {len(va)}, test {len(te)}', flush=True)
    res = {tag: run_version(tag, mod, tr, va, te) for tag, mod in VERSIONS}
    rows = []
    for k, d in (('train', tr), ('val', va), ('test', te)):
        for tag in ('a', 'b'):
            rows += PL.metrics(d, res[tag][1][k], k, f'physics only ({tag.upper()})')
            rows += PL.metrics(d, res[tag][0][k], k, f'version {tag.upper()} (3-seed avg)')
        fin = (res['a'][0][k] + res['b'][0][k]) / 2
        fin['is'] = res['a'][0][k]['is']
        rows += PL.metrics(d, fin, k, 'FINAL')
        if k == 'test':
            te[CURRENTS].add_prefix('spice_').join(fin.add_prefix('pred_')).to_csv(os.path.join(OUT, 'test_predictions.csv'))
            PL.plot_parity(te, fin)
    m = pd.DataFrame(rows); m.to_csv(os.path.join(OUT, 'metrics.csv'), index=False, float_format='%.5g')
    print('\nFINAL MODEL')
    print(m[m.model == 'FINAL'][['split', 'current', 'mean_accuracy', 'median_err', 'p95_err', 'within_1pct', 'r2_log']]
          .to_string(index=False, float_format='%.4f'))
    print(f'\nDone in {time.time()-t0:.0f}s. Results in {os.path.abspath(OUT)}')

if __name__ == '__main__':
    main()
