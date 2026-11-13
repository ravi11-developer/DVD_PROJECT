"""
pipeline.py - shared building blocks (table building, target transform, metrics, plots).
Used by train_final.py; not run directly.
"""
import os, time, json
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.metrics import r2_score
from non_common import load_split, pct_err, CURRENTS, K_Q
PB = None   # physics module, set by train_final.py

OUT = os.environ.get('NMOS_ON_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 7, 123]
N_JOBS = 1          # 1 = exactly reproducible; -1 = faster (results may differ very slightly)
PARAMS = dict(objective='huber', alpha=0.005, learning_rate=0.08, num_leaves=31, min_child_samples=20,
              linear_tree=True, linear_lambda=0.1, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
              reg_lambda=0.01, n_estimators=45000, verbose=-1, n_jobs=N_JOBS)

def build_table(d, P):
    """Long table (4 rows per sample) + targets and the numbers needed to turn predictions back into currents."""
    ph = PB.components(P, d); n = len(d)
    Fd, Fs, C, Id, Ig, Is, Ib = ph['Fd'], ph['Fs'], ph['C'], ph['Id'], ph['Ig'], ph['Is'], ph['Ib']
    T = d.temp.values + 273.15; vt = K_Q * T; vd, vs = d.vd.values, d.vs.values
    common = pd.DataFrame({'vdd': d.vdd.values, 'temp': d.temp.values, 'log_w': np.log10(d.w.values),
        'vth0': d.vth0.values, 'toxe': d.toxe.values, 'ndep': d.ndep.values, 'u0': d.u0.values, 'xj': d.xj.values,
        'v_high': np.maximum(vd, vs), 'v_low': np.minimum(vd, vs), 'abs_vds': np.abs(vd - vs), 'inv_vt': 1 / vt,
        'vth_vt': d.vth0.values / vt, 'vds_vt': np.abs(vd - vs) / vt, 'log_C': np.log10(C), 'log_ibphys': np.log10(np.abs(Ib))})
    def frame(v_this, v_other, F_this, F_other, I_phys, cur, base, y):
        X = common.copy()
        X['v_this'] = v_this; X['v_other'] = v_other; X['v_diff'] = v_this - v_other
        X['this_vt'] = v_this / vt; X['e_this'] = 1 - np.exp(-v_this / vt)
        X['log_Fthis'] = np.log10(F_this); X['log_Fother'] = np.log10(F_other)
        X['share'] = C / (C + F_this); X['r_phys'] = I_phys / (C + F_this)
        X['cur'] = cur; X['phys'] = base; X['y'] = y        # cur: 0 = terminal current (Id/Is), 1 = Ig, 2 = Ib
        return X
    L = pd.concat([frame(vd, vs, Fd, Fs, Id, 0, np.log10(np.abs(Id) + 1e-40), np.log10(np.abs(d.id.values))),
                   frame(vd, vs, Fd, Fs, Id, 1, np.log10(-Ig), np.log10(-d.ig.values)),
                   frame(vs, vd, Fs, Fd, Is, 0, np.log10(np.abs(Is) + 1e-40), np.log10(np.abs(d['is'].values))),
                   frame(vd, vs, Fd, Fs, Id, 2, np.log10(np.abs(Ib)), np.log10(np.abs(d.ib.values)))], ignore_index=True)
    scale = np.r_[C + Fd, np.ones(n), C + Fs, np.ones(n)]
    phys_r = np.r_[Id / (C + Fd), np.zeros(n), Is / (C + Fs), np.zeros(n)]
    true_r = np.r_[d.id.values / (C + Fd), np.zeros(n), d['is'].values / (C + Fs), np.zeros(n)]
    term = np.r_[np.ones(n, bool), np.zeros(n, bool), np.ones(n, bool), np.zeros(n, bool)]
    target = np.where(term, true_r - phys_r, (L.y - L.phys).values * np.log(10))
    return dict(L=L, t=target, term=term, scale=scale, phys_r=phys_r, n=n, index=d.index)

def to_currents(B, out):
    """Turn model outputs back into the four signed currents."""
    n = B['n']
    sgn = np.r_[np.ones(n), -np.ones(n), np.ones(n), np.ones(n)]   # Ig < 0, Ib > 0 in the mirrored (PMOS-ON-like) view
    v = np.where(B['term'], (B['phys_r'] + out) * B['scale'], sgn * 10 ** (B['L'].phys.values + out / np.log(10)))
    return pd.DataFrame({'id': v[:n], 'ig': v[n:2*n], 'is': v[2*n:3*n], 'ib': v[3*n:]}, index=B['index'])

def metrics(d, p, split, model):
    rows = []
    for c in CURRENTS:
        t, q = d[c].values, p[c].values; e = pct_err(t, q)
        rows.append(dict(model=model, split=split, current=c.upper(), mean_accuracy=100 - np.minimum(e, 100).mean(),
            median_err=np.median(e), p95_err=np.percentile(e, 95), max_err=e.max(), within_1pct=(e < 1).mean() * 100,
            r2_log=r2_score(np.log10(np.abs(t)), np.log10(np.abs(q))), r2_linear=r2_score(t, q),
            sign_accuracy=(np.sign(q) == np.sign(t)).mean() * 100))
    return rows

def main():
    t0 = time.time()
    df, tr, va, te = load_split()
    print(f'rows: train {len(tr)}, val {len(va)}, test {len(te)}', flush=True)
    print('Fitting physics formulas on the training set ...', flush=True)
    P = PB.fit_physics(tr); joblib.dump(P, os.path.join(OUT, 'physics_params.joblib'))

    rows = []
    for split, d in [('train', tr), ('val', va), ('test', te)]:    # ablation: physics alone, no ML
        ph = PB.components(P, d)
        rows += metrics(d, pd.DataFrame({'id': ph['Id'], 'ig': ph['Ig'], 'is': ph['Is'], 'ib': ph['Ib']}, index=d.index),
                        split, 'physics only')

    BT, BV, BE = build_table(tr, P), build_table(va, P), build_table(te, P)
    F = [c for c in BT['L'].columns if c != 'y']
    outs = {k: [] for k in ('train', 'val', 'test')}
    for s in SEEDS:
        ts = time.time()
        reg = lgb.LGBMRegressor(**PARAMS, random_state=s).fit(
            BT['L'][F], BT['t'], eval_set=[(BV['L'][F], BV['t'])], categorical_feature=['cur'],
            callbacks=[lgb.early_stopping(300, verbose=False)])
        joblib.dump(reg, os.path.join(OUT, f'lgbm_seed{s}.joblib'), compress=3)
        for k, B in (('train', BT), ('val', BV), ('test', BE)): outs[k].append(reg.predict(B['L'][F]))
        print(f'seed {s}: {reg.best_iteration_} trees, {time.time()-ts:.0f}s', flush=True)
    joblib.dump(F, os.path.join(OUT, 'feature_list.joblib'))

    preds = {}
    for k, B, d in (('train', BT, tr), ('val', BV, va), ('test', BE, te)):
        preds[k] = to_currents(B, np.mean(outs[k], axis=0))
        rows += metrics(d, preds[k], k, 'physics + LightGBM (3-seed avg)')
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, 'metrics.csv'), index=False, float_format='%.5g')
    out_test = te[CURRENTS].add_prefix('spice_').join(preds['test'].add_prefix('pred_'))
    out_test.to_csv(os.path.join(OUT, 'test_predictions.csv'))

    show = res[res.model.str.startswith('physics +')]
    print('\nFINAL MODEL (physics + LightGBM, 3 seeds averaged)')
    print(show[['split', 'current', 'mean_accuracy', 'median_err', 'p95_err', 'within_1pct', 'r2_log']].to_string(index=False, float_format='%.4f'))
    plot_parity(te, preds['test'])
    print(f'\nDone in {time.time()-t0:.0f}s. Results in {os.path.abspath(OUT)}')

def plot_parity(d, p):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    for a, c in zip(ax.ravel(), CURRENTS):
        t, q = np.abs(d[c].values), np.abs(p[c].values); e = pct_err(d[c].values, p[c].values)
        a.loglog(t, q, '.', ms=3, alpha=0.5); lo, hi = min(t.min(), q.min()), max(t.max(), q.max())
        a.plot([lo, hi], [lo, hi], 'k--', lw=1)
        a.set_title(f'|I_{c[1]}|  test: median {np.median(e):.3f}%, p95 {np.percentile(e, 95):.2f}%')
        a.set_xlabel('SPICE (A)'); a.set_ylabel('ML prediction (A)')
    fig.suptitle('NMOS ON: predicted vs SPICE (test set)'); fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'parity_test.png'), dpi=150); plt.close(fig)

if __name__ == '__main__':
    main()
