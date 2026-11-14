"""
physics_baseline.py  -  the physics "first guess" for PMOS OFF leakage (mirrored view, see pmoff_common.py).

Same formulas as the NMOS OFF model; only the normalisation constants are the PMOS nominals from
model/22nm_MGK.pm (toxe 0.67 nm, |vth0| 0.25399 V, ndep 4.4e18, u0 0.0023, xj 7.2 nm).

Idea: LightGBM learns only a small CORRECTION on top of these formulas.
Every formula is fitted on the TRAINING rows only (no validation/test leakage),
and is fitted DIRECTLY against Id, Ig, Is and Ib. No current is ever built by
adding or subtracting other measured currents (e.g. no Id - Is anywhere).

Mechanisms (mirrored view: Vg' = Vb' = 0):
  Fd, Fs : gate-tunnelling current at the drain / source edge   (ln_gate)
  C      : subthreshold channel current, flows high -> low      (ln_channel)
  Ib     : junction + high-temperature + field terms            (ib_model)
  Physics guess:  Id = sign(Vd-Vs)*C + Fd,  Is = -sign(Vd-Vs)*C + Fs,  Ig = -(Fd+Fs)

Main function used: scipy.optimize.least_squares. It is better than a plain
curve_fit here because it handles many parameters, lets us write our own
residuals (relative errors), supports robust losses (soft_l1) and scales
parameters automatically (x_scale='jac').
"""
import numpy as np
from scipy.optimize import least_squares

K_Q = 8.617333e-5          # Boltzmann constant / electron charge (V per K)

# ---------------------------------------------------------------- inputs
def norm_inputs(d):
    """Put every input on a similar, small scale (helps the fitting)."""
    T = d.temp.values + 273.15
    return dict(vd=d.vd.values, vs=d.vs.values, lw=np.log(d.w.values / 1e-7), T=T, tau=(T - 300) / 100,
                tox=(d.toxe.values * 1e9 - 0.67) / 0.067, vth=d.vth0.values, vn=(d.vth0.values - 0.25399) / 0.025399,
                nd=(d.ndep.values / 4.4e18 - 1) / 0.1, u0=(d.u0.values / 0.0023 - 1) / 0.1, xj=(d.xj.values / 7.2e-9 - 1) / 0.1)

def quad_terms(z):
    """All first- and second-order terms (x_i and x_i*x_j) of the columns of z."""
    cols = [z[:, i] for i in range(z.shape[1])]
    for i in range(z.shape[1]):
        for j in range(i, z.shape[1]):
            cols.append(z[:, i] * z[:, j])
    return np.c_[cols].T

# ---------------------------------------------------------------- gate tunnelling
NG = 18
def ln_gate(g, V, c):
    """ln of gate-tunnelling current at one edge with voltage V (drain or source)."""
    lv = np.log(V)
    return (g[0] + c['lw'] * g[13] + g[1] * lv + g[2] * V + g[3] * V**2 + g[4] * V**3
            + c['tox'] * (g[5] + g[6] * V + g[7] * V**2) + g[8] * c['tox']**2
            + c['tau'] * (g[9] + g[10] * V) + g[11] * c['tau']**2 + g[12] * c['vn']
            + g[14] * c['lw']**2 + g[15] * c['lw'] * V + g[16] * c['lw'] * c['tox']
            + g[17] * np.log(np.exp(c['lw']) + 0.05))

# ---------------------------------------------------------------- subthreshold channel
NC = 26
def ln_chan_core(p, c):
    """Subthreshold channel: body effect, DIBL, temperature, width and process terms."""
    vh = np.maximum(c['vd'], c['vs']); vl = np.minimum(c['vd'], c['vs']); vds = vh - vl
    vt = K_Q * c['T']; t = c['tau']; lw = c['lw']; phi = 0.8
    n = p[4] * (1 + p[11] * t + p[16] * vl + p[17] * vds + p[18] * lw + p[24] * t * t)
    vth = (c['vth'] * (1 + p[7] * c['nd'] + p[8] * c['tox'] + p[12] * c['xj']) + p[9] * t + p[19] * t * t
           + p[1] * (np.sqrt(phi + vl) - np.sqrt(phi)) - p[3] * vds - p[10] * vds**2 - p[13] * vds * vl
           + p[20] * lw + p[21] * lw**2 + p[22] * c['nd'] * vl + p[25] * vds * t)
    return (p[0] + lw * p[15] + p[23] * lw**2 + (-vl - vth) / (n * vt)
            + np.log1p(-np.exp(-vds / vt) + 1e-300) + p[5] * np.log(c['T'] / 300)
            + p[6] * c['nd'] + p[2] * c['u0'] + p[14] * c['tox'])

def chan_quad(c):
    """Inputs of the smooth quadratic correction to the channel formula."""
    vh = np.maximum(c['vd'], c['vs']); vl = np.minimum(c['vd'], c['vs'])
    z = np.c_[vl / 0.5 - 1, (vh - vl) / 0.5 - 1, c['tau'], c['vn'], c['nd'], c['tox'], c['u0'], c['xj'], c['lw']]
    return quad_terms(z)

# ---------------------------------------------------------------- body current
def ib_model(p, d):
    """|Ib|: two junction terms + high-temperature activated term + field term."""
    T = d.temp.values + 273.15; vt = K_Q * T; t = (T - 300) / 100; lw = np.log(d.w.values / 1e-7)
    vd, vs = d.vd.values, d.vs.values
    isat = np.exp(p[0] + p[1] * t + p[2] * t * t); n = p[3]
    jd = isat * (1 - np.exp(-vd / (n * vt)))
    js = isat * np.exp(p[4]) * (1 - np.exp(-vs / (n * vt)))
    hot = np.exp(p[5] + p[6] * lw - p[7] * (1 / vt - 1 / (K_Q * 398))) * (np.exp(p[8] * vd) + np.exp(p[8] * vs))
    fld = np.exp(p[9] + p[10] * lw + p[11] * t) * (np.exp(p[12] * vd + p[13] * vd**2) + np.exp(p[12] * vs + p[13] * vs**2))
    return jd + js + hot + fld

# ---------------------------------------------------------------- evaluate
def components(P, d):
    """Return physics Id, Ig, Is, Ib and the parts Fd, Fs, C for a dataframe d."""
    c = norm_inputs(d); g, p, q = P['gate'], P['chan'], P['chan_q']
    Fd = np.exp(ln_gate(g, c['vd'], c)); Fs = np.exp(ln_gate(g, c['vs'], c))
    C = np.exp(ln_chan_core(p, c) + chan_quad(c) @ q); sg = np.sign(c['vd'] - c['vs'])
    return dict(Id=sg * C + Fd, Ig=-(Fd + Fs), Is=-sg * C + Fs, Ib=-ib_model(P['ib'], d), Fd=Fd, Fs=Fs, C=C)

# ---------------------------------------------------------------- fitting (training rows only)
def _res_dsg(g, p, q, c, Q, d):
    """Residuals: Id and Is as signed errors scaled by the size of their parts
    (stays well-behaved when Id ~ 0), Ig as a log ratio."""
    Fd = np.exp(ln_gate(g, c['vd'], c)); Fs = np.exp(ln_gate(g, c['vs'], c))
    C = np.exp(ln_chan_core(p, c) + (Q @ q if q is not None else 0)); sg = np.sign(c['vd'] - c['vs'])
    sc = C + np.maximum(Fd, Fs)
    ok = np.isfinite(d.ig.values)                 # Ig below the simulator noise floor is NaN: skip it
    return np.r_[(sg * C + Fd - d.id.values) / sc, (-sg * C + Fs - d['is'].values) / sc,
                 np.log((Fd + Fs)[ok] / -d.ig.values[ok])]

def fit_physics(tr, verbose=True):
    c = norm_inputs(tr); n = len(tr); Q = chan_quad(c)
    # Stage 1: gate tunnelling, fitted on Ig only (Ig depends only on the gate terms)
    g0 = np.zeros(NG); g0[0] = -18; g0[1] = 2; g0[13] = 1; g0[5] = -2
    okg = np.isfinite(tr.ig.values)               # skip noise-floor (NaN) Ig values
    rg = lambda g: np.log((np.exp(ln_gate(g, c['vd'], c)) + np.exp(ln_gate(g, c['vs'], c)))[okg] / -tr.ig.values[okg])
    g = least_squares(rg, g0, max_nfev=20000, x_scale='jac').x
    if verbose: print('  gate formula fitted', flush=True)
    # Stage 2: channel, starting level chosen by a quick scan, then fitted on Id and Is
    p = np.zeros(NC); p[1] = 0.3; p[3] = 0.05; p[4] = 1.3; p[15] = 1; p[5] = 2
    best = None
    for p00 in np.arange(-16, -4, 1.0):
        p[0] = p00; v = np.sum(_res_dsg(g, p, None, c, Q, tr)**2)
        if best is None or v < best[0]: best = (v, p00)
    p[0] = best[1]
    p = least_squares(lambda pp: _res_dsg(g, pp, None, c, Q, tr)[:2 * n], p, max_nfev=20000, x_scale='jac').x
    # Stage 3: polish gate + channel together
    th = least_squares(lambda x: _res_dsg(x[:NG], x[NG:], None, c, Q, tr), np.r_[g, p], max_nfev=20000, x_scale='jac').x
    g, p = th[:NG], th[NG:]
    if verbose: print('  channel formula fitted', flush=True)
    # Stage 4: smooth quadratic correction to the channel (gate kept fixed)
    nq = Q.shape[1]
    x = least_squares(lambda x: _res_dsg(g, x[:NC], x[NC:], c, Q, tr)[:2 * n], np.r_[p, np.zeros(nq)],
                      max_nfev=3000, x_scale='jac').x
    p, q = x[:NC], x[NC:]
    if verbose: print('  channel quadratic correction fitted', flush=True)
    # Stage 5: body current (robust soft_l1 loss, a few deterministic restarts)
    okb = np.isfinite(tr.ib.values)               # skip noise-floor (NaN) Ib values
    rb = lambda pp: np.log(ib_model(pp, tr)[okb] / -tr.ib.values[okb])
    pb0 = np.array([np.log(5e-15), 0, 0, 1, 0, np.log(1e-15), 1, 1.1, 1, np.log(1e-17), 1, 0, 5, 0])
    bestb = None
    for s in range(6):
        start = pb0 + np.random.default_rng(s).normal(0, 0.3, len(pb0)) * (s > 0)
        f = least_squares(rb, start, loss='soft_l1', f_scale=0.01, max_nfev=20000)
        if bestb is None or f.cost < bestb.cost: bestb = f
    if verbose: print('  body-current formula fitted', flush=True)
    return dict(gate=g, chan=p, chan_q=q, ib=bestb.x)
