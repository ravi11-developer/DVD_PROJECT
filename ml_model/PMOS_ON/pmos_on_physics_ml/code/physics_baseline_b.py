"""
physics_baseline_b.py  -  PHYSICS VERSION B, the physics "first guess" for PMOS ON (Vg = 0, Vb = VDD).

LightGBM learns only a small CORRECTION on top of these formulas. Every formula is fitted on the
TRAINING rows only, directly against Id, Ig, Is and Ib. No current is built by adding or subtracting
other measured currents.

Mechanisms:
  C        : channel current, EKV-style formula (one smooth equation from subthreshold to strong
             inversion) with body effect, DIBL and mobility / velocity-saturation degradation,
             plus a smooth quadratic correction                              (ln_chan_core, zch)
  Gd, Gs   : gate tunnelling at the drain / source edge + a gate-to-channel term split between them
                                                                             (gate_parts)
  Ib       : junction floor + gate-to-body tunnelling through the inverted channel ends
             (BSIM4-style inversion charge), times a smooth correction       (ib_model, charge)
  Gb       : gate-to-body tunnelling (Vgb = -VDD, independent of Vd and Vs), part of Ig; Ib gets its
             own fitted floor of the same form. This fixed the large Ig / Ib errors in rows where the
             transistor is barely on.                                        (g_body)
  Physics guess:  Id = sign(Vd-Vs)*C + Gd,  Is = -sign(Vd-Vs)*C + Gs,  Ig = -(total gate current),  Ib = ib_model

Main function used: scipy.optimize.least_squares (custom residuals, robust soft_l1 loss for Ib,
automatic parameter scaling). numpy.linalg.lstsq gives exact linear starting points.
"""
import numpy as np
from scipy.optimize import least_squares
K_Q=8.617333e-5
def nc(d):
    T=d.temp.values+273.15; vh=np.maximum(d.vd.values,d.vs.values); vl=np.minimum(d.vd.values,d.vs.values)
    return dict(vd=d.vd.values,vs=d.vs.values,vb=d.vb.values,vh=vh,vl=vl,vds=vh-vl,vbr=d.vb.values-vh,T=T,vt=K_Q*T,tau=(T-300)/100,
        lw=np.log(d.w.values/1e-7),vth=np.abs(d.vth0.values),vn=(np.abs(d.vth0.values)-0.2533)/0.02533,
        nd=(d.ndep.values/4.4e18-1)/0.1,tox=(d.toxe.values/6.688e-10-1)/0.1,u0=(d.u0.values/2.302e-3-1)/0.1,xj=(d.xj.values/7.2e-9-1)/0.1,vdn=(d.vdd.values-0.9)/0.09,evds=np.exp(-np.abs(d.vd.values-d.vs.values)/0.03))
def quad(z):
    cols=[z[:,i] for i in range(z.shape[1])]
    for i in range(z.shape[1]):
        for j in range(i,z.shape[1]): cols.append(z[:,i]*z[:,j])
    return np.c_[cols].T
# ---- gate: edge tunnelling at each terminal (V = terminal voltage, gate at 0) + gate-to-channel term
NG=18
def ln_gate(g,V,c):
    lv=np.log(np.maximum(V,1e-9))
    return (g[0]+c['lw']*g[13]+g[1]*lv+g[2]*V+g[3]*V**2+g[4]*V**3+c['tox']*(g[5]+g[6]*V+g[7]*V**2)+g[8]*c['tox']**2
            +c['tau']*(g[9]+g[10]*V)+g[11]*c['tau']**2+g[12]*c['vn']+g[14]*c['lw']**2+g[15]*c['lw']*V+g[16]*c['lw']*c['tox']+g[17]*np.log(np.exp(c['lw'])+0.05))
NH=12
def charge(h,V,c):
    """inversion charge at a channel end at voltage V (gate at 0): smooth turn-on at |Vth| (BSIM4-style softplus)"""
    nig=0.5+np.abs(h[7]); vth_e=c['vth']*(1+h[10])+h[8]*c['tau']+h[9]*(np.sqrt(0.8+np.maximum(c['vb']-V,0))-np.sqrt(0.8))
    return nig*c['vt']*np.logaddexp(0,(V-vth_e)/(nig*c['vt']))
def zgc(c): return np.c_[c['vh']/0.5-1,c['vds']/0.5-1,c['vbr']/0.5-1,c['tau'],c['vn'],c['tox'],c['lw']]
NGC=1+35   # gate-to-channel term: level + quadratic over 7 inputs
def gate_parts(gp,c,Qg=None):
    """edge tunnelling at each terminal + gate-to-channel term split between drain and source"""
    if Qg is None: Qg=quad(zgc(c))
    g=gp[:NG]; gc=gp[NG:NG+NGC]; k=gp[NG+NGC]
    Fd=np.exp(np.clip(ln_gate(g,c['vd'],c),-120,10)); Fs=np.exp(np.clip(ln_gate(g,c['vs'],c),-120,10))
    Gc=np.exp(np.clip(gc[0]+Qg@gc[1:],-120,10)); fd=1/(1+np.exp(np.clip(-k*(c['vd']-c['vs'])/0.1,-50,50)))
    Gb=g_body(gp[NG+NGC+1:],c)
    return Fd+Gc*fd, Fs+Gc*(1-fd), Fd+Fs+Gc+Gb
NGB=6
def g_body(gb,c):
    return np.exp(np.clip(gb[0]+gb[1]*c['tox']+gb[2]*c['tau']+gb[3]*c['lw']+gb[4]*c['vn']+gb[5]*c['vdn'],-120,10))
NGP=NG+NGC+1+NGB
# ---- channel: EKV-style (subthreshold <-> strong inversion), body effect, DIBL, mobility/vsat degradation
NC=20
def ln_chan_core(p,c):
    vt=c['vt']; t=c['tau']; vds=c['vds']
    vth=(c['vth']*(1+p[9]*c['nd']+p[10]*c['tox']+p[11]*c['xj'])+p[5]*t+p[3]*(np.sqrt(0.8+c['vbr'])-np.sqrt(0.8))-p[4]*vds+p[12]*c['lw']+p[18]*t*t)
    n=0.3+np.abs(p[1])*np.exp(np.clip(p[13]*t+p[14]*c['vbr'],-3,3))
    x=(c['vh']-vth)/n
    s1=np.logaddexp(0,x/(2*vt)); s2=np.logaddexp(0,(x-vds)/(2*vt))
    core=np.log(np.maximum((s1-s2)*(s1+s2),1e-300))
    vov=vt*np.logaddexp(0,(c['vh']-vth)/vt)
    deg=np.log1p(np.abs(p[7])*vov+np.abs(p[8])*vds+np.abs(p[19])*vov*vds)
    return p[0]+np.log(n)+2*np.log(vt)+core-deg+p[6]*np.log(c['T']/300)+p[15]*c['lw']+p[16]*c['u0']+p[17]*c['tox']+p[2]*c['nd']
def zch(c): return np.c_[c['vh']/0.5-1,c['vds']/0.5-1,c['vbr']/0.5-1,c['tau'],c['vn'],c['nd'],c['tox'],c['u0'],c['xj'],c['lw']]
# ---- body current: junction floor + gate-to-body tunnelling through the inverted channel ends, times a smooth correction
def zb(c): return np.c_[c['vh']/0.5-1,c['vl']/0.5-1,c['vbr']/0.5-1,c['tau'],c['vn'],c['tox'],c['lw'],c['nd'],c['evds']]
NB=19
def ib_model(b,c,Qb):
    jf=np.exp(b[0])*(2-np.exp(-(c['vb']-c['vd'])/c['vt'])-np.exp(-(c['vb']-c['vs'])/c['vt']))
    h=np.r_[0,0,0,0,0,0,0,b[9],b[10],b[11],b[12],0]
    def end(V): return np.exp(np.clip(b[1]+b[2]*V+b[3]*V**2+b[4]*c['tau']+b[5]*c['tox']+b[6]*c['lw']+b[7]*c['vn']+b[8]*V*c['tau'],-120,10))*charge(h,V,c)
    return (jf+end(c['vd'])+end(c['vs'])+np.exp(np.clip(b[13]+b[14]*c['tox']+b[15]*c['tau']+b[16]*c['lw']+b[17]*c['vn']+b[18]*c['vdn'],-120,10)))*np.exp(np.clip(Qb@b[NB:],-20,20))
def components(P,d):
    c=nc(d); Gd,Gs,Gt=gate_parts(P['gate'],c)
    C=np.exp(np.clip(ln_chan_core(P['chan'],c)+quad(zch(c))@P['chan_q'],-120,10)); sg=np.sign(c['vd']-c['vs'])
    return dict(Id=sg*C+Gd,Is=-sg*C+Gs,Ig=-Gt,Ib=ib_model(P['ib'],c,quad(zb(c))),Fd=Gd,Fs=Gs,C=C)
def _res(gp,p,q,c,Qg,Qc,d,n):
    Gd,Gs,Gt=gate_parts(gp,c,Qg); C=np.exp(np.clip(ln_chan_core(p,c)+(Qc@q if q is not None else 0),-120,10)); sg=np.sign(c['vd']-c['vs'])
    sc=C+np.maximum(Gd,Gs)
    return np.r_[(sg*C+Gd-d.id.values)/sc,(-sg*C+Gs-d['is'].values)/sc,np.log(Gt/-d.ig.values)]
def fit_physics(tr,log=print):
    c=nc(tr); n=len(tr); Qg=quad(zgc(c)); Qc=quad(zch(c)); Qb=quad(zb(c))
    gp=np.zeros(NGP); gp[0]=-26; gp[1]=2; gp[13]=1; gp[5]=-2; gp[NG+NGC+1]=np.log(np.percentile(-tr.ig.values,2))
    A=np.c_[np.ones(n),Qg]; gp[NG:NG+NGC]=np.linalg.lstsq(A,np.log(-tr.ig.values),rcond=None)[0]   # exact linear start
    rg=lambda x: np.log(gate_parts(x,c,Qg)[2]/-tr.ig.values)
    gp=least_squares(rg,gp,max_nfev=4000).x; log('  gate fitted, Ig median err %.3f%%'%(np.median(np.abs(rg(gp)))*100))
    p=np.zeros(NC); p[1]=1.3; p[3]=0.2; p[4]=0.05; p[6]=-1.5; p[7]=1; p[8]=1; p[15]=1
    best=None
    for p00 in np.arange(-6,8,1.0):
        p[0]=p00; v=np.sum(_res(gp,p,None,c,Qg,Qc,tr,n)[:2*n]**2)
        if best is None or v<best[0]: best=(v,p00)
    p[0]=best[1]
    p=least_squares(lambda x: _res(gp,x,None,c,Qg,Qc,tr,n)[:2*n],p,max_nfev=6000,x_scale='jac').x
    log('  channel core fitted')
    x=least_squares(lambda x: _res(x[NC:],x[:NC],None,c,Qg,Qc,tr,n),np.r_[p,gp],max_nfev=3000,x_scale='jac').x
    p=x[:NC]; gp=x[NC:]
    nq=Qc.shape[1]
    x=least_squares(lambda x: _res(gp,x[:NC],x[NC:],c,Qg,Qc,tr,n)[:2*n],np.r_[p,np.zeros(nq)],max_nfev=400,x_scale='jac').x
    p,q=x[:NC],x[NC:]; log('  channel quadratic correction fitted')
    rb=lambda b: np.log(ib_model(b,c,Qb)/tr.ib.values)
    b0=np.zeros(NB+Qb.shape[1]); b0[0]=np.log(1e-15); b0[1]=-22; b0[9]=1.0; b0[13]=np.log(1e-14)
    b0=least_squares(lambda x: rb(np.r_[x,np.zeros(Qb.shape[1])]),b0[:NB],loss='soft_l1',f_scale=0.01,max_nfev=3000).x
    b0=np.r_[b0,np.zeros(Qb.shape[1])]
    b=least_squares(rb,b0,loss='soft_l1',f_scale=0.01,max_nfev=2000).x; log('  body current fitted')
    return dict(gate=gp,chan=p,chan_q=q,ib=b)
