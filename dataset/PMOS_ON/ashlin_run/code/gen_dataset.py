"""Part A step 1: draw random PVT samples and simulate single FETs with ngspice.
usage: gen_dataset.py N [outdir] [case ...]   e.g. gen_dataset.py 10000 data pmos_on   (seed fixed per case)"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from spice_lib import *

CASES = {  # kind, gate level, body level (in units of VDD)
    "nmos_on": ("nmos", 1, 0), "nmos_off": ("nmos", 0, 0),
    "pmos_on": ("pmos", 0, 1), "pmos_off": ("pmos", 1, 1)}

def draw_corner(rng, kind):
    """Process params +/-10% of nominal (separate per device type)."""
    return {p: NOM[kind][p] * rng.uniform(0.9, 1.1) for p in PCOLS}

def draw(case, n, seed):
    kind, gl, bl = CASES[case]
    rng = np.random.default_rng(seed)
    vdd = rng.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM, n)          # VDD first
    d = dict(vdd=vdd, temp=rng.uniform(-40, 125, n), w=rng.uniform(1, 8, n) * LMIN,
             vd=rng.uniform(0, 1, n) * vdd, vs=rng.uniform(0, 1, n) * vdd,
             vg=gl * vdd, vb=bl * vdd)
    for p in PCOLS: d[p] = NOM[kind][p] * rng.uniform(0.9, 1.1, n)
    return pd.DataFrame(d)

def sim(args):
    kind, row = args
    s = dict(row)
    nl = OPTS + "\n.temp %.4f\n" % s["temp"] + fet_block(0, kind, s)
    r = run_ngspice(nl, ["i(vd0)", "i(vg0)", "i(vs0)", "i(vb0)"], "fet")
    return [-r["i(vd0)"], -r["i(vg0)"], -r["i(vs0)"], -r["i(vb0)"]]

if __name__ == "__main__":
    n = int(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else "data"
    os.makedirs(out, exist_ok=True)
    only = sys.argv[3:] or list(CASES)          # optional: case names, e.g. pmos_on
    for k, case in enumerate(CASES):
        if case not in only: continue
        kind = CASES[case][0]
        df = draw(case, n, seed=1000 + k)
        with Pool(2) as p:
            res = p.map(sim, [(kind, r) for r in df.to_dict("records")], chunksize=50)
        df[["id", "ig", "is", "ib"]] = np.array(res)
        cols = ["vdd","temp","w","vd","vs","vg","vb"] + PCOLS + ["id","ig","is","ib"]
        df[cols].to_csv(os.path.join(out, case + ".csv"), index=False)
        # example netlist (first sample) for the deliverables
        s = df.iloc[0].to_dict()
        open(os.path.join(ROOT, "netlists", "fet_%s.cir" % case), "w").write(
            "* single %s, case %s, sample 0\n%s\n.temp %.4f\n%s\n.control\nop\nprint i(vd0) i(vg0) i(vs0) i(vb0)\n.endc\n.end\n"
            % (kind, case, OPTS, s["temp"], fet_block(0, kind, s)))
        print(case, "done", len(df), flush=True)
