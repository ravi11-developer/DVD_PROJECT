"""
05_augment_inputs.py - extra training samples at the operating points transistors have inside gates.

Why: in a static CMOS gate an ON transistor only carries the leakage of the OFF network, so its
|Vds| is microvolts to millivolts, and the transistor at the bottom of a stack has a terminal exactly
on the rail. The original datasets draw Vd, Vs uniformly in [0, VDD]: only ~2 % of samples have
|Vds| < 10 mV and none has a terminal exactly on a rail. These extra samples cover that region.

Everything except Vd and Vs is drawn exactly as in 01_<case>_inputs.py (VDD +-10 %, -40..125 C,
W = 1..8 Lmin, process parameters nominal +-10 %, Vg/Vb fixed by the case).

Vd / Vs (per sample):
  * the terminal at the "source rail" (0 for NMOS, VDD for PMOS): with probability 1/2 exactly on the
    rail, otherwise uniform within RAIL_BAND of it (ON) or anywhere in [0, VDD] (OFF);
  * the other terminal is offset by |Vds| drawn LOG-UNIFORM between VDS_MIN and VDS_MAX[case]
    (every decade equally often), towards the inside of [0, VDD];
  * which of the two is the drain is random (the device is symmetric).

Output: dataset/<CASE>/<case>_aug_inputs.csv  ->  simulate with
    python dataset/<CASE>/02_<case>_simulate.py --inputs <case>_aug_inputs.csv --output <case>_aug.csv --simdir sim_<case>_aug
Run (from anywhere):  python dataset/05_augment_inputs.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VDD_NOM, L = 0.9, 22e-9
NOM = {"nmos": dict(vth0=0.3692, toxe=6.5e-10, ndep=1.2e19, u0=0.0181, xj=7.2e-9),
       "pmos": dict(vth0=-0.25399, toxe=6.7e-10, ndep=4.4e18, u0=0.0023, xj=7.2e-9)}
CASES = {  # case: (mos, n extra samples, vg, vb, seed)
    "nmos_on": ("nmos", 5000, "vdd", "0", 1101),
    "nmos_off": ("nmos", 3000, "0", "0", 1102),
    "pmos_on": ("pmos", 5000, "0", "vdd", 1103),
    "pmos_off": ("pmos", 3000, "vdd", "vdd", 1104),
}
VDS_MIN = 1e-6                                     # 1 uV
VDS_MAX = {"on": 3e-2, "off": 0.9}                 # ON: up to 30 mV; OFF: up to the full supply
RAIL_BAND = 0.1                                    # ON: non-rail terminal base within 0.1 V of the rail


def draw(case):
    mos, n, vg, vb, seed = CASES[case]
    state = "on" if case.endswith("_on") else "off"
    rng = np.random.default_rng(seed)
    vdd = rng.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM, n)

    # distance of the "anchor" terminal from its source rail
    on_rail = rng.random(n) < 0.5
    band = RAIL_BAND if state == "on" else vdd
    anchor_off = np.where(on_rail, 0.0, rng.uniform(0, 1, n) * band)
    vds = 10 ** rng.uniform(np.log10(VDS_MIN), np.log10(VDS_MAX[state]), n)
    other_off = anchor_off + vds
    # keep inside [0, VDD]: if the offset would leave the supply range, put it on the other side of the
    # anchor; if that does not fit either, use the far rail (largest possible |Vds|, never Vds = 0)
    out = other_off > vdd
    below = anchor_off - vds
    other_off[out] = np.where(below[out] >= 0, below[out], vdd[out])
    # convert "distance from the source rail" into node voltages
    if mos == "nmos":
        anchor, other = anchor_off, other_off
    else:
        anchor, other = vdd - anchor_off, vdd - other_off
    swap = rng.random(n) < 0.5                     # which terminal is the drain
    vd = np.where(swap, other, anchor)
    vs = np.where(swap, anchor, other)

    w = rng.uniform(1 * L, 8 * L, n)
    temp = rng.uniform(-40.0, 125.0, n)
    proc = {}
    for p, v in NOM[mos].items():
        lo, hi = sorted((0.9 * v, 1.1 * v))
        proc[p] = rng.uniform(lo, hi, n)
    df = pd.DataFrame({"vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs,
                       "vg": vdd.copy() if vg == "vdd" else np.zeros(n),
                       "vb": vdd.copy() if vb == "vdd" else np.zeros(n), **proc})
    assert ((df.vd >= 0) & (df.vd <= df.vdd) & (df.vs >= 0) & (df.vs <= df.vdd)).all()
    return df


def main():
    for case in CASES:
        df = draw(case)
        out = HERE / case.upper() / f"{case}_aug_inputs.csv"
        df.to_csv(out, index=False)
        vds = (df.vd - df.vs).abs()
        rail = (df.vs == 0) | (df.vd == 0) if case.startswith("nmos") else (df.vs == df.vdd) | (df.vd == df.vdd)
        print(f"{case:9s} {len(df):5d} rows -> {out.relative_to(HERE.parent)}   "
              f"|Vds| < 1 mV: {(vds < 1e-3).mean() * 100:4.1f} %   terminal on rail: {rail.mean() * 100:4.1f} %")


if __name__ == "__main__":
    main()
