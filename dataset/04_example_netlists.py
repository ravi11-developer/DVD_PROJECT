"""
04_example_netlists.py - deliverable 2 (transistor part): one single-transistor netlist per case.

Writes dataset/<CASE>/example_<case>.net for the first row of <case>.csv, runs it in ngspice and
checks that the four currents match the dataset row. Every terminal has its own voltage source and
the required .options line is included, as the project document asks.

Run (from anywhere):  python dataset/04_example_netlists.py
"""
import os, re, shutil, subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = (HERE.parent / "model" / "22nm_MGK.pm").resolve()
NGSPICE = os.environ.get("NGSPICE") or shutil.which("ngspice") or "ngspice"
L = 22e-9
CASES = {"NMOS_ON": "nmos", "NMOS_OFF": "nmos", "PMOS_ON": "pmos", "PMOS_OFF": "pmos"}


def netlist(case, mos, r):
    inst = "Mn" if mos == "nmos" else "Mp"
    return f"""* {case.replace('_', ' ')} - single-transistor example netlist (22nm_MGK, row 0 of {case.lower()}.csv)
* Vg/Vb fixed by the case; Vd, Vs, VDD, T, W and the process parameters are one random sample.
.include "../../model/22nm_MGK.pm"
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
.temp {r.temp:.9g}

{inst} drain gate source body {mos} W={r.w:.9g} L={L:.4g}
Vd drain  0 {r.vd:.9g}
Vg gate   0 {r.vg:.9g}
Vs source 0 {r.vs:.9g}
Vb body   0 {r.vb:.9g}

.control
* process sample (nominal +-10 %)
altermod {mos} vth0={r.vth0:.9g}
altermod {mos} toxe={r.toxe:.9g}
altermod {mos} ndep={r.ndep:.9g}
altermod {mos} u0={r.u0:.9g}
altermod {mos} xj={r.xj:.9g}
op
* currents INTO each terminal (dataset sign convention) = -i(source)
let id = -i(vd)
let ig = -i(vg)
let is = -i(vs)
let ib = -i(vb)
print id ig is ib
.endc
.end
"""


def run(path):
    out = subprocess.run([NGSPICE, "-b", path.name], cwd=path.parent, capture_output=True, text=True).stdout
    vals = {}
    for k in ("id", "ig", "is", "ib"):
        m = re.search(rf"^{k}\s*=\s*([-+0-9.eE]+)", out, re.M)
        if not m:
            raise RuntimeError(f"{path.name}: could not read {k} from ngspice output:\n{out[-800:]}")
        vals[k] = float(m.group(1))
    return vals


def main():
    for case, mos in CASES.items():
        d = pd.read_csv(HERE / case / f"{case.lower()}.csv")
        d.columns = d.columns.str.strip()
        r = d.iloc[0]
        path = HERE / case / f"example_{case.lower()}.net"
        path.write_text(netlist(case, mos, r))
        sim = run(path)
        errs = {c: abs(sim[c] - r[c]) / abs(r[c]) * 100 for c in ("id", "ig", "is", "ib")}
        worst = max(errs.values())
        print(f"{case:8s} {path.relative_to(HERE.parent)}  max |sim - dataset| = {worst:.2e} %  "
              f"({'OK' if worst < 1e-2 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
