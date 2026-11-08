import sys; sys.path.insert(0, __file__.rsplit("/",1)[0])
from spice_lib import *
for kind, vg, vb in [("nmos", 0.9, 0.0), ("nmos", 0.0, 0.0), ("pmos", 0.0, 0.9), ("pmos", 0.9, 0.9)]:
    s = dict(NOM[kind], w=88e-9, temp=100.0, vd=0.45, vs=0.0 if kind == "nmos" else 0.9, vg=vg, vb=vb)
    nl = OPTS + "\n.temp 100\n" + fet_block(0, kind, s)
    r = run_ngspice(nl, ["i(vd0)", "i(vg0)", "i(vs0)", "i(vb0)"])
    # SPICE i(V) flows + -> - through the source, so current INTO the device terminal = -i(V)
    I = [-r["i(vd0)"], -r["i(vg0)"], -r["i(vs0)"], -r["i(vb0)"]]
    print(kind, "Vg=%.1f Vb=%.1f" % (vg, vb), "Id Ig Is Ib =", ["%.3e" % x for x in I], "sum=%.2e" % sum(I))
