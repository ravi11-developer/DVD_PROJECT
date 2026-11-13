"""Step 3 - break every gate operating point into its transistors and simulate
each transistor ALONE in SPICE at exactly the same terminal voltages
(the "oracle").

Why:
  * It builds data/gate_transistors.csv, the table the ML step (05) predicts
    on: one row per transistor with the 12 model features, the model case
    (nmos_on/off, pmos_on/off) and which terminals count toward leakage.
  * Summing the oracle currents with the same terminal bookkeeping must give
    back the gate SPICE leakage. That proves the voltage mapping and the sign
    convention are right, so any later ML-vs-SPICE error is model error (or
    the vg-at-logic-level assumption), not bookkeeping.

Currents use the dataset sign convention: + = current INTO the device terminal
(= -i(V) of the terminal's source).
"""
import numpy as np
import pandas as pd

from circuits import expand_gate_rows, leakage_from_terminals
from config import GATE_SPICE_CSV, GATE_TRANSISTORS_CSV, L, PROC_PARAMS, SIM_DIR, ensure_dirs
from spice_utils import control_block, header, run_jobs

BATCH = 1500
VEC = ("v(drain) v(gate) v(source) v(body) i(vd) i(vg) i(vs) i(vb) "
       "@{m}[vth0] @{m}[toxe] v(ref) tc")
VEC_NAMES = ["rb_vd", "rb_vg", "rb_vs", "rb_vb", "i_vd", "i_vg", "i_vs", "i_vb",
             "rb_vth0", "rb_toxe", "rb_vdd", "rb_temp"]


def netlist(ttype, rows, out_name):
    model, inst = ("nmos", "mn") if ttype == "n" else ("pmos", "mp")
    circ = (f"{inst.upper()} drain gate source body {model} W=1e-7 L={L:.4e}\n"
            "Vd drain 0 0\nVg gate 0 0\nVs source 0 0\nVb body 0 0\nVref ref 0 0\n\n")
    body = []
    vec = VEC.format(m=model)
    for r in rows.itertuples():
        body.append("\n".join(
            [f"altermod {model} {p}={getattr(r, p):.10e}" for p in PROC_PARAMS]
            + [f"alter {inst} w={r.w:.10e}",
               f"alter vref dc={r.vdd:.10e}",
               f"alter vd dc={r.vd_raw:.10e}", f"alter vg dc={r.vg_spice:.10e}",
               f"alter vs dc={r.vs_raw:.10e}", f"alter vb dc={r.vb:.10e}",
               f"option temp={r.temp:.10e}", f"let tc = {r.temp:.10e}",
               "op", f"wrdata {out_name} {vec}"]) + "\n")
    return header(f"single {model} oracle") + circ + control_block("".join(body))


def main():
    ensure_dirs()
    gs = pd.read_csv(GATE_SPICE_CSV, dtype={"inputs": str})
    tr = expand_gate_rows(gs)
    print(f"{len(gs)} gate operating points -> {len(tr)} transistor operating points")

    run_dir = SIM_DIR / "oracle"
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs, idx = [], []
    for ttype in ("n", "p"):
        sub = tr[tr["type"] == ttype]
        for k in range(0, len(sub), BATCH):
            chunk = sub.iloc[k:k + BATCH]
            out = f"{ttype}_{k // BATCH:03d}_out.txt"
            net = run_dir / f"{ttype}_{k // BATCH:03d}.net"
            net.write_text(netlist(ttype, chunk, out))
            jobs.append((net, run_dir / out, len(chunk), len(VEC_NAMES)))
            idx.append(chunk.index.to_numpy())
    data = run_jobs(jobs, label="oracle")
    res = pd.DataFrame(data, columns=VEC_NAMES, index=np.concatenate(idx)).sort_index()

    # read-back checks
    for a, b in (("rb_vd", "vd_raw"), ("rb_vg", "vg_spice"), ("rb_vs", "vs_raw"), ("rb_vb", "vb"),
                 ("rb_vth0", "vth0"), ("rb_vdd", "vdd"), ("rb_temp", "temp")):
        m = float((res[a] - tr[b]).abs().max())
        if m > 1e-6 * max(1.0, float(tr[b].abs().max())):
            raise RuntimeError(f"read-back mismatch {a} vs {b}: {m}")

    for k in ("d", "g", "s", "b"):
        tr[f"i{k}_spice"] = -res[f"i_v{k}"].to_numpy()

    # bookkeeping check: oracle leakage vs gate SPICE leakage
    tmp = tr.rename(columns={f"i{k}_spice": f"i{k}" for k in "dgsb"})
    tr["leak_src_spice"], tr["leak_snk_spice"] = leakage_from_terminals(tmp)
    agg = tr.groupby(["gate", "inputs", "corner"], sort=False)[["leak_src_spice", "leak_snk_spice"]].sum()
    chk = gs.set_index(["gate", "inputs", "corner"])[["i_leak_spice"]].join(agg)
    for c in ("leak_src_spice", "leak_snk_spice"):
        rel = ((chk[c] - chk["i_leak_spice"]).abs() / chk["i_leak_spice"].abs())
        print(f"Oracle {c[5:8]}-side sum vs gate SPICE: median rel diff {rel.median():.2e}, max {rel.max():.2e}")

    keep = (["gate", "inputs", "corner", "tr", "type", "case"]
            + ["vdd", "temp", "w", "vd", "vs", "vg", "vb"] + PROC_PARAMS
            + ["vd_raw", "vs_raw", "vg_spice", "node_d", "node_g", "node_s", "node_b"]
            + [f"src_{k}" for k in "dgsb"] + [f"snk_{k}" for k in "dgsb"]
            + [f"i{k}_spice" for k in "dgsb"])
    tr[keep].to_csv(GATE_TRANSISTORS_CSV, index=False)
    print(f"Saved {len(tr)} rows -> {GATE_TRANSISTORS_CSV}")
    print(tr["case"].value_counts().to_string())


if __name__ == "__main__":
    main()
