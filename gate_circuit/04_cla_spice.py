"""Step 4 - SPICE reference for the 3-bit carry-lookahead adder (Part C).

* Flattens the gate-level CLA (circuits.cla_gate_list: 24 NAND2 + 9 AND2 +
  6 OR2 = 39 gates, 186 transistors) to ONE transistor-level netlist.
* Writes an example netlist (nominal corner, all 128 vectors) to
  netlists/cla3.net  -> deliverable 2b
* Simulates all 128 input vectors at the same 200 corners as the gates.
* Records every gate-output net voltage, every source current and the
  leakage (same definition as the gates):
      I_leak = -i(Vdd) - sum_{inputs at 1} i(Vin)
* Checks the SPICE outputs S0..S2, C3 against a logic simulation.

Output: data/cla_spice.csv
"""
import numpy as np
import pandas as pd

from circuits import CLA_INPUTS, CLA_OUTPUTS, cla_eval, cla_gate_list, cla_transistors, cla_vectors
from config import (CLA_SPICE_CSV, CORNERS_CSV, NETLIST_DIR, NMOS_NOM, PMOS_NOM, PROC_PARAMS,
                    SIM_DIR, VDD_NOM, WN, WP, L, ensure_dirs)
from spice_utils import control_block, corner_cmds, header, run_jobs

CORNERS_PER_FILE = 4
GATE_NETS = [g[3] for g in cla_gate_list()]


def circuit_lines():
    lines = [f"* {len(cla_gate_list())} gates, {len(cla_transistors())} transistors"]
    for name, tt, d, g, s, b in cla_transistors():
        model, w = ("nmos", WN) if tt == "n" else ("pmos", WP)
        lines.append(f"{name} {d} {g} {s} {b} {model} W={w:.4e} L={L:.4e}")
    lines.append("Vdd vdd 0 0")
    lines += [f"V{n} {n} 0 0" for n in CLA_INPUTS]
    return "\n".join(lines) + "\n\n"


VEC = ([f"v({n})" for n in CLA_INPUTS] + [f"v({n})" for n in GATE_NETS] + ["i(vdd)"]
       + [f"i(v{n})" for n in CLA_INPUTS]
       + ["v(vdd)", "tc", "@nmos[vth0]", "@pmos[vth0]"])
NAMES = ([f"rb_{n}" for n in CLA_INPUTS] + [f"v_{n}" for n in GATE_NETS] + ["i_vdd"]
         + [f"i_v{n}" for n in CLA_INPUTS]
         + ["rb_vdd", "rb_temp", "rb_n_vth0", "rb_p_vth0"])


def vector_cmds(bits, vdd):
    return "".join(f"alter v{n} dc={bits[n] * vdd:.10e}\n" for n in CLA_INPUTS)


def wrdata_cmd(out_name):
    return f"wrdata {out_name} " + " ".join(VEC) + "\n"


def example_netlist():
    nom = {"vdd": VDD_NOM, "temp": 27.0,
           **{"n_" + p: NMOS_NOM[p] for p in PROC_PARAMS},
           **{"p_" + p: PMOS_NOM[p] for p in PROC_PARAMS}}
    body = corner_cmds(nom)
    for k, bits in enumerate(cla_vectors()):
        lab = " ".join(f"{n}={bits[n]}" for n in CLA_INPUTS)
        body += f"* vector {k}: {lab}\n" + vector_cmds(bits, VDD_NOM) + "op\n"
        body += "print " + " ".join(f"v({n})" for n in CLA_OUTPUTS) + " i(vdd)\n"
    return (header(f"3-bit CLA - 22nm_MGK, Wn={WN:.2e} Wp={WP:.2e}")
            + circuit_lines() + control_block(body))


def main():
    ensure_dirs()
    (NETLIST_DIR / "cla3.net").write_text(example_netlist())
    corners = pd.read_csv(CORNERS_CSV)
    vecs = cla_vectors()
    run_dir = SIM_DIR / "cla"
    run_dir.mkdir(parents=True, exist_ok=True)
    circ = circuit_lines()

    jobs, rows = [], []
    for k in range(0, len(corners), CORNERS_PER_FILE):
        chunk = corners.iloc[k:k + CORNERS_PER_FILE]
        out = f"cla_{k // CORNERS_PER_FILE:03d}_out.txt"
        body = ""
        for _, c in chunk.iterrows():
            body += corner_cmds(c)
            for vi, bits in enumerate(vecs):
                body += vector_cmds(bits, c["vdd"]) + "op\n" + wrdata_cmd(out)
                rows.append((int(c["corner"]), vi))
        net = run_dir / f"cla_{k // CORNERS_PER_FILE:03d}.net"
        net.write_text(header("3-bit CLA batch") + circ + control_block(body))
        jobs.append((net, run_dir / out, len(chunk) * len(vecs), len(VEC)))

    print(f"Running {len(jobs)} CLA netlists ({len(rows)} operating points) ...")
    data = run_jobs(jobs, label="cla")
    df = pd.DataFrame(data, columns=NAMES)
    df.insert(0, "corner", [r[0] for r in rows])
    df.insert(1, "vector", [r[1] for r in rows])
    for n in CLA_INPUTS:
        df[n] = [vecs[v][n] for v in df["vector"]]
    df = df.merge(corners, on="corner", how="left")

    worst = max(float(((df[a] - df[b]).abs() / df[b].abs()).max())
                for a, b in (("rb_vdd", "vdd"), ("rb_temp", "temp"),
                             ("rb_n_vth0", "n_vth0"), ("rb_p_vth0", "p_vth0")))
    print(f"Largest corner read-back mismatch: {worst:.2e}")
    if worst > 1e-6:
        raise RuntimeError("Corner parameters were not applied as intended")

    # logic check of every gate output net
    logic = {v: cla_eval(bits)[0] for v, bits in enumerate(vecs)}
    max_dev = 0.0
    for n in GATE_NETS:
        ideal = np.array([logic[v][n] for v in df["vector"]]) * df["vdd"].to_numpy()
        dev = np.abs(df[f"v_{n}"].to_numpy() - ideal)
        max_dev = max(max_dev, float(dev.max()))
        if (dev > 0.5 * df["vdd"].to_numpy()).any():
            raise RuntimeError(f"net {n} is on the wrong logic level")
    print(f"All 39 gate outputs on the correct logic level; max |V - ideal| = {max_dev:.3e} V")
    for n in CLA_OUTPUTS:
        df[f"{n}_logic"] = [logic[v][n] for v in df["vector"]]
    # adder check: A + B + Cin = S + 8*C3
    A = df.a0 + 2 * df.a1 + 4 * df.a2
    B = df.b0 + 2 * df.b1 + 4 * df.b2
    S = df.s0_logic + 2 * df.s1_logic + 4 * df.s2_logic + 8 * df.c3_logic
    assert (A + B + df.c0 == S).all(), "CLA logic is not an adder"

    leak = -df["i_vdd"].to_numpy().copy()
    for n in CLA_INPUTS:
        leak -= np.where(df[n] == 1, df[f"i_v{n}"], 0.0)
    df["i_leak_spice"] = leak
    df = df.drop(columns=[c for c in df.columns if c.startswith("rb_")])
    df.to_csv(CLA_SPICE_CSV, index=False)
    print(f"Saved {len(df)} rows -> {CLA_SPICE_CSV}")
    print("CLA SPICE leakage: median {:.3e} A, min {:.3e}, max {:.3e}".format(
        df.i_leak_spice.median(), df.i_leak_spice.min(), df.i_leak_spice.max()))


if __name__ == "__main__":
    main()
