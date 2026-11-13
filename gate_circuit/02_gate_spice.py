"""Step 2 - SPICE reference for the 5 gates (18 gate/input cases x 200 corners).

* Writes one example netlist per gate (nominal corner, every input
  combination) to netlists/  -> deliverable 2b
* Simulates every gate as a complete transistor netlist (AND2 / OR2 are full
  gates, not NAND2 + INV added) at every corner and input combination.
* Records VDD, temperature, process draws, every internal node voltage, the
  output voltage and every source current.

Leakage (same definition everywhere):
    I_leak = current delivered by the VDD source
           + current delivered by every input source held at logic 1
           = -i(Vdd) - sum_{inputs at 1} i(Vin)      (ngspice sign convention)

Output: data/gate_spice.csv
"""
import numpy as np
import pandas as pd

from circuits import ALL_INTERNAL, GATE_ORDER, GATES, gate_input_combos, gate_logic
from config import (CORNERS_CSV, GATE_SPICE_CSV, NETLIST_DIR, NMOS_NOM, PMOS_NOM,
                    PROC_PARAMS, SIM_DIR, VDD_NOM, WN, WP, L, ensure_dirs)
from spice_utils import control_block, corner_cmds, header, run_jobs


def circuit_lines(gate):
    g = GATES[gate]
    lines = []
    for name, tt, d, gg, s, b in g["tr"]:
        model, w = ("nmos", WN) if tt == "n" else ("pmos", WP)
        lines.append(f"{name} {d} {gg} {s} {b} {model} W={w:.4e} L={L:.4e}")
    lines.append("Vdd vdd 0 0")
    for i in g["inputs"]:
        lines.append(f"V{i} {i} 0 0")
    return "\n".join(lines) + "\n\n"


def vectors(gate):
    g = GATES[gate]
    nodes = ["out"] + g["internal"]
    vec = ([f"v({i})" for i in g["inputs"]] + [f"v({n})" for n in nodes]
           + ["i(vdd)"] + [f"i(v{i})" for i in g["inputs"]]
           + ["v(vdd)", "tc", "@nmos[vth0]", "@pmos[vth0]", "@nmos[toxe]", "@pmos[toxe]"])
    names = ([f"rb_{i}" for i in g["inputs"]] + [f"v_{n}" for n in nodes]
             + ["i_vdd"] + [f"i_v{i}" for i in g["inputs"]]
             + ["rb_vdd", "rb_temp", "rb_n_vth0", "rb_p_vth0", "rb_n_toxe", "rb_p_toxe"])
    return vec, names


def input_cmds(gate, ins, vdd):
    return "".join(f"alter v{i} dc={ins[i] * vdd:.10e}\n" for i in GATES[gate]["inputs"])


def example_netlist(gate):
    """Human-readable netlist: nominal corner, every input combination."""
    nom = {"vdd": VDD_NOM, "temp": 27.0,
           **{"n_" + p: NMOS_NOM[p] for p in PROC_PARAMS},
           **{"p_" + p: PMOS_NOM[p] for p in PROC_PARAMS}}
    body = corner_cmds(nom)
    vec, _ = vectors(gate)
    for label, ins in gate_input_combos(gate):
        body += f"* inputs {label}\n" + input_cmds(gate, ins, VDD_NOM) + "op\n"
        body += "print " + " ".join(vec[:len(vec) - 6]) + "\n"
    return header(f"{gate} - 22nm_MGK, Wn={WN:.2e} Wp={WP:.2e}") + circuit_lines(gate) + control_block(body)


def main():
    ensure_dirs()
    corners = pd.read_csv(CORNERS_CSV)
    run_dir = SIM_DIR / "gates"
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs, plan = [], []
    for gate in GATE_ORDER:
        (NETLIST_DIR / f"{gate.lower()}.net").write_text(example_netlist(gate))
        vec, names = vectors(gate)
        combos = gate_input_combos(gate)
        body, rows = "", []
        for _, c in corners.iterrows():
            body += corner_cmds(c)
            for label, ins in combos:
                body += input_cmds(gate, ins, c["vdd"]) + "op\n"
                body += f"wrdata {gate.lower()}_out.txt " + " ".join(vec) + "\n"
                rows.append((int(c["corner"]), label, ins))
        net = run_dir / f"{gate.lower()}.net"
        net.write_text(header(f"{gate} batch") + circuit_lines(gate) + control_block(body))
        jobs.append((net, run_dir / f"{gate.lower()}_out.txt", len(rows), len(vec)))
        plan.append((gate, names, rows))

    print(f"Running {len(jobs)} gate netlists ...")
    datas = run_jobs(jobs, label="gates", stack=False)   # all gates in parallel
    results = []
    for (gate, names, rows), data in zip(plan, datas):
        d = pd.DataFrame(data, columns=names)
        d.insert(0, "gate", gate)
        d.insert(1, "inputs", [r[1] for r in rows])
        d.insert(2, "corner", [r[0] for r in rows])
        results.append(d)
    df = pd.concat(results, ignore_index=True)

    # attach corner parameters (the truth), check read-backs
    df = df.merge(corners, on="corner", how="left")
    checks = {"rb_vdd": "vdd", "rb_temp": "temp", "rb_n_vth0": "n_vth0",
              "rb_p_vth0": "p_vth0", "rb_n_toxe": "n_toxe", "rb_p_toxe": "p_toxe"}
    worst = max(float(((df[a] - df[b]).abs() / df[b].abs()).max()) for a, b in checks.items())
    print(f"Largest corner read-back mismatch: {worst:.2e}")
    if worst > 1e-6:
        raise RuntimeError("Corner parameters were not applied as intended")

    # logic bits, expected output, leakage
    df["a"] = df["inputs"].str[0].astype(int)
    df["b"] = np.where(df["gate"] == "INV", np.nan, df["inputs"].str[-1].astype(float))
    df["out_logic"] = [gate_logic(g, {"a": int(s[0]), "b": int(s[-1])})["out"]
                       for g, s in zip(df["gate"], df["inputs"])]
    leak = -df["i_vdd"].to_numpy().copy()
    leak -= np.where(df["a"] == 1, df["i_va"], 0.0)
    leak -= np.where(df["b"] == 1, df["i_vb"].fillna(0.0), 0.0)
    df["i_leak_spice"] = leak

    out_err = (df["v_out"] - df["out_logic"] * df["vdd"]).abs()
    print(f"Output |V(out) - ideal| : median {out_err.median():.2e} V, max {out_err.max():.2e} V")
    if (out_err > 0.5 * df["vdd"]).any():
        raise RuntimeError("A gate output is on the wrong logic level")
    if (df["i_leak_spice"] <= 0).any():
        print("WARNING: non-positive leakage found in", int((df.i_leak_spice <= 0).sum()), "rows")

    for n in ALL_INTERNAL:
        if f"v_{n}" not in df:
            df[f"v_{n}"] = np.nan
    cols = (["gate", "inputs", "a", "b", "corner", "vdd", "temp"]
            + [c for c in corners.columns if c[:2] in ("n_", "p_")]
            + [f"v_{n}" for n in ALL_INTERNAL] + ["out_logic", "i_vdd", "i_va", "i_vb", "i_leak_spice"])
    df[cols].to_csv(GATE_SPICE_CSV, index=False)
    print(f"Saved {len(df)} rows -> {GATE_SPICE_CSV}")
    print(df.groupby(["gate", "inputs"], sort=False)["i_leak_spice"].median().to_string())


if __name__ == "__main__":
    main()
