"""Gate topologies, logic functions, the 3-bit CLA gate list, and the mapping
from a recorded gate operating point to the per-transistor terminal voltages
that are fed to the transistor ML models.

Node naming inside a gate
    'vdd'  supply rail             '0'   ground
    'a','b' gate inputs (driven to exactly 0 or VDD)
    'out'  gate output
    'x'    node between the series NMOS of a NAND2 (also inside AND2)
    'y'    node between the series PMOS of a NOR2  (also inside OR2)
    'nd'   NAND2 output inside AND2      'nr'  NOR2 output inside OR2
"""
import itertools

import numpy as np
import pandas as pd

from config import WN, WP, PROC_PARAMS

# (instance, type, drain, gate, source, body)
_NAND_CORE = [
    ("MPA", "p", "{o}", "a", "vdd", "vdd"),
    ("MPB", "p", "{o}", "b", "vdd", "vdd"),
    ("MNA", "n", "{o}", "a", "x", "0"),      # top of the series stack
    ("MNB", "n", "x", "b", "0", "0"),        # bottom of the series stack
]
_NOR_CORE = [
    ("MPA", "p", "y", "a", "vdd", "vdd"),    # top of the series stack
    ("MPB", "p", "{o}", "b", "y", "vdd"),    # bottom of the series stack
    ("MNA", "n", "{o}", "a", "0", "0"),
    ("MNB", "n", "{o}", "b", "0", "0"),
]


def _fmt(core, o):
    return [tuple(f.format(o=o) for f in t) for t in core]


GATES = {
    "INV": dict(inputs=["a"], internal=[],
                tr=[("MP", "p", "out", "a", "vdd", "vdd"),
                    ("MN", "n", "out", "a", "0", "0")]),
    "NAND2": dict(inputs=["a", "b"], internal=["x"], tr=_fmt(_NAND_CORE, "out")),
    "NOR2": dict(inputs=["a", "b"], internal=["y"], tr=_fmt(_NOR_CORE, "out")),
    "AND2": dict(inputs=["a", "b"], internal=["x", "nd"],
                 tr=_fmt(_NAND_CORE, "nd") + [("MPI", "p", "out", "nd", "vdd", "vdd"),
                                              ("MNI", "n", "out", "nd", "0", "0")]),
    "OR2": dict(inputs=["a", "b"], internal=["y", "nr"],
                tr=_fmt(_NOR_CORE, "nr") + [("MPI", "p", "out", "nr", "vdd", "vdd"),
                                            ("MNI", "n", "out", "nr", "0", "0")]),
}
GATE_ORDER = ["INV", "NAND2", "NOR2", "AND2", "OR2"]
ALL_INTERNAL = ["out", "x", "y", "nd", "nr"]     # every node whose voltage is recorded


def gate_logic(gate, ins):
    """Logic value (0/1) of every logic node of `gate` for input dict `ins`."""
    a = ins.get("a", 0)
    b = ins.get("b", 0)
    if gate == "INV":
        return {"out": 1 - a}
    if gate == "NAND2":
        return {"out": 1 - (a & b)}
    if gate == "NOR2":
        return {"out": 1 - (a | b)}
    if gate == "AND2":
        nd = 1 - (a & b)
        return {"nd": nd, "out": 1 - nd}
    if gate == "OR2":
        nr = 1 - (a | b)
        return {"nr": nr, "out": 1 - nr}
    raise ValueError(gate)


def gate_input_combos(gate):
    """List of (label, {input: bit}) in the order 0, 1 (INV) or 00, 01, 10, 11.
    The label's first character is input A."""
    names = GATES[gate]["inputs"]
    out = []
    for bits in itertools.product([0, 1], repeat=len(names)):
        out.append(("".join(map(str, bits)), dict(zip(names, bits))))
    return out


# ============================================================ gate -> transistors
def _case(ttype, g_logic):
    if ttype == "n":
        return "nmos_on" if g_logic else "nmos_off"
    return "pmos_off" if g_logic else "pmos_on"


def expand_gate_rows(gs):
    """Turn gate_spice rows into one row per transistor.

    Returns a DataFrame with, for every transistor:
      * the 12 model features (vdd, temp, w, vd, vs, vg, vb, vth0..xj)
        - vd / vs come from the SPICE-recorded node voltages (clipped to [0, VDD])
        - vg is the *logic level* (0 or VDD), as the models were trained with
          vg fixed at 0 or VDD
      * vg_spice / vd_raw / vs_raw: the exact SPICE node voltages (used by the
        single-transistor SPICE oracle)
      * case: nmos_on / nmos_off / pmos_on / pmos_off
      * src_d/g/s/b = 1 if that terminal sits on a node held at VDD by a source
        (the VDD rail or an input at logic 1)  -> leakage = current INTO the
        device summed over these terminals
      * snk_d/g/s/b = 1 if that terminal sits on a node held at 0 by a source
        (ground or an input at logic 0)       -> leakage = current OUT of the
        device summed over these terminals (equal to the above by KCL)
    """
    frames = []
    for (gate, inp), sub in gs.groupby(["gate", "inputs"], sort=False):
        ins = {n: int(ch) for n, ch in zip(GATES[gate]["inputs"], inp)}
        logic = gate_logic(gate, ins)
        logic.update(ins)
        vdd = sub["vdd"].to_numpy()

        def node_v(node):
            if node == "vdd":
                return vdd
            if node == "0":
                return np.zeros_like(vdd)
            if node in ins:
                return ins[node] * vdd
            return sub[f"v_{node}"].to_numpy()

        def on_src(node):
            return int(node == "vdd" or (node in ins and ins[node] == 1))

        def on_snk(node):
            return int(node == "0" or (node in ins and ins[node] == 0))

        for name, ttype, d, g, s, b in GATES[gate]["tr"]:
            pre = "n_" if ttype == "n" else "p_"
            gl = logic[g]
            vd_raw, vs_raw = node_v(d), node_v(s)
            f = pd.DataFrame({
                "gate": gate, "inputs": inp, "corner": sub["corner"].to_numpy(),
                "tr": name, "type": ttype, "case": _case(ttype, gl),
                "vdd": vdd, "temp": sub["temp"].to_numpy(),
                "w": WN if ttype == "n" else WP,
                "vd": np.clip(vd_raw, 0.0, vdd), "vs": np.clip(vs_raw, 0.0, vdd),
                "vg": gl * vdd, "vb": node_v(b),
                **{p: sub[pre + p].to_numpy() for p in PROC_PARAMS},
                "vd_raw": vd_raw, "vs_raw": vs_raw, "vg_spice": node_v(g),
                "node_d": d, "node_g": g, "node_s": s, "node_b": b,
                "src_d": on_src(d), "src_g": on_src(g), "src_s": on_src(s), "src_b": on_src(b),
                "snk_d": on_snk(d), "snk_g": on_snk(g), "snk_s": on_snk(s), "snk_b": on_snk(b),
            })
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def leakage_from_terminals(df, cur_prefix=""):
    """Sum terminal currents (dataset sign: + = into the device) into leakage.

    Returns (I_src, I_snk) per row:
      I_src = sum of currents INTO the device at terminals on VDD-level source nodes
      I_snk = sum of currents OUT of the device at terminals on 0-level source nodes
    """
    t = ["d", "g", "s", "b"]
    I = {k: df[f"{cur_prefix}i{k}"].to_numpy() for k in t}
    src = sum(df[f"src_{k}"].to_numpy() * I[k] for k in t)
    snk = sum(df[f"snk_{k}"].to_numpy() * -I[k] for k in t)
    return src, snk


# ============================================================ 3-bit CLA
CLA_INPUTS = ["a0", "a1", "a2", "b0", "b1", "b2", "c0"]
CLA_OUTPUTS = ["s0", "s1", "s2", "c3"]


def cla_gate_list():
    """Gate-level 3-bit carry-lookahead adder, only INV/NAND2/NOR2/AND2/OR2.

        P_i = A_i xor B_i         (XOR = 4 x NAND2)
        G_i = A_i . B_i           (AND2)
        C1  = G0 + P0.C0
        C2  = G1 + P1.G0 + P1.P0.C0
        C3  = G2 + P2.G1 + P2.P1.G0 + P2.P1.P0.C0
        S_i = P_i xor C_i
    Product terms are shared (t0 = P0.C0, u1 = P1.G0, t1 = P1.t0 ...) so every
    carry is computed directly from P, G, C0 (no ripple between carries).

    Returns a list of (instance, gate_type, [input nets], output net) in
    topological order.
    """
    G = []

    def xor(name, a, b, out):
        n1, n2, n3 = f"{name}_n1", f"{name}_n2", f"{name}_n3"
        G.extend([
            (f"{name}a", "NAND2", [a, b], n1),
            (f"{name}b", "NAND2", [a, n1], n2),
            (f"{name}c", "NAND2", [b, n1], n3),
            (f"{name}d", "NAND2", [n2, n3], out),
        ])

    for i in range(3):
        xor(f"xp{i}", f"a{i}", f"b{i}", f"p{i}")
        G.append((f"ag{i}", "AND2", [f"a{i}", f"b{i}"], f"g{i}"))
    # C1
    G.append(("at0", "AND2", ["p0", "c0"], "t0"))          # P0.C0
    G.append(("oc1", "OR2", ["g0", "t0"], "c1"))
    # C2
    G.append(("au1", "AND2", ["p1", "g0"], "u1"))          # P1.G0
    G.append(("at1", "AND2", ["p1", "t0"], "t1"))          # P1.P0.C0
    G.append(("oc2a", "OR2", ["g1", "u1"], "o2a"))
    G.append(("oc2", "OR2", ["o2a", "t1"], "c2"))
    # C3
    G.append(("av2", "AND2", ["p2", "g1"], "v2"))          # P2.G1
    G.append(("aw2", "AND2", ["p2", "u1"], "w2"))          # P2.P1.G0
    G.append(("ax2", "AND2", ["p2", "t1"], "x2"))          # P2.P1.P0.C0
    G.append(("oc3a", "OR2", ["g2", "v2"], "o3a"))
    G.append(("oc3b", "OR2", ["o3a", "w2"], "o3b"))
    G.append(("oc3", "OR2", ["o3b", "x2"], "c3"))
    # sums
    xor("xs0", "p0", "c0", "s0")
    xor("xs1", "p1", "c1", "s1")
    xor("xs2", "p2", "c2", "s2")
    return G


_LOGIC2 = {"INV": lambda a, b: 1 - a, "NAND2": lambda a, b: 1 - (a & b),
           "NOR2": lambda a, b: 1 - (a | b), "AND2": lambda a, b: a & b,
           "OR2": lambda a, b: a | b}


def cla_eval(bits):
    """Logic-simulate the CLA. `bits` = dict input net -> 0/1.
    Returns (net values dict, list of (instance, gate_type, input label))."""
    v = dict(bits)
    used = []
    for inst, gt, ins, out in cla_gate_list():
        a = v[ins[0]]
        b = v[ins[1]] if len(ins) > 1 else 0
        v[out] = _LOGIC2[gt](a, b)
        used.append((inst, gt, "".join(str(v[n]) for n in ins)))
    return v, used


def cla_vectors():
    """All 128 input vectors as a list of dicts.
    Vector index k: bit j of k is the value of CLA_INPUTS[j]
    (bit 0 = a0, bit 1 = a1, ..., bit 6 = c0)."""
    vecs = []
    for k in range(2 ** len(CLA_INPUTS)):
        vecs.append({n: (k >> j) & 1 for j, n in enumerate(CLA_INPUTS)})
    return vecs


def cla_transistors():
    """Flatten the CLA to transistors: list of (inst_name, type, d, g, s, b)."""
    out = []
    for inst, gt, ins, o in cla_gate_list():
        local = {"vdd": "vdd", "0": "0", "out": o, "a": ins[0]}
        if len(ins) > 1:
            local["b"] = ins[1]
        for n in GATES[gt]["internal"]:
            local[n] = f"{inst}_{n}"
        for name, tt, d, g, s, b in GATES[gt]["tr"]:
            out.append((f"M{inst}_{name[1:]}", tt, local[d], local[g], local[s], local[b]))
    return out
