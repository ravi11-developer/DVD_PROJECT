"""Shared helpers: model-card patching, single-FET netlists, ngspice runner."""
import re, subprocess, os, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = next(p for p in [os.path.join(ROOT, "model", "22nm_MGK.pm"),
                         os.path.join(ROOT, "..", "..", "..", "model", "22nm_MGK.pm")] if os.path.exists(p))
LMIN = 22e-9
VDD_NOM = 0.9
OPTS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
PCOLS = ["vth0", "toxe", "ndep", "u0", "xj"]

def _cards():
    txt = open(MODEL).read()
    i = txt.index(".model  pmos")
    return txt[:i], txt[i:]
_N, _P = _cards()

def nominal(card):
    return {p: float(re.search(r"\b%s\s*=\s*(\S+)" % p, card).group(1)) for p in PCOLS}
NOM = {"nmos": nominal(_N), "pmos": nominal(_P)}

def model_card(name, kind, over):
    """Return a .model card named `name` with overridden parameters."""
    card = _N if kind == "nmos" else _P
    card = re.sub(r"\.model\s+\w+\s+\w+", ".model %s %s" % (name, kind), card, count=1)
    for p, v in over.items():
        card = re.sub(r"\b%s(\s*)=(\s*)\S+" % p, lambda m: "%s%s=%s%.6e" % (p, m.group(1), m.group(2), v), card, count=1)
    return card.strip()

def fet_block(i, kind, s):
    """Single FET with its own V source on every terminal. s: dict of sample values.
    Uses instance dtemp = T - 27 for per-sample temperature."""
    d, g, so, b = ("d%d" % i, "g%d" % i, "s%d" % i, "b%d" % i)
    return "\n".join([
        model_card("m%d" % i, kind, {p: s[p] for p in PCOLS}),
        "M%d %s %s %s %s m%d W=%.6e L=%.6e" % (i, d, g, so, b, i, s["w"], LMIN),
        "Vd%d %s 0 %.6f" % (i, d, s["vd"]), "Vg%d %s 0 %.6f" % (i, g, s["vg"]),
        "Vs%d %s 0 %.6f" % (i, so, s["vs"]), "Vb%d %s 0 %.6f" % (i, b, s["vb"])])

def run_ngspice(netlist, prints, tag="run"):
    """Run netlist (op analysis), print vectors -> returns dict name->float."""
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "n.cir"); o = os.path.join(td, "o.txt")
        ctl = ".control\nset width=200\nop\nprint " + " ".join(prints) + " > " + o + "\n.endc\n"
        open(f, "w").write("* %s\n%s\n%s\n.end\n" % (tag, netlist, ctl))
        r = subprocess.run(["ngspice", "-b", f], capture_output=True, text=True)
        if not os.path.exists(o):
            raise RuntimeError(r.stdout[-2000:] + r.stderr[-500:])
        out = {}
        for ln in open(o):
            m = re.match(r"\s*([\w#\.\(\)]+)\s*=\s*(\S+)", ln)
            if m: out[m.group(1).lower()] = float(m.group(2))
        return out
