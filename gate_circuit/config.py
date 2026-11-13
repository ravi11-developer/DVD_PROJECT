"""Shared settings for Part B (gates) and Part C (3-bit CLA).

Every script in this folder imports from here, so changing a value here
changes it everywhere.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent          # gate_circuit/
REPO = HERE.parent
MODEL_FILE = REPO / "model" / "22nm_MGK.pm"
NGSPICE = os.environ.get("NGSPICE", "ngspice")  # override with env var if needed

DATA_DIR = HERE / "data"            # exported CSVs
NETLIST_DIR = HERE / "netlists"     # example netlists (deliverable 2b)
SIM_DIR = HERE / "sim"              # ngspice batch scratch (not committed)
RESULTS_DIR = HERE / "results"      # tables + plots (deliverables 6, 7, 8)

# ---------------------------------------------------------------- technology
VDD_NOM = 0.9
L = 22e-9
WN = 2 * L          # every NMOS in every gate
WP = 4 * L          # every PMOS in every gate

# Nominal values from model/22nm_MGK.pm (same as dataset/*_inputs.py)
NMOS_NOM = dict(vth0=0.3692, toxe=6.5e-10, ndep=1.2e19, u0=0.0181, xj=7.2e-9)
PMOS_NOM = dict(vth0=-0.25399, toxe=6.7e-10, ndep=4.4e18, u0=0.0023, xj=7.2e-9)
PROC_PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]

VDD_SPREAD = 0.10   # VDD uniform in VDD_NOM * (1 +- 0.10)
PROC_SPREAD = 0.10  # each process parameter uniform in nominal * (1 +- 0.10)
T_MIN, T_MAX = -40.0, 125.0

# ---------------------------------------------------------------- corners
N_CORNERS = 200
CORNER_SEED = 2026
CORNERS_CSV = DATA_DIR / "corners.csv"

# ---------------------------------------------------------------- SPICE
SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# ---------------------------------------------------------------- exported data
GATE_SPICE_CSV = DATA_DIR / "gate_spice.csv"              # one row per gate/input/corner
GATE_TRANSISTORS_CSV = DATA_DIR / "gate_transistors.csv"  # one row per transistor (+ SPICE oracle currents)
CLA_SPICE_CSV = DATA_DIR / "cla_spice.csv"                # one row per CLA input vector/corner

# Outputs of the ML / analysis scripts (05..08)
GATE_ML_CSV = DATA_DIR / "gate_ml.csv"
GATE_TRANSISTORS_ML_CSV = DATA_DIR / "gate_transistors_ml.csv"
CLA_ML_CSV = DATA_DIR / "cla_ml.csv"

# Column order the transistor models were trained on (dataset/*.csv)
FEATURES = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
            "vth0", "toxe", "ndep", "u0", "xj"]
CURRENTS = ["id", "ig", "is", "ib"]


def ensure_dirs():
    for d in (DATA_DIR, NETLIST_DIR, SIM_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
