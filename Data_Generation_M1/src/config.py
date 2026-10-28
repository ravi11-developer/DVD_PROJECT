#!/usr/bin/env python3
"""
config.py — Configuration for the leakage current estimation project.
=====================================================================
Modify the USER SETTINGS below to match your team assignment.
Everything else is derived automatically.
"""

import os
import re

# ╔════════════════════════════════════════════════════════════════╗
# ║  USER SETTINGS — Modify these for your team assignment        ║
# ╚════════════════════════════════════════════════════════════════╝

MODEL_FILE   = "model/22nm_MGK.pm"    # Your assigned PTM model file (relative to project root)
NOMINAL_VDD  = 0.9                     # Your assigned nominal VDD (Volts)
TECH_NODE_NM = 22                      # Technology node in nm (16, 22, 32, or 45)

NUM_SAMPLES  = 10000                   # Number of random samples per transistor case
RANDOM_SEED  = 42                      # Fixed seed for reproducibility
NUM_WORKERS  = 8                       # Number of parallel NGSPICE processes

# ╔════════════════════════════════════════════════════════════════╗
# ║  DERIVED SETTINGS — Usually no need to modify                 ║
# ╚════════════════════════════════════════════════════════════════╝

# Paths (all relative to project root)
PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_DIR, MODEL_FILE)
DATA_DIR     = os.path.join(PROJECT_DIR, "data")
TEMP_DIR     = os.path.join(PROJECT_DIR, "netlists", "temp")

# Physical parameters
LMIN = TECH_NODE_NM * 1e-9                              # Minimum channel length (m)

# Parameter sweep ranges
VDD_LO, VDD_HI   = NOMINAL_VDD * 0.9, NOMINAL_VDD * 1.1  # VDD ± 10%
TEMP_LO, TEMP_HI  = -40.0, 125.0                          # Temperature range (°C)
W_LO, W_HI        = 1 * LMIN, 8 * LMIN                    # Width: 1×Lmin to 8×Lmin
PROCESS_VAR        = 0.10                                  # Process variation ± 10%

# Transistor case definitions
# vg_is_vdd: True  → gate is tied to VDD  (transistor ON for NMOS, OFF for PMOS)
# vb_is_vdd: True  → body is tied to VDD  (PMOS substrate bias)
CASES = {
    "nmos_on":  {"mos_type": "nmos", "vg_is_vdd": True,  "vb_is_vdd": False},
    "nmos_off": {"mos_type": "nmos", "vg_is_vdd": False, "vb_is_vdd": False},
    "pmos_on":  {"mos_type": "pmos", "vg_is_vdd": False, "vb_is_vdd": True},
    "pmos_off": {"mos_type": "pmos", "vg_is_vdd": True,  "vb_is_vdd": True},
}

# SPICE simulation options (required by project spec)
SPICE_OPTIONS = "GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9"
NGSPICE_TIMEOUT = 30  # Max seconds per single simulation


# ╔════════════════════════════════════════════════════════════════╗
# ║  MODEL FILE PARSER                                            ║
# ╚════════════════════════════════════════════════════════════════╝

PROCESS_PARAMS = ['vth0', 'toxe', 'ndep', 'u0', 'xj']

def parse_nominal_params(filepath):
    """
    Parse a PTM .pm model file and extract nominal values of the
    5 process parameters (vth0, toxe, ndep, u0, xj) for both NMOS and PMOS.

    Returns:
        dict: {"nmos": {"vth0": float, ...}, "pmos": {"vth0": float, ...}}
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Split the file into NMOS section and PMOS section
    # The PMOS model starts with ".model  pmos" (with variable whitespace)
    pmos_match = re.search(r'\.model\s+pmos', content, re.IGNORECASE)
    if pmos_match:
        nmos_text = content[:pmos_match.start()]
        pmos_text = content[pmos_match.start():]
    else:
        nmos_text = content
        pmos_text = ""

    def _extract(text):
        """Extract the 5 process parameters from a section of model text."""
        params = {}
        for p in PROCESS_PARAMS:
            # Match patterns like: vth0 = 0.3692  or  toxe = 6.5e-010
            m = re.search(
                rf'{p}\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
                text, re.IGNORECASE
            )
            if m:
                params[p] = float(m.group(1))
            else:
                raise ValueError(f"Could not find parameter '{p}' in model text")
        return params

    return {
        "nmos": _extract(nmos_text),
        "pmos": _extract(pmos_text),
    }
