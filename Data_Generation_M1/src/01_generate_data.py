#!/usr/bin/env python3
"""
01_generate_data.py — PVT Variation Dataset Generator for NGSPICE
==================================================================
This script automates the generation of 10,000 SPICE simulation samples
for each of the 4 single-transistor operating states:
  - NMOS ON
  - NMOS OFF
  - PMOS ON
  - PMOS OFF

Workflow:
  1. Draws random parameters (VDD, Temp, W, Vd, Vs, vth0, toxe, ndep, u0, xj)
     according to the project specification (Uniform distributions).
  2. Synthesizes a standalone SPICE netlist with custom model parameters for each sample.
  3. Executes NGSPICE in batch mode in parallel across multiple CPU cores.
  4. Parses terminal currents (Id, Ig, Is, Ib), verifies KCL (Id + Ig + Is + Ib ≈ 0).
  5. Exports data to CSV files in the `data/` directory.

NOTE: Do NOT run this code automatically if requested to only generate files.
To execute manually later:
    python src/01_generate_data.py --help
    python src/01_generate_data.py --samples 10000 --workers 8
"""

import os
import sys
import re
import argparse
import subprocess
import tempfile
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

# Import project settings and parser from config.py
try:
    from config import (
        PROJECT_DIR, MODEL_PATH, DATA_DIR, LMIN,
        VDD_LO, VDD_HI, TEMP_LO, TEMP_HI, W_LO, W_HI,
        PROCESS_VAR, CASES, SPICE_OPTIONS, NGSPICE_TIMEOUT,
        NUM_SAMPLES, RANDOM_SEED, NUM_WORKERS,
        parse_nominal_params
    )
except ImportError:
    from src.config import (
        PROJECT_DIR, MODEL_PATH, DATA_DIR, LMIN,
        VDD_LO, VDD_HI, TEMP_LO, TEMP_HI, W_LO, W_HI,
        PROCESS_VAR, CASES, SPICE_OPTIONS, NGSPICE_TIMEOUT,
        NUM_SAMPLES, RANDOM_SEED, NUM_WORKERS,
        parse_nominal_params
    )


# ╔════════════════════════════════════════════════════════════════╗
# ║  PARAMETER SAMPLING                                            ║
# ╚════════════════════════════════════════════════════════════════╝

def generate_parameter_samples(case_name, n_samples, seed, nominal_params):
    """
    Generate independent random samples for all input parameters.

    Returns:
        pd.DataFrame: Matrix of 12 input features.
    """
    rng = np.random.default_rng(seed)
    case_info = CASES[case_name]
    mos_type = case_info["mos_type"]
    nom = nominal_params[mos_type]

    # 1. Operating conditions
    # VDD is drawn first because Vd and Vs ranges depend on it
    vdd = rng.uniform(VDD_LO, VDD_HI, n_samples)
    temp = rng.uniform(TEMP_LO, TEMP_HI, n_samples)
    w = rng.uniform(W_LO, W_HI, n_samples)
    vd = rng.uniform(0.0, 1.0, n_samples) * vdd
    vs = rng.uniform(0.0, 1.0, n_samples) * vdd

    # 2. Gate and Body voltages (fixed per operating case)
    vg = vdd.copy() if case_info["vg_is_vdd"] else np.zeros(n_samples)
    vb = vdd.copy() if case_info["vb_is_vdd"] else np.zeros(n_samples)

    # 3. Process parameters: nominal ± 10%
    # Note: vth0 for PMOS is negative in BSIM4, so preserve its sign properly
    vth0_nom = nom["vth0"]
    vth0_delta = abs(vth0_nom) * PROCESS_VAR
    vth0 = rng.uniform(vth0_nom - vth0_delta, vth0_nom + vth0_delta, n_samples)

    toxe = rng.uniform(nom["toxe"] * (1 - PROCESS_VAR), nom["toxe"] * (1 + PROCESS_VAR), n_samples)
    ndep = rng.uniform(nom["ndep"] * (1 - PROCESS_VAR), nom["ndep"] * (1 + PROCESS_VAR), n_samples)
    u0   = rng.uniform(nom["u0"]   * (1 - PROCESS_VAR), nom["u0"]   * (1 + PROCESS_VAR), n_samples)
    xj   = rng.uniform(nom["xj"]   * (1 - PROCESS_VAR), nom["xj"]   * (1 + PROCESS_VAR), n_samples)

    df = pd.DataFrame({
        "vdd": vdd,
        "temp": temp,
        "w": w,
        "vd": vd,
        "vs": vs,
        "vg": vg,
        "vb": vb,
        "vth0": vth0,
        "toxe": toxe,
        "ndep": ndep,
        "u0": u0,
        "xj": xj,
    })
    return df


# ╔════════════════════════════════════════════════════════════════╗
# ║  NETLIST GENERATION & SIMULATION                               ║
# ╚════════════════════════════════════════════════════════════════╝

def build_netlist_text(sample_idx, case_name, row, base_model_path):
    """
    Synthesize an NGSPICE netlist for a single transistor operating point.
    Uses 'altermod' inside a .control block to dynamically override BSIM4 parameters.
    """
    case_info = CASES[case_name]
    mos_type = case_info["mos_type"]
    device_inst = "Mn" if mos_type == "nmos" else "Mp"

    netlist = f"""* Leakage Current Simulation - {case_name.upper()} Sample {sample_idx}
.INCLUDE "{base_model_path}"
.OPTIONS {SPICE_OPTIONS}

.PARAM Lmin = {LMIN}
.PARAM Wdev = {row['w']:.6e}

* Transistor Instance: Drain Gate Source Body Model
{device_inst} drain gate source body {mos_type} W={{Wdev}} L={{Lmin}}

* Terminal Voltage Sources
Vd drain  0  {row['vd']:.6f}
Vg gate   0  {row['vg']:.6f}
Vs source 0  {row['vs']:.6f}
Vb body   0  {row['vb']:.6f}

.TEMP {row['temp']:.2f}

.CONTROL
set temp = {row['temp']:.2f}

* Inject sampled BSIM4 process parameters
altermod {mos_type} vth0 = {row['vth0']:.6e}
altermod {mos_type} toxe = {row['toxe']:.6e}
altermod {mos_type} ndep = {row['ndep']:.6e}
altermod {mos_type} u0   = {row['u0']:.6e}
altermod {mos_type} xj   = {row['xj']:.6e}

* Run DC operating point analysis
op

* Print current measurements
print I(Vd) I(Vg) I(Vs) I(Vb)
.ENDC

.END
"""
    return netlist


def simulate_single_sample(args):
    """
    Worker function executed by multiprocessing pool:
      1. Writes netlist to a temporary file.
      2. Invokes NGSPICE in batch mode.
      3. Parses currents from stdout.
      4. Cleans up temporary file.
    """
    sample_idx, case_name, row_dict, base_model_path = args

    # Create temporary netlist file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.net', delete=False) as tf:
        netlist_path = tf.name
        content = build_netlist_text(sample_idx, case_name, row_dict, base_model_path)
        tf.write(content)

    id_val, ig_val, is_val, ib_val = None, None, None, None

    try:
        # Run NGSPICE in batch mode (-b)
        cmd = ["ngspice", "-b", netlist_path]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=NGSPICE_TIMEOUT
        )
        output = proc.stdout

        # Regex patterns to parse currents: e.g. "i(vd) = -1.234567e-08"
        # In SPICE, I(Vx) is current entering the source (out of the transistor terminal).
        # Standard convention for transistor terminals: current ENTERING the terminal is positive.
        # Thus, I_terminal = -I(Vsource).
        match_vd = re.search(r'i\(vd\)\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', output, re.IGNORECASE)
        match_vg = re.search(r'i\(vg\)\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', output, re.IGNORECASE)
        match_vs = re.search(r'i\(vs\)\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', output, re.IGNORECASE)
        match_vb = re.search(r'i\(vb\)\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', output, re.IGNORECASE)

        if match_vd and match_vg and match_vs and match_vb:
            id_val = -float(match_vd.group(1))
            ig_val = -float(match_vg.group(1))
            is_val = -float(match_vs.group(1))
            ib_val = -float(match_vb.group(1))
        else:
            # Fallback regex for tabular or alternative NGSPICE print output
            patterns = {
                'vd': r'vd#branch\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
                'vg': r'vg#branch\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
                'vs': r'vs#branch\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
                'vb': r'vb#branch\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
            }
            vals = {}
            for k, pat in patterns.items():
                m = re.search(pat, output, re.IGNORECASE)
                if m:
                    vals[k] = -float(m.group(1))
            if len(vals) == 4:
                id_val, ig_val, is_val, ib_val = vals['vd'], vals['vg'], vals['vs'], vals['vb']

    except subprocess.TimeoutExpired:
        print(f"Warning: Simulation timeout on sample {sample_idx}")
    except Exception as e:
        print(f"Error on sample {sample_idx}: {e}")
    finally:
        # Always remove temp netlist file
        if os.path.exists(netlist_path):
            os.remove(netlist_path)

    return sample_idx, id_val, ig_val, is_val, ib_val


# ╔════════════════════════════════════════════════════════════════╗
# ║  DATASET GENERATION PIPELINE                                  ║
# ╚════════════════════════════════════════════════════════════════╝

def generate_case_dataset(case_name, n_samples, n_workers, seed, nominal_params):
    """
    Runs the complete simulation batch for a given transistor state.
    """
    print(f"\n{'='*70}")
    print(f"  Generating dataset for: {case_name.upper()} ({n_samples} samples)")
    print(f"{'='*70}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out_csv = os.path.join(DATA_DIR, f"{case_name}.csv")

    # 1. Sample inputs
    df_inputs = generate_parameter_samples(case_name, n_samples, seed, nominal_params)

    # 2. Prepare task arguments for multiprocessing
    tasks = [
        (i, case_name, row.to_dict(), MODEL_PATH)
        for i, row in df_inputs.iterrows()
    ]

    # 3. Run parallel simulations
    start_time = time.time()
    results = []

    print(f"[*] Starting {n_workers} worker processes...")
    with Pool(processes=n_workers) as pool:
        # Use chunksize for optimal IPC throughput
        chunksize = max(1, n_samples // (n_workers * 10))
        for idx, (s_idx, id_v, ig_v, is_v, ib_v) in enumerate(pool.imap_unordered(simulate_single_sample, tasks, chunksize=chunksize)):
            results.append((s_idx, id_v, ig_v, is_v, ib_v))
            if (idx + 1) % 500 == 0 or (idx + 1) == n_samples:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (n_samples - (idx + 1)) / rate if rate > 0 else 0
                print(f"    Progress: {idx + 1}/{n_samples} ({100 * (idx + 1) / n_samples:.1f}%) | "
                      f"Speed: {rate:.1f} sim/s | Remaining: {remaining/60:.1f} min")

    total_time = time.time() - start_time
    print(f"[✓] Finished simulations in {total_time:.2f} seconds ({n_samples / total_time:.1f} sim/s)")

    # 4. Assemble results into DataFrame
    res_df = pd.DataFrame(results, columns=["sample_idx", "id", "ig", "is", "ib"])
    res_df = res_df.sort_values("sample_idx").reset_index(drop=True)

    # Combine inputs with measured currents
    full_df = df_inputs.copy()
    full_df["id"] = res_df["id"]
    full_df["ig"] = res_df["ig"]
    full_df["is"] = res_df["is"]
    full_df["ib"] = res_df["ib"]

    # 5. Check for convergence failures
    failed_count = full_df["id"].isna().sum()
    if failed_count > 0:
        print(f"[!] Warning: {failed_count} samples failed to simulate or parse. Dropping failed rows.")
        full_df = full_df.dropna().reset_index(drop=True)

    # 6. Kirchhoff's Current Law (KCL) verification: Id + Ig + Is + Ib ≈ 0
    kcl_residual = np.abs(full_df["id"] + full_df["ig"] + full_df["is"] + full_df["ib"])
    print(f"[*] KCL Check (|Id + Ig + Is + Ib|):")
    print(f"    Max KCL error  : {kcl_residual.max():.3e} A")
    print(f"    Mean KCL error : {kcl_residual.mean():.3e} A")

    # 7. Save dataset to CSV
    # Required columns order from specification:
    col_order = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]
    full_df = full_df[col_order]
    full_df.to_csv(out_csv, index=False)
    print(f"[✓] Successfully saved dataset to: {out_csv} ({len(full_df)} rows)")

    return full_df


def main():
    parser = argparse.ArgumentParser(description="Generate PVT variation datasets using NGSPICE.")
    parser.add_argument("--cases", nargs="+", default=["nmos_on", "nmos_off", "pmos_on", "pmos_off"],
                        choices=["nmos_on", "nmos_off", "pmos_on", "pmos_off"],
                        help="Operating cases to generate (default: all 4)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES,
                        help=f"Number of samples per case (default: {NUM_SAMPLES})")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"Number of parallel workers (default: {NUM_WORKERS})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    args = parser.parse_args()

    print("======================================================================")
    print("   PVT VARIATION AWARE LEAKAGE ESTIMATION — DATA GENERATOR")
    print("======================================================================")
    print(f" Model File     : {MODEL_PATH}")
    print(f" Target Samples : {args.samples} per case")
    print(f" Worker Cores   : {args.workers}")
    print(f" Random Seed    : {args.seed}")
    print(f" Cases          : {', '.join(args.cases)}")
    print("======================================================================")

    # Ensure model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model file not found at: {MODEL_PATH}")
        print("    Please check the MODEL_FILE setting in src/config.py")
        sys.exit(1)

    # Parse nominal process parameters from model file
    print("[*] Parsing nominal model parameters...")
    nominal_params = parse_nominal_params(MODEL_PATH)
    print(f"    NMOS Nominals: {nominal_params['nmos']}")
    print(f"    PMOS Nominals: {nominal_params['pmos']}")

    # Generate each requested case
    case_seeds = {
        "nmos_on":  args.seed + 1,
        "nmos_off": args.seed + 2,
        "pmos_on":  args.seed + 3,
        "pmos_off": args.seed + 4,
    }

    for case_name in args.cases:
        generate_case_dataset(
            case_name=case_name,
            n_samples=args.samples,
            n_workers=args.workers,
            seed=case_seeds[case_name],
            nominal_params=nominal_params
        )

    print("\n[✓] All requested datasets have been generated successfully!")


if __name__ == "__main__":
    main()
