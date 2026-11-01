#!/usr/bin/env python3
"""
01_data_check.py — Data Verification & Trend Plot Generator
============================================================
Fulfills Project Deliverable 3:
  1. Performs Kirchhoff's Current Law check for each case:
         Id + Ig + Is + Ib ≈ 0
  2. Generates the 3 required trend plots:
     - OFF current (log10 |Id|) vs Temperature
     - OFF current (log10 |Id|) vs vth0
     - Gate current (log10 |Ig|) vs toxe
  3. Saves summary statistics and publication-quality figures to `plots/`.

NOTE: Run this script after generating CSV datasets in `data/`.
Usage:
    python src/01_data_check.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from config import PROJECT_DIR, DATA_DIR, CASES
except ImportError:
    from src.config import PROJECT_DIR, DATA_DIR, CASES

PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")


def run_kirchhoff_check(df, case_name):
    """
    Verify Kirchhoff's Current Law: sum(Id, Ig, Is, Ib) ≈ 0.
    """
    kcl = df["id"] + df["ig"] + df["is"] + df["ib"]
    abs_kcl = np.abs(kcl)
    max_err = abs_kcl.max()
    mean_err = abs_kcl.mean()
    median_err = abs_kcl.median()

    # Relative KCL error compared to max terminal current in that sample
    max_current = df[["id", "ig", "is", "ib"]].abs().max(axis=1)
    # Avoid div by zero
    rel_err = abs_kcl / np.maximum(max_current, 1e-18)
    max_rel_err = rel_err.max()

    print(f"\n--- KCL Check for {case_name.upper()} ({len(df)} samples) ---")
    print(f"  Absolute KCL Error -> Max: {max_err:.3e} A | Mean: {mean_err:.3e} A | Median: {median_err:.3e} A")
    print(f"  Relative KCL Error -> Max: {max_rel_err * 100:.4f}% | Mean: {rel_err.mean() * 100:.6f}%")

    return {
        "case": case_name,
        "samples": len(df),
        "max_kcl_abs": max_err,
        "mean_kcl_abs": mean_err,
        "max_kcl_rel_pct": max_rel_err * 100
    }


def generate_trend_plots():
    """
    Generate the 3 required Deliverable 3 trend plots:
      1. OFF current vs Temperature
      2. OFF current vs vth0
      3. Gate current vs toxe
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    nmos_off_path = os.path.join(DATA_DIR, "nmos_off.csv")
    pmos_off_path = os.path.join(DATA_DIR, "pmos_off.csv")
    nmos_on_path  = os.path.join(DATA_DIR, "nmos_on.csv")

    if not os.path.exists(nmos_off_path):
        print(f"[!] File not found: {nmos_off_path}. Please run data generation first.")
        return

    df_nmos_off = pd.read_csv(nmos_off_path)
    df_pmos_off = pd.read_csv(pmos_off_path) if os.path.exists(pmos_off_path) else None
    df_nmos_on  = pd.read_csv(nmos_on_path) if os.path.exists(nmos_on_path) else None

    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

    # =========================================================================
    # Plot 1: OFF current vs Temperature (Subthreshold leakage is exponential with T)
    # =========================================================================
    plt.figure(figsize=(8, 5))
    plt.scatter(df_nmos_off["temp"], np.abs(df_nmos_off["id"]), alpha=0.35, s=8, label="NMOS OFF (|Id|)", color="#1f77b4")
    if df_pmos_off is not None:
        plt.scatter(df_pmos_off["temp"], np.abs(df_pmos_off["id"]), alpha=0.35, s=8, label="PMOS OFF (|Id|)", color="#d62728")
    plt.yscale("log")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("OFF Drain Leakage Current |Id| (A)")
    plt.title("Trend 1: Subthreshold OFF Current vs. Temperature")
    plt.legend()
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "trend1_off_current_vs_temp.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"[✓] Saved Plot 1: {plot1_path}")

    # =========================================================================
    # Plot 2: OFF current vs Threshold Voltage vth0
    # =========================================================================
    plt.figure(figsize=(8, 5))
    plt.scatter(df_nmos_off["vth0"], np.abs(df_nmos_off["id"]), alpha=0.35, s=8, label="NMOS OFF (|Id|)", color="#2ca02c")
    if df_pmos_off is not None:
        plt.scatter(df_pmos_off["vth0"].abs(), np.abs(df_pmos_off["id"]), alpha=0.35, s=8, label="PMOS OFF (|Id| vs |vth0|)", color="#9467bd")
    plt.yscale("log")
    plt.xlabel("Threshold Voltage vth0 (V)")
    plt.ylabel("OFF Drain Leakage Current |Id| (A)")
    plt.title("Trend 2: Subthreshold OFF Current vs. Threshold Voltage (vth0)")
    plt.legend()
    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "trend2_off_current_vs_vth0.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"[✓] Saved Plot 2: {plot2_path}")

    # =========================================================================
    # Plot 3: Gate current vs Oxide Thickness toxe (Direct tunneling is exponential with toxe)
    # =========================================================================
    plt.figure(figsize=(8, 5))
    # Use NMOS ON or NMOS OFF gate tunneling current
    target_df = df_nmos_on if df_nmos_on is not None else df_nmos_off
    label_txt = "NMOS ON (|Ig|)" if df_nmos_on is not None else "NMOS OFF (|Ig|)"
    plt.scatter(target_df["toxe"] * 1e10, np.abs(target_df["ig"]), alpha=0.35, s=8, label=label_txt, color="#ff7f0e")
    plt.yscale("log")
    plt.xlabel("Oxide Thickness toxe (Å)")
    plt.ylabel("Gate Tunneling Leakage |Ig| (A)")
    plt.title("Trend 3: Gate Leakage Current vs. Oxide Thickness (toxe)")
    plt.legend()
    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, "trend3_gate_current_vs_toxe.png")
    plt.savefig(plot3_path)
    plt.close()
    print(f"[✓] Saved Plot 3: {plot3_path}")


def main():
    print("======================================================================")
    print("         DATA CHECK & TREND PLOTS (DELIVERABLE 3)")
    print("======================================================================")

    kcl_results = []
    for case_name in CASES.keys():
        csv_file = os.path.join(DATA_DIR, f"{case_name}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            res = run_kirchhoff_check(df, case_name)
            kcl_results.append(res)
        else:
            print(f"[-] {case_name}.csv not yet created in data/")

    if kcl_results:
        summary_df = pd.DataFrame(kcl_results)
        print("\n" + "="*50)
        print("KCL Verification Summary:")
        print(summary_df.to_string(index=False))
        print("="*50)

        print("\n[*] Generating required trend plots...")
        generate_trend_plots()
        print("\n[✓] Data verification and plotting complete!")
    else:
        print("\nNo dataset files found yet. Run `01_generate_data.py` when you are ready to simulate.")


if __name__ == "__main__":
    main()
