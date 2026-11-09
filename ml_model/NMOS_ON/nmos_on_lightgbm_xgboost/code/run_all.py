"""
run_all.py  -  Master Pipeline Runner for NMOS ON (Physics-Guided LightGBM & XGBoost)
Project: ML-Driven PVT Variation Aware Leakage Estimation (22nm_MGK, VDD = 0.9 V)
Case: NMOS ON (Vg = VDD, Vb = 0)

Executes all 4 steps sequentially:
    Step 0: Data Quality & Physical Trend Checks (~10 s)
    Step 1: Physics-Guided LightGBM with Linear Trees (~2-4 min)
    Step 2: Physics-Guided XGBoost (hist) (~5-10 min)
    Step 3: Side-by-Side Model Comparison Tables & Combined JSON (~10 s)

Usage:
    python run_all.py
"""

import os
import sys
import time
import subprocess

STEPS = [
    ("Step 0: Data Quality & Physical Trend Checks", "step0_data_check.py"),
    ("Step 1: Physics-Guided LightGBM (Linear Trees)", "step1_lightgbm.py"),
    ("Step 2: Physics-Guided XGBoost (hist)", "step2_xgboost.py"),
    ("Step 3: Model Comparison & JSON Report", "step3_compare_models.py"),
]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    total_start = time.time()
    print("=" * 85)
    print("      NMOS_ON PHYSICS-GUIDED MACHINE LEARNING PIPELINE (TARGET: >= 99% ACCURACY)")
    print("=" * 85)

    for idx, (title, script_name) in enumerate(STEPS):
        script_path = os.path.join(THIS_DIR, script_name)
        print(f"\n[{idx+1}/{len(STEPS)}] Running: {title} ...")
        t0 = time.time()
        ret = subprocess.call([sys.executable, script_path], cwd=THIS_DIR)
        if ret != 0:
            print(f"\n❌ Error occurred in {script_name} (exit code {ret}). Stopping pipeline.")
            sys.exit(ret)
        print(f"✓ Completed {script_name} in {time.time() - t0:.1f} s.")

    elapsed = time.time() - total_start
    results_dir = os.path.abspath(os.path.join(THIS_DIR, "..", "results"))
    print("\n" + "=" * 85)
    print(f"✓ PIPELINE COMPLETE in {elapsed/60:.2f} minutes!")
    print(f"✓ Results, metrics tables, and diagnostic plots are saved in:")
    print(f"  {results_dir}")
    print(f"  ├── step0_data_check/ : KCL check and trend plots")
    print(f"  ├── lightgbm/         : LightGBM metrics, parity, and overfit curves")
    print(f"  ├── xgboost/          : XGBoost metrics, parity, and overfit curves")
    print(f"  └── comparison/       : Side-by-side CSVs, PNG tables, and team JSON report")
    print("=" * 85)


if __name__ == "__main__":
    main()
