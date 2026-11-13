"""Step 1 - draw the 200 PVT corners shared by every gate (Part B) and the
CLA (Part C).

A corner = VDD, temperature, one NMOS draw of (vth0, toxe, ndep, u0, xj) and
a separate PMOS draw. VDD is drawn first.

Output: data/corners.csv
"""
import numpy as np
import pandas as pd

from config import (CORNER_SEED, CORNERS_CSV, N_CORNERS, NMOS_NOM, PMOS_NOM,
                    PROC_PARAMS, PROC_SPREAD, T_MAX, T_MIN, VDD_NOM, VDD_SPREAD,
                    ensure_dirs)


def main():
    ensure_dirs()
    rng = np.random.default_rng(CORNER_SEED)
    n = N_CORNERS
    df = pd.DataFrame({"corner": np.arange(n)})
    df["vdd"] = rng.uniform(VDD_NOM * (1 - VDD_SPREAD), VDD_NOM * (1 + VDD_SPREAD), n)
    df["temp"] = rng.uniform(T_MIN, T_MAX, n)
    for prefix, nom in (("n_", NMOS_NOM), ("p_", PMOS_NOM)):
        for p in PROC_PARAMS:
            lo, hi = sorted((nom[p] * (1 - PROC_SPREAD), nom[p] * (1 + PROC_SPREAD)))
            df[prefix + p] = rng.uniform(lo, hi, n)
    df.to_csv(CORNERS_CSV, index=False)
    print(f"Saved {len(df)} corners -> {CORNERS_CSV}")
    print(df.describe().loc[["min", "max"]].T)


if __name__ == "__main__":
    main()
