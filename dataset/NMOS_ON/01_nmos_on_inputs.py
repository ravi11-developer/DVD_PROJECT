from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES = 10000
SEED      = 42

VDD_NOM = 0.9
L       = 22e-9

W_MIN, W_MAX = 1 * L, 8 * L
T_MIN, T_MAX = -40.0, 125.0

VTH0_NOM = 0.3692
TOXE_NOM = 6.5e-10
NDEP_NOM = 1.2e19
U0_NOM   = 0.0181
XJ_NOM   = 7.2e-9

rng = np.random.default_rng(SEED)

vdd = rng.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM, N_SAMPLES)

vd = rng.uniform(0.0, vdd)
vs = rng.uniform(0.0, vdd)

w    = rng.uniform(W_MIN, W_MAX, N_SAMPLES)
temp = rng.uniform(T_MIN, T_MAX, N_SAMPLES)

vth0 = rng.uniform(0.9 * VTH0_NOM, 1.1 * VTH0_NOM, N_SAMPLES)
toxe = rng.uniform(0.9 * TOXE_NOM, 1.1 * TOXE_NOM, N_SAMPLES)
ndep = rng.uniform(0.9 * NDEP_NOM, 1.1 * NDEP_NOM, N_SAMPLES)
u0   = rng.uniform(0.9 * U0_NOM,   1.1 * U0_NOM,   N_SAMPLES)
xj   = rng.uniform(0.9 * XJ_NOM,   1.1 * XJ_NOM,   N_SAMPLES)

vg = vdd.copy()
vb = np.zeros(N_SAMPLES)

df = pd.DataFrame({
    "vdd": vdd, "temp": temp, "w": w,
    "vd": vd, "vs": vs, "vg": vg, "vb": vb,
    "vth0": vth0, "toxe": toxe, "ndep": ndep, "u0": u0, "xj": xj,
})

print("Shape (rows, columns):", df.shape)
print(df.describe().loc[["min", "max"]].T)
print("All vd <= vdd ?", bool((df.vd <= df.vdd).all()))
print("All vs <= vdd ?", bool((df.vs <= df.vdd).all()))

df.to_csv(Path(__file__).resolve().parent / "nmos_on_inputs.csv", index=False)
print("Saved nmos_on_inputs.csv")
