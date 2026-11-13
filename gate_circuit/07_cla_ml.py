"""Step 7 - hierarchical ML prediction of the 3-bit CLA leakage (Part C, step 7).

For every input vector:
  * logic-simulate the CLA to get the logic inputs of each of its 39 gates;
  * look up that gate's ML leakage for that input combination at the same
    corner (from data/gate_ml.csv, i.e. the Part B gate models, which call the
    transistor models);
  * sum over all gates.
Assumes every gate input is exactly 0 or VDD and ignores loading between gates.

Also computes the same sum with the isolated-gate SPICE leakage
(`i_leak_gatesum_spice`). Comparing that with the full CLA netlist isolates the
error of the hierarchical composition itself from the ML model error.

Output: data/cla_ml.csv
"""
import numpy as np
import pandas as pd

from circuits import CLA_INPUTS, cla_eval, cla_vectors
from config import CLA_ML_CSV, CLA_SPICE_CSV, GATE_ML_CSV


def main():
    gm = pd.read_csv(GATE_ML_CSV, dtype={"inputs": str})
    cla = pd.read_csv(CLA_SPICE_CSV)
    gm["case"] = gm["gate"] + "_" + gm["inputs"]
    cases = sorted(gm["case"].unique())
    col = {c: i for i, c in enumerate(cases)}

    # how many gates of each (type, input combo) each vector uses: (128 x 18)
    vecs = cla_vectors()
    counts = np.zeros((len(vecs), len(cases)))
    for v, bits in enumerate(vecs):
        for _, gt, label in cla_eval(bits)[1]:
            counts[v, col[f"{gt}_{label}"]] += 1
    pd.DataFrame(counts.astype(int), columns=cases).assign(vector=range(len(vecs))) \
        .to_csv(CLA_ML_CSV.with_name("cla_gate_usage.csv"), index=False)

    # gate leakage per corner: (200 x 18)
    ml_tab = gm.pivot(index="corner", columns="case", values="i_leak_ml")[cases]
    sp_tab = gm.pivot(index="corner", columns="case", values="i_leak_spice")[cases]
    ml_sum = ml_tab.to_numpy() @ counts.T          # (200 x 128)
    sp_sum = sp_tab.to_numpy() @ counts.T

    corners = ml_tab.index.to_numpy()
    cpos = {c: i for i, c in enumerate(corners)}
    ci = cla["corner"].map(cpos).to_numpy()
    vi = cla["vector"].to_numpy()
    out = cla[["corner", "vector"] + CLA_INPUTS + ["vdd", "temp"]
              + [c for c in cla.columns if c[:2] in ("n_", "p_")] + ["i_leak_spice"]].copy()
    out["n_ones"] = cla[CLA_INPUTS].sum(axis=1)
    out["i_leak_ml"] = ml_sum[ci, vi]
    out["i_leak_gatesum_spice"] = sp_sum[ci, vi]
    out["backend"] = gm["backend"].iloc[0]
    out["side"] = gm["side"].iloc[0]
    out.to_csv(CLA_ML_CSV, index=False)

    r = 100 * (out.i_leak_ml - out.i_leak_spice).abs() / out.i_leak_spice
    print(f"CLA hierarchical ML vs full SPICE: median |err| {r.median():.3f}%, "
          f"p95 {np.percentile(r, 95):.3f}%")
    print(f"Saved {CLA_ML_CSV}")


if __name__ == "__main__":
    main()
