"""Step 5 - ML prediction of gate leakage (Part B, step 4).

For every gate / input combination / corner already simulated in SPICE:
  * every transistor's terminal voltages come from the recorded SPICE node
    voltages (data/gate_transistors.csv, built by step 3);
  * the transistor's case (nmos_on/off, pmos_on/off) is chosen from the logic
    level of its gate node, and vg is set to that logic level (0 or VDD);
  * the four transistor models predict [id, ig, is, ib];
  * gate leakage = sum of the predicted currents over the terminals sitting on
    a VDD-level source node (VDD rail or input at logic 1)      -> "src" side.
    The same quantity can also be computed from the ground side
    (terminals on GND or inputs at logic 0)                       -> "snk" side.
    Both are equal in SPICE (KCL); with ML they differ, so both are saved and
    --side picks which one becomes the reported `i_leak_ml`.

Usage
    python 05_gate_ml.py                    # trained models (leakage_models.py)
    python 05_gate_ml.py --backend oracle   # pipeline test with SPICE transistor currents
    python 05_gate_ml.py --side snk         # report the ground-side sum instead

Outputs: data/gate_ml.csv, data/gate_transistors_ml.csv
"""
import argparse

import numpy as np
import pandas as pd

import leakage_models
from circuits import leakage_from_terminals
from config import (FEATURES, GATE_ML_CSV, GATE_SPICE_CSV, GATE_TRANSISTORS_CSV,
                    GATE_TRANSISTORS_ML_CSV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ml", "oracle"], default="ml")
    ap.add_argument("--side", choices=["src", "snk"], default="src")
    args = ap.parse_args()

    tr = pd.read_csv(GATE_TRANSISTORS_CSV, dtype={"inputs": str})
    gs = pd.read_csv(GATE_SPICE_CSV, dtype={"inputs": str})

    pred = np.zeros((len(tr), 4))
    for case, idx in tr.groupby("case").groups.items():
        if args.backend == "oracle":
            pred[tr.index.get_indexer(idx)] = tr.loc[idx, [f"i{k}_spice" for k in "dgsb"]].to_numpy()
        else:
            pred[tr.index.get_indexer(idx)] = leakage_models.predict(case, tr.loc[idx, FEATURES])
        print(f"  predicted {len(idx):6d} transistors for {case}")
    for j, k in enumerate("dgsb"):
        tr[f"i{k}_ml"] = pred[:, j]

    # per-transistor leakage contributions (ML and SPICE oracle)
    tr["leak_src_ml"], tr["leak_snk_ml"] = leakage_from_terminals(
        tr.rename(columns={f"i{k}_ml": f"i{k}" for k in "dgsb"}))
    tr["leak_src_spice"], tr["leak_snk_spice"] = leakage_from_terminals(
        tr.rename(columns={f"i{k}_spice": f"i{k}" for k in "dgsb"}))
    tr.to_csv(GATE_TRANSISTORS_ML_CSV, index=False)

    keys = ["gate", "inputs", "corner"]
    agg = tr.groupby(keys, sort=False)[["leak_src_ml", "leak_snk_ml"]].sum().reset_index()
    agg = agg.rename(columns={"leak_src_ml": "i_leak_ml_src", "leak_snk_ml": "i_leak_ml_snk"})
    out = gs.merge(agg, on=keys, how="left")
    out["i_leak_ml"] = out[f"i_leak_ml_{args.side}"]
    out["backend"] = args.backend
    out["side"] = args.side
    out.to_csv(GATE_ML_CSV, index=False)

    bad = (out["i_leak_ml"] <= 0).sum()
    if bad:
        print(f"WARNING: {bad} gate cases have non-positive ML leakage")
    r = 100 * (out.i_leak_ml - out.i_leak_spice).abs() / out.i_leak_spice
    print(f"backend={args.backend} side={args.side}: median |err| {r.median():.3f}%, "
          f"p95 {np.percentile(r, 95):.3f}%")
    print(f"Saved {GATE_ML_CSV} and {GATE_TRANSISTORS_ML_CSV}")


if __name__ == "__main__":
    main()
