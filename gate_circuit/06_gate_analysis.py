"""Step 6 - gate results table (deliverable 6), gate parity plot
(deliverable 7) and error analysis for the report.

Reads  : data/gate_ml.csv, data/gate_transistors_ml.csv   (from step 5)
Writes : results/gate/
    gate_results.csv / .txt        18 gate/input rows + ALL: SPICE median, ML median,
                                   bias, median error, p95 error, mean accuracy
    gate_summary_by_gate.csv/.txt  same metrics per gate
    gate_side_comparison.csv       src-side vs snk-side ML leakage accuracy
    gate_parity.png                ML vs SPICE, all 3,600 simulations, log axes
    gate_parity_by_gate.png        one panel per gate, input combos colour+marker
    gate_error_by_case.png         |error| distribution per gate/input
    gate_error_vs_corner.png       |error| vs temperature, VDD, NMOS/PMOS vth0
    transistor_errors.csv/.txt     per model case and terminal: ML vs single-transistor
                                   SPICE error, and share of total leakage error
    gate_worst_cases.csv           the 30 worst gate predictions with their corner
"""
import numpy as np
import pandas as pd

from analysis_utils import (error_box, error_vs, metrics, metrics_table, parity_by_group,
                            parity_plot, rel_err, write_table)
from circuits import GATE_ORDER
from config import GATE_ML_CSV, GATE_TRANSISTORS_ML_CSV, RESULTS_DIR


def main():
    out = RESULTS_DIR / "gate"
    out.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(GATE_ML_CSV, dtype={"inputs": str})
    tr = pd.read_csv(GATE_TRANSISTORS_ML_CSV, dtype={"inputs": str})
    backend, side = g["backend"].iloc[0], g["side"].iloc[0]
    tag = f"(backend={backend}, leakage side={side})"
    if backend == "oracle":
        print("NOTE: backend=oracle -> these are SPICE transistor currents, NOT the ML models.")

    g["gate"] = pd.Categorical(g["gate"], GATE_ORDER, ordered=True)
    g = g.sort_values(["gate", "inputs", "corner"]).reset_index(drop=True)
    g["gate"] = g["gate"].astype(str)
    g["case_label"] = g["gate"] + " " + g["inputs"]
    g["err_pct"] = rel_err(g.i_leak_ml, g.i_leak_spice)

    # ---------------- deliverable 6: table
    tab = metrics_table(g, ["gate", "inputs"], "i_leak_ml", "i_leak_spice")
    write_table(tab, out / "gate_results.csv",
                f"Gate leakage, ML vs SPICE, 200 corners per row {tag}")
    write_table(metrics_table(g, ["gate"], "i_leak_ml", "i_leak_spice"),
                out / "gate_summary_by_gate.csv", f"Per gate {tag}")

    side_rows = []
    for s in ("src", "snk"):
        m = metrics(g[f"i_leak_ml_{s}"], g.i_leak_spice)
        side_rows.append({"side": s, **m})
    pd.DataFrame(side_rows).to_csv(out / "gate_side_comparison.csv", index=False)
    print(pd.DataFrame(side_rows)[["side", "bias_pct", "median_err_pct", "p95_err_pct",
                                   "mean_acc_pct"]].to_string(index=False))

    body = tab[tab.gate != "ALL"]
    best = body.nsmallest(3, "median_err_pct")[["gate", "inputs", "median_err_pct", "p95_err_pct"]]
    worst = body.nlargest(3, "median_err_pct")[["gate", "inputs", "median_err_pct", "p95_err_pct"]]
    print("\nMost accurate cases:\n" + best.to_string(index=False))
    print("Least accurate cases:\n" + worst.to_string(index=False))

    # ---------------- deliverable 7: parity plots
    m = metrics(g.i_leak_ml, g.i_leak_spice)
    note = (f"median err {m['median_err_pct']:.2f}%\np95 err    {m['p95_err_pct']:.2f}%\n"
            f"bias       {m['bias_pct']:+.2f}%\nmean acc   {m['mean_acc_pct']:.2f}%")
    parity_plot(g.i_leak_spice, g.i_leak_ml, out / "gate_parity.png",
                f"Gate leakage: ML vs SPICE, all 18 cases x 200 corners", note)
    parity_by_group(g, "gate", "inputs", "i_leak_spice", "i_leak_ml",
                    out / "gate_parity_by_gate.png", f"Gate leakage parity per gate {tag}",
                    panel_order=GATE_ORDER)

    # ---------------- error analysis
    error_box(g, "case_label", "err_pct", out / "gate_error_by_case.png",
              f"Gate leakage error per gate / input {tag}",
              order=list(dict.fromkeys(g.case_label)))
    error_vs(g, ["temp", "vdd", "n_vth0", "p_vth0"], "err_pct", out / "gate_error_vs_corner.png",
             "Gate leakage error vs corner conditions",
             xlabels={"temp": "temperature [°C]", "vdd": "VDD [V]",
                      "n_vth0": "NMOS vth0 [V]", "p_vth0": "PMOS vth0 [V]"})

    # transistor-level: which model / terminal causes the error
    rows = []
    for case, d in tr.groupby("case"):
        for k in "dgsb":
            sp, ml = d[f"i{k}_spice"].to_numpy(), d[f"i{k}_ml"].to_numpy()
            mask = np.abs(sp) > 0
            a = np.abs(rel_err(ml[mask], sp[mask]))
            rows.append({"case": case, "terminal": "i" + k, "n": int(mask.sum()),
                         "median_abs_A": float(np.median(np.abs(sp))),
                         "median_err_pct": float(np.median(a)),
                         "p95_err_pct": float(np.percentile(a, 95))})
    te = pd.DataFrame(rows)
    col = f"leak_{side}"
    abs_err = (tr[f"{col}_ml"] - tr[f"{col}_spice"]).abs()
    share = abs_err.groupby(tr["case"]).sum() / abs_err.sum() * 100
    te["share_of_gate_leak_abs_err_pct"] = te["case"].map(share)
    write_table(te, out / "transistor_errors.csv",
                "Transistor terminal currents inside the gates: ML vs single-transistor SPICE")

    worst = g.reindex(g.err_pct.abs().sort_values(ascending=False).index).head(30)
    worst.to_csv(out / "gate_worst_cases.csv", index=False)
    print(f"\nAll gate results written to {out}")


if __name__ == "__main__":
    main()
