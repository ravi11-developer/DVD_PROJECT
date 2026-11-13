"""Step 8 - 3-bit CLA results (deliverable 8) and comparison with gate level.

Reads  : data/cla_ml.csv (step 7), results/gate/gate_results.csv (step 6)
Writes : results/cla/
    cla_results.csv / .txt        overall + per number-of-ones-in-input rows:
                                  SPICE median, ML median, bias, median err, p95 err
    cla_results_per_vector.csv    same metrics for each of the 128 input vectors
    cla_error_decomposition.csv   ML vs full SPICE  |  SPICE gate-sum vs full SPICE
                                  (composition error)  |  ML vs SPICE gate-sum (model error)
    cla_vs_gate_level.csv/.txt    CLA accuracy next to gate-level accuracy
    cla_parity.png                ML vs SPICE, all 25,600 points, log axes
    cla_parity_gatesum.png        SPICE gate-sum vs full SPICE (composition only)
    cla_error_vs_corner.png       |error| vs temperature, VDD, vth0
    cla_error_by_ones.png         |error| vs number of 1s in the input vector
    cla_worst_cases.csv           30 worst predictions with their corner
"""
import pandas as pd

from analysis_utils import (error_box, error_vs, metrics, metrics_table, parity_plot, rel_err,
                            write_table)
from config import CLA_ML_CSV, RESULTS_DIR


def main():
    out = RESULTS_DIR / "cla"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(CLA_ML_CSV)
    backend, side = d["backend"].iloc[0], d["side"].iloc[0]
    tag = f"(backend={backend}, leakage side={side})"
    if backend == "oracle":
        print("NOTE: backend=oracle -> gate leakages are SPICE transistor sums, NOT the ML models.")
    d["err_pct"] = rel_err(d.i_leak_ml, d.i_leak_spice)

    # ---------------- deliverable 8: table
    t = metrics_table(d, ["n_ones"], "i_leak_ml", "i_leak_spice")
    t["n_ones"] = t["n_ones"].astype(str)
    write_table(t, out / "cla_results.csv",
                f"3-bit CLA leakage, hierarchical ML vs full SPICE netlist {tag}\n"
                f"(rows grouped by number of 1s in the 7-bit input vector; ALL = 128 vectors x 200 corners)")
    metrics_table(d, ["vector"], "i_leak_ml", "i_leak_spice").to_csv(
        out / "cla_results_per_vector.csv", index=False)

    dec = pd.DataFrame([
        {"comparison": "ML hierarchical vs full SPICE", **metrics(d.i_leak_ml, d.i_leak_spice)},
        {"comparison": "SPICE gate-sum vs full SPICE (composition error)",
         **metrics(d.i_leak_gatesum_spice, d.i_leak_spice)},
        {"comparison": "ML hierarchical vs SPICE gate-sum (model error)",
         **metrics(d.i_leak_ml, d.i_leak_gatesum_spice)},
    ])
    write_table(dec, out / "cla_error_decomposition.csv", "Where the CLA error comes from")

    gate_csv = RESULTS_DIR / "gate" / "gate_results.csv"
    if gate_csv.exists():
        gall = pd.read_csv(gate_csv).query("gate == 'ALL'").iloc[0]
        cmp_ = pd.DataFrame([
            {"level": "gates (18 cases x 200)", **{k: gall[k] for k in
             ["bias_pct", "median_err_pct", "p95_err_pct", "max_err_pct", "mean_acc_pct"]}},
            {"level": "3-bit CLA (128 x 200)", **{k: v for k, v in metrics(d.i_leak_ml, d.i_leak_spice).items()
             if k in ["bias_pct", "median_err_pct", "p95_err_pct", "max_err_pct", "mean_acc_pct"]}},
        ])
        write_table(cmp_, out / "cla_vs_gate_level.csv", "Gate level vs circuit level")
    else:
        print("(run 06_gate_analysis.py first to get the gate-vs-CLA comparison)")

    # ---------------- plots
    m = metrics(d.i_leak_ml, d.i_leak_spice)
    note = (f"median err {m['median_err_pct']:.2f}%\np95 err    {m['p95_err_pct']:.2f}%\n"
            f"bias       {m['bias_pct']:+.2f}%\nmean acc   {m['mean_acc_pct']:.2f}%")
    parity_plot(d.i_leak_spice, d.i_leak_ml, out / "cla_parity.png",
                "3-bit CLA leakage: hierarchical ML vs full SPICE", note)
    mc = metrics(d.i_leak_gatesum_spice, d.i_leak_spice)
    parity_plot(d.i_leak_spice, d.i_leak_gatesum_spice, out / "cla_parity_gatesum.png",
                "3-bit CLA: sum of isolated-gate SPICE vs full SPICE",
                f"median err {mc['median_err_pct']:.2f}%\np95 err    {mc['p95_err_pct']:.2f}%\n"
                f"bias       {mc['bias_pct']:+.2f}%")
    error_vs(d, ["temp", "vdd", "n_vth0", "p_vth0"], "err_pct", out / "cla_error_vs_corner.png",
             "CLA leakage error vs corner conditions",
             xlabels={"temp": "temperature [°C]", "vdd": "VDD [V]",
                      "n_vth0": "NMOS vth0 [V]", "p_vth0": "PMOS vth0 [V]"})
    d["ones_label"] = d["n_ones"].astype(str) + " ones"
    error_box(d, "ones_label", "err_pct", out / "cla_error_by_ones.png",
              "CLA leakage error vs number of 1s in the input",
              order=[f"{k} ones" for k in range(8)])

    worst = d.reindex(d.err_pct.abs().sort_values(ascending=False).index).head(30)
    worst.to_csv(out / "cla_worst_cases.csv", index=False)
    print(f"All CLA results written to {out}")


if __name__ == "__main__":
    main()
