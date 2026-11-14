"""
make_report_assets.py - refresh everything report.tex reads, from the current project outputs.

    report/figures/     copies of every figure the report shows
    report/generated/   tables (tab_*.tex) and numbers.tex (one \\newcommand per number)

report.tex only \\input{}s these files, so after re-running any stage just run this script
(or ./run_all.sh report) and recompile. Nothing here compiles LaTeX.

Gate and test-circuit material is taken only from the ML run (backend = ml). Until that run exists
the tables say "pending" and the figures fall back to the \\includeimage placeholder box.

Run (from anywhere):  python report/make_report_assets.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
FIG, GEN = REPORT / "figures", REPORT / "generated"
GC = ROOT / "gate_circuit"
PENDING = r"\textit{[pending]}"
MODEL_DIRS = {
    "NMOS ON": (ROOT / "ml_model/NMOS_ON/nmos_on_code/results", "FINAL"),
    "NMOS OFF": (ROOT / "ml_model/NMOS_OFF/nmos_off_physics_ml/results", "physics + LightGBM (3-seed avg)"),
    "PMOS ON": (ROOT / "ml_model/PMOS_ON/pmos_on_physics_ml/results", "FINAL"),
    "PMOS OFF": (ROOT / "ml_model/PMOS_OFF/pmos_off_physics_ml/results", "physics + LightGBM (3-seed avg)"),
}
PHYS_LABEL = {"NMOS ON": "physics only (A)", "NMOS OFF": "physics only", "PMOS ON": "physics only (A)",
              "PMOS OFF": "physics only"}
numbers = {}
notes = []


# ---------------------------------------------------------------- helpers
def num(name, value, fmt="{:.2f}"):
    numbers[name] = value if isinstance(value, str) else fmt.format(value)


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def sci(v, digits=2):
    m, e = f"{v:.{digits}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def write_table(name, header, rows, align, caption=None):
    lines = [r"\begin{tabular}{" + align + "}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    lines += [r"\hline" if r == "HLINE" else " & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}"]
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n")


def pending_table(name, text):
    (GEN / f"{name}.tex").write_text(r"\fbox{\parbox{0.8\linewidth}{\centering " + text + "}}\n")


def copy_fig(src, name):
    src = Path(src)
    if src.exists():
        shutil.copy2(src, FIG / name)
    else:
        (FIG / name).unlink(missing_ok=True)
        notes.append(f"figure missing: {src.relative_to(ROOT)}")


def read_metrics(folder):
    m = pd.read_csv(folder / "metrics.csv", skipinitialspace=True)
    m.columns = m.columns.str.strip()
    for c in ("model", "split", "current"):
        m[c] = m[c].astype(str).str.strip()
    return m


def backend(csv):
    return pd.read_csv(csv, usecols=["backend"], nrows=1)["backend"].iloc[0] if csv.exists() else None


# ---------------------------------------------------------------- sections
def data_check():
    dc = ROOT / "dataset/data_check"
    for f in ("trend_off_current_vs_temp.png", "trend_off_current_vs_vth0.png", "trend_gate_current_vs_toxe.png"):
        copy_fig(dc / f, f)
    k = pd.read_csv(dc / "kcl_check.csv")
    write_table("tab_kcl", ["Case", r"max $|\Sigma I|$ (A)", r"mean $|\Sigma I|$ (A)",
                            "median rel.", "max rel.", "All pass"],
                [[r.case, sci(r.max_abs_residual_A), sci(r.mean_abs_residual_A),
                  sci(r.median_rel_residual), sci(r.max_rel_residual),
                  "yes" if r.rows_passing == r.rows else "no"] for r in k.itertuples()],
                "lrrrrr")
    num("KclWorstRel", sci(k.max_rel_residual.max()), "{}")
    num("KclWorstAbs", sci(k.max_abs_residual_A.max()), "{}")
    t = pd.read_csv(dc / "trend_summary.csv")
    g = lambda plot, case: t[(t["plot"] == plot) & (t.case == case)].spearman.iloc[0]
    num("RhoTempN", g("OFF current vs temperature", "NMOS OFF"))
    num("RhoTempP", g("OFF current vs temperature", "PMOS OFF"))
    vb = t[t["plot"].str.startswith("OFF current vs |vth0| (") & ~t["plot"].str.contains("all")]
    num("RhoVthMin", vb.spearman.min()); num("RhoVthMax", vb.spearman.max())
    tox = t[t["plot"] == "gate current vs toxe"]
    num("RhoToxMin", tox.spearman.min()); num("RhoToxMax", tox.spearman.max())
    rows_of = lambda case: str(int(t[(t["plot"] == "OFF current vs temperature") & (t.case == case)].rows.iloc[0]))
    num("OffRowsN", rows_of("NMOS OFF")); num("OffRowsP", rows_of("PMOS OFF"))


def models():
    rows_test, rows_abl, accs = [], [], []
    for case, (folder, label) in MODEL_DIRS.items():
        copy_fig(folder / "parity_test.png", f"parity_{case.lower().replace(' ', '_')}.png")
        if not (folder / "metrics.csv").exists():
            notes.append(f"metrics missing for {case}"); continue
        m = read_metrics(folder)
        te = m[(m.model == label) & (m.split == "test")].set_index("current")
        va = m[(m.model == label) & (m.split == "val")].set_index("current")
        ph = m[(m.model == PHYS_LABEL[case]) & (m.split == "val")].set_index("current")
        for cur in ("ID", "IG", "IS", "IB"):
            r = te.loc[cur]
            accs.append((r.mean_accuracy, f"{case} {cur}"))
            rows_test.append([case if cur == "ID" else "", r"$I_{" + cur[1].lower() + "}$",
                              f"{r.median_err:.3f}", f"{r.p95_err:.3f}", f"{r.mean_accuracy:.2f}", f"{r.r2_log:.5f}"])
            rows_abl.append([case if cur == "ID" else "", r"$I_{" + cur[1].lower() + "}$",
                             f"{ph.loc[cur].median_err:.2f}", f"{ph.loc[cur].mean_accuracy:.2f}",
                             f"{va.loc[cur].median_err:.3f}", f"{va.loc[cur].p95_err:.3f}", f"{va.loc[cur].mean_accuracy:.2f}"])
    write_table("tab_models_test", ["Case", "Current", "Median err (\\%)", "p95 err (\\%)", "Mean acc. (\\%)",
                                    r"$R^2$ (log)"], rows_test, "llrrrr")
    write_table("tab_models_ablation", ["Case", "Current", "Phys. median (\\%)", "Phys. mean acc. (\\%)",
                                        "ML median (\\%)", "ML p95 (\\%)", "ML mean acc. (\\%)"],
                rows_abl, "llrrrrr")
    if accs:
        lo = min(accs)
        num("ModelMinAcc", lo[0]); num("ModelMinAccWhere", lo[1], "{}")
        num("ModelMaxAcc", max(accs)[0])
    sel = ROOT / "ml_model/PMOS_OFF/model_selection.csv"
    s = pd.read_csv(sel)
    piv = s.pivot(index="current", columns="model", values="val_mean_accuracy_pct").loc[["ID", "IG", "IS", "IB"]]
    piv.columns = [c.split(" (")[0] for c in piv.columns]           # drop file names from the headers
    cols = list(piv.columns)
    write_table("tab_pmos_off_selection", ["Current"] + [tex_escape(c) for c in cols],
                [[r"$I_{" + i[1].lower() + "}$"] + [f"{piv.loc[i, c]:.2f}" for c in cols] for i in piv.index],
                "l" + "r" * len(cols))


def gates():
    names = ["gate_parity.png", "gate_parity_by_gate.png", "gate_error_by_case.png", "gate_error_vs_corner.png"]
    ok = backend(GC / "data/gate_ml.csv") == "ml"
    for f in names:
        copy_fig(GC / "results/gate" / f, f) if ok else (FIG / f).unlink(missing_ok=True)
    keys = ["GateWorstBias", "GateMeanAcc", "GateMedErr", "GatePninetyfive", "GateMaxErr", "GateBias", "GateBest", "GateWorst",
            "GateWorstAcc", "GateWorstPninetyfive", "GateBestGate", "GateWorstGate", "GateErrHot", "GateErrCold",
            "GateNinetyNineCount"]
    if not ok:
        for k in keys: num(k, PENDING, "{}")
        pending_table("tab_gate", "Gate results pending: run the ML gate stage (\\texttt{./run_all.sh from gates}).")
        pending_table("tab_gate_by_gate", "pending")
        pending_table("tab_transistor_in_gate", "pending")
        for k in ("PmosOffShare", "PmosOnShare", "NmosOnShare", "OnShare", "OnVdsMedian", "OnVdsNinety", "OnTrainSmallVds"): num(k, PENDING, "{}")
        notes.append("gate results pending (no ML gate run yet)")
        return
    g = pd.read_csv(GC / "results/gate/gate_results.csv", dtype={"inputs": str})
    rows = []
    for r in g.itertuples():
        lab = "All 18 cases" if r.gate == "ALL" else r.gate
        rows.append([lab, "" if r.gate == "ALL" else r.inputs, int(r.n), sci(r.spice_median_A), sci(r.ml_median_A),
                     f"{r.bias_pct:+.3f}", f"{r.median_err_pct:.3f}", f"{r.p95_err_pct:.3f}", f"{r.mean_acc_pct:.2f}"])
    rows.insert(len(rows) - 1, "HLINE")
    write_table("tab_gate", ["Gate", "In", "n", "SPICE median (A)", "ML median (A)", "Bias (\\%)",
                             "Median err (\\%)", "p95 err (\\%)", "Mean acc. (\\%)"], rows, "llrrrrrrr")
    allr = g[g.gate == "ALL"].iloc[0]
    cases = g[g.gate != "ALL"].copy()
    cases["label"] = cases.gate + " " + cases.inputs.astype(str)
    num("GateMeanAcc", allr.mean_acc_pct); num("GateMedErr", allr.median_err_pct, "{:.3f}")
    num("GatePninetyfive", allr.p95_err_pct); num("GateMaxErr", allr.max_err_pct); num("GateBias", allr.bias_pct, "{:+.3f}")
    best, worst = cases.loc[cases.median_err_pct.idxmin()], cases.loc[cases.mean_acc_pct.idxmin()]
    num("GateBest", f"{best.label} (median {best.median_err_pct:.3f}\\,\\%)", "{}")
    num("GateWorst", worst.label, "{}"); num("GateWorstAcc", worst.mean_acc_pct)
    num("GateWorstBias", worst.bias_pct, "{:+.2f}")
    num("GateWorstPninetyfive", worst.p95_err_pct)
    num("GateNinetyNineCount", str(int((cases.mean_acc_pct >= 99).sum())))
    s = pd.read_csv(GC / "results/gate/gate_summary_by_gate.csv")
    s = s[s.gate != "ALL"]
    num("GateBestGate", s.loc[s.mean_acc_pct.idxmax()].gate, "{}"); num("GateWorstGate", s.loc[s.mean_acc_pct.idxmin()].gate, "{}")
    write_table("tab_gate_by_gate", ["Gate", "n", "Bias (\\%)", "Median err (\\%)", "p95 err (\\%)", "Mean acc. (\\%)"],
                [[r.gate, int(r.n), f"{r.bias_pct:+.3f}", f"{r.median_err_pct:.3f}", f"{r.p95_err_pct:.3f}",
                  f"{r.mean_acc_pct:.2f}"] for r in s.itertuples()], "lrrrrr")
    d = pd.read_csv(GC / "data/gate_ml.csv")
    e = (d.i_leak_ml - d.i_leak_spice).abs() / d.i_leak_spice * 100
    num("GateErrHot", np.median(e[d.temp >= 85]), "{:.3f}"); num("GateErrCold", np.median(e[d.temp <= 0]), "{:.3f}")
    transistors_in_gates()


def transistors_in_gates():
    """Per-case terminal errors inside the gates and the |Vds| the ON transistors see there."""
    te = pd.read_csv(GC / "results/gate/transistor_errors.csv")
    rows, share = [], {}
    for case in ("nmos_on", "nmos_off", "pmos_on", "pmos_off"):
        sub = te[te.case == case].set_index("terminal")
        share[case] = sub.share_of_gate_leak_abs_err_pct.iloc[0]
        rows.append([case.upper().replace("_", " ")] +
                    [f"{sub.loc[k].median_err_pct:.2f} / {sub.loc[k].p95_err_pct:.1f}" for k in ("id", "ig", "is", "ib")] +
                    [f"{share[case]:.1f}"])
    write_table("tab_transistor_in_gate", ["Case", r"$I_d$", r"$I_g$", r"$I_s$", r"$I_b$", "Share of gate error (\\%)"],
                rows, "lrrrrr")
    num("PmosOnShare", share["pmos_on"], "{:.0f}"); num("NmosOnShare", share["nmos_on"], "{:.0f}")
    num("PmosOffShare", share["pmos_off"], "{:.0f}")
    num("OnShare", share["pmos_on"] + share["nmos_on"], "{:.0f}")
    tr = pd.read_csv(GC / "data/gate_transistors_ml.csv")
    on = tr[tr.case.str.endswith("_on")]
    vds = (on.vd - on.vs).abs() * 1e3
    num("OnVdsMedian", vds.median(), "{:.2f}"); num("OnVdsNinety", vds.quantile(0.9), "{:.1f}")
    frac = []
    for c in ("NMOS_ON", "PMOS_ON"):
        d = pd.read_csv(ROOT / "dataset" / c / f"{c.lower()}.csv")
        frac.append(((d.vd - d.vs).abs() < 0.01).mean() * 100)
    num("OnTrainSmallVds", np.mean(frac), "{:.1f}")


def cla():
    names = ["cla_parity.png", "cla_error_by_ones.png", "cla_error_vs_corner.png", "cla_parity_gatesum.png"]
    ok = backend(GC / "data/cla_ml.csv") == "ml"
    for f in names:
        copy_fig(GC / "results/cla" / f, f) if ok else (FIG / f).unlink(missing_ok=True)
    keys = ["ClaMeanAcc", "ClaMedErr", "ClaPninetyfive", "ClaMaxErr", "ClaBias", "ClaCompMedErr", "ClaCompMeanAcc",
            "ClaModelMedErr", "ClaModelMeanAcc", "ClaErrHot", "ClaErrCold"]
    if not ok:
        for k in keys: num(k, PENDING, "{}")
        for t in ("tab_cla", "tab_cla_decomp"):
            pending_table(t, "Test-circuit results pending: run the ML gate stage (\\texttt{./run_all.sh from gates}).")
        notes.append("CLA results pending (no ML run yet)")
        return
    c = pd.read_csv(GC / "results/cla/cla_results.csv", dtype={"n_ones": str})
    rows = [["All 128 vectors" if r.n_ones == "ALL" else r.n_ones, int(r.n), sci(r.spice_median_A), sci(r.ml_median_A),
             f"{r.bias_pct:+.3f}", f"{r.median_err_pct:.3f}", f"{r.p95_err_pct:.3f}", f"{r.mean_acc_pct:.2f}"]
            for r in c.itertuples()]
    write_table("tab_cla", ["Inputs at 1", "n", "SPICE median (A)", "ML median (A)", "Bias (\\%)",
                            "Median err (\\%)", "p95 err (\\%)", "Mean acc. (\\%)"], rows, "lrrrrrrr")
    allr = c[c.n_ones == "ALL"].iloc[0]
    num("ClaMeanAcc", allr.mean_acc_pct); num("ClaMedErr", allr.median_err_pct, "{:.3f}")
    num("ClaPninetyfive", allr.p95_err_pct); num("ClaMaxErr", allr.max_err_pct); num("ClaBias", allr.bias_pct, "{:+.3f}")
    dec = pd.read_csv(GC / "results/cla/cla_error_decomposition.csv")
    write_table("tab_cla_decomp", ["Comparison", "Bias (\\%)", "Median err (\\%)", "p95 err (\\%)", "Mean acc. (\\%)"],
                [[tex_escape(r.comparison), f"{r.bias_pct:+.3f}", f"{r.median_err_pct:.3f}", f"{r.p95_err_pct:.3f}",
                  f"{r.mean_acc_pct:.2f}"] for r in dec.itertuples()], "lrrrr")
    comp = dec[dec.comparison.str.contains("composition")].iloc[0]
    mod = dec[dec.comparison.str.contains("model error")].iloc[0]
    num("ClaCompMedErr", comp.median_err_pct, "{:.3f}"); num("ClaCompMeanAcc", comp.mean_acc_pct)
    num("ClaModelMedErr", mod.median_err_pct, "{:.3f}"); num("ClaModelMeanAcc", mod.mean_acc_pct)
    d = pd.read_csv(GC / "data/cla_ml.csv")
    e = (d.i_leak_ml - d.i_leak_spice).abs() / d.i_leak_spice * 100
    num("ClaErrHot", np.median(e[d.temp >= 85]), "{:.3f}"); num("ClaErrCold", np.median(e[d.temp <= 0]), "{:.3f}")


def augmentation():
    """Before/after table of the gate-region data augmentation (ml_model/augmentation_effect.csv)."""
    f = ROOT / "ml_model/augmentation_effect.csv"
    if not f.exists():
        pending_table("tab_augmentation", "pending"); return
    a = pd.read_csv(f)
    rows = []
    for case in ("nmos_on", "nmos_off", "pmos_on", "pmos_off"):
        for cur in ("ID", "IG", "IS", "IB"):
            g = a[(a.case == case) & (a.current == cur)].set_index("split")
            rows.append([case.upper().replace("_", " ") if cur == "ID" else "", r"$I_{" + cur[1].lower() + "}$",
                         f"{g.loc['test_aug'].before_aug:.2f}", rf"\textbf{{{g.loc['test_aug'].after_aug:.2f}}}",
                         f"{g.loc['test'].before_aug:.2f}", f"{g.loc['test'].after_aug:.2f}"])
    write_table("tab_augmentation", ["Case", "Current", "Gate region before", "Gate region after",
                                     "Orig. test before", "Orig. test after"], rows, "llrrrr")
    on = a[a.case.str.endswith("_on") & a.current.isin(["ID", "IS"]) & (a.split == "test_aug")]
    num("AugOnBeforeMin", on.before_aug.min()); num("AugOnBeforeMax", on.before_aug.max())
    num("AugOnAfterMin", on.after_aug.min()); num("AugOnAfterMax", on.after_aug.max())
    ta = a[a.split == "test_aug"]
    num("AugAfterMin", ta.after_aug.min())


def main():
    FIG.mkdir(exist_ok=True); GEN.mkdir(exist_ok=True)
    data_check(); models(); gates(); cla(); augmentation()
    (GEN / "numbers.tex").write_text("% generated by make_report_assets.py - do not edit\n" +
                                     "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in sorted(numbers.items())))
    print(f"figures   -> {FIG.relative_to(ROOT)}/  ({len(list(FIG.glob('*.png')))} files)")
    print(f"generated -> {GEN.relative_to(ROOT)}/  ({len(list(GEN.glob('*.tex')))} files)")
    for n in notes:
        print("  note:", n)


if __name__ == "__main__":
    main()
