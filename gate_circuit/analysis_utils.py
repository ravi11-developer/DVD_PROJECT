"""Metrics and plotting helpers shared by the gate (06) and CLA (08) analysis."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order, never cycled) + recessive inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#1f1f1e"
INK_2 = "#5f5e5a"
GRID = "#e4e3de"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ------------------------------------------------------------------ metrics
def rel_err(ml, sp):
    """Signed relative error in % : 100 * (ML - SPICE) / SPICE."""
    ml, sp = np.asarray(ml, float), np.asarray(sp, float)
    return 100.0 * (ml - sp) / sp


def metrics(ml, sp):
    """Row of the results table.

    spice_median / ml_median : medians of the leakage over the corners [A]
    bias_pct     : median signed relative error (ML-SPICE)/SPICE  [%]
    median_err   : median |relative error|  [%]
    p95_err      : 95th percentile |relative error|  [%]
    mean_acc     : 100 - mean(min(|rel err|, 100))   [%]   (project target ~99 %)
    """
    r = rel_err(ml, sp)
    a = np.abs(r)
    return {
        "n": int(len(a)),
        "spice_median_A": float(np.median(sp)),
        "ml_median_A": float(np.median(ml)),
        "bias_pct": float(np.median(r)),
        "median_err_pct": float(np.median(a)),
        "p95_err_pct": float(np.percentile(a, 95)),
        "max_err_pct": float(a.max()),
        "mean_acc_pct": float(100.0 - np.minimum(a, 100.0).mean()),
    }


def metrics_table(df, by, ml_col, sp_col, overall_label="ALL"):
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **metrics(g[ml_col], g[sp_col])})
    allrow = {b: overall_label for b in by}
    rows.append({**allrow, **metrics(df[ml_col], df[sp_col])})
    return pd.DataFrame(rows)


def write_table(df, path_csv, title=""):
    """Save CSV + a fixed-width text version that pastes cleanly into a report."""
    df.to_csv(path_csv, index=False)
    fmt = df.copy()
    for c in fmt.columns:
        if c.endswith("_A"):
            fmt[c] = fmt[c].map(lambda v: f"{v:.4e}")
        elif c.endswith("_pct"):
            fmt[c] = fmt[c].map(lambda v: f"{v:+.3f}" if c == "bias_pct" else f"{v:.3f}")
    txt = (title + "\n" if title else "") + fmt.to_string(index=False) + "\n"
    path_csv.with_suffix(".txt").write_text(txt)
    print(txt)


# ------------------------------------------------------------------ plots
def _parity_frame(ax, lo, hi, bands=(5, 10)):
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=INK, lw=1.0, zorder=1, label="ML = SPICE")
    styles = ["--", ":"]
    for b, ls in zip(bands, styles):
        ax.plot(xs, xs * (1 + b / 100), color=INK_2, lw=0.8, ls=ls, zorder=1, label=f"±{b}%")
        ax.plot(xs, xs * (1 - b / 100), color=INK_2, lw=0.8, ls=ls, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("SPICE leakage [A]")
    ax.set_ylabel("ML leakage [A]")


def _limits(*arrs):
    v = np.concatenate([np.asarray(a, float) for a in arrs])
    v = v[v > 0]
    return v.min() / 1.3, v.max() * 1.3


def parity_plot(sp, ml, path, title, note=""):
    """Single-series log-log parity plot."""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lo, hi = _limits(sp, ml)
    _parity_frame(ax, lo, hi)
    ax.scatter(sp, ml, s=10, color=SERIES[0], alpha=0.55, linewidths=0, zorder=2,
               label=f"simulations (n={len(sp)})")
    ax.set_title(title)
    if note:
        ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=9,
                color=INK, family="monospace")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def parity_by_group(df, group, sub, sp_col, ml_col, path, title, panel_order=None):
    """Small multiples: one panel per `group` value, `sub` encoded by colour AND marker."""
    panels = panel_order or list(dict.fromkeys(df[group]))
    n = len(panels)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.2 * nrow), squeeze=False)
    lo, hi = _limits(df[sp_col], df[ml_col])
    for ax, p in zip(axes.flat, panels):
        d = df[df[group] == p]
        _parity_frame(ax, lo, hi)
        for k, (s, g) in enumerate(d.groupby(sub, sort=True)):
            ax.scatter(g[sp_col], g[ml_col], s=12, color=SERIES[k % 8], marker=MARKERS[k % 8],
                       alpha=0.6, linewidths=0, zorder=2, label=f"{sub}={s}")
        m = metrics(d[ml_col], d[sp_col])
        ax.set_title(f"{p}   median err {m['median_err_pct']:.2f}%  p95 {m['p95_err_pct']:.2f}%",
                     fontsize=10)
        ax.legend(loc="lower right", fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def error_vs(df, xcols, err_col, path, title, xlabels=None):
    """|error| (log y) against corner variables, one panel per variable."""
    n = len(xcols)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    y = np.maximum(np.abs(df[err_col].to_numpy()), 1e-4)
    for i, (ax, c) in enumerate(zip(axes.flat, xcols)):
        ax.scatter(df[c], y, s=6, color=SERIES[0], alpha=0.4, linewidths=0)
        # binned median and p95 trend
        bins = np.quantile(df[c], np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(df[c], bins[1:-1]), 0, 9)
        mids = [df[c][idx == k].median() for k in range(10)]
        med = [np.median(y[idx == k]) for k in range(10)]
        p95 = [np.percentile(y[idx == k], 95) for k in range(10)]
        ax.plot(mids, med, color=INK, lw=2, label="binned median")
        ax.plot(mids, p95, color=SERIES[1], lw=2, label="binned p95")
        ax.set_yscale("log")
        ax.set_xlabel((xlabels or {}).get(c, c))
        if i == 0:
            ax.set_ylabel("|ML - SPICE| / SPICE [%]")
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def error_box(df, cat_col, err_col, path, title, order=None):
    """Box plot of |error| per category (log y)."""
    order = order or list(dict.fromkeys(df[cat_col]))
    data = [np.maximum(np.abs(df.loc[df[cat_col] == o, err_col]), 1e-4) for o in order]
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(order) + 2), 4))
    bp = ax.boxplot(data, whis=(5, 95), showfliers=True, patch_artist=True,
                    flierprops=dict(marker="o", markersize=2, markerfacecolor=INK_2,
                                    markeredgewidth=0, alpha=0.5),
                    medianprops=dict(color=INK, lw=1.5))
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=SERIES[0])
    ax.set_xticks(range(1, len(order) + 1), order, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("|ML - SPICE| / SPICE [%]   (whiskers 5-95%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
