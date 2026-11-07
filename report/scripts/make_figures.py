"""Regenerate the paper's figures at IEEE column width from the saved results (no new measurements).

Inputs (all produced by `fvs` and committed under results/full/):
  tables/summary_test_by_level.csv      held-out per-level summaries
  verify/estimation.parquet             estimated vs actual rows for all 2,700 filters
  calibration/threshold_curve.parquet   simulated P1 on the calibration split, per candidate T
  calibration/frozen_router.json        frozen threshold T*
Run from the repository root:  uv run python report/scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "full"
OUT = ROOT / "report" / "figures"

COL_W = 3.5      # IEEE single column (in)
FULL_W = 7.16    # IEEE double column (in)

# Validated categorical palette (same entity colours as results/full/figures)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"B1": "#2a78d6", "B2-plain": "#eb6834", "B2-iter": "#1baf7a", "B3": "#eda100", "P1": "#e87ba4",
         "Oracle": INK2}
MARKER = {"B1": "o", "B2-plain": "s", "B2-iter": "D", "B3": "^", "P1": "P", "Oracle": "_"}
LABEL = {"B1": "B1 exact", "B2-plain": "B2-plain", "B2-iter": "B2-iter", "B3": "B3 planner", "P1": "P1 router",
         "Oracle": "Oracle (latency floor)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE,
})


def plain_log(axis, minor=True):
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1000 else f"{v:g}"))
    if minor:
        axis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())


def sel_axis(ax, sels, label=True):
    ax.set_xscale("log")
    ax.set_xticks(sels)
    ax.set_xticklabels([f"{s * 100:g}" for s in sels])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("filter selectivity (% of rows that pass)")


def line(ax, x, y, m):
    ax.plot(x, y, color=COLOR[m], marker=MARKER[m], label=LABEL[m], markeredgecolor=SURFACE,
            markeredgewidth=0.6, solid_capstyle="round", solid_joinstyle="round")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by = pd.read_csv(RES / "tables" / "summary_test_by_level.csv")
    sels = sorted(by["target_sel"].unique())
    methods = ["B1", "B2-plain", "B2-iter", "B3", "P1", "Oracle"]

    # ---- Fig: latency vs selectivity (p50 | p95), double column --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.1))
    for ax, stat, title in zip(axes, ["p50_ms", "p95_ms"], ["(a) median (p50) latency", "(b) p95 latency"]):
        for m in methods:
            d = by[by["method"] == m].sort_values("target_sel")
            line(ax, d["target_sel"], d[stat], m)
        sel_axis(ax, sels)
        ax.set_yscale("log")
        plain_log(ax.yaxis)
        ax.set_ylabel("latency (ms, log scale)")
        ax.set_title(title, loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "latency_vs_selectivity")

    # ---- Fig: recall and completeness, single column -------------------------------------------
    qm = ["B1", "B2-plain", "B2-iter", "B3", "P1"]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.7), sharex=True)
    for ax, stat, ylab in zip(axes, ["recall", "completeness"], ["mean strict Recall@10", "share complete"]):
        for m in qm:
            d = by[by["method"] == m].sort_values("target_sel")
            line(ax, d["target_sel"], d[stat], m)
        sel_axis(ax, sels, label=(stat == "completeness"))
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel(ylab)
    axes[0].axhline(0.95, color=MUTED, linewidth=0.5)
    axes[1].axhline(0.99, color=MUTED, linewidth=0.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "quality_vs_selectivity")

    # ---- Fig: estimated vs actual rows, single column ------------------------------------------
    est = pd.read_parquet(RES / "verify" / "estimation.parquet")
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.95))
    ax = axes[0]
    ax.scatter(est["actual"], est["est_ours"].clip(lower=1), s=3, color=COLOR["B1"], alpha=0.35, edgecolors="none",
               rasterized=True)
    lim = [60, 1.6e5]
    ax.plot(lim, lim, color=INK2, linewidth=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    plain_log(ax.xaxis, minor=False); plain_log(ax.yaxis, minor=False)
    ax.set_xlim(*lim)
    ax.set_xlabel("actual matching rows $|E|$")
    ax.set_ylabel(r"estimated rows $\hat{M}$")
    ax.set_title("(a) all 2,700 filters (line: $\\hat{M}=|E|$)", loc="left")
    ax = axes[1]
    est["abs_rel"] = est["rel_err_ours"].abs()
    g = est.groupby("target_sel")["abs_rel"]
    ax.plot(g.mean().index, g.mean().values, color=INK, marker="o", markeredgecolor=SURFACE,
            markeredgewidth=0.6, label="mean")
    ax.plot(g.quantile(0.95).index, g.quantile(0.95).values, color=MUTED, marker="s", markeredgecolor=SURFACE,
            markeredgewidth=0.6, label="95th percentile")
    sel_axis(ax, sels)
    ax.set_ylabel(r"$|\hat{M}-|E||\,/\,|E|$")
    ax.set_title("(b) relative estimation error by level", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout(h_pad=0.8)
    save(fig, "estimation")

    # ---- Fig: calibration curve, single column -------------------------------------------------
    curve = pd.read_parquet(RES / "calibration" / "threshold_curve.parquet")
    frozen = json.loads((RES / "calibration" / "frozen_router.json").read_text())
    T = frozen["threshold_T"]
    c = curve[np.isfinite(curve["T"]) & (curve["T"] > 0)]
    feas = curve[curve["feasible"]]
    near = feas[(feas["mean_ms"] <= feas["mean_ms"].min() * 1.01) & np.isfinite(feas["T"])]
    lo, hi = near["T"].min(), near["T"].max()
    cal = est[est["split"] == "calibration"].groupby("target_sel")["est_ours"].median()
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.7), sharex=True)
    ax = axes[0]
    ax.axvspan(lo, hi, color=GRID, alpha=0.7, linewidth=0)
    ax.plot(c["T"], c["mean_ms"], color=INK, label="mean")
    ax.plot(c["T"], c["p95_ms"], color=MUTED, label="p95")
    ax.set_ylabel("simulated P1 latency (ms)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.3, 0.9), ncol=2)
    for s, m in cal.items():
        if s == 0.5:   # 50% and 90% medians are too close to label both; 90% marks the right end
            continue
        ax.text(m, 27.5, f"{s * 100:g}%", ha="center", va="bottom", fontsize=5.5, color=INK2)
    ax.set_ylim(0, 29)
    ax = axes[1]
    ax.axvspan(lo, hi, color=GRID, alpha=0.7, linewidth=0)
    ax.plot(c["T"], c["recall"], color=INK, label="mean strict Recall@10")
    ax.plot(c["T"], c["completeness"], color=MUTED, label="completeness")
    ax.axhline(0.95, color=AXIS, linewidth=0.6)
    ax.set_ylim(0.945, 1.003)
    ax.set_ylabel("quality")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(r"threshold $T$ (estimated rows; route to B1 if $\hat{M}\leq T$)")
    ax.legend(loc="lower right")
    for a in axes:
        a.axvline(T, color=COLOR["P1"], linewidth=1.0)
    axes[0].annotate(f"$T^*$ = {T:,.0f}", (T, 15), xytext=(3, 0), textcoords="offset points", fontsize=6,
                     color=INK)
    fig.tight_layout(h_pad=0.4)
    save(fig, "calibration_curve")


if __name__ == "__main__":
    main()
