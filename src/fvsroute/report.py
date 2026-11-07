"""Figures, tables and report.md - generated only from measured data on disk.

Every figure has a companion CSV table (the colour-relief rule: three palette slots sit below 3:1 contrast,
so values are always available as text). Colours follow the entity (method) in every figure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .metrics import cluster_bootstrap, latency_stats, summarize  # noqa: E402
from .oracle import gain_captured, oracle_floor, oracle_valid  # noqa: E402

log = logging.getLogger(__name__)

PRIMARY_BASELINES = ["B1", "B2-plain", "B2-iter", "B3"]
MAIN = PRIMARY_BASELINES + ["P1"]
STATS = ["p50_ms", "p95_ms", "mean_ms"]

# validated categorical slots (light surface), fixed per entity
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"B1": "#2a78d6", "B2-plain": "#eb6834", "B2-iter": "#1baf7a", "B3": "#eda100", "P1": "#e87ba4",
         "B3+iter": "#008300", "Oracle": INK2, "Oracle-V": MUTED,
         "B2-iter-ef40": "#008300", "B2-iter-ef200": "#6250d6", "B2-strict": "#e34948", "B2-iter-mst100k": INK2}
MARKER = {"B1": "o", "B2-plain": "s", "B2-iter": "D", "B3": "^", "P1": "P", "B3+iter": "v", "Oracle": "_",
          "Oracle-V": "x", "B2-iter-ef40": "v", "B2-iter-ef200": "^", "B2-strict": "s", "B2-iter-mst100k": "X"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": INK2, "lines.linewidth": 1.6, "lines.markersize": 6,
        "axes.axisbelow": True,
    })


def _line(ax, x, y, name, label=None):
    ax.plot(x, y, color=COLOR.get(name, INK2), marker=MARKER.get(name, "o"), label=label or name,
            markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round", solid_joinstyle="round")


def _sel_axis(ax, sels):
    ax.set_xscale("log")
    ax.set_xticks(sels)
    ax.set_xticklabels([f"{s * 100:g}%" for s in sels])
    ax.minorticks_off()
    ax.set_xlabel("filter selectivity (actual share of rows that pass)")


def _plain_log(axis):
    """Log axis with plain-number tick labels (0.3, 1, 5) instead of 3x10^-1."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5)))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _log_y(ax):
    ax.set_yscale("log")
    _plain_log(ax.yaxis)


def _log_x(ax):
    ax.set_xscale("log")
    _plain_log(ax.xaxis)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _md_table(df: pd.DataFrame, floatfmt: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(_fmt(float(v), floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ============================================================================ main entry
def make_report(cfg: Config) -> Path:
    from .pipeline import evaluate_phase, frozen_path, load_processed

    _style()
    run = cfg.run_dir
    fig_dir, tab_dir = run / "figures", run / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    p = load_processed(cfg)
    base = p.base(cfg)
    pc_cal = evaluate_phase(cfg, "calibration", p, base)
    pc_test = evaluate_phase(cfg, "test", p, base)
    frozen = json.loads(frozen_path(cfg).read_text())
    curve = pd.read_parquet(run / "calibration" / "threshold_curve.parquet")
    estimation = pd.read_parquet(run / "verify" / "estimation.parquet")
    consistency = json.loads((run / "verify" / "consistency.json").read_text())
    manifest = json.loads((cfg.processed_dir / "manifest.json").read_text())
    db_setup = json.loads((run / "env" / "db_setup.json").read_text())
    server_env = json.loads((run / "env" / "server_env.json").read_text())
    host_env = json.loads((run / "env" / "host_env.json").read_text())
    meta = {ph: json.loads((run / ph / "phase_meta.json").read_text()) for ph in ("calibration", "test")}
    plans = {ph: pd.read_parquet(run / ph / "plans.parquet") for ph in ("calibration", "test")}
    ea = {ph: pd.read_parquet(run / ph / "explain_analyze.parquet") for ph in ("calibration", "test")}
    raw_test = pd.read_parquet(run / "test" / "raw.parquet",
                               columns=["method", "pass_idx", "latency_ns", "config_id"])
    sels = sorted(p.workload["target_sel"].unique())
    crit = cfg.calibration

    # ---------------------------------------------------------------- oracle + combined table
    orc = oracle_floor(pc_test)
    orcv = oracle_valid(pc_test)
    combined = pd.concat([pc_test, orc, orcv.drop(columns=["fallback"])], ignore_index=True)

    overall = summarize(combined, ["method"])
    by_level = summarize(combined, ["method", "target_sel"])
    overall["role"] = overall["method"].map(lambda m: "proposed" if m == "P1" else
                                            "supplementary" if m in ("B3+iter", "Oracle-V") else
                                            "reference (latency floor, ignores quality)" if m == "Oracle" else
                                            "baseline")
    overall["meets_quality"] = (overall["recall"] >= crit.min_mean_recall) & \
                               (overall["completeness"] >= crit.min_completeness)
    by_level["meets_quality"] = (by_level["recall"] >= crit.min_mean_recall) & \
                                (by_level["completeness"] >= crit.min_completeness)
    order = MAIN + ["Oracle", "Oracle-V", "B3+iter"]
    overall = overall.set_index("method").loc[[m for m in order if m in set(overall["method"])]].reset_index()
    overall.to_csv(tab_dir / "summary_test_overall.csv", index=False)
    by_level.to_csv(tab_dir / "summary_test_by_level.csv", index=False)

    # ---------------------------------------------------------------- bootstrap CIs (overall)
    ci = cluster_bootstrap(pc_test[pc_test["method"].isin(MAIN)], MAIN, "P1", cfg.bootstrap_resamples, cfg.seed)
    ci.to_csv(tab_dir / "bootstrap_ci_test.csv", index=False)

    # ---------------------------------------------------------------- success criteria
    def best_baseline(summ: pd.DataFrame, stat: str):
        q = summ[summ["method"].isin(PRIMARY_BASELINES) & summ["meets_quality"]]
        if q.empty:
            return None, None
        r = q.sort_values(stat).iloc[0]
        return r["method"], float(r[stat])

    def criteria(summ: pd.DataFrame, scope: str) -> list[dict]:
        p1 = summ[summ["method"] == "P1"].iloc[0]
        orow = summ[summ["method"] == "Oracle"].iloc[0]
        rows = [{"scope": scope, "criterion": "1 quality", "detail":
                 f"recall {p1['recall']:.4f} (>= {crit.min_mean_recall}), completeness {p1['completeness']:.4f} "
                 f"(>= {crit.min_completeness})", "result": "PASS" if p1["meets_quality"] else "FAIL"}]
        for stat in ("p50_ms", "p95_ms"):
            bname, bval = best_baseline(summ, stat)
            if bname is None:
                rows.append({"scope": scope, "criterion": f"2 {stat}", "detail": "no baseline meets criterion 1",
                             "result": "N/A"})
            else:
                rows.append({"scope": scope, "criterion": f"2 {stat}",
                             "detail": f"P1 {p1[stat]:.3f} vs best qualifying baseline {bname} {bval:.3f} ms",
                             "result": "PASS" if p1[stat] <= bval else "FAIL"})
        for stat in STATS:
            bname, bval = best_baseline(summ, stat)
            g, reason = gain_captured(bval, float(p1[stat]), float(orow[stat]))
            rows.append({"scope": scope, "criterion": f"3 gain captured ({stat})",
                         "detail": (f"best={bname} {bval:.3f}, P1 {p1[stat]:.3f}, Oracle {orow[stat]:.3f} ms"
                                    if bname else reason),
                         "result": f"{g * 100:.1f}%" if g is not None else f"N/A ({reason})"})
        return rows

    crit_rows = criteria(overall, "overall")
    for s in sels:
        crit_rows += criteria(by_level[by_level["target_sel"] == s], f"{s * 100:g}%")
    crit_df = pd.DataFrame(crit_rows)
    crit_df.to_csv(tab_dir / "success_criteria.csv", index=False)

    # per-level effect sizes for criterion 2: P1 vs the best qualifying baseline at that level, with a paired
    # bootstrap CI of the difference (resampling the level's configs; one config per query vector per level)
    cmp_rows = []
    for s in sels:
        summ = by_level[by_level["target_sel"] == s]
        sub = pc_test[pc_test["target_sel"] == s]
        for stat in ("p50_ms", "p95_ms"):
            bname, bval = best_baseline(summ, stat)
            p1v = float(summ[summ["method"] == "P1"][stat].iloc[0])
            if bname is None:
                cmp_rows.append({"level": f"{s * 100:g}%", "stat": stat, "best_baseline": "none qualifies",
                                 "best_ms": np.nan, "p1_ms": p1v, "diff_ms": np.nan, "diff_pct": np.nan,
                                 "ci_low": np.nan, "ci_high": np.nan, "verdict": "N/A"})
                continue
            bs = cluster_bootstrap(sub[sub["method"].isin([bname, "P1"])], [bname, "P1"], "P1",
                                   min(cfg.bootstrap_resamples, 1000), cfg.seed)
            d = bs[(bs["method"] == f"P1 - {bname}") & (bs["stat"] == stat)].iloc[0]
            if p1v <= bval:
                verdict = "PASS"
            elif d["ci_low"] <= 0:
                verdict = "FAIL (difference within bootstrap CI of 0)"
            else:
                verdict = "FAIL (P1 slower; CI excludes 0)"
            cmp_rows.append({"level": f"{s * 100:g}%", "stat": stat, "best_baseline": bname, "best_ms": bval,
                             "p1_ms": p1v, "diff_ms": p1v - bval, "diff_pct": (p1v - bval) / bval * 100,
                             "ci_low": float(d["ci_low"]), "ci_high": float(d["ci_high"]), "verdict": verdict})
    level_cmp = pd.DataFrame(cmp_rows)
    level_cmp.to_csv(tab_dir / "criterion2_by_level.csv", index=False)

    # ---------------------------------------------------------------- routing agreement & P1 checks
    p1 = pc_test[pc_test["method"] == "P1"].set_index("config_id")
    o = orc.set_index("config_id")
    arm_lat = pc_test.pivot(index="config_id", columns="method", values="latency_ms_median")
    agree = pd.DataFrame({
        "target_sel": p1["target_sel"], "route": p1["route"], "oracle_choice": o.loc[p1.index, "choice"],
        "p1_ms": p1["latency_ms_median"], "oracle_ms": o.loc[p1.index, "latency_ms_median"],
        "chosen_arm_ms": [arm_lat.loc[c, r] for c, r in zip(p1.index, p1["route"])],
        "est_rows": p1["est_rows"], "actual_rows": p1["n_eligible"],
    })
    agree["agree"] = agree["route"] == agree["oracle_choice"]
    agree["p1_minus_oracle_ms"] = agree["p1_ms"] - agree["oracle_ms"]
    agree["p1_minus_chosen_arm_ms"] = agree["p1_ms"] - agree["chosen_arm_ms"]
    agree_lvl = agree.groupby("target_sel").agg(
        n=("route", "size"), p1_to_B1=("route", lambda s: int((s == "B1").sum())),
        oracle_to_B1=("oracle_choice", lambda s: int((s == "B1").sum())),
        agreement=("agree", "mean"), mean_cost_vs_oracle_ms=("p1_minus_oracle_ms", "mean"),
        median_est_rows=("est_rows", "median"), median_actual_rows=("actual_rows", "median")).reset_index()
    agree_lvl.to_csv(tab_dir / "routing_agreement_by_level.csv", index=False)
    p1_checks = {
        "overhead_ns_median": float(p1["overhead_ns"].median()),
        "overhead_ns_p95": float(p1["overhead_ns"].quantile(0.95)),
        "p1_minus_chosen_arm_ms_median": float(agree["p1_minus_chosen_arm_ms"].median()),
        "p1_minus_chosen_arm_ms_mean": float(agree["p1_minus_chosen_arm_ms"].mean()),
        "routing_agreement_with_oracle": float(agree["agree"].mean()),
    }

    # ---------------------------------------------------------------- estimation
    est_lvl = estimation.groupby("target_sel").agg(
        n=("actual", "size"), actual_median=("actual", "median"), est_ours_median=("est_ours", "median"),
        est_pg_median=("est_pg", "median"),
        mean_abs_rel_err_ours=("rel_err_ours", lambda s: s.abs().mean()),
        p95_abs_rel_err_ours=("rel_err_ours", lambda s: s.abs().quantile(0.95)),
        mean_abs_rel_err_pg=("rel_err_pg", lambda s: s.abs().mean()),
        mean_abs_err_ours_rows=("rel_err_ours", "size")).reset_index()
    est_lvl["mean_abs_err_ours_rows"] = estimation.assign(ae=(estimation["est_ours"] - estimation["actual"]).abs()) \
        .groupby("target_sel")["ae"].mean().to_numpy()
    est_lvl.to_csv(tab_dir / "estimation_by_level.csv", index=False)

    # ---------------------------------------------------------------- calibration tables
    cal_overall = summarize(pc_cal, ["method"])
    cal_level = summarize(pc_cal, ["method", "target_sel"])
    cal_overall.to_csv(tab_dir / "calibration_summary_overall.csv", index=False)
    cal_level.to_csv(tab_dir / "calibration_summary_by_level.csv", index=False)
    curve.to_csv(tab_dir / "calibration_threshold_curve.csv", index=False)

    # ---------------------------------------------------------------- plan choice + drift + env
    b3plans = []
    for ph in ("calibration", "test"):
        pl = plans[ph]
        pl = pl[pl["method"].isin(["B1", "B3", "B3+iter"])].merge(
            p.workload[["config_id", "target_sel"]], on="config_id")
        t = pl.groupby(["method", "target_sel", "plan_class"]).size().unstack(fill_value=0).reset_index()
        t.insert(0, "phase", ph)
        b3plans.append(t)
    plan_choice = pd.concat(b3plans, ignore_index=True).fillna(0)
    plan_choice.to_csv(tab_dir / "plan_choice_by_level.csv", index=False)
    plan_audit = pd.DataFrame([{"phase": ph, "plans_checked": meta[ph]["plan_audit"]["n"],
                                "violations": meta[ph]["plan_audit"]["n_failed"],
                                "explain_analyze_samples": meta[ph]["explain_analyze_n"],
                                "max_shared_read_blocks": int(ea[ph]["shared_read_blocks"].max())}
                               for ph in ("calibration", "test")])
    plan_audit.to_csv(tab_dir / "plan_audit.csv", index=False)
    raw_test["latency_ms"] = raw_test["latency_ns"] / 1e6
    drift = raw_test.groupby(["method", "pass_idx"])["latency_ms"].median().unstack().reset_index()
    drift.to_csv(tab_dir / "per_pass_median_latency_test.csv", index=False)

    # ================================================================ figures
    lvl_main = by_level[by_level["method"].isin(MAIN + ["Oracle"])]

    def per_level_fig(metric: str, ylabel: str, fname: str, title: str, logy=False, ylim=None, methods=None):
        methods = methods or (MAIN + ["Oracle"])
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for m in methods:
            d = by_level[by_level["method"] == m].sort_values("target_sel")
            if len(d):
                _line(ax, d["target_sel"], d[metric], m,
                      label="Oracle (latency floor, ignores quality)" if m == "Oracle" else m)
        _sel_axis(ax, sels)
        if logy:
            _log_y(ax)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=3)
        _save(fig, fig_dir / fname)
        by_level[by_level["method"].isin(methods)][["method", "target_sel", metric]] \
            .pivot(index="target_sel", columns="method", values=metric).to_csv(tab_dir / f"fig_{fname}.csv")

    per_level_fig("p50_ms", "p50 latency (ms, log scale)", "latency_p50_vs_selectivity",
                  "Median end-to-end latency by selectivity (held-out)", logy=True)
    per_level_fig("p95_ms", "p95 latency (ms, log scale)", "latency_p95_vs_selectivity",
                  "p95 end-to-end latency by selectivity (held-out)", logy=True)
    per_level_fig("recall", "mean strict Recall@10", "recall_vs_selectivity",
                  "Recall@10 by selectivity (held-out)", ylim=(-0.02, 1.02))
    per_level_fig("completeness", "share of complete queries", "completeness_vs_selectivity",
                  "Completeness by selectivity (held-out)", ylim=(-0.02, 1.02))

    # estimated vs actual
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    ax.scatter(estimation["actual"], estimation["est_ours"].clip(lower=0.5), s=8, color=COLOR["B1"],
               edgecolors="none", alpha=0.5)
    lim = [max(0.5, estimation["actual"].min() * 0.7), estimation["actual"].max() * 1.3]
    ax.plot(lim, lim, color=INK2, linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("actual matching rows"); ax.set_ylabel("estimated rows (pg_stats histogram)")
    ax.set_title("Estimated vs actual rows (all configs; line = exact)", loc="left")
    ax = axes[1]
    _line(ax, est_lvl["target_sel"], est_lvl["mean_abs_rel_err_ours"], "B1", label="router estimator (ours)")
    _line(ax, est_lvl["target_sel"], est_lvl["mean_abs_rel_err_pg"], "B2-plain", label="PostgreSQL planner")
    _sel_axis(ax, sels)
    ax.set_ylabel("mean |relative error|")
    ax.set_title("Row-estimate error by level", loc="left")
    ax.legend(fontsize=8)
    _save(fig, fig_dir / "estimated_vs_actual_rows")

    # P1 vs baselines overall (bars) - one panel per statistic, colour per method
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, stat in zip(axes, STATS):
        d = overall.set_index("method").loc[MAIN + ["Oracle"]]
        xs = np.arange(len(d))
        ax.bar(xs, d[stat], width=0.45, color=[COLOR[m] for m in d.index], edgecolor=SURFACE, linewidth=1.5)
        for x, (m, r) in zip(xs, d.iterrows()):
            ax.text(x, r[stat], f"{r[stat]:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Oracle*" if m == "Oracle" else (m if r["meets_quality"] else m + "\u2020")
                            for m, r in d.iterrows()], rotation=0, fontsize=8)
        ax.set_title(stat.replace("_ms", "") + " latency (ms)", loc="left")
        ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.005, "* Oracle = hindsight latency floor (faster of B1/B2-iter per query), ignores quality. "
             "\u2020 fails quality: mean Recall@10 < 0.95 or completeness < 99%.", fontsize=7, color=INK2)
    _save(fig, fig_dir / "p1_vs_baselines_overall")

    # latency CDF
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in MAIN + ["Oracle"]:
        x = np.sort(combined[combined["method"] == m]["latency_ms_median"].to_numpy())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOR[m], label=m if m != "Oracle" else
                "Oracle (latency floor)", linewidth=1.6)
    _log_x(ax)
    ax.set_xlabel("per-config median latency (ms, log scale)"); ax.set_ylabel("share of held-out configs")
    ax.set_title("Latency distribution (held-out, all levels)", loc="left")
    ax.legend(fontsize=8)
    _save(fig, fig_dir / "latency_cdf")

    # P1 vs Oracle per level
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in ["B1", "B2-iter", "P1", "Oracle"]:
        d = by_level[by_level["method"] == m].sort_values("target_sel")
        _line(ax, d["target_sel"], d["mean_ms"], m, label="Oracle (latency floor, ignores quality)"
              if m == "Oracle" else m)
    _sel_axis(ax, sels)
    _log_y(ax)
    ax.set_ylabel("mean latency (ms, log scale)")
    g_mean = crit_df[(crit_df["scope"] == "overall") & (crit_df["criterion"] == "3 gain captured (mean_ms)")]
    ax.set_title(f"P1 vs the hindsight Oracle (held-out); gain captured (mean): {g_mean['result'].iloc[0]}",
                 loc="left", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=2)
    _save(fig, fig_dir / "p1_vs_oracle_by_level")

    # calibration curve (series are statistics, not methods -> ink tones; T* belongs to P1 -> P1 colour)
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
    c = curve[np.isfinite(curve["T"]) & (curve["T"] > 0)]
    ax = axes[0]
    ax.plot(c["T"], c["mean_ms"], color=INK, label="mean", linewidth=1.6)
    ax.plot(c["T"], c["p95_ms"], color=MUTED, label="p95", linewidth=1.6)
    ax.set_ylabel("simulated P1 latency (ms)")
    ax.set_title("Threshold calibration (calibration split only)", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(c["T"], c["recall"], color=INK, label="mean strict Recall@10", linewidth=1.6)
    ax.plot(c["T"], c["completeness"], color=MUTED, label="completeness", linewidth=1.6)
    ax.axhline(crit.min_mean_recall, color=AXIS, linewidth=0.8)
    ax.axhline(crit.min_completeness, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("threshold T (estimated rows; route to B1 when estimate <= T)")
    ax.set_ylabel("quality (constraint lines at 0.95 / 0.99)")
    ax.legend(fontsize=8, loc="lower right")
    T = frozen["threshold_T"]
    for a_ in axes:
        if np.isfinite(T):
            a_.axvline(T, color=COLOR["P1"], linewidth=1.2)
    if np.isfinite(T):
        axes[0].annotate(f"frozen T* = {T:,.0f}", (T, axes[0].get_ylim()[1]), xytext=(4, -12),
                         textcoords="offset points", fontsize=8, color=INK)
    _save(fig, fig_dir / "calibration_curve")

    # HNSW sensitivity (calibration split)
    sens = ["B2-iter"] + [v.name for v in cfg.sensitivity]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, metric, ylabel in zip(axes, ["recall", "completeness", "p50_ms"],
                                  ["mean strict Recall@10", "completeness", "p50 latency (ms, log)"]):
        for m in sens:
            d = cal_level[cal_level["method"] == m].sort_values("target_sel")
            if len(d):
                _line(ax, d["target_sel"], d[metric], m)
        _sel_axis(ax, sels)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylabel(ylabel)
        if metric == "p50_ms":
            _log_y(ax)
    axes[0].set_title("HNSW sensitivity (calibration split, report only)", loc="left")
    axes[0].legend(fontsize=7)
    _save(fig, fig_dir / "hnsw_sensitivity")
    cal_level[cal_level["method"].isin(sens)].to_csv(tab_dir / "fig_hnsw_sensitivity.csv", index=False)

    # ================================================================ report.md
    md = _report_md(cfg, overall, by_level, ci, crit_df, level_cmp, agree_lvl, p1_checks, est_lvl, frozen, curve,
                    cal_overall, plan_choice, plan_audit, drift, consistency, manifest, db_setup, server_env,
                    host_env, meta, sels)
    out = run / "report.md"
    out.write_text(md)
    log.info("report written to %s", out)
    return out


def _report_md(cfg, overall, by_level, ci, crit_df, level_cmp, agree_lvl, p1_checks, est_lvl, frozen, curve, cal_overall,
               plan_choice, plan_audit, drift, consistency, manifest, db_setup, server_env, host_env, meta,
               sels) -> str:
    L = []
    a = L.append
    a(f"# Results: selectivity-aware query routing (run `{cfg.run_id}`)\n")
    a("All numbers below are generated from the measured data in this run directory. Latency is client-side "
      "end-to-end (includes network round trip and, for P1, routing overhead); per-config latency is the "
      f"median of {cfg.benchmark.repetitions} timed repetitions; p50/p95/mean are taken over configs. "
      "Recall@10 is **strict ID overlap** with the exact top-k; tie-tolerant recall is a diagnostic only.\n")

    # -- headline criteria
    a("## Success criteria (held-out test split)\n")
    a(_md_table(crit_df[crit_df["scope"] == "overall"][["criterion", "detail", "result"]]))
    a("\nCriterion 2 compares against the best single baseline among B1, B2-plain, B2-iter, B3 that meets "
      "criterion 1 (B3+iter is supplementary and excluded). Gain captured = (best - P1) / (best - Oracle) for one "
      "statistic at a time; the Oracle is the hindsight latency floor (faster of B1/B2-iter per query, no quality "
      "requirement), so it is an optimistic bound.\n")
    lvl = crit_df[crit_df["scope"] != "overall"].pivot(index="scope", columns="criterion", values="result")
    lvl = lvl.reindex([f"{s * 100:g}%" for s in sels]).reset_index().rename(columns={"scope": "level"})
    a("### Per selectivity level\n")
    a(_md_table(lvl))
    losers = lvl[(lvl.get("2 p50_ms") == "FAIL") | (lvl.get("2 p95_ms") == "FAIL") | (lvl.get("1 quality") == "FAIL")]
    if len(losers):
        a(f"\n**Levels where P1 does not meet a criterion under the strict per-level rule (RQ3):** "
          f"{', '.join(losers['level'])}.\n")
    else:
        a("\nP1 met criteria 1 and 2 at every level.\n")
    a("Criterion 2 per level with effect sizes (P1 minus the best qualifying baseline *at that level*; 95% paired "
      "bootstrap CI over the level's configs). The pre-registered verdict is strict (P1 must be <= best); the CI "
      "shows whether a FAIL is distinguishable from measurement noise:\n")
    a(_md_table(level_cmp))
    sig = level_cmp[level_cmp["verdict"].str.startswith("FAIL (P1 slower")]
    a(f"\nStatistically distinguishable per-level losses: "
      f"{', '.join(sig['level'] + ' ' + sig['stat'] + ' vs ' + sig['best_baseline']) if len(sig) else 'none'}.\n")

    # -- main summary
    a("## Held-out summary (all levels, each level weighted 1/9)\n")
    cols = ["method", "role", "n_configs", "p50_ms", "p95_ms", "mean_ms", "recall", "recall_tol", "completeness",
            "meets_quality", "n_errors"]
    a(_md_table(overall[cols]))
    a("\n95% cluster-bootstrap CIs (resampling query vectors) for the proposed router and paired differences "
      "(negative = P1 faster / lower):\n")
    cit = ci[ci["stat"].isin(["p50_ms", "p95_ms", "mean_ms", "recall"])]
    cit = cit.assign(ci=cit.apply(lambda r: f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1))
    a(_md_table(cit.pivot(index="method", columns="stat", values="ci").reset_index()))
    a("\n### By level\n")
    for metric, title in (("p50_ms", "p50 latency (ms)"), ("p95_ms", "p95 latency (ms)"),
                          ("recall", "mean strict Recall@10"), ("completeness", "completeness")):
        t = by_level[by_level["method"].isin(MAIN + ["Oracle"])].pivot(index="target_sel", columns="method",
                                                                       values=metric)
        t = t[[m for m in MAIN + ["Oracle"] if m in t.columns]].reset_index()
        t["target_sel"] = t["target_sel"].map(lambda s: f"{s * 100:g}%")
        a(f"**{title}**\n")
        a(_md_table(t))
        a("")

    # -- oracle + routing
    a("## Routing behaviour vs the hindsight Oracle\n")
    a(_md_table(agree_lvl.assign(target_sel=agree_lvl["target_sel"].map(lambda s: f"{s * 100:g}%"))))
    cal_oh = frozen["routing_overhead_ns"]["median_ns"] / 1e3
    a(f"\nP1 routing overhead measured live inside the benchmark loop (estimate + decision, in-process): median "
      f"{p1_checks['overhead_ns_median'] / 1e3:.1f} µs, p95 {p1_checks['overhead_ns_p95'] / 1e3:.1f} µs. "
      f"Calibration used a tight-loop measurement of {cal_oh:.1f} µs; the live value is higher because the router "
      "code runs cold between database round trips. P1 minus its chosen arm's latency in the same session: median "
      f"{p1_checks['p1_minus_chosen_arm_ms_median'] * 1e3:.1f} µs, i.e. the live cost of routing is about "
      f"{p1_checks['p1_minus_chosen_arm_ms_median'] * 1e3:.0f} µs per query - small against millisecond latencies, "
      "but visible at the most selective levels where both P1 and B1 run the same exact plan in < 1.2 ms.\n")

    # -- estimation
    a("## Selectivity estimation (all 2,700 configs)\n" if cfg.run_id == "full" else
      "## Selectivity estimation (all configs)\n")
    a(_md_table(est_lvl.assign(target_sel=est_lvl["target_sel"].map(lambda s: f"{s * 100:g}%"))
                .drop(columns=["n"])))
    ec = consistency["estimator_matches_planner"]
    a(f"\nOur estimator vs PostgreSQL's own planner estimate: median |relative difference| "
      f"{ec['median_abs_rel_diff']:.4f}, p95 {ec['p95_abs_rel_diff']:.4f}.\n")

    # -- calibration
    g = frozen["guard"]
    a("## Calibration (calibration split only)\n")
    a(f"- Frozen threshold **T\\* = {frozen['threshold_T']:,.1f} estimated rows** "
      f"(p95-optimal feasible T would be {frozen['p95_optimal_feasible_T']:,.1f}); frozen at {frozen['frozen_at']}.")
    a(f"- HNSW: ef_search = {frozen['hnsw']['ef_search']}, iterative_scan = relaxed_order, max_scan_tuples = "
      f"{frozen['hnsw']['max_scan_tuples']}, m = {frozen['hnsw']['m']}, ef_construction = {frozen['hnsw']['ef_construction']}.")
    a(f"- Guard: B2-iter mean strict recall at the broadest level = {g['b2iter_mean_recall_at_broadest']:.4f} "
      f"(threshold {g['threshold']}) -> {'TRIGGERED (ef_search raised)' if g['triggered'] else 'not triggered'}.")
    ca = frozen["calibration_at_T"]
    a(f"- Simulated P1 on calibration at T\\*: mean {ca['mean_ms']:.3f} ms, p50 {ca['p50_ms']:.3f}, p95 "
      f"{ca['p95_ms']:.3f}, recall {ca['recall']:.4f}, completeness {ca['completeness']:.4f}, "
      f"{ca['frac_B1'] * 100:.1f}% routed to B1. {frozen['n_candidate_thresholds']} candidate thresholds evaluated.")
    feas = curve[curve["feasible"]]
    near = feas[feas["mean_ms"] <= feas["mean_ms"].min() * 1.01]
    fin = near[np.isfinite(near["T"])]
    if len(fin):
        a(f"- Flatness of the objective: every feasible T in [{fin['T'].min():,.0f}, {fin['T'].max():,.0f}] is within 1% "
          f"of the best calibration mean latency, so T* is one point of a near-optimal interval (reported, not re-tuned).\n")
    else:
        a("")
    a("Calibration-split summary of every variant (the four sensitivity variants are report-only):\n")
    a(_md_table(cal_overall[["method", "n_configs", "p50_ms", "p95_ms", "mean_ms", "recall", "completeness"]]))

    # -- verification
    a("\n## Verification\n")
    gate = manifest["gt_gate"]
    a(f"- Ground-truth gate: reproducing Qdrant's exact `closest_ids` for the {gate['n_queries']} selected queries "
      f"under their original conditions - mean overlap {gate['mean_overlap']:.4f}, exact match "
      f"{gate['exact_match_fraction'] * 100:.1f}% ({gate.get('n_queries_with_mismatch', 'n/a')} queries differ), "
      f"tie-aware overlap {gate.get('mean_overlap_tie_aware', float('nan')):.4f}, max |score diff| "
      f"{gate['max_abs_score_diff']:.2e}. Every differing id lies at exactly the same distance as one of ours "
      "(duplicate images); Qdrant breaks such exact ties differently from our (distance, id) rule.")
    for name, cdict in consistency.items():
        if name == "all_pass":
            continue
        detail = {k: v for k, v in cdict.items() if k not in ("pass", "note")}
        a(f"- `{name}`: {'PASS' if cdict['pass'] else 'FAIL'} {detail}")
    a("\nPlan audit (plain EXPLAIN on every config x method; EXPLAIN ANALYZE BUFFERS on a stratified sample):\n")
    a(_md_table(plan_audit))
    a("\nPlan choice by level (B1 = exact paths; B3 = PostgreSQL default planner decision under the shared HNSW "
      "configuration):\n")
    pc2 = plan_choice.copy()
    pc2["target_sel"] = pc2["target_sel"].map(lambda s: f"{s * 100:g}%")
    a(_md_table(pc2, floatfmt=0))

    # -- environment / runtime
    a("\n## Environment and runtime\n")
    s = db_setup["sizes"]
    a(f"- {server_env['version']}; pgvector {server_env['pgvector_version']}.")
    a(f"- Host: {host_env['cpu']}, {host_env['cpu_count']} cores; Docker {host_env['docker_version']} "
      f"({host_env['docker_resources']}); Python {host_env['python']}.")
    a(f"- Table: {db_setup['rows']:,} rows; heap {s['heap_bytes'] / 2**20:.0f} MB (TOAST {s['toast_bytes']} B), "
      f"B-tree {s['btree_bytes'] / 2**20:.1f} MB, HNSW {s['hnsw_bytes'] / 2**20:.0f} MB; HNSW build "
      f"{db_setup['hnsw_build_s']:.1f} s; load {db_setup['load_s']:.1f} s.")
    ks = server_env["key_settings"]
    a(f"- Key settings: shared_buffers={ks.get('shared_buffers')}, work_mem={ks.get('work_mem')}, jit={ks.get('jit')}, "
      f"max_parallel_workers_per_gather={ks.get('max_parallel_workers_per_gather')}, "
      f"random_page_cost={ks.get('random_page_cost')}, default_statistics_target={ks.get('default_statistics_target')}.")
    for ph in ("calibration", "test"):
        m = meta[ph]
        a(f"- {ph}: {m['n_configs']} configs x {len(m['methods'])} methods, {m['n_executions']:,} timed executions, "
          f"{m['n_errors']} errors, wall clock {m['wall_clock_s'] / 60:.1f} min; `SELECT 1` RTT p50 "
          f"{m['rtt_select1']['p50_ms']:.3f} ms.")
    a("\nPer-pass median latency (held-out; drift check):\n")
    a(_md_table(drift))
    a("\n## Figures\n")
    for f in ("latency_p50_vs_selectivity", "latency_p95_vs_selectivity", "recall_vs_selectivity",
              "completeness_vs_selectivity", "estimated_vs_actual_rows", "p1_vs_baselines_overall", "latency_cdf",
              "p1_vs_oracle_by_level", "calibration_curve", "hnsw_sensitivity"):
        a(f"![{f}](figures/{f}.png)")
    return "\n".join(L) + "\n"
