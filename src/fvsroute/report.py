"""Write the report: summary tables, success criteria, estimation error and figures."""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .benchmark import gain_captured, oracle_floor, summarize
from .search import BASELINES

METHODS = [*BASELINES, "P1", "Oracle"]
METRICS = ("p50_ms", "p95_ms", "mean_ms", "recall", "completeness")


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def success_criteria(summary, cfg, scope):
    router = summary[summary["method"] == "P1"].iloc[0]
    oracle = summary[summary["method"] == "Oracle"].iloc[0]
    baselines = summary[summary["method"].isin(BASELINES) & summary["meets_quality"]]
    rows = [
        {
            "scope": scope,
            "criterion": "quality",
            "result": "PASS" if router["meets_quality"] else "FAIL",
            "detail": f"Recall@10={router['recall']:.4f}; completeness={router['completeness']:.4f}",
        }
    ]
    for metric in ("p50_ms", "p95_ms", "mean_ms"):
        if baselines.empty:
            best_name, best_value = None, None
        else:
            best = baselines.sort_values(metric).iloc[0]
            best_name, best_value = best["method"], float(best[metric])
        if metric != "mean_ms":
            result = "N/A" if best_value is None else "PASS" if router[metric] <= best_value else "FAIL"
            rows.append(
                {
                    "scope": scope,
                    "criterion": metric,
                    "result": result,
                    "detail": "no qualifying baseline"
                    if best_value is None
                    else f"P1 {router[metric]:.3f} ms vs {best_name} {best_value:.3f} ms",
                }
            )
        gain, reason = gain_captured(best_value, float(router[metric]), float(oracle[metric]))
        rows.append(
            {
                "scope": scope,
                "criterion": f"oracle gain ({metric})",
                "result": f"{100 * gain:.1f}%" if gain is not None else "N/A",
                "detail": reason,
            }
        )
    return rows


def plot_calibration(curve, estimation, frozen, figures):
    """Draw the paper's two-panel calibration figure using this run's measurements."""
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    threshold = frozen["threshold_T"]
    visible = curve[np.isfinite(curve["T"]) & (curve["T"] > 0)]
    feasible = curve[curve["feasible"]]
    near = feasible[
        (feasible["mean_ms"] <= feasible["mean_ms"].min() * 1.01) & np.isfinite(feasible["T"])
    ]
    level_estimates = (
        estimation[estimation["split"] == "calibration"].groupby("target_sel")["est_ours"].median()
    )
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.6,
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "xtick.labelcolor": "#0b0b0b",
        "ytick.labelcolor": "#0b0b0b",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True)
        latency, quality = axes
        for ax in axes:
            if not near.empty:
                ax.axvspan(near["T"].min(), near["T"].max(), color="#e1e0d9", alpha=0.7, linewidth=0)
            if 0 < threshold < np.inf:
                ax.axvline(threshold, color="#e87ba4", linewidth=1.0)
        latency.plot(visible["T"], visible["mean_ms"], color="#0b0b0b", label="mean")
        latency.plot(visible["T"], visible["p95_ms"], color="#898781", label="p95")
        latency.set_ylabel("simulated P1 latency (ms)")
        latency.legend(loc="center left", bbox_to_anchor=(0.01, 0.55))
        # Scale to the measured host so higher latencies remain visible.
        upper = max(1.0, visible[["mean_ms", "p95_ms"]].max().max() * 1.13)
        latency.set_ylim(0, upper)
        for level, estimate in level_estimates.items():
            if level != 0.5:  # The 50% and 90% labels would crowd the right edge.
                latency.text(estimate, upper * 0.95, f"{100 * level:g}%", ha="center", va="bottom",
                             fontsize=5.5, color="#52514e")
        if 0 < threshold < np.inf:
            latency.annotate(f"$T^*$ = {threshold:,.0f}", (threshold, upper * 0.52), xytext=(3, 0),
                             textcoords="offset points", fontsize=6, color="#0b0b0b")
        else:
            label = r"$T^*=\infty$ (always B1)" if np.isinf(threshold) else "$T^*=0$ (always B2-iter)"
            latency.text(0.97, 0.52, label, transform=latency.transAxes,
                         ha="right", fontsize=6)
        quality.plot(visible["T"], visible["recall"], color="#0b0b0b", label="mean strict Recall@10")
        quality.plot(visible["T"], visible["completeness"], color="#898781", label="completeness")
        quality.axhline(frozen["constraints"]["min_mean_recall"], color="#c3c2b7", linewidth=0.6)
        quality.set_ylim(0.945, 1.003)
        quality.set_ylabel("quality")
        quality.set_xscale("log")
        quality.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        quality.set_xlabel(r"threshold $T$ (estimated rows; route to B1 if $\hat{M}\leq T$)")
        quality.legend(loc="lower right")
        fig.tight_layout(h_pad=0.4)
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"calibration_curve.{extension}", dpi=300,
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def make_report(cfg):
    source = cfg.run_dir
    test = pd.read_parquet(source / "test/per_config.parquet")
    if set(test["method"]) != set([*BASELINES, "P1"]):
        raise ValueError("report requires the four baselines and P1")
    combined = pd.concat([test, oracle_floor(test)], ignore_index=True)
    overall = summarize(combined, ["method"])
    by_level = summarize(combined, ["method", "target_sel"])
    for summary in (overall, by_level):
        summary["meets_quality"] = (summary["recall"] >= cfg.calibration.min_mean_recall) & (
            summary["completeness"] >= cfg.calibration.min_completeness
        )
    overall = overall.set_index("method").loc[METHODS].reset_index()
    criteria = success_criteria(overall, cfg, "overall")
    for level, rows in by_level.groupby("target_sel"):
        criteria.extend(success_criteria(rows, cfg, f"{100 * level:g}%"))
    criteria = pd.DataFrame(criteria)
    estimation = pd.read_parquet(source / "verify/estimation.parquet")
    estimation = estimation.assign(abs_error=(estimation["est_ours"] - estimation["actual"]).abs())
    errors = (
        estimation.groupby("target_sel")
        .agg(
            actual_median=("actual", "median"),
            estimate_median=("est_ours", "median"),
            mean_absolute_error=("abs_error", "mean"),
            mean_absolute_relative_error=("rel_err_ours", lambda values: values.abs().mean()),
        )
        .reset_index()
    )
    tables, figures = cfg.run_dir / "tables", cfg.run_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    columns = ["method", "n_configs", *METRICS, "meets_quality", "n_errors"]
    overall[columns].to_csv(tables / "summary_test_overall.csv", index=False)
    by_level[["method", "target_sel", "n_configs", *METRICS, "meets_quality", "n_errors"]].to_csv(
        tables / "summary_test_by_level.csv", index=False
    )
    criteria.to_csv(tables / "success_criteria.csv", index=False)
    errors.to_csv(tables / "estimation_by_level.csv", index=False)
    names = {
        "p50_ms": "latency_p50",
        "p95_ms": "latency_p95",
        "recall": "recall",
        "completeness": "completeness",
    }
    labels = {
        "p50_ms": "p50 latency (ms)",
        "p95_ms": "p95 latency (ms)",
        "recall": "Recall@10",
        "completeness": "Share of complete queries",
    }
    levels = sorted(by_level["target_sel"].unique())
    for metric, name in names.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in METHODS:
            values = by_level[by_level["method"] == method].sort_values("target_sel")
            ax.plot(values["target_sel"], values[metric], marker="o", label=method)
        ax.set_xscale("log")
        ax.set_xticks(levels, [f"{100 * level:g}%" for level in levels])
        if metric.endswith("_ms"):
            ax.set_yscale("log")
        ax.set(xlabel="Filter selectivity", ylabel=labels[metric])
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{name}_vs_selectivity.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(estimation["actual"], estimation["est_ours"], s=5, alpha=0.4)
    bounds = [max(1, estimation["actual"].min()), estimation["actual"].max()]
    ax.plot(bounds, bounds, color="black")
    ax.set(xscale="log", yscale="log", xlabel="Actual matching rows", ylabel="Estimated matching rows")
    fig.tight_layout()
    fig.savefig(figures / "estimated_vs_actual_rows.png", dpi=160)
    plt.close(fig)
    frozen = json.loads((source / "calibration/frozen_router.json").read_text())
    plot_calibration(
        pd.read_parquet(source / "calibration/threshold_curve.parquet"), estimation, frozen, figures
    )
    lines = [
        f"# Experiment report: {cfg.run_id}",
        f"Frozen threshold: **{frozen['threshold_T']:,.1f} estimated rows**.",
        "Latency includes execute/fetch and P1 routing overhead. Each query configuration uses the median of its timed repetitions. "
        "Oracle chooses the faster measured B1/B2-iter result without a quality constraint.",
        "## Held-out results",
        markdown_table(overall[columns]),
        "## Success criteria",
        markdown_table(criteria),
        "## Results per selectivity",
        markdown_table(by_level[["method", "target_sel", *METRICS]]),
        "## Row-estimate error",
        markdown_table(errors),
        "FAIL entries are reported even when the overall result passes.",
        "## Figures",
    ]
    for name in [f"{name}_vs_selectivity" for name in names.values()] + [
        "estimated_vs_actual_rows", "calibration_curve"
    ]:
        lines.append(f"![{name}](figures/{name}.png)")
        if name == "calibration_curve":
            lines.append(
                "Calibration split: mean and p95 simulated latency (top), strict Recall@10 and completeness "
                "(bottom). The pink line marks the frozen threshold; the shaded band contains feasible "
                "thresholds within 1% of the minimum mean latency."
            )
    path = cfg.run_dir / "report.md"
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return path
