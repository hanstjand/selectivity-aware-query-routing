"""Offline calibration of the routing threshold T on the calibration split only.

1. Pre-registered guard (decided before T): if B2-iter (primary ef_search) has mean strict Recall@10 below
   `guard_min_recall` at the broadest selectivity level, ef_search is raised to `guard_ef_search` for every
   method; the calibration-only sensitivity variant with that ef_search supplies the B2-iter measurements.
2. For every candidate T, simulate P1 per config (B1 if est_rows <= T else B2-iter) using the measured
   per-config medians plus the measured median routing overhead.
3. Choose T minimising MEAN latency subject to mean strict Recall@10 >= min_mean_recall and
   completeness >= min_completeness; ties -> lower p95, then larger T. T = inf (always B1) is always feasible.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from .config import Config
from .workload import frame_sha256


class LeakageError(RuntimeError):
    pass


def candidate_thresholds(estimates: np.ndarray) -> list[float]:
    e = np.unique(np.asarray(estimates, dtype=float))
    cands = [0.0]
    for a, b in zip(e[:-1], e[1:]):
        cands.append(math.sqrt(a * b) if a > 0 else (a + b) / 2.0)
    cands.append(math.inf)
    return cands


def simulate(arms: pd.DataFrame, T: float, overhead_ms: float) -> dict:
    """`arms` columns: est_rows, lat_B1, lat_H, rec_B1, rec_H, com_B1, com_H (one row per config)."""
    to_b1 = arms["est_rows"].to_numpy() <= T
    lat = np.where(to_b1, arms["lat_B1"], arms["lat_H"]) + overhead_ms
    rec = np.where(to_b1, arms["rec_B1"], arms["rec_H"])
    com = np.where(to_b1, arms["com_B1"], arms["com_H"])
    return {"T": T, "mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)), "recall": float(np.nanmean(rec)),
            "completeness": float(np.mean(com)), "frac_B1": float(to_b1.mean())}


def _guard_variant(cfg: Config) -> str:
    for v in cfg.sensitivity:
        if (v.ef_search == cfg.hnsw.guard_ef_search and v.iterative_scan in (None, "relaxed_order")
                and v.max_scan_tuples is None):
            return v.name
    raise ValueError("guard requires a sensitivity variant with ef_search == guard_ef_search")


def calibrate(cfg: Config, pc_cal: pd.DataFrame, est_rows: pd.Series, overhead_ns: dict,
              stats_sha: str) -> tuple[dict, pd.DataFrame]:
    if (pc_cal["split"] != "calibration").any():
        raise LeakageError("calibration received rows from the held-out split")

    # ---- 1. guard ----------------------------------------------------------------------------
    broad = pc_cal["level_idx"].max()
    b2 = pc_cal[(pc_cal["method"] == "B2-iter") & (pc_cal["level_idx"] == broad)]
    guard_recall = float(np.nanmean(b2["recall"]))
    triggered = guard_recall < cfg.hnsw.guard_min_recall
    hnsw_source = _guard_variant(cfg) if triggered else "B2-iter"
    ef = cfg.hnsw.guard_ef_search if triggered else cfg.hnsw.ef_search

    # ---- 2. per-config arm table ------------------------------------------------------------------
    def arm(name, col):
        return pc_cal[pc_cal["method"] == name].set_index("config_id")[col]
    arms = pd.DataFrame({
        "lat_B1": arm("B1", "latency_ms_median"), "rec_B1": arm("B1", "recall"), "com_B1": arm("B1", "complete"),
        "lat_H": arm(hnsw_source, "latency_ms_median"), "rec_H": arm(hnsw_source, "recall"),
        "com_H": arm(hnsw_source, "complete"),
    })
    arms["est_rows"] = est_rows.reindex(arms.index)
    if arms.isna().drop(columns=["rec_B1", "rec_H"]).any().any():
        raise ValueError("missing per-config measurements for calibration")
    overhead_ms = overhead_ns["median_ns"] / 1e6

    # ---- 3. threshold sweep ----------------------------------------------------------------------
    c = cfg.calibration
    curve = pd.DataFrame([simulate(arms, T, overhead_ms) for T in candidate_thresholds(arms["est_rows"])])
    curve["feasible"] = (curve["recall"] >= c.min_mean_recall) & (curve["completeness"] >= c.min_completeness)
    feas = curve[curve["feasible"]].copy()
    feas["neg_T"] = -feas["T"]
    best = feas.sort_values(["mean_ms", "p95_ms", "neg_T"]).iloc[0]
    T_star = float(best["T"])
    # report the p95-optimal feasible T too (it need not coincide with the mean-optimal one)
    best_p95 = feas.sort_values(["p95_ms", "mean_ms", "neg_T"]).iloc[0]

    by_level = []
    tmp = arms.join(pc_cal.drop_duplicates("config_id").set_index("config_id")[["level_idx", "target_sel"]])
    for lvl, g in tmp.groupby("level_idx"):
        by_level.append({"level_idx": int(lvl), "target_sel": float(g["target_sel"].iloc[0]),
                         **simulate(g, T_star, overhead_ms)})

    sha_cols = ["config_id", "method", "latency_ms_median", "recall", "complete", "split"]
    frozen = {
        "threshold_T": T_star,
        "threshold_is_infinite": math.isinf(T_star),
        "objective": "mean latency subject to mean strict Recall@10 and completeness constraints",
        "constraints": {"min_mean_recall": c.min_mean_recall, "min_completeness": c.min_completeness},
        "calibration_at_T": {k: best[k] for k in ("mean_ms", "p50_ms", "p95_ms", "recall", "completeness",
                                                    "frac_B1")},
        "p95_optimal_feasible_T": float(best_p95["T"]),
        "calibration_by_level_at_T": by_level,
        "hnsw": {"ef_search": ef, "iterative_scan": "relaxed_order",
                 "max_scan_tuples": cfg.hnsw.max_scan_tuples, "scan_mem_multiplier": cfg.hnsw.scan_mem_multiplier,
                 "m": cfg.hnsw.m, "ef_construction": cfg.hnsw.ef_construction},
        "guard": {"broadest_level_idx": int(broad), "b2iter_mean_recall_at_broadest": guard_recall,
                  "threshold": cfg.hnsw.guard_min_recall, "triggered": bool(triggered),
                  "hnsw_measurements_from": hnsw_source},
        "routing_overhead_ns": overhead_ns,
        "n_calibration_configs": int(len(arms)),
        "n_candidate_thresholds": int(len(curve)),
        "stats_snapshot_sha256": stats_sha,
        "calibration_input_sha256": frame_sha256(
            pc_cal.sort_values(["config_id", "method"]).reset_index(drop=True), sha_cols),
        "config_sha256": cfg.source_sha256,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return frozen, curve
