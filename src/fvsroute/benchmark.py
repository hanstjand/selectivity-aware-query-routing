"""Time the required methods, calibrate T, and evaluate held-out queries."""

import gc
import json
import logging
import math
import time

import numpy as np
import pandas as pd
import psycopg
from pgvector import Vector

from . import data, prepare, search
from .config import write_json

log = logging.getLogger(__name__)


def recall_strict(ids, gt_ids, n_eligible: int, k: int) -> float:
    denom = min(k, n_eligible)
    if denom == 0:
        return float("nan")
    return len(set(list(ids)[:k]) & set(gt_ids)) / denom


def is_complete(n_returned: int, n_eligible: int, k: int) -> bool:
    return n_returned == min(k, n_eligible)


def latency_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "mean_ms": float(x.mean()),
    }


def summarize(pc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pc.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        lat = latency_stats(g["latency_ms_median"].to_numpy())
        rows.append(
            {
                **dict(zip(by, key)),
                "n_configs": len(g),
                **lat,
                "recall": float(np.nanmean(g["recall"])) if g["recall"].notna().any() else float("nan"),
                "recall_tol": float(np.nanmean(g["recall_tol"]))
                if g["recall_tol"].notna().any()
                else float("nan"),
                "completeness": float(g["complete"].mean()),
                "n_errors": int(g["n_errors"].sum()),
            }
        )
    return pd.DataFrame(rows)


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
    return {
        "T": T,
        "mean_ms": float(lat.mean()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "recall": float(np.nanmean(rec)),
        "completeness": float(np.mean(com)),
        "frac_B1": float(to_b1.mean()),
    }


TIE_EPS = 1e-6
ARMS = ("B1", "B2-iter")


def recall_tolerant(
    in_filter: np.ndarray,
    dists: np.ndarray,
    kth_dist: float,
    n_eligible: int,
    k: int,
    eps: float = TIE_EPS,
) -> float:
    denom = min(k, n_eligible)
    if denom == 0:
        return float("nan")
    hits = int(np.sum(in_filter[:k] & (dists[:k] <= kth_dist + eps)))
    return min(hits, denom) / denom


def evaluate_executions(
    raw: pd.DataFrame,
    workload: pd.DataFrame,
    gt: pd.DataFrame,
    values: np.ndarray,
    base_normed: np.ndarray,
    queries_normed: dict[int, np.ndarray],
    k: int,
) -> pd.DataFrame:
    """Add recall / recall_tol / complete / filter_ok to every execution row.

    Distinct (config, result) pairs are evaluated once (results are deterministic across repetitions).
    """
    filters = workload.set_index("config_id")
    truth = gt.set_index("config_id")
    cache: dict[tuple, tuple] = {}
    scores = []
    for cid, ids, err in zip(raw["config_id"], raw["ids"], raw["error"]):
        ids_t = tuple(int(i) for i in (ids if ids is not None else ()))
        failed = not pd.isna(err)
        key = (cid, ids_t, not failed)
        if key not in cache:
            window = filters.loc[cid]
            n_eligible = int(window["n_eligible"])
            arr = np.asarray(ids_t, dtype=np.int64)
            in_filter = (values[arr] >= window["lo"]) & (values[arr] <= window["hi"])
            distances = (
                1.0 - base_normed[arr] @ queries_normed[int(window["query_id"])] if arr.size else np.empty(0)
            )
            if failed and n_eligible > 0:
                recall = tolerant = 0.0
            else:
                recall = recall_strict(ids_t, truth.loc[cid, "gt_ids"], n_eligible, k)
                tolerant = recall_tolerant(
                    in_filter,
                    distances,
                    float(truth.loc[cid, "gt_kth_dist"]),
                    n_eligible,
                    k,
                )
            cache[key] = (
                recall,
                tolerant,
                bool(not failed and is_complete(arr.size, n_eligible, k)),
                bool(in_filter.all()),
            )
        scores.append(cache[key])
    out = raw.copy()
    for index, column in enumerate(("recall", "recall_tol", "complete", "filter_ok")):
        out[column] = [score[index] for score in scores]
    return out


def per_config(evaluated: pd.DataFrame, workload: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, method): median latency over timed reps and quality metrics."""
    ev = evaluated.copy()
    ev["latency_ms"] = ev["latency_ns"] / 1e6
    ev["ids_key"] = ev["ids"].map(lambda x: tuple(x) if x is not None else ())
    agg = (
        ev.groupby(["config_id", "method"], sort=False)
        .agg(
            latency_ms_median=("latency_ms", "median"),
            latency_ms_mean=("latency_ms", "mean"),
            reps=("latency_ms", "size"),
            recall=("recall", "mean"),
            recall_tol=("recall_tol", "mean"),
            complete=("complete", "mean"),
            filter_ok=("filter_ok", "all"),
            n_errors=("error", lambda s: int(s.notna().sum())),
            n_distinct_results=("ids_key", "nunique"),
            n_returned=("n_returned", "median"),
        )
        .reset_index()
    )
    if "route" in ev.columns and ev["route"].notna().any():
        p1 = (
            ev[ev["route"].notna()]
            .groupby(["config_id", "method"])
            .agg(
                route=("route", lambda s: s.mode().iloc[0]),
                est_rows=("est_rows", "first"),
                overhead_ns=("overhead_ns", "median"),
            )
            .reset_index()
        )
        agg = agg.merge(p1, on=["config_id", "method"], how="left")
    cols = [
        "config_id",
        "query_id",
        "level_idx",
        "target_sel",
        "actual_sel",
        "n_eligible",
        "lo",
        "hi",
        "split",
    ]
    return agg.merge(workload[cols], on="config_id", how="left")


def _arm_table(pc: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    sub = pc[pc["method"].isin(arms)]
    t = sub.pivot(index="config_id", columns="method", values=["latency_ms_median", "recall", "complete"])
    missing = [a for a in arms if a not in t["latency_ms_median"].columns]
    if missing:
        raise ValueError(f"oracle needs per-config results for {missing}")
    return t


def _assemble(pc: pd.DataFrame, t: pd.DataFrame, choice: pd.Series, name: str) -> pd.DataFrame:
    rows = []
    for cid, arm in choice.items():
        rows.append(
            {
                "config_id": cid,
                "method": name,
                "choice": arm,
                "latency_ms_median": t.loc[cid, ("latency_ms_median", arm)],
                "recall": t.loc[cid, ("recall", arm)],
                "complete": t.loc[cid, ("complete", arm)],
            }
        )
    out = pd.DataFrame(rows)
    meta = pc.drop_duplicates("config_id")[
        ["config_id", "query_id", "level_idx", "target_sel", "actual_sel", "n_eligible", "split"]
    ]
    out = out.merge(meta, on="config_id", how="left")
    out["recall_tol"] = np.nan
    out["n_errors"] = 0
    return out


def oracle_floor(pc: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    t = _arm_table(pc, arms)
    lat = t["latency_ms_median"][list(arms)]
    choice = lat.idxmin(axis=1)  # ties -> first arm (B1)
    return _assemble(pc, t, choice, "Oracle")


def gain_captured(
    s_best: float | None,
    s_p1: float,
    s_oracle: float,
    min_rel_headroom: float = 0.01,
    min_abs_headroom_ms: float = 0.01,
) -> tuple[float | None, str]:
    """(S(best) - S(P1)) / (S(best) - S(Oracle)); all three must be the same statistic on the same configs.

    The ratio is undefined (N/A) when the oracle's headroom over the best baseline is below
    max(min_abs_headroom_ms, min_rel_headroom * S(best)): with (near-)zero headroom the ratio only amplifies noise.
    """
    if s_best is None or not np.isfinite(s_best):
        return None, "no single baseline meets criterion 1"
    denom = s_best - s_oracle
    if denom <= 0:
        return None, "no headroom: oracle not faster than the best baseline"
    if denom < max(min_abs_headroom_ms, min_rel_headroom * s_best):
        return None, f"headroom {denom * 1e3:.1f} us < 1% of best: ratio not meaningful"
    return (s_best - s_p1) / denom, ""


def calibrate(cfg, measurements, estimates, overhead_ns, stats_sha):
    if (measurements["split"] != "calibration").any():
        raise ValueError("calibration received held-out test rows")
    groups = {name: rows.set_index("config_id") for name, rows in measurements.groupby("method", sort=False)}
    exact, hnsw = groups["B1"], groups["B2-iter"]
    arms = pd.DataFrame(
        {
            "lat_B1": exact["latency_ms_median"],
            "rec_B1": exact["recall"],
            "com_B1": exact["complete"],
            "lat_H": hnsw["latency_ms_median"],
            "rec_H": hnsw["recall"],
            "com_H": hnsw["complete"],
        }
    )
    arms["est_rows"] = estimates.reindex(arms.index)
    if arms.isna().drop(columns=["rec_B1", "rec_H"]).any().any():
        raise ValueError("missing per-query calibration measurements")
    curve = pd.DataFrame(
        [
            simulate(arms, threshold, overhead_ns["median_ns"] / 1e6)
            for threshold in candidate_thresholds(arms["est_rows"])
        ]
    )
    curve["feasible"] = (curve["recall"] >= cfg.calibration.min_mean_recall) & (
        curve["completeness"] >= cfg.calibration.min_completeness
    )
    feasible = curve[curve["feasible"]].assign(neg_T=lambda rows: -rows["T"])
    if feasible.empty:
        raise RuntimeError("no threshold meets the calibration quality targets")
    best = feasible.sort_values(["mean_ms", "p95_ms", "neg_T"]).iloc[0]
    frozen = {
        "threshold_T": float(best["T"]),
        "hnsw": vars(cfg.hnsw),
        "constraints": vars(cfg.calibration),
        "calibration_at_T": {
            key: float(best[key])
            for key in (
                "mean_ms",
                "p50_ms",
                "p95_ms",
                "recall",
                "completeness",
                "frac_B1",
            )
        },
        "stats_snapshot_sha256": stats_sha,
        "config_sha256": cfg.source_sha256,
        "routing_overhead_ns": overhead_ns,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return frozen, curve


def measure_overhead_ns(stats, filters, repeats=20):
    samples = []
    for _ in range(repeats):
        for lo, hi in filters:
            start = time.perf_counter_ns()
            search.route(stats, 0.0, lo, hi)
            samples.append(time.perf_counter_ns() - start)
    samples.sort()
    return {
        "n": len(samples),
        "median_ns": samples[len(samples) // 2],
        "p95_ns": samples[int(0.95 * (len(samples) - 1))],
        "mean_ns": sum(samples) / len(samples),
    }


def evaluate_phase(cfg, phase, prepared=None):
    prepared = prepared or prepare.load_processed(cfg)
    raw = pd.read_parquet(cfg.run_dir / phase / "raw.parquet")
    base = data.normalize_rows(prepared.base(cfg))
    queries = {key: data.normalize_rows(vector[None])[0] for key, vector in prepared.queries.items()}
    evaluated = evaluate_executions(
        raw, prepared.workload, prepared.gt, prepared.values, base, queries, cfg.workload.k
    )
    result = per_config(evaluated, prepared.workload)
    if not result["filter_ok"].all():
        raise RuntimeError("a search returned IDs outside its metadata filter")
    result.to_parquet(cfg.run_dir / phase / "per_config.parquet", index=False)
    return result


def step_calibrate(cfg):
    prepared = prepare.load_processed(cfg)
    measurements = pd.read_parquet(cfg.run_dir / "calibration/per_config.parquet")
    snapshot = json.loads((cfg.run_dir / "env/stats_snapshot.json").read_text())
    stats = search.cache_stats(snapshot)
    windows = prepared.workload[prepared.workload["split"] == "calibration"]
    estimates = pd.Series(
        {row.config_id: search.estimate_rows(stats, row.lo, row.hi) for row in windows.itertuples()}
    )
    overhead = measure_overhead_ns(stats, list(zip(windows["lo"], windows["hi"])))
    frozen, curve = calibrate(cfg, measurements, estimates, overhead, search.stats_hash(snapshot))
    write_json(cfg.run_dir / "calibration/frozen_router.json", frozen)
    curve.to_parquet(cfg.run_dir / "calibration/threshold_curve.parquet", index=False)
    log.info("P1 threshold: %.1f estimated rows", frozen["threshold_T"])
    return frozen


def step_bench(cfg, phase):
    if phase not in ("calibration", "test"):
        raise ValueError("phase must be calibration or test")
    prepared = prepare.load_processed(cfg)
    workload = prepared.workload[prepared.workload["split"] == phase].reset_index(drop=True)
    snapshot = json.loads((cfg.run_dir / "env/stats_snapshot.json").read_text())
    stats = search.cache_stats(snapshot)
    threshold = None
    if phase == "test":
        frozen = json.loads((cfg.run_dir / "calibration/frozen_router.json").read_text())
        if frozen["config_sha256"] != cfg.source_sha256 or frozen[
            "stats_snapshot_sha256"
        ] != search.stats_hash(snapshot):
            raise RuntimeError("configuration or statistics changed after calibration")
        threshold = frozen["threshold_T"]
    output = cfg.run_dir / phase
    if (output / "raw.parquet").exists():
        raise RuntimeError("this phase already has measurements; use a new --run-id for a rerun")
    output.mkdir(parents=True, exist_ok=True)
    settings = search.method_settings(cfg)
    methods = list(search.BASELINES) + (["P1"] if phase == "test" else [])
    vectors = {key: Vector(value) for key, value in prepared.queries.items()}
    rows_by_id = {row.config_id: row for row in workload.itertuples(index=False)}
    connections = {}
    start_phase = time.time()
    try:
        for method in search.BASELINES:
            connections[method] = search.connect(cfg.database, settings[method])
        cursors = {method: conn.cursor() for method, conn in connections.items()}
        sql = {
            method: search.query_sql(method, cfg.database.table, cfg.workload.k)
            for method in search.BASELINES
        }
        control = connections["B3"]
        expected = json.loads((cfg.run_dir / "env/db_setup.json").read_text())["stats_fingerprint"]
        if search.stats_fingerprint(control, cfg.database.table) != expected:
            raise RuntimeError("database statistics changed since setup")
        audit = {}
        for method, conn in connections.items():
            wanted = {
                "enable_indexscan": "on",
                "enable_seqscan": "on",
                "enable_bitmapscan": "on",
                "jit": "off",
                **settings[method],
            }
            actual = {name: conn.execute(f"SHOW {name}").fetchone()[0] for name in wanted}
            for name, value in wanted.items():
                if actual[name] != value and not (
                    name.endswith("scan_mem_multiplier") and float(actual[name]) == float(value)
                ):
                    raise RuntimeError(f"{method}: unexpected {name}={actual[name]}")
            audit[method] = actual
        write_json(output / "settings_audit.json", audit)
        prepare.prewarm(control, cfg.database.table)
        plans = []
        for row in workload.itertuples(index=False):
            params = {"lo": row.lo, "hi": row.hi, "q": vectors[row.query_id]}
            for method in methods:
                arm = search.route(stats, threshold, row.lo, row.hi)[0] if method == "P1" else method
                plan = search.classify(search.explain(cursors[arm], sql[arm], params), cfg.database.table)
                expectation = "exact" if arm == "B1" else "hnsw" if arm.startswith("B2") else "any"
                if not search.meets_expectation(expectation, plan):
                    raise RuntimeError(f"{method}/{row.config_id}: unexpected plan {plan['plan_class']}")
                plans.append({"config_id": row.config_id, "method": method, "route": arm, **plan})
        pd.DataFrame(plans).to_parquet(output / "plans.parquet", index=False)

        def run_query(config_id, method):
            row = rows_by_id[config_id]
            params = {"lo": row.lo, "hi": row.hi, "q": vectors[row.query_id]}
            arm, estimate, overhead, error = method, None, None, None
            ids = []
            start = time.perf_counter_ns()
            try:
                if method == "P1":
                    arm, estimate = search.route(stats, threshold, row.lo, row.hi)
                    overhead = time.perf_counter_ns() - start
                ids = search.execute(cursors[arm], sql[arm], params)
            except psycopg.Error as problem:
                error = f"{type(problem).__name__}: {problem}".strip()
                if connections[arm].closed:
                    connections[arm] = search.connect(cfg.database, settings[arm])
                    cursors[arm] = connections[arm].cursor()
            latency = time.perf_counter_ns() - start
            return ids, latency, arm if method == "P1" else None, estimate, overhead, error

        pairs = [(config_id, method) for config_id in workload["config_id"] for method in methods]
        records, passes = [], []
        for repetition in range(cfg.benchmark.warmup_passes + cfg.benchmark.repetitions):
            timed = repetition >= cfg.benchmark.warmup_passes
            order = np.random.default_rng([cfg.seed, 11, int(phase == "test"), repetition]).permutation(
                len(pairs)
            )
            start_pass = time.time()
            gc.collect()
            gc.disable()
            try:
                for order_index, index in enumerate(order):
                    config_id, method = pairs[index]
                    ids, latency, arm, estimate, overhead, error = run_query(config_id, method)
                    if timed:
                        records.append(
                            (
                                repetition - cfg.benchmark.warmup_passes,
                                order_index,
                                config_id,
                                method,
                                latency,
                                len(ids),
                                np.asarray(ids, dtype=np.int32),
                                error,
                                time.time_ns(),
                                arm,
                                estimate,
                                overhead,
                            )
                        )
            finally:
                gc.enable()
            passes.append({"pass": repetition, "timed": timed, "seconds": time.time() - start_pass})
            log.info("%s pass %d finished (%s)", phase, repetition + 1, "timed" if timed else "warm-up")
        if search.stats_fingerprint(control, cfg.database.table) != expected:
            raise RuntimeError("database statistics changed during timing")
    finally:
        for conn in connections.values():
            conn.close()
    raw = pd.DataFrame(
        records,
        columns=[
            "pass_idx",
            "order_idx",
            "config_id",
            "method",
            "latency_ns",
            "n_returned",
            "ids",
            "error",
            "ts_ns",
            "route",
            "est_rows",
            "overhead_ns",
        ],
    )
    raw = raw.merge(workload[["config_id", "query_id", "level_idx"]], on="config_id", how="left")
    raw.insert(0, "phase", phase)
    raw.to_parquet(output / "raw.parquet", index=False)
    meta = {
        "phase": phase,
        "methods": methods,
        "n_configs": len(workload),
        "n_executions": len(raw),
        "n_errors": int(raw["error"].notna().sum()),
        "wall_clock_s": time.time() - start_phase,
        "pass_durations_s": passes,
        "plans_checked": len(plans),
    }
    write_json(output / "phase_meta.json", meta)
    evaluate_phase(cfg, phase, prepared)
    return meta
