"""Pipeline steps behind the `fvs` CLI. Each step reads its inputs from disk and writes its outputs to disk."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset, db as dbmod, groundtruth, workload as wlmod
from .calibrate import calibrate
from .config import Config
from .estimator import ColumnStats, SelectivityEstimator, fetch_stats
from .metrics import evaluate_executions, per_config
from .plans import classify
from .router import Router, measure_overhead_ns
from .runner import PhaseInputs, run_phase
from .strategies import SessionPool, build_strategies

log = logging.getLogger(__name__)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ============================================================================ download / prepare
def step_download(cfg: Config) -> None:
    archive = dataset.download(cfg.dataset.url, cfg.archive_path, cfg.dataset.sha256)
    dataset.extract(archive, cfg.raw_dir)
    log.info("dataset ready in %s", cfg.raw_dir)


def step_prepare(cfg: Config) -> dict:
    t0 = time.time()
    raw = dataset.load_raw(cfg.raw_dir, cfg.dataset.filter_field)
    report = dataset.validate_raw(raw, cfg.dataset.dim, cfg.dataset.imputed_value)
    n = cfg.dataset.n_rows
    if n > raw.n:
        raise dataset.DatasetError(f"n_rows={n} exceeds dataset size {raw.n}")
    base = raw.vectors[:n].astype(np.float32)
    values = raw.filter_values[:n].copy()
    w = cfg.workload

    qids = wlmod.select_queries(raw.test_queries, w.n_queries, cfg.seed)
    split = wlmod.split_queries(qids, w.calibration_fraction, cfg.seed)
    wl = wlmod.generate_filters(values, w.selectivities, qids, split, cfg.seed, cfg.dataset.imputed_value)
    queries = {int(q): raw.test_queries[q] for q in qids}

    # Ground-truth gate: reproduce Qdrant's exact closest_ids (full base, original conditions).
    gate = groundtruth.validate_against_benchmark(
        raw.vectors.astype(np.float32), raw.filter_values, raw.test_queries[qids], raw.test_cond_gt[qids],
        raw.test_closest_ids[qids], raw.test_closest_scores[qids])
    gate["min_required_mean_overlap"] = w.gt_gate_min_overlap
    gate["passed"] = gate["mean_overlap"] >= w.gt_gate_min_overlap
    out = cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "gt_vs_qdrant.json", gate)
    if not gate["passed"]:
        raise dataset.DatasetError(f"ground-truth gate failed: {gate}")

    gt = groundtruth.compute_groundtruth(base, values, queries, wl, w.k)
    merged = wl.merge(gt[["config_id", "n_eligible_gt"]], on="config_id")
    if not (merged["n_eligible"] == merged["n_eligible_gt"]).all():
        raise RuntimeError("eligible counts disagree between workload and ground truth")

    np.save(out / "query_ids.npy", qids)
    np.save(out / "queries.npy", np.stack([queries[int(q)] for q in qids]))
    np.save(out / "filter_values.npy", values)
    wl.to_parquet(out / "workload.parquet", index=False)
    gt.to_parquet(out / "groundtruth.parquet", index=False)

    # per-level selectivity summary (actual vs target)
    lvl = wl.groupby("target_sel").agg(n=("config_id", "size"), m_target=("m_target", "first"),
                                       eligible_min=("n_eligible", "min"), eligible_max=("n_eligible", "max"),
                                       actual_sel_mean=("actual_sel", "mean")).reset_index()
    max_excess = int((wl["n_eligible"] - wl["m_target"]).max())
    manifest = {
        "config": str(cfg.source_path), "config_sha256": cfg.source_sha256, "run_id": cfg.run_id,
        "dataset_sha256": cfg.dataset.sha256, "dataset_validation": report,
        "n_rows_used": n, "n_imputed_rows_used": int((values == cfg.dataset.imputed_value).sum()),
        "n_queries": int(len(qids)), "query_ids": [int(q) for q in qids],
        "split_counts": {s: sum(1 for v in split.values() if v == s) for s in ("calibration", "test")},
        "n_configs": int(len(wl)), "levels": lvl.to_dict(orient="records"),
        "max_eligible_excess_over_target": max_excess,
        "n_configs_tie_at_k": int(gt["tie_at_k"].sum()),
        "n_configs_fewer_than_k_eligible": int((wl["n_eligible"] < w.k).sum()),
        "n_configs_zero_eligible": int((wl["n_eligible"] == 0).sum()),
        "workload_sha256": wlmod.frame_sha256(wl),
        "groundtruth_sha256": wlmod.frame_sha256(gt[["config_id", "gt_ids"]]),
        "gt_gate": gate, "seconds": time.time() - t0,
    }
    _dump(out / "manifest.json", manifest)
    log.info("prepared %d configs (%d queries x %d levels); GT gate mean overlap %.4f; tie_at_k=%d",
             len(wl), len(qids), len(w.selectivities), gate["mean_overlap"], manifest["n_configs_tie_at_k"])
    return manifest


@dataclass
class Processed:
    workload: pd.DataFrame
    gt: pd.DataFrame
    query_ids: np.ndarray
    queries: dict[int, np.ndarray]
    values: np.ndarray

    def base(self, cfg: Config) -> np.ndarray:
        v = np.load(cfg.raw_dir / "vectors.npy", allow_pickle=False)
        return v[: cfg.dataset.n_rows].astype(np.float32)


def load_processed(cfg: Config) -> Processed:
    d = cfg.processed_dir
    if not (d / "workload.parquet").exists():
        raise FileNotFoundError(f"{d} has no prepared workload; run `fvs prepare` first")
    qids = np.load(d / "query_ids.npy")
    qs = np.load(d / "queries.npy")
    return Processed(
        workload=pd.read_parquet(d / "workload.parquet"),
        gt=pd.read_parquet(d / "groundtruth.parquet"),
        query_ids=qids,
        queries={int(q): qs[i] for i, q in enumerate(qids)},
        values=np.load(d / "filter_values.npy"),
    )


# ============================================================================ database
def step_db_setup(cfg: Config) -> dict:
    p = load_processed(cfg)
    base = p.base(cfg)
    report = dbmod.setup_database(cfg, p.values, base)
    env_dir = cfg.run_dir / "env"
    with dbmod.connect(cfg.database) as conn:
        stats = fetch_stats(conn, cfg.database.table, cfg.dataset.filter_field)
        stats.save(env_dir / "stats_snapshot.json")
        report["stats_snapshot_sha256"] = stats.sha256()
        _dump(env_dir / "server_env.json", dbmod.server_environment(conn))
    _dump(env_dir / "host_env.json", dbmod.host_environment())
    _dump(env_dir / "db_setup.json", report)
    log.info("db ready: %d rows, heap %.0f MB, hnsw %.0f MB, hnsw build %.1fs", report["rows"],
             report["sizes"]["heap_bytes"] / 2**20, report["sizes"]["hnsw_bytes"] / 2**20, report["hnsw_build_s"])
    return report


def _db_setup_report(cfg: Config) -> dict:
    path = cfg.run_dir / "env" / "db_setup.json"
    if not path.exists():
        raise FileNotFoundError("run `fvs db-setup` first")
    return json.loads(path.read_text())


def _stats_snapshot(cfg: Config) -> ColumnStats:
    return ColumnStats.load(cfg.run_dir / "env" / "stats_snapshot.json")


# ============================================================================ verify
class VerificationError(RuntimeError):
    pass


def step_verify(cfg: Config) -> dict:
    p = load_processed(cfg)
    base = p.base(cfg)
    base_n = groundtruth.normalize_rows(base)
    t = cfg.database.table
    n = base.shape[0]
    checks: dict = {}
    rng = np.random.default_rng([cfg.seed, 21])
    snap = _stats_snapshot(cfg)

    with dbmod.connect(cfg.database) as conn:
        # 1. ids / row count
        cnt, dist, mn, mx = conn.execute(f"SELECT count(*), count(DISTINCT id), min(id), max(id) FROM {t}").fetchone()
        checks["ids"] = {"rows": cnt, "distinct": dist, "min": mn, "max": mx,
                         "pass": bool(cnt == n and dist == n and mn == 0 and mx == n - 1)}
        # 2. filter values stored exactly
        stored = np.array([r[0] for r in conn.execute(f"SELECT similarity FROM {t} ORDER BY id")], dtype=np.float64)
        checks["filter_values_exact"] = {"pass": bool(np.array_equal(stored, p.values))}
        # 3. vectors stored exactly (sample)
        ids = rng.choice(n, min(1000, n), replace=False)
        rows = conn.execute(f"SELECT id, embedding FROM {t} WHERE id = ANY(%s)", (ids.tolist(),)).fetchall()
        ok = all(np.array_equal(np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32), base[i])
                 for i, v in rows)
        checks["vectors_exact_sample"] = {"n": len(rows), "pass": bool(ok and len(rows) == len(ids))}
        # 4. cosine distance consistency PG <=> vs NumPy
        diffs = []
        for q in p.query_ids[: min(50, len(p.query_ids))]:
            qv = p.queries[int(q)]
            sample = rng.choice(n, min(200, n), replace=False)
            res = conn.execute(f"SELECT id, embedding <=> %b FROM {t} WHERE id = ANY(%s)",
                               (qv, sample.tolist())).fetchall()
            rid = np.array([r[0] for r in res])
            dpg = np.array([r[1] for r in res])
            dnp = 1.0 - base_n[rid] @ groundtruth.normalize_rows(qv[None, :])[0]
            diffs.append(np.abs(dpg - dnp).max())
        checks["cosine_consistency"] = {"max_abs_diff": float(max(diffs)), "tolerance": 1e-6,
                                        "pass": bool(max(diffs) <= 1e-6)}
        # 5. predicate equivalence for the B2 rewrite
        nd = conn.execute(f"SELECT count(*) FROM {t} WHERE similarity + 0.0::float8 IS DISTINCT FROM similarity"
                          ).fetchone()[0]
        nonfinite = conn.execute(f"SELECT count(*) FROM {t} WHERE similarity IN ('NaN','Infinity','-Infinity')"
                                 ).fetchone()[0]
        checks["predicate_equivalence"] = {"rows_where_x_plus_0_differs": nd, "nonfinite_rows": nonfinite,
                                           "pass": bool(nd == 0 and nonfinite == 0)}
        # 6. eligible counts per filter (SQL vs NumPy)
        bad = 0
        for r in p.workload.itertuples(index=False):
            c = conn.execute(f"SELECT count(*) FROM {t} WHERE similarity BETWEEN %s AND %s", (r.lo, r.hi)).fetchone()[0]
            bad += int(c != r.n_eligible)
        checks["eligible_counts"] = {"n_filters": len(p.workload), "mismatches": bad, "pass": bad == 0}
        # 7. statistics snapshot still current
        cur = fetch_stats(conn, t, cfg.dataset.filter_field)
        checks["stats_snapshot_current"] = {"pass": cur.sha256() == snap.sha256()}
        # 8. estimation: ours vs PostgreSQL planner vs actual
        est = SelectivityEstimator(snap)
        erows = []
        for r in p.workload.itertuples(index=False):
            doc = conn.execute(f"EXPLAIN (FORMAT JSON) SELECT id FROM {t} WHERE similarity BETWEEN %s AND %s",
                               (r.lo, r.hi)).fetchone()[0][0]
            erows.append({"config_id": r.config_id, "level_idx": r.level_idx, "target_sel": r.target_sel,
                          "split": r.split, "actual": r.n_eligible, "est_ours": est.estimate_rows(r.lo, r.hi),
                          "est_pg": classify(doc, t)["scan_plan_rows"]})
        estimation = pd.DataFrame(erows)
        estimation["rel_err_ours"] = (estimation["est_ours"] - estimation["actual"]) / estimation["actual"]
        estimation["rel_err_pg"] = (estimation["est_pg"] - estimation["actual"]) / estimation["actual"]
        estimation["ours_vs_pg_rel"] = (estimation["est_ours"] - estimation["est_pg"]) / estimation["est_pg"]
        out = cfg.run_dir / "verify"
        out.mkdir(parents=True, exist_ok=True)
        estimation.to_parquet(out / "estimation.parquet", index=False)
        checks["estimator_matches_planner"] = {
            "median_abs_rel_diff": float(estimation["ours_vs_pg_rel"].abs().median()),
            "p95_abs_rel_diff": float(estimation["ours_vs_pg_rel"].abs().quantile(0.95)),
            "note": "informational: PostgreSQL clamps/rounds estimates slightly differently",
            "pass": True,
        }

    # 9. B1 returns the exact ground truth for every config
    strategies = build_strategies(cfg)
    g = p.gt.set_index("config_id")
    vrows = []
    with SessionPool(cfg, {"B1": strategies["B1"]}) as pool:
        for r in p.workload.itertuples(index=False):
            got = pool.execute("B1", {"lo": r.lo, "hi": r.hi, "q": p.queries[r.query_id]})
            want = g.loc[r.config_id, "gt_ids"]
            denom = min(cfg.workload.k, r.n_eligible)
            strict = len(set(got) & set(want)) / denom if denom else float("nan")
            explained = True
            if strict < 1.0:
                extra = np.array(sorted(set(got) - set(want)), dtype=np.int64)
                qn = groundtruth.normalize_rows(p.queries[r.query_id][None, :])[0]
                d_extra = 1.0 - base_n[extra] @ qn
                explained = bool(np.all(d_extra <= g.loc[r.config_id, "gt_kth_dist"] + groundtruth.TIE_EPS))
            vrows.append({"config_id": r.config_id, "level_idx": r.level_idx, "n_returned": len(got),
                          "strict_recall": strict, "near_tie_explained": explained})
    b1 = pd.DataFrame(vrows)
    b1.to_parquet(cfg.run_dir / "verify" / "b1_vs_gt.parquet", index=False)
    n_miss = int((b1["strict_recall"] < 1.0).sum())
    n_unexplained = int((~b1["near_tie_explained"]).sum())
    checks["b1_equals_ground_truth"] = {"n_configs": len(b1), "strict_mismatches": n_miss,
                                        "unexplained_mismatches": n_unexplained, "pass": n_unexplained == 0}
    all_pass = all(c["pass"] for c in checks.values())
    checks["all_pass"] = all_pass
    _dump(cfg.run_dir / "verify" / "consistency.json", checks)
    for name, c in checks.items():
        if name != "all_pass":
            log.info("verify %-28s %s", name, "PASS" if c["pass"] else f"FAIL {c}")
    if not all_pass:
        raise VerificationError("verification failed; see verify/consistency.json")
    return checks


# ============================================================================ benchmark
def frozen_path(cfg: Config) -> Path:
    return cfg.run_dir / "calibration" / "frozen_router.json"


def step_bench(cfg: Config, phase: str) -> dict:
    p = load_processed(cfg)
    setup = _db_setup_report(cfg)
    wl = p.workload[p.workload["split"] == phase].copy()
    if phase == "calibration":
        strategies = build_strategies(cfg, include_sensitivity=True)
        router = None
    elif phase == "test":
        fpath = frozen_path(cfg)
        if not fpath.exists():
            raise FileNotFoundError("held-out evaluation requires calibration/frozen_router.json (run `fvs calibrate`)")
        frozen = json.loads(fpath.read_text())
        snap = _stats_snapshot(cfg)
        if frozen["stats_snapshot_sha256"] != snap.sha256():
            raise RuntimeError("frozen router was calibrated against different statistics")
        if frozen["config_sha256"] != cfg.source_sha256:
            raise RuntimeError("config changed after calibration")
        strategies = build_strategies(cfg, ef_search=frozen["hnsw"]["ef_search"], include_supplementary=True)
        router = Router(SelectivityEstimator(snap), float(frozen["threshold_T"]))
    else:
        raise ValueError(f"unknown phase {phase!r}")
    inp = PhaseInputs(phase, wl, {q: p.queries[q] for q in wl["query_id"].unique()}, strategies, router)
    return run_phase(cfg, inp, cfg.run_dir / phase, setup["stats_fingerprint"])


def evaluate_phase(cfg: Config, phase: str, p: Processed | None = None, base: np.ndarray | None = None
                   ) -> pd.DataFrame:
    """Per-config metrics for a finished phase (cached as per_config.parquet)."""
    p = p or load_processed(cfg)
    base = base if base is not None else p.base(cfg)
    raw = pd.read_parquet(cfg.run_dir / phase / "raw.parquet")
    base_n = groundtruth.normalize_rows(base)
    qn = {q: groundtruth.normalize_rows(v[None, :])[0] for q, v in p.queries.items()}
    ev = evaluate_executions(raw, p.workload, p.gt, p.values, base_n, qn, cfg.workload.k)
    pc = per_config(ev, p.workload)
    pc.to_parquet(cfg.run_dir / phase / "per_config.parquet", index=False)
    return pc


# ============================================================================ calibrate
def step_calibrate(cfg: Config) -> dict:
    p = load_processed(cfg)
    pc = evaluate_phase(cfg, "calibration", p)
    pc_cal = pc[pc["split"] == "calibration"]
    snap = _stats_snapshot(cfg)
    est = SelectivityEstimator(snap)
    cal_wl = p.workload[p.workload["split"] == "calibration"]
    est_rows = pd.Series({r.config_id: est.estimate_rows(r.lo, r.hi) for r in cal_wl.itertuples(index=False)})
    overhead = measure_overhead_ns(Router(est, 0.0), list(zip(cal_wl["lo"], cal_wl["hi"])))
    frozen, curve = calibrate(cfg, pc_cal, est_rows, overhead, snap.sha256())
    out = cfg.run_dir / "calibration"
    curve.to_parquet(out / "threshold_curve.parquet", index=False)
    _dump(frozen_path(cfg), frozen)
    log.info("frozen T* = %s (ef_search=%d, guard triggered=%s); calibration mean %.3f ms, recall %.4f, "
             "completeness %.4f, frac B1 %.2f", frozen["threshold_T"], frozen["hnsw"]["ef_search"],
             frozen["guard"]["triggered"], frozen["calibration_at_T"]["mean_ms"],
             frozen["calibration_at_T"]["recall"], frozen["calibration_at_T"]["completeness"],
             frozen["calibration_at_T"]["frac_B1"])
    return frozen
