"""Download and prepare data, load PostgreSQL, and check correctness."""

from __future__ import annotations

import logging
import json
import platform
import importlib.metadata
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from . import data, search
from .config import Config, connection_string, write_json
from .search import connect, stats_fingerprint

log = logging.getLogger(__name__)
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def ensure_database(db) -> None:
    with psycopg.connect(connection_string(db, "postgres"), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db.dbname,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{db.dbname}"')
            log.info("created database %s", db.dbname)
    with psycopg.connect(connection_string(db), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")


def wait_for_server(db, timeout_s: float = 120) -> None:
    deadline = time.time() + timeout_s
    while True:
        try:
            with psycopg.connect(connection_string(db, "postgres"), connect_timeout=3):
                return
        except psycopg.OperationalError:
            if time.time() > deadline:
                raise
            time.sleep(1)


def _render(sql_file: str, **kw) -> list[str]:
    text = (SQL_DIR / sql_file).read_text().format(**kw)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def create_schema(conn: psycopg.Connection, table: str, dim: int) -> None:
    for stmt in _render("01_schema.sql", table=table, dim=dim):
        conn.execute(stmt)


def load_items(conn: psycopg.Connection, table: str, values: np.ndarray, vectors: np.ndarray) -> float:
    """Binary COPY of (id = row index, similarity, embedding float32). Returns seconds."""
    if not np.isfinite(values).all():
        raise ValueError("filter values must be finite (predicate-equivalence precondition)")
    t0 = time.perf_counter()
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} (id, similarity, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int4", "float8", "vector"])
            for i in range(vectors.shape[0]):
                cp.write_row((i, float(values[i]), vectors[i]))
    return time.perf_counter() - t0


def build_indexes(conn: psycopg.Connection, table: str, m: int, ef_construction: int) -> dict:
    stmts = _render("02_indexes.sql", table=table, m=m, ef_construction=ef_construction)
    times = {}
    for name, stmt in zip(["btree", "hnsw"], stmts):
        t0 = time.perf_counter()
        conn.execute(stmt)
        times[f"{name}_build_s"] = time.perf_counter() - t0
        log.info("built %s index in %.1fs", name, times[f"{name}_build_s"])
    return times


def vacuum_analyze(conn: psycopg.Connection, table: str) -> float:
    t0 = time.perf_counter()
    conn.execute(f"VACUUM (FREEZE, ANALYZE) {table}")
    return time.perf_counter() - t0


def prewarm(conn: psycopg.Connection, table: str) -> dict:
    out = {}
    for rel in (table, f"{table}_similarity_btree", f"{table}_embedding_hnsw", f"{table}_pkey"):
        out[rel] = conn.execute("SELECT pg_prewarm(%s)", (rel,)).fetchone()[0]
    return out


def relation_sizes(conn: psycopg.Connection, table: str) -> dict:
    q = """
    SELECT pg_relation_size(%(t)s::regclass)                                AS heap_bytes,
           COALESCE(pg_relation_size(reltoastrelid), 0)                     AS toast_bytes,
           pg_relation_size(%(b)s::regclass)                                AS btree_bytes,
           pg_relation_size(%(h)s::regclass)                                AS hnsw_bytes,
           pg_relation_size(%(p)s::regclass)                                AS pkey_bytes,
           relpages, reltuples
      FROM pg_class WHERE oid = %(t)s::regclass
    """
    row = conn.execute(
        q,
        {
            "t": table,
            "b": f"{table}_similarity_btree",
            "h": f"{table}_embedding_hnsw",
            "p": f"{table}_pkey",
        },
    ).fetchone()
    keys = [
        "heap_bytes",
        "toast_bytes",
        "btree_bytes",
        "hnsw_bytes",
        "pkey_bytes",
        "relpages",
        "reltuples",
    ]
    return {k: (float(v) if k == "reltuples" else int(v)) for k, v in zip(keys, row)}


def step_download(cfg: Config) -> None:
    archive = data.download(cfg.dataset.url, cfg.archive_path, cfg.dataset.sha256)
    data.extract(archive, cfg.raw_dir)
    log.info("dataset ready in %s", cfg.raw_dir)


def step_prepare(cfg: Config) -> dict:
    t0 = time.time()
    raw = data.load_raw(cfg.raw_dir, cfg.dataset.filter_field)
    report = data.validate_raw(raw, cfg.dataset.dim, cfg.dataset.imputed_value)
    n = cfg.dataset.n_rows
    if n > raw.n:
        raise data.DatasetError(f"n_rows={n} exceeds dataset size {raw.n}")
    base = raw.vectors[:n].astype(np.float32)
    values = raw.filter_values[:n].copy()
    w = cfg.workload

    qids = data.select_queries(raw.test_queries, w.n_queries, cfg.seed)
    split = data.split_queries(qids, w.calibration_fraction, cfg.seed)
    wl = data.generate_filters(values, w.selectivities, qids, split, cfg.seed, cfg.dataset.imputed_value)
    queries = {int(q): raw.test_queries[q] for q in qids}

    # Ground-truth gate: reproduce Qdrant's exact closest_ids (full base, original conditions).
    gate = data.validate_against_benchmark(
        raw.vectors.astype(np.float32),
        raw.filter_values,
        raw.test_queries[qids],
        raw.test_cond_gt[qids],
        raw.test_closest_ids[qids],
        raw.test_closest_scores[qids],
    )
    gate["min_required_mean_overlap"] = w.gt_gate_min_overlap
    gate["passed"] = gate["mean_overlap"] >= w.gt_gate_min_overlap
    out = cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "gt_vs_qdrant.json", gate)
    if not gate["passed"]:
        raise data.DatasetError(f"ground-truth gate failed: {gate}")

    gt = data.compute_groundtruth(base, values, queries, wl, w.k)
    merged = wl.merge(gt[["config_id", "n_eligible_gt"]], on="config_id")
    if not (merged["n_eligible"] == merged["n_eligible_gt"]).all():
        raise RuntimeError("eligible counts disagree between workload and ground truth")

    np.save(out / "query_ids.npy", qids)
    np.save(out / "queries.npy", np.stack([queries[int(q)] for q in qids]))
    np.save(out / "filter_values.npy", values)
    wl.to_parquet(out / "workload.parquet", index=False)
    gt.to_parquet(out / "groundtruth.parquet", index=False)

    # per-level selectivity summary (actual vs target)
    lvl = (
        wl.groupby("target_sel")
        .agg(
            n=("config_id", "size"),
            m_target=("m_target", "first"),
            eligible_min=("n_eligible", "min"),
            eligible_max=("n_eligible", "max"),
            actual_sel_mean=("actual_sel", "mean"),
        )
        .reset_index()
    )
    max_excess = int((wl["n_eligible"] - wl["m_target"]).max())
    manifest = {
        "config": str(cfg.source_path),
        "config_sha256": cfg.source_sha256,
        "run_id": cfg.run_id,
        "dataset_sha256": cfg.dataset.sha256,
        "dataset_validation": report,
        "n_rows_used": n,
        "n_imputed_rows_used": int((values == cfg.dataset.imputed_value).sum()),
        "n_queries": int(len(qids)),
        "query_ids": [int(q) for q in qids],
        "split_counts": {s: sum(1 for v in split.values() if v == s) for s in ("calibration", "test")},
        "n_configs": int(len(wl)),
        "levels": lvl.to_dict(orient="records"),
        "max_eligible_excess_over_target": max_excess,
        "n_configs_tie_at_k": int(gt["tie_at_k"].sum()),
        "n_configs_fewer_than_k_eligible": int((wl["n_eligible"] < w.k).sum()),
        "n_configs_zero_eligible": int((wl["n_eligible"] == 0).sum()),
        "workload_sha256": data.frame_sha256(wl),
        "groundtruth_sha256": data.frame_sha256(gt[["config_id", "gt_ids"]]),
        "gt_gate": gate,
        "seconds": time.time() - t0,
    }
    write_json(out / "manifest.json", manifest)
    log.info(
        "prepared %d configs (%d queries x %d levels); GT gate mean overlap %.4f; tie_at_k=%d",
        len(wl),
        len(qids),
        len(w.selectivities),
        gate["mean_overlap"],
        manifest["n_configs_tie_at_k"],
    )
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


def step_db_setup(cfg):
    prepared = load_processed(cfg)
    vectors = prepared.base(cfg)
    wait_for_server(cfg.database)
    ensure_database(cfg.database)
    with connect(cfg.database) as conn:
        create_schema(conn, cfg.database.table, cfg.dataset.dim)
        report = {"load_s": load_items(conn, cfg.database.table, prepared.values, vectors)}
        report.update(build_indexes(conn, cfg.database.table, cfg.hnsw.m, cfg.hnsw.ef_construction))
        report["vacuum_analyze_s"] = vacuum_analyze(conn, cfg.database.table)
        report["rows"] = conn.execute(f"SELECT count(*) FROM {cfg.database.table}").fetchone()[0]
        if report["rows"] != len(vectors):
            raise RuntimeError("loaded row count does not match the dataset")
        report["sizes"] = relation_sizes(conn, cfg.database.table)
        report["stats_fingerprint"] = stats_fingerprint(conn, cfg.database.table)
        stats = search.fetch_stats(conn, cfg.database.table, cfg.dataset.filter_field)
        stats["sha256"] = search.stats_hash(stats)
        write_json(cfg.run_dir / "env/stats_snapshot.json", stats)
        server = {
            "version": conn.execute("SELECT version()").fetchone()[0],
            "pgvector_version": conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname='vector'"
            ).fetchone()[0],
            "settings": {
                name: conn.execute(f"SHOW {name}").fetchone()[0]
                for name in (
                    "shared_buffers",
                    "work_mem",
                    "maintenance_work_mem",
                    "jit",
                    "default_statistics_target",
                )
            },
        }
    host = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "psycopg",
                "pgvector",
                "pandas",
                "pyarrow",
                "matplotlib",
                "pyyaml",
            )
        },
    }
    write_json(cfg.run_dir / "env/server_env.json", server)
    write_json(cfg.run_dir / "env/host_env.json", host)
    write_json(cfg.run_dir / "env/db_setup.json", report)
    return report


def step_verify(cfg):
    prepared = load_processed(cfg)
    base = data.normalize_rows(prepared.base(cfg))
    truth = prepared.gt.set_index("config_id")
    table = cfg.database.table
    checks = {"eligible_count_errors": 0, "unexplained_exact_errors": 0, "exact_near_ties": 0}
    rows = []
    rng = np.random.default_rng([cfg.seed, 21])
    settings = search.method_settings(cfg)["B1"]
    with connect(cfg.database) as conn, connect(cfg.database, settings) as exact:
        checks["rows"] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        checks["predicate_mismatches"] = conn.execute(
            f"SELECT count(*) FROM {table} WHERE similarity + 0.0::float8 IS DISTINCT FROM similarity"
        ).fetchone()[0]
        stats = search.fetch_stats(conn, table, cfg.dataset.filter_field)
        cached = search.cache_stats(stats)
        snapshot = json.loads((cfg.run_dir / "env/stats_snapshot.json").read_text())
        if search.stats_hash(stats) != search.stats_hash(snapshot):
            raise RuntimeError("database statistics changed after setup")
        max_difference = 0.0
        for query_id in prepared.query_ids[:50]:
            query = prepared.queries[int(query_id)]
            ids = rng.choice(len(base), min(200, len(base)), replace=False)
            distances = conn.execute(
                f"SELECT id, embedding <=> %b FROM {table} WHERE id = ANY(%s)",
                (query, ids.tolist()),
            ).fetchall()
            for item_id, pg_distance in distances:
                numpy_distance = 1.0 - base[item_id] @ data.normalize_rows(query[None])[0]
                max_difference = max(max_difference, abs(pg_distance - numpy_distance))
        checks["max_cosine_difference"] = float(max_difference)
        for row in prepared.workload.itertuples(index=False):
            count = conn.execute(
                f"SELECT count(*) FROM {table} WHERE similarity BETWEEN %s AND %s",
                (row.lo, row.hi),
            ).fetchone()[0]
            checks["eligible_count_errors"] += int(count != row.n_eligible)
            params = {"lo": row.lo, "hi": row.hi, "q": prepared.queries[row.query_id]}
            result = search.execute(exact.cursor(), search.query_sql("B1", table, cfg.workload.k), params)
            if len(result) != min(cfg.workload.k, row.n_eligible):
                checks["unexplained_exact_errors"] += 1
            extra = np.array(sorted(set(result) - set(truth.loc[row.config_id, "gt_ids"])), dtype=int)
            if len(extra):
                query = data.normalize_rows(prepared.queries[row.query_id][None])[0]
                near_tie = np.all(
                    1.0 - base[extra] @ query <= truth.loc[row.config_id, "gt_kth_dist"] + data.TIE_EPS
                )
                checks["exact_near_ties"] += int(near_tie)
                checks["unexplained_exact_errors"] += int(not near_tie)
            plan = conn.execute(
                f"EXPLAIN (FORMAT JSON) SELECT id FROM {table} WHERE similarity BETWEEN %s AND %s",
                (row.lo, row.hi),
            ).fetchone()[0][0]
            rows.append(
                {
                    "config_id": row.config_id,
                    "level_idx": row.level_idx,
                    "target_sel": row.target_sel,
                    "split": row.split,
                    "actual": count,
                    "est_ours": search.estimate_rows(cached, row.lo, row.hi),
                    "est_pg": search.classify(plan, table)["scan_plan_rows"],
                }
            )
    estimation = pd.DataFrame(rows)
    for name in ("ours", "pg"):
        estimation[f"rel_err_{name}"] = (estimation[f"est_{name}"] - estimation["actual"]) / estimation[
            "actual"
        ].replace(0, np.nan)
    output = cfg.run_dir / "verify"
    output.mkdir(parents=True, exist_ok=True)
    estimation.to_parquet(output / "estimation.parquet", index=False)
    checks["all_pass"] = (
        checks["rows"] == cfg.dataset.n_rows
        and max_difference <= 1e-6
        and checks["predicate_mismatches"]
        == checks["eligible_count_errors"]
        == checks["unexplained_exact_errors"]
        == 0
    )
    write_json(output / "consistency.json", checks)
    if not checks["all_pass"]:
        raise RuntimeError("verification failed; see verify/consistency.json")
    return checks
