"""PostgreSQL access: connections, schema, bulk load, indexes, statistics and environment capture."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import Config, DatabaseCfg

log = logging.getLogger(__name__)
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def hnsw_index_name(table: str) -> str:
    return f"{table}_embedding_hnsw"


def btree_index_name(table: str) -> str:
    return f"{table}_similarity_btree"


def connect(db: DatabaseCfg, dbname: str | None = None, settings: dict[str, str] | None = None,
            register: bool = True) -> psycopg.Connection:
    """Autocommit connection with server-side prepared statements disabled.

    `prepare_threshold=None` means every execution is planned with its actual parameter values
    (custom plans), so the planner never switches to a generic plan mid-benchmark.
    """
    conn = psycopg.connect(db.conninfo(dbname), autocommit=True, prepare_threshold=None)
    if register:
        register_vector(conn)
        # load the pgvector library so hnsw.* settings are real GUCs (not placeholders) before SET
        conn.execute("SELECT '[1]'::vector")
    conn.execute(f"SET statement_timeout = {int(db.statement_timeout_ms)}")
    for name, value in (settings or {}).items():
        conn.execute(f"SET {name} = {value}")
    return conn


def ensure_database(db: DatabaseCfg) -> None:
    with psycopg.connect(db.conninfo("postgres"), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db.dbname,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{db.dbname}"')
            log.info("created database %s", db.dbname)
    with psycopg.connect(db.conninfo(), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")


def wait_for_server(db: DatabaseCfg, timeout_s: float = 120) -> None:
    deadline = time.time() + timeout_s
    while True:
        try:
            with psycopg.connect(db.conninfo("postgres"), connect_timeout=3):
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
    for rel in (table, btree_index_name(table), hnsw_index_name(table), f"{table}_pkey"):
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
    row = conn.execute(q, {"t": table, "b": btree_index_name(table), "h": hnsw_index_name(table),
                           "p": f"{table}_pkey"}).fetchone()
    keys = ["heap_bytes", "toast_bytes", "btree_bytes", "hnsw_bytes", "pkey_bytes", "relpages", "reltuples"]
    return {k: (float(v) if k == "reltuples" else int(v)) for k, v in zip(keys, row)}


def stats_fingerprint(conn: psycopg.Connection, table: str) -> str:
    """Hash of the planner statistics for the table; changes if ANALYZE runs again."""
    rows = conn.execute(
        """SELECT attname, null_frac, n_distinct, most_common_vals::text, most_common_freqs::text,
                  histogram_bounds::text, correlation
             FROM pg_stats WHERE schemaname = current_schema() AND tablename = %s ORDER BY attname""",
        (table,),
    ).fetchall()
    rel = conn.execute("SELECT reltuples, relpages FROM pg_class WHERE oid = %s::regclass", (table,)).fetchone()
    import hashlib
    return hashlib.sha256(json.dumps([rows, list(rel)], default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- environment
KEY_SETTINGS = (
    "server_version", "shared_buffers", "work_mem", "maintenance_work_mem", "effective_cache_size",
    "random_page_cost", "seq_page_cost", "cpu_tuple_cost", "cpu_operator_cost", "jit",
    "max_parallel_workers_per_gather", "max_parallel_workers", "max_parallel_maintenance_workers",
    "default_statistics_target", "plan_cache_mode", "enable_seqscan", "enable_indexscan",
    "enable_bitmapscan", "hnsw.ef_search", "hnsw.iterative_scan", "hnsw.max_scan_tuples",
    "hnsw.scan_mem_multiplier",
)


def server_environment(conn: psycopg.Connection) -> dict:
    env = {
        "version": conn.execute("SELECT version()").fetchone()[0],
        "pgvector_version": conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0],
        "key_settings": {},
        "non_default_settings": {},
    }
    for name in KEY_SETTINGS:
        try:
            env["key_settings"][name] = conn.execute(f"SHOW {name}").fetchone()[0]
        except psycopg.Error:
            env["key_settings"][name] = None
    for name, setting, unit, source in conn.execute(
            "SELECT name, setting, unit, source FROM pg_settings "
            "WHERE source NOT IN ('default', 'override') ORDER BY name"):
        env["non_default_settings"][name] = {"setting": setting, "unit": unit, "source": source}
    return env


def host_environment() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # noqa: BLE001 - environment capture must never abort a run
            return f"unavailable: {e}"

    pkgs = {}
    for p in ("numpy", "psycopg", "psycopg-binary", "pgvector", "pandas", "pyarrow", "matplotlib", "pyyaml"):
        try:
            pkgs[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pkgs[p] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": run(["sysctl", "-n", "machdep.cpu.brand_string"]) if platform.system() == "Darwin" else platform.processor(),
        "cpu_count": os.cpu_count(),
        "mem_bytes": run(["sysctl", "-n", "hw.memsize"]) if platform.system() == "Darwin" else None,
        "docker_version": run(["docker", "version", "--format", "{{.Server.Version}}"]),
        "docker_resources": run(["docker", "info", "--format", "cpus={{.NCPU}} mem={{.MemTotal}}"]),
        "image": run(["docker", "inspect", "--format", "{{.Config.Image}} {{.Image}}", "fvs-postgres"]),
        "packages": pkgs,
    }


def rtt_baseline(conn: psycopg.Connection, n: int = 300) -> dict:
    """Client round-trip time of `SELECT 1` - the fixed per-query overhead shared by all methods."""
    cur = conn.cursor()
    lat = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        cur.execute("SELECT 1")
        cur.fetchall()
        lat.append((time.perf_counter_ns() - t0) / 1e6)
    a = np.asarray(lat)
    return {"n": n, "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean())}


def setup_database(cfg: Config, values: np.ndarray, vectors: np.ndarray) -> dict:
    """Create schema, load, index, VACUUM (FREEZE, ANALYZE), prewarm. Returns a build report."""
    db = cfg.database
    wait_for_server(db)
    ensure_database(db)
    with connect(db) as conn:
        create_schema(conn, db.table, cfg.dataset.dim)
        report = {"load_s": load_items(conn, db.table, values, vectors)}
        report.update(build_indexes(conn, db.table, cfg.hnsw.m, cfg.hnsw.ef_construction))
        report["vacuum_analyze_s"] = vacuum_analyze(conn, db.table)
        report["sizes"] = relation_sizes(conn, db.table)
        report["prewarm_blocks"] = prewarm(conn, db.table)
        report["stats_fingerprint"] = stats_fingerprint(conn, db.table)
        n = conn.execute(f"SELECT count(*) FROM {db.table}").fetchone()[0]
        if n != vectors.shape[0]:
            raise RuntimeError(f"loaded {n} rows, expected {vectors.shape[0]}")
        report["rows"] = int(n)
    return report
