"""Search strategies: SQL text + session settings per method, and a session pool.

Every method runs the same query shape
    SELECT id FROM items WHERE <predicate> ORDER BY embedding <=> $q LIMIT k
on its own persistent session whose settings never change during a run.

* B1 exact      : enable_indexscan=off (pgvector's documented way to get exact search). HNSW is only reachable
                  through an index scan, so it is impossible; the planner still chooses between a B-tree
                  bitmap scan and a (parallel) seq scan for the filter. Both are exact.
* B2 HNSW       : predicate rewritten as (similarity + 0.0::float8) BETWEEN lo AND hi - semantically identical for
                  finite float8 values (x + 0.0 == x) but not B-tree indexable - plus enable_seqscan=off, so the
                  only remaining path is the HNSW index scan. Plain = iterative_scan off; iter = relaxed_order.
* B3 planner    : the unmodified query, no enable_* changes: PostgreSQL's default plan choice under the shared
                  HNSW configuration (ef_search like every other method; iterative_scan at its default, off).
* B3+iter       : supplementary only - B3 with iterative_scan=relaxed_order.
All HNSW-related GUCs (ef_search, max_scan_tuples, scan_mem_multiplier) are identical across methods unless a
calibration-only sensitivity variant overrides them explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import psycopg

from .config import Config, HnswCfg

PRED_PLAIN = "similarity BETWEEN %(lo)s AND %(hi)s"
PRED_NOSARG = "(similarity + 0.0::float8) BETWEEN %(lo)s AND %(hi)s"

# roles
PRIMARY = "primary"
SENSITIVITY = "sensitivity"
SUPPLEMENTARY = "supplementary"

# plan expectations
EXPECT_EXACT = "exact"    # must not touch the HNSW index
EXPECT_HNSW = "hnsw"      # must use the HNSW index scan
EXPECT_ANY = "any"        # recorded only

AUDITED_SETTINGS = (
    "enable_indexscan", "enable_seqscan", "enable_bitmapscan", "enable_indexonlyscan",
    "hnsw.ef_search", "hnsw.iterative_scan", "hnsw.max_scan_tuples", "hnsw.scan_mem_multiplier",
    "jit", "work_mem", "max_parallel_workers_per_gather", "statement_timeout", "plan_cache_mode",
    "random_page_cost", "effective_cache_size",
)


@dataclass(frozen=True)
class Strategy:
    name: str
    predicate: str
    settings: tuple[tuple[str, str], ...]
    expect: str
    role: str
    description: str = ""

    def sql(self, table: str, k: int) -> str:
        return (f"SELECT id FROM {table} WHERE {self.predicate} "
                f"ORDER BY embedding <=> %(q)b LIMIT {int(k)}")

    @property
    def settings_dict(self) -> dict[str, str]:
        return dict(self.settings)


def _hnsw_settings(h: HnswCfg, ef_search: int, iterative: str, max_scan_tuples: int | None = None):
    return (
        ("hnsw.ef_search", str(ef_search)),
        ("hnsw.iterative_scan", iterative),
        ("hnsw.max_scan_tuples", str(max_scan_tuples if max_scan_tuples is not None else h.max_scan_tuples)),
        ("hnsw.scan_mem_multiplier", str(h.scan_mem_multiplier)),
    )


def build_strategies(cfg: Config, ef_search: int | None = None, include_sensitivity: bool = False,
                     include_supplementary: bool = False) -> dict[str, Strategy]:
    """All baseline strategies for one phase. `ef_search` defaults to the shared primary value."""
    h = cfg.hnsw
    ef = ef_search if ef_search is not None else h.ef_search
    s: dict[str, Strategy] = {}
    s["B1"] = Strategy("B1", PRED_PLAIN, (("enable_indexscan", "off"),) + _hnsw_settings(h, ef, "off"),
                       EXPECT_EXACT, PRIMARY, "exact: filter (B-tree bitmap or seq scan) + exact distance sort")
    s["B2-plain"] = Strategy("B2-plain", PRED_NOSARG, (("enable_seqscan", "off"),) + _hnsw_settings(h, ef, "off"),
                             EXPECT_HNSW, PRIMARY, "HNSW index scan, post-filter, no iterative scan")
    s["B2-iter"] = Strategy("B2-iter", PRED_NOSARG,
                            (("enable_seqscan", "off"),) + _hnsw_settings(h, ef, "relaxed_order"),
                            EXPECT_HNSW, PRIMARY, "HNSW index scan with iterative scan (relaxed_order)")
    s["B3"] = Strategy("B3", PRED_PLAIN, _hnsw_settings(h, ef, "off"), EXPECT_ANY, PRIMARY,
                       "PostgreSQL default planner decision (shared HNSW config, iterative scan off)")
    if include_supplementary:
        s["B3+iter"] = Strategy("B3+iter", PRED_PLAIN, _hnsw_settings(h, ef, "relaxed_order"), EXPECT_ANY,
                                SUPPLEMENTARY, "default planner decision with iterative scan enabled")
    if include_sensitivity:
        for v in cfg.sensitivity:
            mode = v.iterative_scan or "relaxed_order"
            s[v.name] = Strategy(
                v.name, PRED_NOSARG,
                (("enable_seqscan", "off"),) + _hnsw_settings(h, v.ef_search or ef, mode, v.max_scan_tuples),
                EXPECT_HNSW, SENSITIVITY, f"sensitivity: ef={v.ef_search or ef} mode={mode} "
                f"max_scan_tuples={v.max_scan_tuples or h.max_scan_tuples}")
    return s


@dataclass
class Session:
    strategy: Strategy
    conn: psycopg.Connection
    cur: psycopg.Cursor
    sql: str
    explain_sql: str
    explain_analyze_sql: str


@dataclass
class SessionPool:
    """One persistent connection per strategy, configured once."""
    cfg: Config
    strategies: dict[str, Strategy]
    sessions: dict[str, Session] = field(default_factory=dict)

    def open(self) -> "SessionPool":
        from .db import connect
        for name, st in self.strategies.items():
            conn = connect(self.cfg.database, settings=st.settings_dict)
            sql = st.sql(self.cfg.database.table, self.cfg.workload.k)
            self.sessions[name] = Session(st, conn, conn.cursor(), sql, "EXPLAIN (FORMAT JSON) " + sql,
                                          "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)
        return self

    def reconnect(self, name: str) -> None:
        from .db import connect
        old = self.sessions[name]
        try:
            old.conn.close()
        except Exception:  # noqa: BLE001
            pass
        st = old.strategy
        conn = connect(self.cfg.database, settings=st.settings_dict)
        self.sessions[name] = Session(st, conn, conn.cursor(), old.sql, old.explain_sql, old.explain_analyze_sql)

    def close(self) -> None:
        for s in self.sessions.values():
            s.conn.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ---- execution ------------------------------------------------------
    def execute(self, name: str, params: dict) -> list[int]:
        s = self.sessions[name]
        s.cur.execute(s.sql, params)
        return [r[0] for r in s.cur.fetchall()]

    def explain(self, name: str, params: dict, analyze: bool = False) -> dict:
        s = self.sessions[name]
        s.cur.execute(s.explain_analyze_sql if analyze else s.explain_sql, params)
        return s.cur.fetchone()[0][0]

    # ---- settings audit ---------------------------------------------------
    def audit_settings(self) -> dict:
        """Read every relevant GUC back from each session and compare with the intended value."""
        report, mismatches = {}, []
        for name, s in self.sessions.items():
            actual = {g: s.conn.execute(f"SHOW {g}").fetchone()[0] for g in AUDITED_SETTINGS}
            expected = dict(s.strategy.settings)
            for g in ("enable_indexscan", "enable_seqscan", "enable_bitmapscan", "enable_indexonlyscan"):
                expected.setdefault(g, "on")
            expected["jit"] = "off"
            for g, want in expected.items():
                if _norm(actual.get(g)) != _norm(want):
                    mismatches.append(f"{name}: {g}={actual.get(g)!r}, expected {want!r}")
            report[name] = actual
        return {"sessions": report, "mismatches": mismatches, "ok": not mismatches}


def _norm(v) -> str:
    s = str(v).strip().lower()
    try:
        f = float(s)
        return repr(f)
    except ValueError:
        return s


def time_call(fn, *args):
    t0 = time.perf_counter_ns()
    out = fn(*args)
    return out, time.perf_counter_ns() - t0
