"""The four baselines, cached row estimates, and the B4 routing rule."""

import bisect
import hashlib
import json
from itertools import accumulate
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from .config import connection_string

BASELINES = ("B1", "B2-plain", "B2-iter", "B3")
PRED_PLAIN = "similarity BETWEEN %(lo)s AND %(hi)s"
# This equivalent expression prevents the B-tree from taking over a forced HNSW query.
PRED_NOSARG = "(similarity + 0.0::float8) BETWEEN %(lo)s AND %(hi)s"
SCAN_NODES = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Bitmap Index Scan"}


def connect(database, settings=None, dbname=None, register=True):
    conn = psycopg.connect(
        connection_string(database, dbname),
        autocommit=True,
        prepare_threshold=None,
    )
    if register:
        register_vector(conn)
        conn.execute("SELECT '[1]'::vector")
    conn.execute(f"SET statement_timeout = {int(database.statement_timeout_ms)}")
    for name, value in (settings or {}).items():
        conn.execute(f"SET {name} = {value}")
    return conn


def method_settings(cfg):
    common = {
        "hnsw.ef_search": str(cfg.hnsw.ef_search),
        "hnsw.max_scan_tuples": str(cfg.hnsw.max_scan_tuples),
        "hnsw.scan_mem_multiplier": str(float(cfg.hnsw.scan_mem_multiplier)),
        "hnsw.iterative_scan": "off",
    }
    return {
        "B1": {**common, "enable_indexscan": "off"},
        "B2-plain": {**common, "enable_seqscan": "off"},
        "B2-iter": {**common, "enable_seqscan": "off", "hnsw.iterative_scan": "relaxed_order"},
        "B3": common,
    }


def query_sql(method, table, k):
    predicate = PRED_NOSARG if method in ("B2-plain", "B2-iter") else PRED_PLAIN
    return f"SELECT id FROM {table} WHERE {predicate} ORDER BY embedding <=> %(q)b LIMIT {int(k)}"


def execute(cursor, sql, params):
    cursor.execute(sql, params)
    return [row[0] for row in cursor.fetchall()]


def explain(cursor, sql, params):
    cursor.execute("EXPLAIN (FORMAT JSON) " + sql, params)
    return cursor.fetchone()[0][0]


def fetch_stats(conn, table, column):
    rows, pages = conn.execute(
        "SELECT reltuples, relpages FROM pg_class WHERE oid = %s::regclass",
        (table,),
    ).fetchone()
    row = conn.execute(
        """SELECT null_frac, n_distinct, most_common_vals::text::float8[],
                  most_common_freqs::float8[], histogram_bounds::text::float8[]
             FROM pg_stats
            WHERE schemaname = current_schema() AND tablename = %s AND attname = %s""",
        (table, column),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"no statistics for {table}.{column}; run ANALYZE first")
    nulls, distinct, values, frequencies, bounds = row
    return dict(
        table=table,
        column=column,
        reltuples=float(rows),
        relpages=int(pages),
        null_frac=float(nulls),
        n_distinct=float(distinct),
        mcv_vals=values or [],
        mcv_freqs=frequencies or [],
        hist_bounds=bounds or [],
    )


def stats_hash(stats):
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in stats.items() if key != "sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()


def stats_fingerprint(conn, table):
    rows = conn.execute(
        """SELECT attname, null_frac, n_distinct, most_common_vals::text, most_common_freqs::text,
                  histogram_bounds::text, correlation
             FROM pg_stats WHERE schemaname = current_schema() AND tablename = %s ORDER BY attname""",
        (table,),
    ).fetchall()
    relation = conn.execute(
        "SELECT reltuples, relpages FROM pg_class WHERE oid = %s::regclass",
        (table,),
    ).fetchone()
    return hashlib.sha256(json.dumps([rows, list(relation)], default=str).encode()).hexdigest()


def cache_stats(stats):
    pairs = sorted(zip(stats["mcv_vals"], stats["mcv_freqs"]))
    cumulative = [0.0, *accumulate(frequency for _, frequency in pairs)]
    return {
        "rows": stats["reltuples"],
        "histogram": stats["hist_bounds"],
        "values": [value for value, _ in pairs],
        "cumulative": cumulative,
        "histogram_weight": max(0.0, 1.0 - stats["null_frac"] - cumulative[-1]),
    }


def histogram_fraction(histogram, value):
    buckets = len(histogram) - 1
    if buckets < 1:
        return 0.5
    if value <= histogram[0]:
        return 0.0
    if value >= histogram[-1]:
        return 1.0
    index = bisect.bisect_right(histogram, value) - 1
    lo, hi = histogram[index : index + 2]
    fraction = (value - lo) / (hi - lo) if hi > lo else 0.5
    return (index + fraction) / buckets


def estimate_rows(stats, lo, hi):
    if hi < lo:
        return 0.0
    values, cumulative = stats["values"], stats["cumulative"]
    upper = cumulative[bisect.bisect_right(values, hi)]
    upper += stats["histogram_weight"] * histogram_fraction(stats["histogram"], hi)
    lower = cumulative[bisect.bisect_left(values, lo)]
    lower += stats["histogram_weight"] * histogram_fraction(stats["histogram"], lo)
    return stats["rows"] * min(1.0, max(0.0, upper - lower))


def route(stats, threshold, lo, hi):
    estimate = estimate_rows(stats, lo, hi)
    return ("B1" if estimate <= threshold else "B2-iter"), estimate


def walk(node: dict) -> Iterator[dict]:
    yield node
    for child in node.get("Plans", []) or []:
        yield from walk(child)


def classify(explain_doc: dict, table: str) -> dict:
    """Summarise an `EXPLAIN (FORMAT JSON)` document (the element inside the top-level list).

    plan_class is one of: hnsw | btree_bitmap | btree_index | seq | other
    """
    root = explain_doc["Plan"]
    nodes = list(walk(root))
    types = [n["Node Type"] for n in nodes]
    index_names = {n.get("Index Name") for n in nodes if n["Node Type"] in ("Index Scan", "Index Only Scan")}
    bitmap_names = {n.get("Index Name") for n in nodes if n["Node Type"] == "Bitmap Index Scan"}
    uses_hnsw = f"{table}_embedding_hnsw" in index_names
    uses_btree_bitmap = f"{table}_similarity_btree" in bitmap_names
    uses_btree_index = f"{table}_similarity_btree" in index_names
    uses_seq = any(n["Node Type"] == "Seq Scan" and n.get("Relation Name") == table for n in nodes)
    if uses_hnsw:
        plan_class = "hnsw"
    elif uses_btree_bitmap:
        plan_class = "btree_bitmap"
    elif uses_btree_index:
        plan_class = "btree_index"
    elif uses_seq:
        plan_class = "seq"
    else:
        plan_class = "other"
    scan = next(
        (n for n in nodes if n["Node Type"] in SCAN_NODES and n["Node Type"] != "Bitmap Index Scan"),
        None,
    )
    out = {
        "plan_class": plan_class,
        "uses_hnsw": uses_hnsw,
        "uses_btree": uses_btree_bitmap or uses_btree_index,
        "uses_seq": uses_seq,
        "parallel": any(t in ("Gather", "Gather Merge") for t in types),
        "node_types": ">".join(types),
        "total_cost": root.get("Total Cost"),
        "scan_plan_rows": scan.get("Plan Rows") if scan else None,
    }
    if "Execution Time" in explain_doc:  # EXPLAIN ANALYZE
        out.update(
            {
                "planning_ms": explain_doc.get("Planning Time"),
                "execution_ms": explain_doc.get("Execution Time"),
                "scan_actual_rows": scan.get("Actual Rows") if scan else None,
                "scan_actual_loops": scan.get("Actual Loops") if scan else None,
                "rows_removed_by_filter": sum(int(n.get("Rows Removed by Filter", 0) or 0) for n in nodes),
                "shared_hit_blocks": root.get("Shared Hit Blocks"),
                "shared_read_blocks": root.get("Shared Read Blocks"),
                "root_actual_rows": root.get("Actual Rows"),
            }
        )
    return out


def meets_expectation(expect: str, summary: dict) -> bool:
    if expect == "exact":
        return not summary["uses_hnsw"]
    if expect == "hnsw":
        return summary["uses_hnsw"] and not summary["uses_btree"] and not summary["uses_seq"]
    return True
