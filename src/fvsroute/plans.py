"""EXPLAIN plan classification and intended-plan assertions."""

from __future__ import annotations

from typing import Iterator

SCAN_NODES = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Bitmap Index Scan"}


def walk(node: dict) -> Iterator[dict]:
    yield node
    for child in node.get("Plans", []) or []:
        yield from walk(child)


def classify(explain_doc: dict, table: str) -> dict:
    """Summarise an `EXPLAIN (FORMAT JSON)` document (the element inside the top-level list).

    plan_class is one of: hnsw | btree_bitmap | btree_index | seq | other
    """
    hnsw_idx = f"{table}_embedding_hnsw"
    btree_idx = f"{table}_similarity_btree"
    root = explain_doc["Plan"]
    nodes = list(walk(root))
    types = [n["Node Type"] for n in nodes]
    uses_hnsw = any(n.get("Index Name") == hnsw_idx and n["Node Type"] in ("Index Scan", "Index Only Scan")
                    for n in nodes)
    uses_btree_bitmap = any(n.get("Index Name") == btree_idx and n["Node Type"] == "Bitmap Index Scan"
                            for n in nodes)
    uses_btree_index = any(n.get("Index Name") == btree_idx and n["Node Type"] in ("Index Scan", "Index Only Scan")
                           for n in nodes)
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
    scan = next((n for n in nodes if n["Node Type"] in SCAN_NODES and n["Node Type"] != "Bitmap Index Scan"), None)
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
        out.update({
            "planning_ms": explain_doc.get("Planning Time"),
            "execution_ms": explain_doc.get("Execution Time"),
            "scan_actual_rows": scan.get("Actual Rows") if scan else None,
            "scan_actual_loops": scan.get("Actual Loops") if scan else None,
            "rows_removed_by_filter": sum(int(n.get("Rows Removed by Filter", 0) or 0) for n in nodes),
            "shared_hit_blocks": root.get("Shared Hit Blocks"),
            "shared_read_blocks": root.get("Shared Read Blocks"),
            "root_actual_rows": root.get("Actual Rows"),
        })
    return out


def meets_expectation(expect: str, summary: dict) -> bool:
    if expect == "exact":
        return not summary["uses_hnsw"]
    if expect == "hnsw":
        return summary["uses_hnsw"] and not summary["uses_btree"] and not summary["uses_seq"]
    return True
