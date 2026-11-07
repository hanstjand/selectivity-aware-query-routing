import pytest

from fvsroute.config import load_config
from fvsroute.plans import classify, meets_expectation
from fvsroute.strategies import PRED_NOSARG, PRED_PLAIN, build_strategies

from .conftest import SMOKE

T = "items"


def _doc(plan, **extra):
    return {"Plan": plan, **extra}


HNSW_PLAN = _doc({"Node Type": "Limit", "Plans": [
    {"Node Type": "Index Scan", "Index Name": "items_embedding_hnsw", "Relation Name": "items", "Plan Rows": 5,
     "Filter": "((similarity + '0'::double precision) >= 0.3)"}]})
BITMAP_PLAN = _doc({"Node Type": "Limit", "Plans": [{"Node Type": "Sort", "Plans": [
    {"Node Type": "Bitmap Heap Scan", "Relation Name": "items", "Plan Rows": 100, "Plans": [
        {"Node Type": "Bitmap Index Scan", "Index Name": "items_similarity_btree"}]}]}]})
PARALLEL_SEQ_PLAN = _doc({"Node Type": "Limit", "Plans": [{"Node Type": "Gather Merge", "Plans": [
    {"Node Type": "Sort", "Plans": [{"Node Type": "Seq Scan", "Relation Name": "items", "Plan Rows": 37500,
                                     "Parallel Aware": True}]}]}]})
BTREE_INDEX_PLAN = _doc({"Node Type": "Limit", "Plans": [{"Node Type": "Sort", "Plans": [
    {"Node Type": "Index Scan", "Index Name": "items_similarity_btree", "Relation Name": "items", "Plan Rows": 9}]}]})
ANALYZED = _doc({"Node Type": "Limit", "Actual Rows": 10, "Shared Hit Blocks": 120, "Shared Read Blocks": 0,
                 "Plans": [{"Node Type": "Index Scan", "Index Name": "items_embedding_hnsw",
                            "Relation Name": "items", "Plan Rows": 5, "Actual Rows": 10, "Actual Loops": 1,
                            "Rows Removed by Filter": 4321}]},
                Planning_Time=0.1, **{"Planning Time": 0.12, "Execution Time": 3.4})


@pytest.mark.parametrize("doc,cls,par", [(HNSW_PLAN, "hnsw", False), (BITMAP_PLAN, "btree_bitmap", False),
                                         (PARALLEL_SEQ_PLAN, "seq", True), (BTREE_INDEX_PLAN, "btree_index", False)])
def test_classify(doc, cls, par):
    s = classify(doc, T)
    assert s["plan_class"] == cls and s["parallel"] == par


def test_expectations():
    assert meets_expectation("exact", classify(BITMAP_PLAN, T))
    assert meets_expectation("exact", classify(PARALLEL_SEQ_PLAN, T))
    assert not meets_expectation("exact", classify(HNSW_PLAN, T))
    assert meets_expectation("hnsw", classify(HNSW_PLAN, T))
    assert not meets_expectation("hnsw", classify(BITMAP_PLAN, T))
    assert meets_expectation("any", classify(BTREE_INDEX_PLAN, T))


def test_analyze_fields():
    s = classify(ANALYZED, T)
    assert s["execution_ms"] == 3.4 and s["rows_removed_by_filter"] == 4321 and s["shared_read_blocks"] == 0


@pytest.fixture(scope="module")
def cfg():
    return load_config(SMOKE)


def test_strategy_sql_and_settings(cfg):
    s = build_strategies(cfg, include_sensitivity=True, include_supplementary=True)
    assert PRED_PLAIN in s["B1"].sql("items", 10) and "LIMIT 10" in s["B1"].sql("items", 10)
    assert s["B1"].settings_dict["enable_indexscan"] == "off"
    for name in ("B2-plain", "B2-iter", "B2-iter-ef40", "B2-strict"):
        assert s[name].predicate == PRED_NOSARG and s[name].settings_dict["enable_seqscan"] == "off"
    assert s["B2-plain"].settings_dict["hnsw.iterative_scan"] == "off"
    assert s["B2-iter"].settings_dict["hnsw.iterative_scan"] == "relaxed_order"
    assert s["B2-strict"].settings_dict["hnsw.iterative_scan"] == "strict_order"
    assert s["B2-iter-mst100k"].settings_dict["hnsw.max_scan_tuples"] == "100000"
    # B3 has no plan-forcing settings and the shared ef_search
    assert not any(k.startswith("enable_") for k in s["B3"].settings_dict)
    shared = {k: v for k, v in s["B2-iter"].settings_dict.items() if k in ("hnsw.ef_search", "hnsw.max_scan_tuples")}
    for name in ("B1", "B2-plain", "B3", "B3+iter"):
        for k, v in shared.items():
            assert s[name].settings_dict[k] == v, (name, k)
    assert s["B3+iter"].role == "supplementary" and s["B2-iter-ef40"].role == "sensitivity"


def test_ef_override_applies_to_all_methods(cfg):
    s = build_strategies(cfg, ef_search=200, include_supplementary=True)
    assert {st.settings_dict["hnsw.ef_search"] for st in s.values()} == {"200"}


def test_configs_load():
    full = load_config(SMOKE.parent / "full.yaml")
    assert full.workload.n_queries == 300 and len(full.workload.selectivities) == 9
    assert full.dataset.n_rows == 100_000 and full.benchmark.repetitions == 5
