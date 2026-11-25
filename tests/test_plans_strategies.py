import pytest
from unittest.mock import Mock

from fvsroute.config import load_config
from fvsroute.search import classify, meets_expectation
from fvsroute.search import PRED_NOSARG, PRED_PLAIN, method_settings, query_sql, execute, explain

from .conftest import SMOKE

T = "items"


def _doc(plan, **extra):
    return {"Plan": plan, **extra}


HNSW_PLAN = _doc(
    {
        "Node Type": "Limit",
        "Plans": [
            {
                "Node Type": "Index Scan",
                "Index Name": "items_embedding_hnsw",
                "Relation Name": "items",
                "Plan Rows": 5,
                "Filter": "((similarity + '0'::double precision) >= 0.3)",
            }
        ],
    }
)
BITMAP_PLAN = _doc(
    {
        "Node Type": "Limit",
        "Plans": [
            {
                "Node Type": "Sort",
                "Plans": [
                    {
                        "Node Type": "Bitmap Heap Scan",
                        "Relation Name": "items",
                        "Plan Rows": 100,
                        "Plans": [
                            {
                                "Node Type": "Bitmap Index Scan",
                                "Index Name": "items_similarity_btree",
                            }
                        ],
                    }
                ],
            }
        ],
    }
)
PARALLEL_SEQ_PLAN = _doc(
    {
        "Node Type": "Limit",
        "Plans": [
            {
                "Node Type": "Gather Merge",
                "Plans": [
                    {
                        "Node Type": "Sort",
                        "Plans": [
                            {
                                "Node Type": "Seq Scan",
                                "Relation Name": "items",
                                "Plan Rows": 37500,
                                "Parallel Aware": True,
                            }
                        ],
                    }
                ],
            }
        ],
    }
)
BTREE_INDEX_PLAN = _doc(
    {
        "Node Type": "Limit",
        "Plans": [
            {
                "Node Type": "Sort",
                "Plans": [
                    {
                        "Node Type": "Index Scan",
                        "Index Name": "items_similarity_btree",
                        "Relation Name": "items",
                        "Plan Rows": 9,
                    }
                ],
            }
        ],
    }
)
ANALYZED = _doc(
    {
        "Node Type": "Limit",
        "Actual Rows": 10,
        "Shared Hit Blocks": 120,
        "Shared Read Blocks": 0,
        "Plans": [
            {
                "Node Type": "Index Scan",
                "Index Name": "items_embedding_hnsw",
                "Relation Name": "items",
                "Plan Rows": 5,
                "Actual Rows": 10,
                "Actual Loops": 1,
                "Rows Removed by Filter": 4321,
            }
        ],
    },
    Planning_Time=0.1,
    **{"Planning Time": 0.12, "Execution Time": 3.4},
)


@pytest.mark.parametrize(
    "doc,cls,par",
    [
        (HNSW_PLAN, "hnsw", False),
        (BITMAP_PLAN, "btree_bitmap", False),
        (PARALLEL_SEQ_PLAN, "seq", True),
        (BTREE_INDEX_PLAN, "btree_index", False),
    ],
)
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


def test_configs_load():
    full = load_config(SMOKE.parent / "full.yaml")
    assert full.workload.n_queries == 300 and len(full.workload.selectivities) == 9
    assert full.dataset.n_rows == 100_000 and full.benchmark.repetitions == 5


def test_baseline_sql_and_fixed_settings(cfg):
    settings = method_settings(cfg)
    assert list(settings) == ["B1", "B2-plain", "B2-iter", "B3"]
    assert PRED_PLAIN in query_sql("B1", "items", 10)
    assert settings["B1"]["enable_indexscan"] == "off"
    for method in ("B2-plain", "B2-iter"):
        assert PRED_NOSARG in query_sql(method, "items", 10)
        assert settings[method]["enable_seqscan"] == "off"
    assert settings["B2-plain"]["hnsw.iterative_scan"] == "off"
    assert settings["B2-iter"]["hnsw.iterative_scan"] == "relaxed_order"
    assert not any(name.startswith("enable_") for name in settings["B3"])
    assert {values["hnsw.ef_search"] for values in settings.values()} == {str(cfg.hnsw.ef_search)}


def test_plain_cursor_execution_and_explain():
    cursor = Mock()
    cursor.fetchall.return_value = [(7,), (9,)]
    cursor.fetchone.return_value = ([{"Plan": {"Node Type": "Limit"}}],)
    sql = query_sql("B2-iter", "items", 10)
    params = {"lo": 0.3, "hi": 0.4, "q": "query"}
    assert execute(cursor, sql, params) == [7, 9]
    cursor.execute.assert_called_with(sql, params)
    assert explain(cursor, sql, params) == {"Plan": {"Node Type": "Limit"}}
    cursor.execute.assert_called_with("EXPLAIN (FORMAT JSON) " + sql, params)
