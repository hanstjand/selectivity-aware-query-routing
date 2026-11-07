"""Integration tests against the smoke database (real data prefix). Run `fvs all --config configs/smoke.yaml` first."""

import numpy as np
import pytest

from fvsroute import db as dbmod
from fvsroute import groundtruth as gt
from fvsroute.estimator import SelectivityEstimator, fetch_stats
from fvsroute.pipeline import load_processed
from fvsroute.plans import classify, meets_expectation
from fvsroute.router import Router
from fvsroute.strategies import SessionPool, build_strategies

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def ctx(smoke_ready):
    cfg = smoke_ready
    p = load_processed(cfg)
    return cfg, p, p.base(cfg)


def test_cosine_distance_consistency(ctx):
    cfg, p, base = ctx
    bn = gt.normalize_rows(base)
    with dbmod.connect(cfg.database) as conn:
        for q in p.query_ids[:5]:
            rows = conn.execute("SELECT id, embedding <=> %b FROM items WHERE id < 300", (p.queries[int(q)],)).fetchall()
            ids = np.array([r[0] for r in rows])
            dpg = np.array([r[1] for r in rows])
            dnp = 1 - bn[ids] @ gt.normalize_rows(p.queries[int(q)][None])[0]
            assert np.abs(dpg - dnp).max() < 1e-6


def test_eligible_counts_and_predicate_equivalence(ctx):
    cfg, p, _ = ctx
    with dbmod.connect(cfg.database) as conn:
        assert conn.execute("SELECT count(*) FROM items WHERE similarity + 0.0::float8 IS DISTINCT FROM similarity"
                            ).fetchone()[0] == 0
        for r in p.workload.head(40).itertuples():
            a = conn.execute("SELECT count(*) FROM items WHERE similarity BETWEEN %s AND %s", (r.lo, r.hi)).fetchone()[0]
            b = conn.execute("SELECT count(*) FROM items WHERE (similarity + 0.0::float8) BETWEEN %s AND %s",
                             (r.lo, r.hi)).fetchone()[0]
            assert a == b == r.n_eligible


def test_b1_returns_exact_ground_truth_and_b2_plans_use_hnsw(ctx):
    cfg, p, _ = ctx
    s = build_strategies(cfg)
    g = p.gt.set_index("config_id")
    with SessionPool(cfg, s) as pool:
        assert pool.audit_settings()["ok"]
        for r in p.workload.itertuples():
            params = {"lo": r.lo, "hi": r.hi, "q": p.queries[r.query_id]}
            # strict Recall@10 is set-based; order may differ at float32/float64 near-ties
            assert set(pool.execute("B1", params)) == set(g.loc[r.config_id, "gt_ids"])
            assert meets_expectation("exact", classify(pool.explain("B1", params), "items"))
            for m in ("B2-plain", "B2-iter"):
                summ = classify(pool.explain(m, params), "items")
                assert summ["plan_class"] == "hnsw", (m, r.config_id, summ["node_types"])


def test_estimator_tracks_planner_estimate(ctx):
    cfg, p, _ = ctx
    with dbmod.connect(cfg.database) as conn:
        est = SelectivityEstimator(fetch_stats(conn, "items", "similarity"))
        rel = []
        for r in p.workload[p.workload["target_sel"] >= 0.01].itertuples():
            doc = conn.execute("EXPLAIN (FORMAT JSON) SELECT id FROM items WHERE similarity BETWEEN %s AND %s",
                               (r.lo, r.hi)).fetchone()[0][0]
            pg = classify(doc, "items")["scan_plan_rows"]
            rel.append(abs(est.estimate_rows(r.lo, r.hi) - pg) / pg)
    assert np.median(rel) < 0.05


def test_router_loads_statistics_with_two_queries(ctx):
    cfg, _, _ = ctx
    with dbmod.connect(cfg.database) as conn:
        calls = []
        orig = conn.execute

        def counting(*a, **k):
            calls.append(a[0])
            return orig(*a, **k)
        conn.execute = counting
        r = Router.from_connection(conn, "items", "similarity", threshold=100)
        n = len(calls)
        for _ in range(50):
            r.route(0.31, 0.33)
        assert n == 2 and len(calls) == n
