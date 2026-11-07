import math

import numpy as np
import pytest

from fvsroute.estimator import ColumnStats, SelectivityEstimator
from fvsroute.router import Router, measure_overhead_ns


def _stats(hist, mcv=(), freqs=(), null_frac=0.0, reltuples=100_000.0):
    return ColumnStats("items", "similarity", reltuples, 1000, null_frac, -1.0, tuple(mcv), tuple(freqs),
                       tuple(hist))


def test_uniform_histogram_gives_window_width():
    est = SelectivityEstimator(_stats(np.linspace(0, 1, 101)))
    assert est.selectivity(0.2, 0.3) == pytest.approx(0.1, abs=1e-12)
    assert est.selectivity(0.0, 1.0) == pytest.approx(1.0)
    assert est.estimate_rows(0.25, 0.75) == pytest.approx(50_000)


def test_interpolates_inside_unequal_buckets():
    est = SelectivityEstimator(_stats([0.0, 0.1, 0.5, 1.0]))  # 3 equi-depth buckets
    # half of bucket 2 (0.1..0.5) = 1/6
    assert est.selectivity(0.1, 0.3) == pytest.approx(1 / 6)


def test_mcv_and_null_fraction():
    est = SelectivityEstimator(_stats(np.linspace(0.3, 0.5, 11), mcv=[0.0], freqs=[0.01], null_frac=0.0))
    # the MCV at 0.0 is outside [0.3, 0.5]; histogram weight is 0.99
    assert est.selectivity(0.3, 0.5) == pytest.approx(0.99)
    assert est.selectivity(-1, 0.0) == pytest.approx(0.01)
    est2 = SelectivityEstimator(_stats(np.linspace(0, 1, 11), null_frac=0.2))
    assert est2.selectivity(0, 1) == pytest.approx(0.8)


def test_out_of_range_and_reversed():
    est = SelectivityEstimator(_stats(np.linspace(0.3, 0.5, 11)))
    assert est.selectivity(0.6, 0.7) == 0.0
    assert est.selectivity(0.0, 0.1) == 0.0
    assert est.selectivity(0.45, 0.4) == 0.0


def test_monotone_in_window_width():
    est = SelectivityEstimator(_stats(np.sort(np.random.default_rng(0).uniform(0.3, 0.5, 101))))
    widths = np.linspace(0.001, 0.2, 50)
    sel = [est.selectivity(0.3, 0.3 + w) for w in widths]
    assert all(b >= a for a, b in zip(sel, sel[1:]))


def test_snapshot_roundtrip(tmp_path):
    s = _stats(np.linspace(0, 1, 5), mcv=[0.0], freqs=[0.1])
    s.save(tmp_path / "snap.json")
    s2 = ColumnStats.load(tmp_path / "snap.json")
    assert s2 == s and s2.sha256() == s.sha256()


def test_router_boundary_inclusive():
    est = SelectivityEstimator(_stats(np.linspace(0, 1, 101), reltuples=1000))
    r = Router(est, threshold=100.0)
    assert r.route(0.0, 0.1)[0] == "B1"          # est == 100 == T -> exact
    assert r.route(0.0, 0.1001)[0] == "B2-iter"  # just above T
    assert Router(est, math.inf).route(0, 1)[0] == "B1"
    assert Router(est, 0.0).route(0.2, 0.9)[0] == "B2-iter"


class _CountingConn:
    def __init__(self):
        self.calls = 0

    def execute(self, sql, params=None):
        self.calls += 1
        conn = self

        class R:
            def fetchone(self_inner):
                if "pg_class" in sql:
                    return (1000.0, 10)
                return (0.0, -1.0, [0.0], [0.01], [0.3, 0.4, 0.5])
        return R()


def test_router_reads_statistics_once_and_routes_without_db():
    conn = _CountingConn()
    r = Router.from_connection(conn, "items", "similarity", threshold=10)
    after_load = conn.calls
    assert after_load == 2  # pg_class + pg_stats, once
    for _ in range(100):
        r.route(0.31, 0.32)
    assert conn.calls == after_load
    oh = measure_overhead_ns(r, [(0.3, 0.4)], repeats=10)
    assert oh["n"] == 10 and oh["median_ns"] > 0
