import math
import json

import numpy as np
import pytest

from fvsroute.search import cache_stats, estimate_rows, fetch_stats, route, stats_hash
from fvsroute.benchmark import measure_overhead_ns
from fvsroute.config import write_json


def _stats(hist, mcv=(), freqs=(), null_frac=0.0, reltuples=100_000.0):
    return dict(
        table="items",
        column="similarity",
        reltuples=reltuples,
        relpages=1000,
        null_frac=null_frac,
        n_distinct=-1.0,
        mcv_vals=list(mcv),
        mcv_freqs=list(freqs),
        hist_bounds=list(hist),
    )


def test_uniform_histogram_gives_window_width():
    stats = cache_stats(_stats(np.linspace(0, 1, 101)))
    assert estimate_rows(stats, 0.2, 0.3) == pytest.approx(10_000, abs=1e-10)
    assert estimate_rows(stats, 0, 1) == 100_000
    assert estimate_rows(stats, 0.25, 0.75) == 50_000


def test_interpolates_inside_unequal_buckets():
    stats = cache_stats(_stats([0.0, 0.1, 0.5, 1.0]))
    assert estimate_rows(stats, 0.1, 0.3) == pytest.approx(100_000 / 6)


def test_mcv_and_null_fraction():
    stats = cache_stats(_stats(np.linspace(0.3, 0.5, 11), mcv=[0.0], freqs=[0.01]))
    assert estimate_rows(stats, 0.3, 0.5) == pytest.approx(99_000)
    assert estimate_rows(stats, -1, 0) == pytest.approx(1_000)
    stats = cache_stats(_stats(np.linspace(0, 1, 11), null_frac=0.2))
    assert estimate_rows(stats, 0, 1) == pytest.approx(80_000)


def test_out_of_range_and_reversed():
    stats = cache_stats(_stats(np.linspace(0.3, 0.5, 11)))
    for lo, hi in [(0.6, 0.7), (0.0, 0.1), (0.45, 0.4)]:
        assert estimate_rows(stats, lo, hi) == 0.0


def test_monotone_in_window_width():
    stats = cache_stats(_stats(np.sort(np.random.default_rng(0).uniform(0.3, 0.5, 101))))
    estimates = [estimate_rows(stats, 0.3, 0.3 + width) for width in np.linspace(0.001, 0.2, 50)]
    assert all(right >= left for left, right in zip(estimates, estimates[1:]))


def test_snapshot_roundtrip(tmp_path):
    original = _stats(np.linspace(0, 1, 5), mcv=[0.0], freqs=[0.1])
    write_json(tmp_path / "stats.json", original)
    loaded = json.loads((tmp_path / "stats.json").read_text())
    assert stats_hash(original) == stats_hash(loaded)
    loaded["sha256"] = stats_hash(loaded)
    assert stats_hash(original) == stats_hash(loaded)


def test_router_boundary_inclusive():
    stats = cache_stats(_stats(np.linspace(0, 1, 101), reltuples=1000))
    assert route(stats, 100.0, 0.0, 0.1)[0] == "B1"
    assert route(stats, 100.0, 0.0, 0.1001)[0] == "B2-iter"
    assert route(stats, math.inf, 0, 1)[0] == "B1"
    assert route(stats, 0.0, 0.2, 0.9)[0] == "B2-iter"


def test_statistics_are_read_once_then_routing_uses_no_database():
    calls = []

    class Connection:
        def execute(self, sql, params=None):
            calls.append(sql)

            class Result:
                def fetchone(self):
                    return (1000.0, 10) if "pg_class" in sql else (0.0, -1.0, [0.0], [0.01], [0.3, 0.4, 0.5])

            return Result()

    stats = cache_stats(fetch_stats(Connection(), "items", "similarity"))
    for _ in range(100):
        route(stats, 10.0, 0.31, 0.32)
    assert len(calls) == 2
    overhead = measure_overhead_ns(stats, [(0.3, 0.4)], repeats=10)
    assert overhead["n"] == 10 and overhead["median_ns"] > 0
