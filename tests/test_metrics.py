import math

import numpy as np
import pytest
import pandas as pd

from fvsroute import metrics as M


def test_strict_recall_denominator_is_min_k_eligible():
    gt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert M.recall_strict([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], gt, 500, 10) == 1.0
    assert M.recall_strict([1, 2, 3, 4, 5, 11, 12, 13, 14, 15], gt, 500, 10) == 0.5
    assert M.recall_strict([1, 2, 3], gt[:3], 3, 10) == 1.0           # |E| = 3 < k
    assert M.recall_strict([1, 2], gt[:3], 3, 10) == 2 / 3            # missing one eligible row
    assert math.isnan(M.recall_strict([], [], 0, 10))                 # |E| = 0 -> undefined


def test_strict_recall_counts_only_first_k():
    assert M.recall_strict(list(range(12)), list(range(10)), 100, 10) == 1.0


def test_tolerant_recall_accepts_near_ties_only_inside_filter():
    in_f = np.array([True, True, False])
    d = np.array([0.1, 0.2000005, 0.15])
    assert M.recall_tolerant(in_f, d, kth_dist=0.2, n_eligible=100, k=3) == 2 / 3
    assert M.recall_tolerant(np.array([True]), np.array([0.3]), 0.2, 100, 1) == 0.0
    assert math.isnan(M.recall_tolerant(np.zeros(0, bool), np.zeros(0), float("nan"), 0, 10))


def test_completeness_definition():
    assert M.is_complete(10, 500, 10)
    assert not M.is_complete(4, 500, 10)
    assert M.is_complete(3, 3, 10)
    assert M.is_complete(0, 0, 10)
    assert not M.is_complete(1, 0, 10)


def _pc(n_q=10, methods=("A", "B")):
    rng = np.random.default_rng(0)
    rows = []
    for q in range(n_q):
        for lvl in range(3):
            for m in methods:
                rows.append({"config_id": f"q{q}_l{lvl}", "query_id": q, "level_idx": lvl, "target_sel": 0.1 * (lvl + 1),
                             "method": m, "latency_ms_median": rng.uniform(1, 2) + (1 if m == "B" else 0),
                             "recall": 1.0 if m == "A" else 0.8, "recall_tol": 1.0, "complete": 1.0,
                             "n_errors": 0})
    return pd.DataFrame(rows)


def test_summarize_percentiles_over_configs():
    pc = _pc()
    s = M.summarize(pc, ["method"]).set_index("method")
    a = pc[pc.method == "A"]["latency_ms_median"].to_numpy()
    assert s.loc["A", "p50_ms"] == np.percentile(a, 50)
    assert s.loc["A", "p95_ms"] == np.percentile(a, 95)
    assert s.loc["B", "recall"] == pytest.approx(0.8) and s.loc["A", "completeness"] == 1.0


def test_cluster_bootstrap_reproducible_and_brackets_point_estimate():
    pc = _pc(30)
    a = M.cluster_bootstrap(pc, ["A", "B"], "A", 300, seed=1)
    b = M.cluster_bootstrap(pc, ["A", "B"], "A", 300, seed=1)
    pd.testing.assert_frame_equal(a, b)
    row = a[(a.method == "A") & (a.stat == "mean_ms")].iloc[0]
    point = pc[pc.method == "A"]["latency_ms_median"].mean()
    assert row.ci_low <= point <= row.ci_high
    diff = a[(a.method == "A - B") & (a.stat == "mean_ms")].iloc[0]
    assert diff.ci_high < 0  # A is ~1 ms faster than B in every resample
