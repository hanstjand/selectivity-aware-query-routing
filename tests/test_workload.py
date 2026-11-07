import numpy as np

from fvsroute import workload as wl


def test_select_queries_dedupes_and_is_deterministic():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(50, 4)).astype(np.float32)
    q[10] = q[3]          # duplicate vector
    a = wl.select_queries(q, 20, seed=7)
    b = wl.select_queries(q, 20, seed=7)
    assert np.array_equal(a, b)
    assert len(set(a.tolist())) == 20
    assert 10 not in a.tolist()  # the later duplicate is never selected (first occurrence kept)
    assert not np.array_equal(a, wl.select_queries(q, 20, seed=8))


def test_split_is_query_level_disjoint_and_sized():
    qids = np.arange(300)
    s = wl.split_queries(qids, 0.7, seed=3)
    cal = [q for q, v in s.items() if v == "calibration"]
    test = [q for q, v in s.items() if v == "test"]
    assert len(cal) == 210 and len(test) == 90
    assert set(cal).isdisjoint(test) and set(cal) | set(test) == set(range(300))
    assert s == wl.split_queries(qids, 0.7, seed=3)


def _values_with_zero_block(n=10_000, zeros=15, ties=True):
    rng = np.random.default_rng(5)
    v = rng.uniform(0.3, 0.5, n)
    if ties:
        v[100:110] = v[99]  # a tie block
    v[:zeros] = 0.0
    return v


def test_filters_hit_target_counts_and_skip_imputed_block():
    v = _values_with_zero_block()
    sels = (0.001, 0.01, 0.1, 0.5, 0.9)
    qids = np.arange(30)
    split = {int(q): "calibration" for q in qids}
    f = wl.generate_filters(v, sels, qids, split, seed=1, imputed_value=0.0)
    assert len(f) == 30 * len(sels)
    _, counts = np.unique(v[v > 0], return_counts=True)
    tie_bound = 2 * (counts.max() - 1)
    excess = f["n_eligible"] - f["m_target"]
    assert (excess >= 0).all() and (excess <= tie_bound).all()
    assert (f["lo"] > 0).all(), "a window included imputed zeros"
    for r in f.itertuples():  # recorded count equals the SQL BETWEEN semantics
        assert r.n_eligible == int(wl.eligible_mask(v, r.lo, r.hi).sum())


def test_filters_are_deterministic_and_independent_of_query_count():
    v = _values_with_zero_block()
    sels = (0.01, 0.25)
    small = wl.generate_filters(v, sels, np.arange(5), {i: "test" for i in range(5)}, 9, 0.0)
    big = wl.generate_filters(v, sels, np.arange(50), {i: "test" for i in range(50)}, 9, 0.0)
    merged = small.merge(big, on="config_id", suffixes=("_s", "_b"))
    assert len(merged) == len(small)
    assert (merged["lo_s"] == merged["lo_b"]).all() and (merged["hi_s"] == merged["hi_b"]).all()


def test_tiny_selectivity_has_at_least_one_row():
    v = np.linspace(0.3, 0.5, 500)
    f = wl.generate_filters(v, (0.001,), np.arange(3), {i: "test" for i in range(3)}, 1, None)
    assert (f["m_target"] == 1).all() and (f["n_eligible"] >= 1).all()


def test_frame_sha_changes_with_content():
    import pandas as pd
    a = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    b = a.copy()
    b.loc[1, "x"] = 3
    assert wl.frame_sha256(a) == wl.frame_sha256(a.copy())
    assert wl.frame_sha256(a) != wl.frame_sha256(b)
