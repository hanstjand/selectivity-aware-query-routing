import numpy as np
import pandas as pd

from fvsroute import data as gt


def _naive_topk(base, values, q, lo, hi, k):
    rows = []
    for i in range(base.shape[0]):
        if lo <= values[i] <= hi:
            x = base[i].astype(np.float64)
            qq = q.astype(np.float64)
            d = 1 - np.dot(x, qq) / np.sqrt(np.dot(x, x) * np.dot(qq, qq))
            rows.append((d, i))
    rows.sort()
    return [i for _, i in rows[:k]]


def _workload(values, qids, ranges):
    recs = []
    for q in qids:
        for li, (lo, hi) in enumerate(ranges):
            recs.append(
                {
                    "config_id": f"q{q}_l{li}",
                    "query_id": q,
                    "level_idx": li,
                    "lo": lo,
                    "hi": hi,
                    "n_eligible": int(((values >= lo) & (values <= hi)).sum()),
                }
            )
    return pd.DataFrame(recs)


def test_matches_naive_bruteforce_on_unnormalised_vectors():
    rng = np.random.default_rng(0)
    base = (rng.normal(size=(400, 16)) * rng.uniform(0.2, 5, (400, 1))).astype(np.float32)
    values = rng.uniform(0, 1, 400)
    queries = {q: (rng.normal(size=16) * 3).astype(np.float32) for q in range(5)}
    ranges = [(0.0, 1.0), (0.2, 0.3), (0.5, 0.52), (0.9, 0.9001)]
    w = _workload(values, list(queries), ranges)
    res = gt.compute_groundtruth(base, values, queries, w, k=10, chunk=2).set_index("config_id")
    for r in w.itertuples():
        want = _naive_topk(base, values, queries[r.query_id], r.lo, r.hi, 10)
        assert list(res.loc[r.config_id, "gt_ids"]) == want
        assert res.loc[r.config_id, "n_eligible_gt"] == r.n_eligible


def test_ties_broken_by_id_and_flagged():
    base = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)  # rows 0,1,2 identical
    values = np.ones(5)
    w = _workload(values, [0], [(0, 2)])
    res = gt.compute_groundtruth(base, values, {0: np.array([1, 0], np.float32)}, w, k=2)
    assert list(res.loc[0, "gt_ids"]) == [0, 1]
    assert bool(res.loc[0, "tie_at_k"])  # row 2 has the same distance as the 2nd neighbour


def test_fewer_than_k_and_zero_eligible():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(50, 4)).astype(np.float32)
    values = np.arange(50, dtype=float)
    w = _workload(values, [0], [(10, 12), (100, 200)])
    res = gt.compute_groundtruth(base, values, {0: rng.normal(size=4).astype(np.float32)}, w, k=10)
    r = res.set_index("config_id")
    assert sorted(r.loc["q0_l0", "gt_ids"]) == [10, 11, 12]
    assert list(r.loc["q0_l1", "gt_ids"]) == [] and r.loc["q0_l1", "n_eligible_gt"] == 0
    assert np.isnan(r.loc["q0_l1", "gt_kth_dist"])


def test_cosine_matches_pgvector_definition():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 8)).astype(np.float32)
    q = rng.normal(size=(3, 8)).astype(np.float32)
    d = gt.cosine_distances(gt.normalize_rows(x), q)
    for i in range(3):
        for j in range(30):
            ref = 1 - np.dot(x[j], q[i]) / np.sqrt(np.dot(x[j], x[j]) * np.dot(q[i], q[i]))
            assert abs(d[i, j] - ref) < 1e-6
    assert d.min() >= 0 and d.max() <= 2


def test_benchmark_validation_detects_wrong_ids():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(200, 8)).astype(np.float32)
    values = rng.uniform(0.3, 0.4, 200)
    qs = rng.normal(size=(6, 8)).astype(np.float32)
    cond = np.array([np.nan, 0.35, np.nan, 0.32, 0.38, np.nan])
    bn = gt.normalize_rows(base)
    ids, scores = [], []
    for i in range(6):
        cand = np.flatnonzero(values > cond[i]) if np.isfinite(cond[i]) else np.arange(200)
        top, d, _ = gt.topk_among(gt.cosine_distances(bn, qs[i : i + 1])[0], cand, 5)
        ids.append(top)
        scores.append(1 - d)
    ids, scores = np.array(ids), np.array(scores)
    ok = gt.validate_against_benchmark(base, values, qs, cond, ids, scores)
    assert ok["mean_overlap"] == 1.0 and ok["max_abs_score_diff"] < 1e-9
    bad_ids = ids.copy()
    bad_ids[:, 0] = (bad_ids[:, 0] + 1) % 200  # shifted id mapping
    bad = gt.validate_against_benchmark(base, values, qs, cond, bad_ids, scores)
    assert bad["mean_overlap"] < 1.0
