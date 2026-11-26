import math

import numpy as np
import pandas as pd
import pytest

from fvsroute.benchmark import calibrate, candidate_thresholds
from fvsroute.config import load_config
from fvsroute.benchmark import gain_captured, oracle_floor

from .conftest import SMOKE

LEVELS = [0.001, 0.01, 0.1, 0.5, 0.9]
N = 100_000


def _pc(split="calibration", b2_recall_broad=1.0, n_q=20):
    """Synthetic per-config table: B1 latency grows with selectivity, B2-iter shrinks; B2 recall is poor
    for selective filters - so the optimal threshold lies between the 1% and 10% levels."""
    rows = []
    for q in range(n_q):
        for li, s in enumerate(LEVELS):
            cid = f"q{q}_l{li}"
            common = {
                "config_id": cid,
                "query_id": q,
                "level_idx": li,
                "target_sel": s,
                "split": split,
                "n_eligible": int(s * N),
            }
            rows.append(
                {**common, "method": "B1", "latency_ms_median": 0.3 + 20 * s, "recall": 1.0, "complete": 1.0}
            )
            b2_rec = {0.001: 0.6, 0.01: 0.9}.get(s, 1.0 if s < 0.9 else b2_recall_broad)
            rows.append(
                {
                    **common,
                    "method": "B2-iter",
                    "latency_ms_median": 5.0 if s < 0.05 else 1.0,
                    "recall": b2_rec,
                    "complete": 1.0,
                }
            )
            rows.append(
                {
                    **common,
                    "method": "B2-iter-ef200",
                    "latency_ms_median": 6.0 if s < 0.05 else 1.2,
                    "recall": min(1.0, b2_rec + 0.05),
                    "complete": 1.0,
                }
            )
    pc = pd.DataFrame(rows)
    est = pd.Series({r.config_id: r.target_sel * N for r in pc.drop_duplicates("config_id").itertuples()})
    return pc, est


@pytest.fixture(scope="module")
def cfg():
    return load_config(SMOKE)


def test_candidate_thresholds_cover_extremes():
    c = candidate_thresholds(np.array([10.0, 100.0, 100.0, 1000.0]))
    assert c[0] == 0.0 and math.isinf(c[-1])
    assert c[1] == pytest.approx(math.sqrt(10 * 100)) and c[2] == pytest.approx(math.sqrt(100 * 1000))


def test_calibration_finds_crossover_and_respects_constraints(cfg):
    pc, est = _pc()
    frozen, curve = calibrate(cfg, pc, est, {"median_ns": 5000}, "sha")
    T = frozen["threshold_T"]
    assert 1_000 < T < 10_000, T  # between the 1% and 10% estimates
    assert frozen["calibration_at_T"]["recall"] >= 0.95
    assert frozen["hnsw"]["ef_search"] == cfg.hnsw.ef_search
    # every infeasible threshold is flagged; T = inf is always feasible
    assert curve.loc[curve["T"] == math.inf, "feasible"].iloc[0]
    assert (curve[curve["T"] == 0]["recall"] < 0.95).all() == (~curve[curve["T"] == 0]["feasible"]).all()


def test_calibration_rejects_test_rows(cfg):
    pc, est = _pc(split="test")
    with pytest.raises(ValueError, match="held-out"):
        calibrate(cfg, pc, est, {"median_ns": 5000}, "sha")


def _test_pc():
    rows = []
    #            cfg   B1 lat  B2 lat  B2 recall  B2 complete
    data = [
        ("c1", 1.0, 3.0, 1.0, 1.0),  # B1 faster
        ("c2", 4.0, 1.0, 1.0, 1.0),  # B2 faster and valid
        ("c3", 4.0, 1.0, 0.9, 1.0),  # B2 faster but inexact
        ("c4", 4.0, 1.0, 1.0, 0.0),
    ]  # B2 faster but incomplete
    for i, (cid, l1, l2, r2, c2) in enumerate(data):
        common = {
            "config_id": cid,
            "query_id": i,
            "level_idx": 0,
            "target_sel": 0.1,
            "actual_sel": 0.1,
            "n_eligible": 100,
            "split": "test",
        }
        rows.append({**common, "method": "B1", "latency_ms_median": l1, "recall": 1.0, "complete": 1.0})
        rows.append({**common, "method": "B2-iter", "latency_ms_median": l2, "recall": r2, "complete": c2})
    return pd.DataFrame(rows)


def test_oracle_floor_ignores_quality():
    o = oracle_floor(_test_pc()).set_index("config_id")
    assert o["choice"].tolist() == ["B1", "B2-iter", "B2-iter", "B2-iter"]
    assert o["latency_ms_median"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert o.loc["c3", "recall"] == 0.9 and o.loc["c4", "complete"] == 0.0  # quality reported, not enforced


def test_oracle_floor_is_a_lower_bound_for_any_router():
    pc = _test_pc()
    o = oracle_floor(pc)["latency_ms_median"].to_numpy()
    lat = pc.pivot(index="config_id", columns="method", values="latency_ms_median").to_numpy()
    rng = np.random.default_rng(0)
    for _ in range(20):
        pick = lat[np.arange(4), rng.integers(0, 2, 4)]
        assert (o <= pick).all()
        for stat in (np.mean, lambda x: np.percentile(x, 50), lambda x: np.percentile(x, 95)):
            assert stat(o) <= stat(pick)


def test_gain_captured_cases():
    assert gain_captured(10.0, 6.0, 2.0) == (0.5, "")
    assert gain_captured(10.0, 10.0, 2.0)[0] == 0.0
    assert gain_captured(10.0, 1.0, 2.0)[0] > 1.0  # the router beats the floor only by noise or a quality trade-off
    assert gain_captured(None, 6.0, 2.0)[0] is None
    assert gain_captured(2.0, 3.0, 2.0)[0] is None  # no headroom
    assert gain_captured(2.0, 2.0, 2.0 - 1e-12)[0] is None  # numerically zero headroom
    assert gain_captured(10.0, 9.99, 9.95)[0] is None  # headroom < 1% of best
