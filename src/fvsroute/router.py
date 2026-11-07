"""P1: selectivity-aware router.

    est_rows = reltuples * s_hat(lo, hi)          (cached pg_stats, no COUNT(*), no DB call)
    est_rows <= T  ->  B1 (exact)
    est_rows >  T  ->  B2-iter (HNSW with iterative scan)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .estimator import ColumnStats, SelectivityEstimator, fetch_stats

EXACT_ARM = "B1"
HNSW_ARM = "B2-iter"


@dataclass
class Router:
    estimator: SelectivityEstimator
    threshold: float
    exact_arm: str = EXACT_ARM
    hnsw_arm: str = HNSW_ARM

    @classmethod
    def from_connection(cls, conn, table: str, column: str, threshold: float) -> "Router":
        """Load statistics once (two catalog queries) and cache them for the router's lifetime."""
        return cls(SelectivityEstimator(fetch_stats(conn, table, column)), threshold)

    @classmethod
    def from_snapshot(cls, path: Path, threshold: float) -> "Router":
        return cls(SelectivityEstimator(ColumnStats.load(path)), threshold)

    def route(self, lo: float, hi: float) -> tuple[str, float]:
        est = self.estimator.estimate_rows(lo, hi)
        return (self.exact_arm if est <= self.threshold else self.hnsw_arm), est


def measure_overhead_ns(router: Router, filters: list[tuple[float, float]], repeats: int = 20) -> dict:
    """In-process cost of estimate + decision, per call (used in calibration and reported)."""
    samples = []
    for _ in range(repeats):
        for lo, hi in filters:
            t0 = time.perf_counter_ns()
            router.route(lo, hi)
            samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    n = len(samples)
    return {"n": n, "median_ns": samples[n // 2], "p95_ns": samples[int(0.95 * (n - 1))],
            "mean_ns": sum(samples) / n}


def load_frozen(path: Path) -> dict:
    return json.loads(Path(path).read_text())
