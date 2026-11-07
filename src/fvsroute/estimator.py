"""Selectivity estimation for `col BETWEEN lo AND hi` from PostgreSQL's own statistics (pg_stats).

Mirrors the structure of PostgreSQL's scalarineqsel() / clauselist range combination:
    P(x <= c) = sum(MCV freqs with value <= c) + (1 - null_frac - sum(MCV freqs)) * hist_frac(c)
    sel(lo <= x <= hi) = P(x <= hi) - P(x < lo)                      (clamped to [0, 1])
where hist_frac(c) is the fraction of the equi-depth histogram population below c, interpolating
linearly inside the bucket that contains c. Estimated rows = reltuples * sel.

The statistics are read once (after ANALYZE) and cached in memory; estimates need no database call.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ColumnStats:
    table: str
    column: str
    reltuples: float
    relpages: int
    null_frac: float
    n_distinct: float
    mcv_vals: tuple[float, ...]
    mcv_freqs: tuple[float, ...]
    hist_bounds: tuple[float, ...]

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**asdict(self), "sha256": self.sha256()}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "ColumnStats":
        d = json.loads(Path(path).read_text())
        d.pop("sha256", None)
        for k in ("mcv_vals", "mcv_freqs", "hist_bounds"):
            d[k] = tuple(d[k])
        return cls(**d)


def fetch_stats(conn, table: str, column: str) -> ColumnStats:
    """Read pg_class + pg_stats (one round trip each). Raises if the column has never been analyzed."""
    rel = conn.execute("SELECT reltuples, relpages FROM pg_class WHERE oid = %s::regclass", (table,)).fetchone()
    row = conn.execute(
        """SELECT null_frac, n_distinct,
                  most_common_vals::text::float8[], most_common_freqs::float8[],
                  histogram_bounds::text::float8[]
             FROM pg_stats
            WHERE schemaname = current_schema() AND tablename = %s AND attname = %s""",
        (table, column),
    ).fetchone()
    if rel is None or row is None:
        raise RuntimeError(f"no statistics for {table}.{column}; run ANALYZE first")
    null_frac, n_distinct, mcv_vals, mcv_freqs, hist = row
    return ColumnStats(
        table=table, column=column, reltuples=float(rel[0]), relpages=int(rel[1]),
        null_frac=float(null_frac), n_distinct=float(n_distinct),
        mcv_vals=tuple(float(v) for v in (mcv_vals or [])),
        mcv_freqs=tuple(float(f) for f in (mcv_freqs or [])),
        hist_bounds=tuple(float(b) for b in (hist or [])),
    )


class SelectivityEstimator:
    def __init__(self, stats: ColumnStats):
        self.stats = stats
        pairs = sorted(zip(stats.mcv_vals, stats.mcv_freqs))
        self._mcv_vals = [p[0] for p in pairs]
        cum, acc = [0.0], 0.0
        for _, f in pairs:
            acc += f
            cum.append(acc)
        self._mcv_cum = cum                      # cum[i] = sum of freqs of the i smallest MCVs
        self._mcv_total = acc
        self._hist = list(stats.hist_bounds)
        self._nb = len(self._hist) - 1           # number of buckets
        self._hist_weight = max(0.0, 1.0 - stats.null_frac - self._mcv_total)
        self._reltuples = stats.reltuples

    def _hist_frac(self, c: float) -> float:
        h, nb = self._hist, self._nb
        if nb < 1:
            return 0.5  # no histogram: PostgreSQL falls back to a default; never hit for our column
        if c <= h[0]:
            return 0.0
        if c >= h[-1]:
            return 1.0
        i = bisect.bisect_right(h, c) - 1        # h[i] <= c < h[i+1]
        lo, hi = h[i], h[i + 1]
        within = (c - lo) / (hi - lo) if hi > lo else 0.5
        return (i + within) / nb

    def _mcv_le(self, c: float) -> float:
        return self._mcv_cum[bisect.bisect_right(self._mcv_vals, c)]

    def _mcv_lt(self, c: float) -> float:
        return self._mcv_cum[bisect.bisect_left(self._mcv_vals, c)]

    def selectivity(self, lo: float, hi: float) -> float:
        if hi < lo:
            return 0.0
        p_le_hi = self._mcv_le(hi) + self._hist_weight * self._hist_frac(hi)
        p_lt_lo = self._mcv_lt(lo) + self._hist_weight * self._hist_frac(lo)
        return min(1.0, max(0.0, p_le_hi - p_lt_lo))

    def estimate_rows(self, lo: float, hi: float) -> float:
        return self._reltuples * self.selectivity(lo, hi)
