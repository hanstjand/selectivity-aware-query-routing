"""Workload generation: query selection, quantile range filters and the query-level split.

All randomness is derived from the global seed through numpy's SeedSequence-style integer lists,
so each (query, level) filter is independent of how many other queries/levels exist.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# stream identifiers keep the random streams of the different steps independent
_STREAM_QUERIES = 1
_STREAM_SPLIT = 2
_STREAM_FILTERS = 3


def select_queries(test_queries: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deduplicate test query vectors and draw `n` distinct ones.

    Returns the original tests.jsonl line indices (sorted); they serve as stable query ids.
    """
    _, first_idx = np.unique(test_queries, axis=0, return_index=True)
    unique_idx = np.sort(first_idx)
    if n > unique_idx.size:
        raise ValueError(f"requested {n} queries but only {unique_idx.size} distinct test vectors")
    rng = np.random.default_rng([seed, _STREAM_QUERIES])
    chosen = rng.choice(unique_idx, size=n, replace=False)
    return np.sort(chosen).astype(np.int64)


def split_queries(query_ids: np.ndarray, calibration_fraction: float, seed: int) -> dict[int, str]:
    """Split by query vector (not by query/filter pair) so no query vector appears in both splits."""
    rng = np.random.default_rng([seed, _STREAM_SPLIT])
    perm = rng.permutation(np.asarray(query_ids))
    n_cal = int(round(calibration_fraction * len(perm)))
    split = {int(q): "calibration" for q in perm[:n_cal]}
    split.update({int(q): "test" for q in perm[n_cal:]})
    return split


def generate_filters(
    values: np.ndarray,
    selectivities: tuple[float, ...],
    query_ids: np.ndarray,
    split: dict[int, str],
    seed: int,
    imputed_value: float | None,
) -> pd.DataFrame:
    """One closed range [lo, hi] per (query, selectivity level).

    A window of m = round(s * N) consecutive values of the sorted attribute is placed at a seeded
    random offset. Offsets start after the imputed block (rows equal to `imputed_value` are the
    nulls that Qdrant filled with 0) so those rows never fall inside a window; they stay in the table.
    The actual eligible count is recomputed exactly (boundary ties can only add rows).
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    sorted_vals = np.sort(values, kind="stable")
    start_min = 0
    if imputed_value is not None:
        start_min = int(np.searchsorted(sorted_vals, imputed_value, side="right")) \
            if (sorted_vals == imputed_value).any() else 0
        # only meaningful if the imputed value is the minimum (true for the 0.0 fill)
        if start_min and sorted_vals[0] != imputed_value:
            raise ValueError("imputed value is not the minimum of the attribute")

    rows = []
    for qid in np.asarray(query_ids, dtype=np.int64):
        for li, s in enumerate(selectivities):
            m = max(1, int(round(s * n)))
            hi_start = n - m  # last valid start index
            if hi_start < start_min:
                raise ValueError(f"selectivity {s} too large for the non-imputed range")
            rng = np.random.default_rng([seed, _STREAM_FILTERS, int(qid), li])
            a = int(rng.integers(start_min, hi_start + 1))
            lo = float(sorted_vals[a])
            hi = float(sorted_vals[a + m - 1])
            n_eligible = int(np.searchsorted(sorted_vals, hi, side="right")
                             - np.searchsorted(sorted_vals, lo, side="left"))
            rows.append({
                "config_id": f"q{int(qid)}_l{li}",
                "query_id": int(qid),
                "level_idx": li,
                "target_sel": s,
                "m_target": m,
                "lo": lo,
                "hi": hi,
                "n_eligible": n_eligible,
                "actual_sel": n_eligible / n,
                "split": split[int(qid)],
            })
    return pd.DataFrame(rows)


def eligible_mask(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Exactly the SQL predicate `value BETWEEN lo AND hi` (closed interval)."""
    return (values >= lo) & (values <= hi)


def frame_sha256(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Content hash of a DataFrame (column order and values), independent of parquet metadata."""
    cols = columns or list(df.columns)
    h = hashlib.sha256()
    h.update(",".join(cols).encode())
    h.update(pd.util.hash_pandas_object(df[cols].astype(str), index=False).values.tobytes())
    return h.hexdigest()
