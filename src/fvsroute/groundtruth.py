"""Exact filtered top-k ground truth with NumPy.

Cosine distance is computed exactly as pgvector's `<=>` defines it:
    d(x, q) = 1 - (x . q) / sqrt(|x|^2 |q|^2),  similarity clipped to [-1, 1].
pgvector accumulates in float32; we accumulate in float64 from the same float32 inputs, so the two differ
only by float rounding (~1e-7). Ordering is (distance, id) ascending - a deterministic tie-break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIE_EPS = 1e-6


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return x64 / np.linalg.norm(x64, axis=1, keepdims=True)


def cosine_distances(base_normed: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """(n_queries, N) float64 cosine distances. `base_normed` must come from normalize_rows."""
    qn = normalize_rows(queries)
    sim = qn @ base_normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def topk_among(dist_row: np.ndarray, candidate_ids: np.ndarray, k: int):
    """Exact top-k of `candidate_ids` by (distance, id). Returns (ids, dists, next_dist or nan)."""
    n = candidate_ids.size
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.float64), float("nan")
    d = dist_row[candidate_ids]
    if n > k + 1:
        kth = np.partition(d, k)[k]          # (k+1)-th smallest distance
        keep = d <= kth                      # contains the top-k plus all boundary ties
        cand, dc = candidate_ids[keep], d[keep]
    else:
        cand, dc = candidate_ids, d
    order = np.lexsort((cand, dc))
    cand, dc = cand[order], dc[order]
    nxt = float(dc[k]) if dc.size > k else float("nan")
    return cand[:k].astype(np.int64), dc[:k], nxt


def compute_groundtruth(
    base: np.ndarray,
    values: np.ndarray,
    queries: dict[int, np.ndarray],
    workload: pd.DataFrame,
    k: int,
    chunk: int = 32,
) -> pd.DataFrame:
    """Exact filtered top-k for every workload config."""
    base_n = normalize_rows(base)
    sorted_idx = np.argsort(values, kind="stable")
    sorted_vals = values[sorted_idx]
    out = []
    qids = sorted(workload["query_id"].unique())
    by_q = {q: g for q, g in workload.groupby("query_id")}
    for start in range(0, len(qids), chunk):
        block = qids[start:start + chunk]
        dist = cosine_distances(base_n, np.stack([queries[q] for q in block]))
        for row_i, q in enumerate(block):
            drow = dist[row_i]
            for r in by_q[q].itertuples(index=False):
                left = np.searchsorted(sorted_vals, r.lo, side="left")
                right = np.searchsorted(sorted_vals, r.hi, side="right")
                cand = sorted_idx[left:right]
                ids, dists, nxt = topk_among(drow, cand, k)
                kth = float(dists[-1]) if dists.size else float("nan")
                out.append({
                    "config_id": r.config_id,
                    "gt_ids": ids.tolist(),
                    "gt_dists": dists.tolist(),
                    "gt_kth_dist": kth,
                    "gt_next_dist": nxt,
                    "n_eligible_gt": int(cand.size),
                    "tie_at_k": bool(np.isfinite(nxt) and (nxt - kth) < TIE_EPS),
                })
    return pd.DataFrame(out)


def validate_against_benchmark(
    base: np.ndarray,
    values: np.ndarray,
    test_queries: np.ndarray,
    cond_gt: np.ndarray,
    closest_ids: np.ndarray,
    closest_scores: np.ndarray,
) -> dict:
    """Reproduce Qdrant's exact `closest_ids` under the tests' original conditions.

    Validates the id mapping (row index == Qdrant point id), the vectors and the cosine formula.
    Qdrant's condition is `value > gt` (strict) or no filter.
    """
    k = closest_ids.shape[1]
    base_n = normalize_rows(base)
    overlaps, tie_overlaps, exact, score_err = [], [], 0, []
    for start in range(0, len(test_queries), 64):
        dist = cosine_distances(base_n, test_queries[start:start + 64])
        for j in range(dist.shape[0]):
            i = start + j
            gt = cond_gt[i]
            cand = np.flatnonzero(values > gt) if np.isfinite(gt) else np.arange(values.size)
            ids, dists, _ = topk_among(dist[j], cand, k)
            theirs = closest_ids[i]
            ov = len(set(ids.tolist()) & set(theirs.tolist())) / k
            overlaps.append(ov)
            exact += int(ov == 1.0)
            # tie-aware: one of Qdrant's ids also counts if it is eligible and exactly as close as our k-th
            eligible = (values[theirs] > gt) if np.isfinite(gt) else np.ones(k, bool)
            tied = eligible & (dist[j][theirs] <= dists[-1] + TIE_EPS)
            tie_overlaps.append(float(np.mean(np.isin(theirs, ids) | tied)))
            # Qdrant score = cosine similarity; compare with our distance on Qdrant's own ids
            score_err.append(float(np.max(np.abs((1.0 - closest_scores[i]) - dist[j][theirs]))))
    return {
        "n_queries": len(overlaps),
        "mean_overlap": float(np.mean(overlaps)),
        "min_overlap": float(np.min(overlaps)),
        "exact_match_fraction": exact / len(overlaps),
        "mean_overlap_tie_aware": float(np.mean(tie_overlaps)),
        "n_queries_with_mismatch": int(sum(o < 1.0 for o in overlaps)),
        "max_abs_score_diff": float(np.max(score_err)),
    }
