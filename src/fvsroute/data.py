"""Read the dataset, generate filters, and compute exact reference answers."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
MEMBERS = ("vectors.npy", "payloads.jsonl", "tests.jsonl")
TIE_EPS = 1e-6
_STREAM_QUERIES, _STREAM_SPLIT, _STREAM_FILTERS = 1, 2, 3


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, corrupt or inconsistent."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> Path:
    """Download `url` to `dest` (skipped if already present with the right hash)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        got = sha256_file(dest)
        if got == expected_sha256:
            log.info("archive present, sha256 OK: %s", dest)
            return dest
        log.warning("archive hash mismatch (%s); re-downloading", got)
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "fvsroute-benchmark"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    got = sha256_file(tmp)
    if got != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise DatasetError(f"sha256 mismatch: expected {expected_sha256}, got {got}")
    tmp.rename(dest)
    log.info("downloaded %s (%d bytes), sha256 OK", dest, dest.stat().st_size)
    return dest


def extract(archive: Path, out_dir: Path) -> Path:
    """Extract only the three expected regular files; reject anything else in the archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        names = sorted(m.name for m in members)
        if names != sorted(MEMBERS):
            raise DatasetError(f"unexpected archive members: {names}")
        for m in members:
            if not m.isfile():
                raise DatasetError(f"archive member is not a regular file: {m.name}")
        for m in members:
            target = out_dir / m.name
            if target.exists() and target.stat().st_size == m.size:
                continue
            tf.extract(m, out_dir, filter="data")
    return out_dir


@dataclass
class RawDataset:
    vectors: np.ndarray  # float16 (N, d) exactly as distributed
    filter_values: np.ndarray  # float64 (N,)
    test_queries: np.ndarray  # float32 (T, d)
    test_cond_gt: np.ndarray  # float64 (T,), NaN when the test is unfiltered
    test_closest_ids: np.ndarray  # int64 (T, k)
    test_closest_scores: np.ndarray  # float64 (T, k)

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])


def _parse_condition(cond: dict, field: str) -> float:
    """Return the `gt` threshold of a Qdrant range condition on `field`, NaN if unfiltered."""
    if not cond:
        return float("nan")
    clauses = cond.get("and", [])
    if len(clauses) != 1 or field not in clauses[0]:
        raise DatasetError(f"unsupported condition: {cond}")
    rng = clauses[0][field].get("range", {})
    if set(rng) != {"gt"}:
        raise DatasetError(f"unsupported range condition: {rng}")
    return float(rng["gt"])


def load_raw(raw_dir: Path, filter_field: str = "similarity") -> RawDataset:
    vec_path = raw_dir / "vectors.npy"
    if not vec_path.exists():
        raise DatasetError(f"{vec_path} missing; run `fvs download` first")
    vectors = np.load(vec_path, allow_pickle=False)

    values = []
    with open(raw_dir / "payloads.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            v = rec.get(filter_field)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise DatasetError(f"payload row {i}: {filter_field}={v!r} is not numeric")
            values.append(float(v))
    filter_values = np.asarray(values, dtype=np.float64)

    q, gt, ids, scores = [], [], [], []
    with open(raw_dir / "tests.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            q.append(rec["query"])
            gt.append(_parse_condition(rec["conditions"], filter_field))
            ids.append(rec["closest_ids"])
            scores.append(rec["closest_scores"])
    lengths = {len(x) for x in ids}
    if len(lengths) != 1:
        raise DatasetError(f"tests have varying closest_ids lengths: {sorted(lengths)}")
    return RawDataset(
        vectors=vectors,
        filter_values=filter_values,
        test_queries=np.asarray(q, dtype=np.float32),
        test_cond_gt=np.asarray(gt, dtype=np.float64),
        test_closest_ids=np.asarray(ids, dtype=np.int64),
        test_closest_scores=np.asarray(scores, dtype=np.float64),
    )


def validate_raw(raw: RawDataset, dim: int, imputed_value: float | None) -> dict:
    """Hard integrity checks. Returns a JSON-serialisable report; raises DatasetError on failure."""
    errors: list[str] = []
    v = raw.vectors
    if v.ndim != 2 or v.shape[1] != dim:
        errors.append(f"vectors shape {v.shape}, expected (N, {dim})")
    if not np.issubdtype(v.dtype, np.floating):
        errors.append(f"vectors dtype {v.dtype} is not floating")
    v32 = v.astype(np.float32)
    if not np.isfinite(v32).all():
        errors.append("vectors contain non-finite values")
    if v.dtype == np.float16 and not np.array_equal(v32.astype(np.float16), v):
        errors.append("float16 -> float32 conversion is not lossless")
    norms = np.linalg.norm(v32.astype(np.float64), axis=1)
    if (norms == 0).any():
        errors.append(f"{int((norms == 0).sum())} zero vectors (cosine undefined)")
    if raw.filter_values.shape[0] != raw.n:
        errors.append(f"{raw.filter_values.shape[0]} payload rows != {raw.n} vectors")
    if not np.isfinite(raw.filter_values).all():
        errors.append("filter values contain non-finite entries")
    tq = raw.test_queries
    if tq.ndim != 2 or tq.shape[1] != dim:
        errors.append(f"test query shape {tq.shape}")
    if not np.isfinite(tq).all():
        errors.append("test queries contain non-finite values")
    ids = raw.test_closest_ids
    if ids.size and (ids.min() < 0 or ids.max() >= raw.n):
        errors.append(f"closest_ids out of range [{ids.min()}, {ids.max()}]")
    if errors:
        raise DatasetError("; ".join(errors))

    _, first_idx = np.unique(tq, axis=0, return_index=True)
    n_imputed = int((raw.filter_values == imputed_value).sum()) if imputed_value is not None else 0
    nz = (
        raw.filter_values[raw.filter_values != imputed_value]
        if imputed_value is not None
        else raw.filter_values
    )
    vals, counts = np.unique(nz, return_counts=True)
    return {
        "n_vectors": raw.n,
        "dim": int(v.shape[1]),
        "dtype_source": str(v.dtype),
        "dtype_loaded": "float32",
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "filter_min": float(raw.filter_values.min()),
        "filter_max": float(raw.filter_values.max()),
        "filter_distinct": int(np.unique(raw.filter_values).size),
        "filter_imputed_value": imputed_value,
        "filter_imputed_rows": n_imputed,
        "filter_nonimputed_min": float(nz.min()),
        "filter_max_tie_nonimputed": int(counts.max()),
        "n_tests": int(tq.shape[0]),
        "n_tests_distinct": int(first_idx.size),
        "n_tests_filtered": int(np.isfinite(raw.test_cond_gt).sum()),
        "closest_ids_min": int(ids.min()),
        "closest_ids_max": int(ids.max()),
    }


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
    if imputed_value is not None and (sorted_vals == imputed_value).any():
        # Skip the zero-filled nulls, which must be at the start of the sorted values.
        start_min = int(np.searchsorted(sorted_vals, imputed_value, side="right"))
        if sorted_vals[0] != imputed_value:
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
            n_eligible = int(
                np.searchsorted(sorted_vals, hi, side="right") - np.searchsorted(sorted_vals, lo, side="left")
            )
            rows.append(
                {
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
                }
            )
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
    if candidate_ids.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.float64), float("nan")
    distances = dist_row[candidate_ids]
    if candidate_ids.size > k + 1:
        # Keep an extra neighbor and all boundary ties before sorting by distance and ID.
        cutoff = np.partition(distances, k)[k]
        keep = distances <= cutoff
        candidate_ids = candidate_ids[keep]
        distances = distances[keep]
    order = np.lexsort((candidate_ids, distances))
    candidate_ids = candidate_ids[order]
    distances = distances[order]
    next_distance = float(distances[k]) if distances.size > k else float("nan")
    return candidate_ids[:k].astype(np.int64), distances[:k], next_distance


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
        block = qids[start : start + chunk]
        dist = cosine_distances(base_n, np.stack([queries[q] for q in block]))
        for row_i, q in enumerate(block):
            drow = dist[row_i]
            for r in by_q[q].itertuples(index=False):
                left = np.searchsorted(sorted_vals, r.lo, side="left")
                right = np.searchsorted(sorted_vals, r.hi, side="right")
                cand = sorted_idx[left:right]
                ids, dists, nxt = topk_among(drow, cand, k)
                kth = float(dists[-1]) if dists.size else float("nan")
                out.append(
                    {
                        "config_id": r.config_id,
                        "gt_ids": ids.tolist(),
                        "gt_dists": dists.tolist(),
                        "gt_kth_dist": kth,
                        "gt_next_dist": nxt,
                        "n_eligible_gt": int(cand.size),
                        "tie_at_k": bool(np.isfinite(nxt) and (nxt - kth) < TIE_EPS),
                    }
                )
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
        dist = cosine_distances(base_n, test_queries[start : start + 64])
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
