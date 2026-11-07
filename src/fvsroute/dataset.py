"""Dataset acquisition, loading and integrity validation (Qdrant laion-small-clip).

Facts verified while planning (re-checked here at run time):
  * vectors.npy   float16, shape (100000, 512), approximately unit norm
  * payloads.jsonl one JSON object per vector; `similarity` = LAION CLIP image-text score
  * tests.jsonl   5000 queries {query, conditions, closest_ids, closest_scores}; closest_ids are
                  row indices into vectors.npy (Qdrant uploaded with ids = range(N)), computed with exact search.
"""

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

log = logging.getLogger(__name__)

MEMBERS = ("vectors.npy", "payloads.jsonl", "tests.jsonl")


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, corrupt or inconsistent."""


# --------------------------------------------------------------------------- download
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


# --------------------------------------------------------------------------- loading
@dataclass
class RawDataset:
    vectors: np.ndarray          # float16 (N, d) exactly as distributed
    filter_values: np.ndarray    # float64 (N,)
    test_queries: np.ndarray     # float32 (T, d)
    test_cond_gt: np.ndarray     # float64 (T,), NaN when the test is unfiltered
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


# --------------------------------------------------------------------------- validation
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
    nz = raw.filter_values[raw.filter_values != imputed_value] if imputed_value is not None else raw.filter_values
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
