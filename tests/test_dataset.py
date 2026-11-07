import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from fvsroute import dataset
from fvsroute.dataset import DatasetError


def _make_raw(tmp_path, n=20, d=8, n_tests=4, vec_dtype=np.float16, values=None, ids_shift=0, extra=None):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(n, d)).astype(vec_dtype)
    values = values if values is not None else rng.uniform(0.3, 0.5, n)
    files = {}
    buf = io.BytesIO()
    np.save(buf, vecs)
    files["vectors.npy"] = buf.getvalue()
    files["payloads.jsonl"] = "".join(json.dumps({"similarity": float(v), "caption": "x"}) + "\n"
                                      for v in values).encode()
    tests = []
    for i in range(n_tests):
        cond = {"and": [{"similarity": {"range": {"gt": 0.35}}}]} if i % 2 else {}
        tests.append({"query": rng.normal(size=d).astype(np.float16).astype(float).tolist(), "conditions": cond,
                      "closest_ids": [int(j + ids_shift) for j in range(3)], "closest_scores": [0.9, 0.8, 0.7]})
    files["tests.jsonl"] = "".join(json.dumps(t) + "\n" for t in tests).encode()
    if extra:
        files.update(extra)
    archive = tmp_path / "ds.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return archive, vecs


def test_extract_load_validate_roundtrip(tmp_path):
    archive, vecs = _make_raw(tmp_path)
    out = dataset.extract(archive, tmp_path / "raw")
    raw = dataset.load_raw(out)
    assert raw.vectors.dtype == np.float16 and raw.vectors.shape == (20, 8)
    rep = dataset.validate_raw(raw, dim=8, imputed_value=0.0)
    assert rep["n_vectors"] == 20 and rep["n_tests"] == 4 and rep["n_tests_filtered"] == 2
    # float16 -> float32 is exact
    assert np.array_equal(raw.vectors.astype(np.float32).astype(np.float16), vecs)
    assert np.isnan(raw.test_cond_gt[0]) and raw.test_cond_gt[1] == 0.35


def test_extract_rejects_unexpected_members(tmp_path):
    archive, _ = _make_raw(tmp_path, extra={"evil.sh": b"echo hi"})
    with pytest.raises(DatasetError, match="unexpected archive members"):
        dataset.extract(archive, tmp_path / "raw")


def test_validate_catches_dimension_mismatch(tmp_path):
    archive, _ = _make_raw(tmp_path)
    raw = dataset.load_raw(dataset.extract(archive, tmp_path / "raw"))
    with pytest.raises(DatasetError, match="shape"):
        dataset.validate_raw(raw, dim=16, imputed_value=0.0)


def test_validate_catches_nonfinite_and_count_mismatch(tmp_path):
    archive, _ = _make_raw(tmp_path)
    raw = dataset.load_raw(dataset.extract(archive, tmp_path / "raw"))
    raw.vectors = raw.vectors.copy()
    raw.vectors[0, 0] = np.nan
    raw.filter_values = raw.filter_values[:-1]
    with pytest.raises(DatasetError) as e:
        dataset.validate_raw(raw, dim=8, imputed_value=0.0)
    assert "non-finite" in str(e.value) and "payload rows" in str(e.value)


def test_validate_catches_out_of_range_ids(tmp_path):
    archive, _ = _make_raw(tmp_path, ids_shift=100)
    raw = dataset.load_raw(dataset.extract(archive, tmp_path / "raw"))
    with pytest.raises(DatasetError, match="closest_ids out of range"):
        dataset.validate_raw(raw, dim=8, imputed_value=0.0)


def test_non_numeric_payload_rejected(tmp_path):
    archive, _ = _make_raw(tmp_path)
    out = dataset.extract(archive, tmp_path / "raw")
    (out / "payloads.jsonl").write_text('{"similarity": "high"}\n')
    with pytest.raises(DatasetError, match="not numeric"):
        dataset.load_raw(out)


def test_download_verifies_sha256(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"benchmark-bytes")
    good = hashlib.sha256(b"benchmark-bytes").hexdigest()
    dest = tmp_path / "dl" / "a.tgz"
    dataset.download(src.as_uri(), dest, good)
    assert dest.read_bytes() == b"benchmark-bytes"
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        dataset.download(src.as_uri(), tmp_path / "dl" / "b.tgz", "0" * 64)
    assert not (tmp_path / "dl" / "b.tgz").exists()
