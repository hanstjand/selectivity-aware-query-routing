"""Exercise the complete timing/calibration path with a deterministic database stand-in."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fvsroute import benchmark, data, prepare, search
from fvsroute.config import load_config, write_json
from .conftest import SMOKE


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    cfg = replace(load_config(SMOKE), results_dir=tmp_path, run_id="test")
    cfg.workload.k = 2
    cfg.dataset.n_rows = 4
    vectors = np.array([[1, 0], [0, 1], [0.7, 0.7], [-1, 0]], dtype=np.float32)
    values = np.array([0.1, 0.2, 0.4, 0.8])
    queries = {0: vectors[0], 1: vectors[1], 2: vectors[2], 3: vectors[3]}
    split = {0: "calibration", 1: "calibration", 2: "test", 3: "test"}
    workload = data.generate_filters(values, (0.25, 0.75), np.arange(4), split, cfg.seed, None)
    truth = data.compute_groundtruth(vectors, values, queries, workload, k=2)
    prepared = SimpleNamespace(
        workload=workload,
        gt=truth,
        queries=queries,
        values=values,
        query_ids=np.arange(4),
        base=lambda cfg: vectors,
    )
    monkeypatch.setattr(prepare, "load_processed", lambda cfg: prepared)
    monkeypatch.setattr(prepare, "prewarm", lambda conn, table: None)
    monkeypatch.setattr(search, "stats_fingerprint", lambda conn, table: "stable")
    stats = dict(
        table="items",
        column="similarity",
        reltuples=4.0,
        relpages=1,
        null_frac=0.0,
        n_distinct=-1.0,
        mcv_vals=[],
        mcv_freqs=[],
        hist_bounds=[0.0, 1.0],
    )
    write_json(cfg.run_dir / "env/stats_snapshot.json", stats)
    write_json(cfg.run_dir / "env/db_setup.json", {"stats_fingerprint": "stable"})
    answers = {}
    ground_truth = truth.set_index("config_id")
    for row in workload.itertuples(index=False):
        answers[row.lo, row.hi, queries[row.query_id].tobytes()] = ground_truth.loc[row.config_id, "gt_ids"]
    connections = []

    class Connection:
        def __init__(self, settings):
            self.settings = {
                "enable_indexscan": "on",
                "enable_seqscan": "on",
                "enable_bitmapscan": "on",
                "jit": "off",
                **settings,
            }
            self.closed = False

        def execute(self, sql):
            return SimpleNamespace(fetchone=lambda: (self.settings[sql.removeprefix("SHOW ")],))

        def cursor(self):
            return Cursor(self)

        def close(self):
            self.closed = True

    class Cursor:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, params):
            if sql.startswith("EXPLAIN"):
                hnsw = self.connection.settings.get("enable_seqscan") == "off"
                node = {"Node Type": "Index Scan" if hnsw else "Seq Scan", "Relation Name": "items"}
                if hnsw:
                    node["Index Name"] = "items_embedding_hnsw"
                self.result = ([{"Plan": node}],)
            else:
                key = params["lo"], params["hi"], params["q"].to_numpy().tobytes()
                self.result = [(item,) for item in answers[key]]

        def fetchone(self):
            return self.result

        def fetchall(self):
            return self.result

    def connect(database, settings):
        connection = Connection(settings)
        connections.append(connection)
        return connection

    monkeypatch.setattr(search, "connect", connect)
    return cfg, prepared, connections


def test_calibration_and_held_out_timing_produce_required_results(experiment):
    cfg, prepared, connections = experiment
    benchmark.step_bench(cfg, "calibration")
    benchmark.step_calibrate(cfg)
    benchmark.step_bench(cfg, "test")
    assert all(connection.closed for connection in connections)
    for phase, count in [("calibration", 4), ("test", 5)]:
        raw = pd.read_parquet(cfg.run_dir / phase / "raw.parquet")
        assert len(raw) == 4 * count * cfg.benchmark.repetitions
        assert raw["pass_idx"].nunique() == cfg.benchmark.repetitions
        assert raw["error"].isna().all()
        scored = pd.read_parquet(cfg.run_dir / phase / "per_config.parquet")
        assert (scored["recall"] == 1.0).all()
        assert (scored["complete"] == 1.0).all()
    router = raw[raw["method"] == "B4"]
    assert (router["overhead_ns"] > 0).all()
    assert (router["overhead_ns"] <= router["latency_ns"]).all()
    frozen = json.loads((cfg.run_dir / "calibration/frozen_router.json").read_text())
    stats = search.cache_stats(json.loads((cfg.run_dir / "env/stats_snapshot.json").read_text()))
    windows = prepared.workload.set_index("config_id")
    for row in router.itertuples(index=False):
        window = windows.loc[row.config_id]
        assert row.route == search.route(stats, frozen["threshold_T"], window.lo, window.hi)[0]


def test_held_out_phase_rejects_changed_configuration(experiment):
    cfg, _, connections = experiment
    benchmark.step_bench(cfg, "calibration")
    benchmark.step_calibrate(cfg)
    opened = len(connections)
    cfg.source_sha256 = "changed"
    with pytest.raises(RuntimeError, match="changed after calibration"):
        benchmark.step_bench(cfg, "test")
    assert len(connections) == opened


def test_benchmark_refuses_to_overwrite_measurements(experiment):
    cfg, _, _ = experiment
    benchmark.step_bench(cfg, "calibration")
    with pytest.raises(RuntimeError, match="already has measurements"):
        benchmark.step_bench(cfg, "calibration")
