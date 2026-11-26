"""The committed measurements in results/full/ must reproduce the reported threshold, routes and tables."""

import json
import shutil
from dataclasses import replace

import pandas as pd

from fvsroute.benchmark import calibrate
from fvsroute.config import load_config
from fvsroute.report import make_report
from fvsroute.search import BASELINES, cache_stats, route

from .conftest import ROOT

SOURCE = ROOT / "results/full"


def test_saved_measurements_reproduce_threshold_and_all_routes():
    cfg = load_config(ROOT / "configs/full.yaml")
    saved = json.loads((SOURCE / "calibration/frozen_router.json").read_text())
    measurements = pd.read_parquet(SOURCE / "calibration/per_config.parquet")
    measurements = measurements[measurements["method"].isin(BASELINES)]
    estimation = pd.read_parquet(SOURCE / "verify/estimation.parquet")
    frozen, curve = calibrate(
        cfg,
        measurements,
        estimation.set_index("config_id")["est_ours"],
        saved["routing_overhead_ns"],
        saved["stats_snapshot_sha256"],
    )
    assert frozen["threshold_T"] == saved["threshold_T"]
    for name, value in frozen["hnsw"].items():
        assert value == saved["hnsw"][name]
    expected = pd.read_parquet(SOURCE / "calibration/threshold_curve.parquet")
    pd.testing.assert_frame_equal(curve, expected, rtol=1e-14, atol=1e-14)
    stats = cache_stats(json.loads((SOURCE / "env/stats_snapshot.json").read_text()))
    results = pd.read_parquet(SOURCE / "test/per_config.parquet")
    router = results[results["method"] == "P1"]
    assert len(router) == 810
    for row in router.itertuples(index=False):
        method, estimate = route(stats, frozen["threshold_T"], row.lo, row.hi)
        assert method == row.route
        assert estimate == row.est_rows


def test_report_regenerates_committed_tables(tmp_path):
    cfg = replace(load_config(ROOT / "configs/full.yaml"), results_dir=tmp_path)
    for part in ("test", "calibration", "verify"):
        shutil.copytree(SOURCE / part, cfg.run_dir / part)
    make_report(cfg)
    for name in ("summary_test_overall", "summary_test_by_level", "success_criteria", "estimation_by_level"):
        actual = pd.read_csv(cfg.run_dir / f"tables/{name}.csv")
        expected = pd.read_csv(SOURCE / f"tables/{name}.csv")
        pd.testing.assert_frame_equal(actual, expected, rtol=1e-13, atol=1e-13)
    overall = pd.read_csv(cfg.run_dir / "tables/summary_test_overall.csv")
    assert overall["method"].tolist() == [*BASELINES, "P1", "Oracle"]
    assert len(list((cfg.run_dir / "figures").glob("*.png"))) == 6
