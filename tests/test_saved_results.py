"""Required outputs must reproduce the historical experiment's numerical results."""

import json
from dataclasses import replace

import pandas as pd

from fvsroute.benchmark import calibrate
from fvsroute.config import load_config
from fvsroute.search import cache_stats, route
from fvsroute.report import make_report

from .conftest import ROOT


def test_saved_measurements_reproduce_threshold_and_all_routes():
    cfg = load_config(ROOT / "configs/full.yaml")
    source = ROOT / "results/full"
    saved = json.loads((source / "calibration/frozen_router.json").read_text())
    measurements = pd.read_parquet(source / "calibration/per_config.parquet")
    measurements = measurements[measurements["method"].isin(["B1", "B2-plain", "B2-iter", "B3"])]
    estimation = pd.read_parquet(source / "verify/estimation.parquet")
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
    expected = pd.read_parquet(source / "calibration/threshold_curve.parquet")
    pd.testing.assert_frame_equal(curve, expected, rtol=1e-14, atol=1e-14)
    stats = cache_stats(json.loads((source / "env/stats_snapshot.json").read_text()))
    results = pd.read_parquet(source / "test/per_config.parquet")
    for row in results[results["method"] == "P1"].itertuples(index=False):
        method, estimate = route(stats, frozen["threshold_T"], row.lo, row.hi)
        assert method == row.route
        assert estimate == row.est_rows


def test_short_report_preserves_required_results_and_excludes_extra_methods(tmp_path):
    cfg = replace(load_config(ROOT / "configs/full.yaml"), results_dir=tmp_path, run_id="replay")
    make_report(cfg, str(ROOT / "results/full"))
    expected = pd.read_csv(ROOT / "results/full/tables/summary_test_overall.csv")
    expected["method"] = expected["method"].replace({"P1": "B4"})
    actual = pd.read_csv(cfg.run_dir / "tables/summary_test_overall.csv")
    expected = expected[expected["method"].isin(actual["method"])][actual.columns]
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True), rtol=1e-13, atol=1e-13
    )
    assert actual["method"].tolist() == ["B1", "B2-plain", "B2-iter", "B3", "B4", "Oracle"]
    assert len(list((cfg.run_dir / "figures").glob("*.png"))) == 6
    provenance = json.loads((cfg.run_dir / "provenance.json").read_text())
    assert provenance["mode"] == "saved measurements"
    assert provenance["source_run"] == "full"
