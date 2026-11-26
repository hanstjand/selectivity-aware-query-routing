"""End-to-end: the whole pipeline (`fvs all`) on the smoke configuration, in a separate database and results dir."""

import json

import pandas as pd
import pytest
import yaml

from .conftest import ROOT, SMOKE, _db_reachable

pytestmark = pytest.mark.db


def test_full_pipeline_smoke(tmp_path, smoke_cfg):
    if not (smoke_cfg.raw_dir / "vectors.npy").exists() or not _db_reachable(smoke_cfg):
        pytest.skip("needs the downloaded dataset and a running PostgreSQL")
    d = yaml.safe_load(SMOKE.read_text())
    d["run_id"] = "pytest-e2e"
    d["database"]["dbname"] = "fvs_pytest"
    d["paths"] = {"data_dir": str(ROOT / "data"), "results_dir": str(tmp_path / "results")}
    cfg_path = tmp_path / "configs" / "e2e.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(yaml.safe_dump(d))

    from fvsroute.main import main

    main(["all", "--config", str(cfg_path)])

    run = tmp_path / "results" / "pytest-e2e"
    for rel in (
        "env/db_setup.json",
        "env/stats_snapshot.json",
        "verify/consistency.json",
        "calibration/raw.parquet",
        "calibration/plans.parquet",
        "calibration/frozen_router.json",
        "test/raw.parquet",
        "test/per_config.parquet",
        "report.md",
        "figures/latency_p50_vs_selectivity.png",
        "tables/summary_test_overall.csv",
        "tables/success_criteria.csv",
    ):
        assert (run / rel).exists(), rel
    assert json.loads((run / "verify/consistency.json").read_text())["all_pass"]
    s = pd.read_csv(run / "tables/summary_test_overall.csv").set_index("method")
    assert s.loc["B1", "recall"] == 1.0 and s.loc["B1", "completeness"] == 1.0
    assert {"B1", "B2-plain", "B2-iter", "B3", "B4", "Oracle"} <= set(s.index)
    # frozen threshold precedes the held-out measurements
    frozen = json.loads((run / "calibration/frozen_router.json").read_text())
    raw = pd.read_parquet(run / "test/raw.parquet", columns=["ts_ns"])
    frozen_ns = pd.Timestamp(frozen["frozen_at"]).value
    assert raw["ts_ns"].min() > frozen_ns
