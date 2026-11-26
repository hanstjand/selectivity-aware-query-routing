"""Read the YAML settings and keep file paths in one place."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import yaml


@dataclass
class Config:
    run_id: str
    seed: int
    dataset: SimpleNamespace
    workload: SimpleNamespace
    database: SimpleNamespace
    hnsw: SimpleNamespace
    benchmark: SimpleNamespace
    calibration: SimpleNamespace
    data_dir: Path
    results_dir: Path
    source_path: Path
    source_sha256: str

    @property
    def archive_path(self):
        return self.data_dir / "raw" / f"{self.dataset.name}.tgz"

    @property
    def raw_dir(self):
        return self.data_dir / "raw" / self.dataset.name

    @property
    def processed_dir(self):
        return self.data_dir / "processed" / self.run_id

    @property
    def run_dir(self):
        return self.results_dir / self.run_id

    def with_run_id(self, name):
        return replace(self, run_id=name)


def load_config(path, run_id=None):
    path = Path(path).resolve()
    contents = path.read_bytes()
    settings = yaml.safe_load(contents)
    root = path.parent.parent
    sections = {
        name: SimpleNamespace(**settings[name])
        for name in ("dataset", "workload", "database", "hnsw", "benchmark", "calibration")
    }
    workload = sections["workload"]
    if not 0 < workload.calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if any(not 0 < value < 1 for value in workload.selectivities):
        raise ValueError("selectivities must be in (0, 1)")
    if sorted(workload.selectivities) != list(workload.selectivities):
        raise ValueError("selectivities must be sorted ascending")
    if sections["benchmark"].repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    paths = settings.get("paths", {})
    return Config(
        run_id or settings["run_id"],
        settings["seed"],
        **sections,
        data_dir=root / paths.get("data_dir", "data"),
        results_dir=root / paths.get("results_dir", "results"),
        source_path=path,
        source_sha256=hashlib.sha256(contents).hexdigest(),
    )


def connection_string(database, dbname=None):
    return (
        f"host={database.host} port={database.port} user={database.user} "
        f"password={database.password} dbname={dbname or database.dbname}"
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()), encoding="utf-8")
