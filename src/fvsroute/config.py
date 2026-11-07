"""Experiment configuration: every experimental parameter lives in a YAML file under configs/."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetCfg:
    url: str
    sha256: str
    name: str
    n_rows: int
    dim: int
    filter_field: str
    imputed_value: float | None


@dataclass(frozen=True)
class WorkloadCfg:
    n_queries: int
    selectivities: tuple[float, ...]
    k: int
    calibration_fraction: float
    gt_gate_min_overlap: float


@dataclass(frozen=True)
class DatabaseCfg:
    host: str
    port: int
    user: str
    password: str
    dbname: str
    table: str
    statement_timeout_ms: int

    def conninfo(self, dbname: str | None = None) -> str:
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={dbname or self.dbname}"
        )


@dataclass(frozen=True)
class HnswCfg:
    m: int
    ef_construction: int
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float
    guard_ef_search: int
    guard_min_recall: float


@dataclass(frozen=True)
class SensitivityVariant:
    name: str
    ef_search: int | None = None
    iterative_scan: str | None = None
    max_scan_tuples: int | None = None


@dataclass(frozen=True)
class BenchmarkCfg:
    warmup_passes: int
    repetitions: int
    analyze_sample_per_level: int


@dataclass(frozen=True)
class CalibrationCfg:
    min_mean_recall: float
    min_completeness: float


@dataclass(frozen=True)
class Config:
    run_id: str
    seed: int
    dataset: DatasetCfg
    workload: WorkloadCfg
    database: DatabaseCfg
    hnsw: HnswCfg
    sensitivity: tuple[SensitivityVariant, ...]
    benchmark: BenchmarkCfg
    calibration: CalibrationCfg
    bootstrap_resamples: int
    data_dir: Path
    results_dir: Path
    source_path: Path
    source_sha256: str

    # ---- derived paths -------------------------------------------------
    @property
    def archive_path(self) -> Path:
        return self.data_dir / "raw" / f"{self.dataset.name}.tgz"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / self.dataset.name

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed" / self.run_id

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id

    def with_run_id(self, run_id: str) -> "Config":
        return replace(self, run_id=run_id)


def load_config(path: str | Path, run_id: str | None = None) -> Config:
    path = Path(path)
    raw_bytes = path.read_bytes()
    d = yaml.safe_load(raw_bytes)
    base = path.resolve().parent.parent  # configs/<file>.yaml -> project root

    ds = d["dataset"]
    wl = d["workload"]
    db = d["database"]
    hn = d["hnsw"]
    bm = d["benchmark"]
    cal = d["calibration"]
    paths = d.get("paths", {})

    cfg = Config(
        run_id=str(d["run_id"]),
        seed=int(d["seed"]),
        dataset=DatasetCfg(
            url=ds["url"],
            sha256=ds["sha256"].lower(),
            name=ds["name"],
            n_rows=int(ds["n_rows"]),
            dim=int(ds["dim"]),
            filter_field=ds["filter_field"],
            imputed_value=None if ds.get("imputed_value") is None else float(ds["imputed_value"]),
        ),
        workload=WorkloadCfg(
            n_queries=int(wl["n_queries"]),
            selectivities=tuple(float(s) for s in wl["selectivities"]),
            k=int(wl["k"]),
            calibration_fraction=float(wl["calibration_fraction"]),
            gt_gate_min_overlap=float(wl["gt_gate_min_overlap"]),
        ),
        database=DatabaseCfg(
            host=db["host"],
            port=int(db["port"]),
            user=db["user"],
            password=db["password"],
            dbname=db["dbname"],
            table=db["table"],
            statement_timeout_ms=int(db["statement_timeout_ms"]),
        ),
        hnsw=HnswCfg(
            m=int(hn["m"]),
            ef_construction=int(hn["ef_construction"]),
            ef_search=int(hn["ef_search"]),
            max_scan_tuples=int(hn["max_scan_tuples"]),
            scan_mem_multiplier=float(hn["scan_mem_multiplier"]),
            guard_ef_search=int(hn["guard_ef_search"]),
            guard_min_recall=float(hn["guard_min_recall"]),
        ),
        sensitivity=tuple(SensitivityVariant(**v) for v in d.get("sensitivity", [])),
        benchmark=BenchmarkCfg(
            warmup_passes=int(bm["warmup_passes"]),
            repetitions=int(bm["repetitions"]),
            analyze_sample_per_level=int(bm["analyze_sample_per_level"]),
        ),
        calibration=CalibrationCfg(
            min_mean_recall=float(cal["min_mean_recall"]),
            min_completeness=float(cal["min_completeness"]),
        ),
        bootstrap_resamples=int(d.get("evaluation", {}).get("bootstrap_resamples", 2000)),
        data_dir=(base / paths.get("data_dir", "data")),
        results_dir=(base / paths.get("results_dir", "results")),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    _validate(cfg)
    if run_id:
        cfg = cfg.with_run_id(run_id)
    return cfg


def _validate(cfg: Config) -> None:
    w = cfg.workload
    if not (0 < w.calibration_fraction < 1):
        raise ValueError("calibration_fraction must be in (0, 1)")
    if any(not (0 < s < 1) for s in w.selectivities):
        raise ValueError("selectivities must be in (0, 1)")
    if list(w.selectivities) != sorted(w.selectivities):
        raise ValueError("selectivities must be sorted ascending")
    if cfg.benchmark.repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    names = [v.name for v in cfg.sensitivity]
    if len(set(names)) != len(names):
        raise ValueError("sensitivity variant names must be unique")
    for v in cfg.sensitivity:
        if v.iterative_scan not in (None, "off", "relaxed_order", "strict_order"):
            raise ValueError(f"bad iterative_scan in {v.name}")
