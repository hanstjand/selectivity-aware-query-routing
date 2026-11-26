from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _db_reachable(cfg) -> bool:
    import psycopg
    from fvsroute.config import connection_string

    try:
        with psycopg.connect(connection_string(cfg.database, "postgres"), connect_timeout=2):
            return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def smoke_cfg():
    from fvsroute.config import load_config

    return load_config(SMOKE)


@pytest.fixture(scope="session")
def smoke_ready(smoke_cfg):
    """Smoke database + prepared data must exist (`fvs all --config configs/smoke.yaml`)."""
    if not (smoke_cfg.raw_dir / "vectors.npy").exists():
        pytest.skip("dataset not downloaded (run `fvs download --config configs/smoke.yaml`)")
    if not _db_reachable(smoke_cfg):
        pytest.skip("PostgreSQL not reachable (run `docker compose up -d`)")
    if not (smoke_cfg.run_dir / "env" / "db_setup.json").exists():
        pytest.skip("smoke database not set up (run `fvs all --config configs/smoke.yaml`)")
    return smoke_cfg
