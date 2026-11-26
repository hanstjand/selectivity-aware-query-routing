from unittest.mock import Mock

import pytest

from fvsroute import main as cli, benchmark, prepare, report


def record_steps(monkeypatch, fail_at=None):
    calls = []

    def record(name, phase=None):
        calls.append((name, phase))
        if name == fail_at:
            raise RuntimeError("step failed")

    for step in ("download", "prepare", "db-setup", "verify"):
        monkeypatch.setattr(
            prepare,
            "step_" + step.replace("-", "_"),
            lambda cfg, name=step: record(name),
        )
    monkeypatch.setattr(benchmark, "step_bench", lambda cfg, phase: record("bench", phase))
    monkeypatch.setattr(benchmark, "step_calibrate", lambda cfg: record("calibrate"))
    monkeypatch.setattr(report, "make_report", lambda cfg, source=None: record("report"))
    return calls


def test_all_runs_calibration_before_held_out_test(monkeypatch):
    calls = record_steps(monkeypatch)
    cli.run_step(None, "all")
    assert calls == [
        ("download", None),
        ("prepare", None),
        ("db-setup", None),
        ("verify", None),
        ("bench", "calibration"),
        ("calibrate", None),
        ("bench", "test"),
        ("report", None),
    ]


def test_failed_calibration_stops_before_test(monkeypatch):
    calls = record_steps(monkeypatch, fail_at="calibrate")
    with pytest.raises(RuntimeError, match="step failed"):
        cli.run_step(None, "all")
    assert calls[-1] == ("calibrate", None)
    assert ("bench", "test") not in calls


def test_bench_requires_phase_before_running_queries(monkeypatch):
    bench = Mock()
    monkeypatch.setattr(benchmark, "step_bench", bench)
    with pytest.raises(SystemExit, match="bench needs --phase"):
        cli.run_step(None, "bench")
    bench.assert_not_called()
