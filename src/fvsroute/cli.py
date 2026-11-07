"""Command-line entry point: `fvs <step> --config configs/<name>.yaml [--run-id ID]`."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import load_config

STEPS = ("download", "prepare", "db-setup", "verify", "bench", "calibrate", "report", "all")


def _setup_logging(cfg, step: str) -> None:
    log_dir = cfg.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / f"{step}-{time.strftime('%Y%m%d-%H%M%S')}.log")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)


def run_step(cfg, step: str, phase: str | None = None) -> None:
    from . import pipeline, report
    log = logging.getLogger("fvs")
    t0 = time.time()
    if step == "download":
        pipeline.step_download(cfg)
    elif step == "prepare":
        pipeline.step_prepare(cfg)
    elif step == "db-setup":
        pipeline.step_db_setup(cfg)
    elif step == "verify":
        pipeline.step_verify(cfg)
    elif step == "bench":
        if phase not in ("calibration", "test"):
            raise SystemExit("bench needs --phase calibration|test")
        pipeline.step_bench(cfg, phase)
    elif step == "calibrate":
        pipeline.step_calibrate(cfg)
    elif step == "report":
        report.make_report(cfg)
    elif step == "all":
        for s, ph in (("download", None), ("prepare", None), ("db-setup", None), ("verify", None),
                      ("bench", "calibration"), ("calibrate", None), ("bench", "test"), ("report", None)):
            run_step(cfg, s, ph)
    log.info("step %s%s finished in %.1fs", step, f" ({phase})" if phase else "", time.time() - t0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fvs", description=__doc__)
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--config", required=True, help="path to configs/<name>.yaml")
    ap.add_argument("--run-id", default=None, help="override run_id (results/<run_id>, data/processed/<run_id>)")
    ap.add_argument("--phase", choices=("calibration", "test"), help="phase for `bench`")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.run_id)
    _setup_logging(cfg, args.step + (f"-{args.phase}" if args.phase else ""))
    run_step(cfg, args.step, args.phase)


if __name__ == "__main__":
    main()
