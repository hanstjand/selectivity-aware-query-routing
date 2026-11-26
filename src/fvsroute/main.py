"""Run the experiment, one step at a time or all at once."""

import argparse
import logging
import time

from . import benchmark, prepare, report
from .config import load_config

PIPELINE = (
    ("download", None),
    ("prepare", None),
    ("db-setup", None),
    ("verify", None),
    ("bench", "calibration"),
    ("calibrate", None),
    ("bench", "test"),
    ("report", None),
)
STEPS = ("all", "download", "prepare", "db-setup", "verify", "bench", "calibrate", "report")


def run_step(cfg, step, phase=None):
    if step == "all":
        for name, benchmark_phase in PIPELINE:
            run_step(cfg, name, benchmark_phase)
    elif step == "bench":
        if phase not in ("calibration", "test"):
            raise SystemExit("bench needs --phase calibration|test")
        benchmark.step_bench(cfg, phase)
    elif step == "report":
        report.make_report(cfg)
    else:
        actions = {
            "download": prepare.step_download,
            "prepare": prepare.step_prepare,
            "db-setup": prepare.step_db_setup,
            "verify": prepare.step_verify,
            "calibrate": benchmark.step_calibrate,
        }
        actions[step](cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", nargs="?", default="all", choices=STEPS)
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--run-id", help="write a separate run under results/<name>")
    parser.add_argument("--phase", choices=("calibration", "test"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.run_id)
    logs = cfg.run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / f"{args.step}-{time.strftime('%Y%m%d-%H%M%S')}.log", encoding="utf-8"),
        ],
        force=True,
    )
    run_step(cfg, args.step, args.phase)


if __name__ == "__main__":
    main()
