"""Benchmark execution for one phase (calibration or held-out test).

Protocol: settings audit -> stats check -> prewarm -> RTT baseline -> plan-shape audit (plain EXPLAIN, 100% of
(config x method)) -> EXPLAIN (ANALYZE, BUFFERS) on a stratified sample of primary methods -> warm-up pass(es)
(discarded) -> timed passes, each a seeded random permutation of all (config x method) pairs -> stats check.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from pgvector import Vector

from . import db as dbmod
from .config import Config
from .plans import classify, meets_expectation
from .router import Router
from .strategies import PRIMARY, SENSITIVITY, SessionPool, Strategy

log = logging.getLogger(__name__)
P1 = "P1"


class PlanAssertionError(RuntimeError):
    pass


@dataclass
class PhaseInputs:
    phase: str
    workload: pd.DataFrame          # configs of this phase only
    queries: dict[int, np.ndarray]  # query_id -> float32 vector
    strategies: dict[str, Strategy]
    router: Router | None = None    # include P1 when given


def _params(row, vec) -> dict:
    return {"lo": float(row.lo), "hi": float(row.hi), "q": vec}


def run_phase(cfg: Config, inp: PhaseInputs, out_dir: Path, expected_stats_fingerprint: str | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_phase = time.time()
    meta: dict = {"phase": inp.phase, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "n_configs": len(inp.workload), "methods": list(inp.strategies) + ([P1] if inp.router else [])}
    wl = inp.workload.reset_index(drop=True)
    vecs = {q: Vector(v) for q, v in inp.queries.items()}
    rows_by_cid = {r.config_id: r for r in wl.itertuples(index=False)}
    methods = meta["methods"]

    with SessionPool(cfg, inp.strategies) as pool, dbmod.connect(cfg.database) as ctl:
        # ---- 1. settings audit -------------------------------------------------------------
        audit = pool.audit_settings()
        (out_dir / "settings_audit.json").write_text(json.dumps(audit, indent=1))
        if not audit["ok"]:
            raise RuntimeError(f"session settings audit failed: {audit['mismatches']}")
        # ---- 2. statistics unchanged since db-setup ------------------------------------------
        fp_start = dbmod.stats_fingerprint(ctl, cfg.database.table)
        if expected_stats_fingerprint and fp_start != expected_stats_fingerprint:
            raise RuntimeError("planner statistics changed since db-setup (was ANALYZE re-run?)")
        meta["stats_fingerprint_start"] = fp_start
        meta["prewarm_blocks"] = dbmod.prewarm(ctl, cfg.database.table)
        meta["rtt_select1"] = dbmod.rtt_baseline(ctl)

        # ---- 3. plan-shape audit (planning only, every config x method) -------------------------
        t0 = time.time()
        plan_rows = []
        for r in wl.itertuples(index=False):
            params = _params(r, vecs[r.query_id])
            for m in methods:
                if m == P1:
                    arm, est = inp.router.route(r.lo, r.hi)
                    st_name, expect = arm, inp.strategies[arm].expect
                else:
                    arm, est, st_name, expect = None, None, m, inp.strategies[m].expect
                summ = classify(pool.explain(st_name, params), cfg.database.table)
                plan_rows.append({"config_id": r.config_id, "level_idx": r.level_idx, "method": m, "route": arm,
                                  "est_rows": est, "expect": expect, "ok": meets_expectation(expect, summ), **summ})
        plans = pd.DataFrame(plan_rows)
        plans.to_parquet(out_dir / "plans.parquet", index=False)
        meta["plan_audit_s"] = time.time() - t0
        bad = plans[~plans["ok"]]
        meta["plan_audit"] = {"n": len(plans), "n_failed": int(len(bad))}
        if len(bad):
            raise PlanAssertionError(f"{len(bad)} plans violate their intended shape, e.g.\n"
                                     f"{bad[['config_id', 'method', 'plan_class', 'node_types']].head()}")
        log.info("plan-shape audit OK: %d plans (%.1fs)", len(plans), meta["plan_audit_s"])

        # ---- 4. EXPLAIN (ANALYZE, BUFFERS) on a stratified sample of non-sensitivity methods ------
        t0 = time.time()
        rng = np.random.default_rng([cfg.seed, 7, 0 if inp.phase == "calibration" else 1])
        picked = []
        for _, g in wl.groupby("level_idx", sort=True):
            take = rng.choice(len(g), min(len(g), cfg.benchmark.analyze_sample_per_level), replace=False)
            picked.extend(g.index[np.sort(take)])
        sample = wl.loc[picked]
        an_dir = out_dir / "explain_analyze"
        an_dir.mkdir(exist_ok=True)
        an_rows = []
        for r in sample.itertuples(index=False):
            params = _params(r, vecs[r.query_id])
            for m in methods:
                if m != P1 and inp.strategies[m].role == SENSITIVITY:
                    continue
                st_name = inp.router.route(r.lo, r.hi)[0] if m == P1 else m
                doc = pool.explain(st_name, params, analyze=True)
                (an_dir / f"{m}__{r.config_id}.json").write_text(json.dumps(doc, indent=1))
                summ = classify(doc, cfg.database.table)
                an_rows.append({"config_id": r.config_id, "level_idx": r.level_idx, "method": m,
                                "executed_arm": st_name, **summ})
        pd.DataFrame(an_rows).to_parquet(out_dir / "explain_analyze.parquet", index=False)
        meta["explain_analyze_s"] = time.time() - t0
        meta["explain_analyze_n"] = len(an_rows)

        # ---- 5. warm-up + timed passes -------------------------------------------------------------
        pairs = [(cid, m) for cid in wl["config_id"] for m in methods]
        records: list[tuple] = []

        def execute(cid: str, m: str):
            r = rows_by_cid[cid]
            params = _params(r, vecs[r.query_id])
            route = est = overhead = None
            err = None
            ids: list[int] = []
            t0 = time.perf_counter_ns()
            try:
                if m == P1:
                    route, est = inp.router.route(r.lo, r.hi)
                    t1 = time.perf_counter_ns()
                    overhead = t1 - t0
                    ids = pool.execute(route, params)
                else:
                    ids = pool.execute(m, params)
            except psycopg.errors.QueryCanceled as e:
                err = f"timeout: {e}".strip()
            except psycopg.Error as e:
                err = f"{type(e).__name__}: {e}".strip()
                name = route if m == P1 else m
                if name and pool.sessions[name].conn.closed:
                    pool.reconnect(name)
            lat = time.perf_counter_ns() - t0
            return ids, lat, route, est, overhead, err

        total_passes = cfg.benchmark.warmup_passes + cfg.benchmark.repetitions
        meta["pass_durations_s"] = []
        for p in range(total_passes):
            timed = p >= cfg.benchmark.warmup_passes
            order = np.random.default_rng([cfg.seed, 11, 0 if inp.phase == "calibration" else 1, p]) \
                .permutation(len(pairs))
            gc.collect()
            gc.disable()
            tp = time.time()
            try:
                for oi, pi in enumerate(order):
                    cid, m = pairs[pi]
                    ids, lat, route, est, overhead, err = execute(cid, m)
                    if timed:
                        records.append((p - cfg.benchmark.warmup_passes, oi, cid, m, lat, len(ids),
                                        np.asarray(ids, dtype=np.int32), err, time.time_ns(), route, est, overhead))
                    if oi and oi % max(1, len(order) // 10) == 0:
                        el = time.time() - tp
                        log.info("%s pass %d/%d (%s): %d/%d, %.0fs elapsed, ~%.0fs left", inp.phase, p + 1,
                                 total_passes, "timed" if timed else "warm-up", oi, len(order), el,
                                 el / oi * (len(order) - oi))
            finally:
                gc.enable()
            meta["pass_durations_s"].append({"pass": p, "timed": timed, "seconds": time.time() - tp})

        fp_end = dbmod.stats_fingerprint(ctl, cfg.database.table)
        meta["stats_fingerprint_end"] = fp_end
        if fp_end != fp_start:
            raise RuntimeError("planner statistics changed during the phase")

    raw = pd.DataFrame.from_records(records, columns=[
        "pass_idx", "order_idx", "config_id", "method", "latency_ns", "n_returned", "ids", "error", "ts_ns",
        "route", "est_rows", "overhead_ns"])
    raw = raw.merge(wl[["config_id", "query_id", "level_idx"]], on="config_id", how="left")
    raw.insert(0, "phase", inp.phase)
    raw.to_parquet(out_dir / "raw.parquet", index=False)
    meta["n_executions"] = len(raw)
    meta["n_errors"] = int(raw["error"].notna().sum())
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["wall_clock_s"] = time.time() - t_phase
    (out_dir / "phase_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log.info("%s phase done: %d executions, %d errors, %.0fs", inp.phase, len(raw), meta["n_errors"],
             meta["wall_clock_s"])
    return meta


def primary_names(strategies: dict[str, Strategy]) -> list[str]:
    return [n for n, s in strategies.items() if s.role == PRIMARY]
