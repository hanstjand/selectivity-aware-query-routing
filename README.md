# Selectivity-Aware Query Routing for Filtered Vector Search

A reproducible PostgreSQL + pgvector study of a simple query router for filtered vector search. For
`SELECT id FROM items WHERE similarity BETWEEN lo AND hi ORDER BY embedding <=> q LIMIT 10`, the router estimates
the number of matching rows from PostgreSQL's own statistics (`pg_stats`) and sends the query to **exact search**
(estimate ≤ T) or to **HNSW with iterative scan** (estimate > T). T is calibrated offline on 70% of the query
vectors and evaluated once on the held-out 30%.

| Method | What runs |
|---|---|
| **B1** Exact | metadata filter (B-tree bitmap or seq scan) + exact cosine sort; HNSW disabled (`enable_indexscan=off`) |
| **B2-plain / B2-iter** | forced HNSW index scan, without / with `hnsw.iterative_scan = relaxed_order` |
| **B3** Planner | the unmodified query: PostgreSQL's default planner decision under the shared HNSW configuration |
| **P1** Router | `est_rows ≤ T → B1`, otherwise `B2-iter` (estimate from cached `pg_stats`, no COUNT(*)) |
| **Oracle** | hindsight latency floor: faster of B1 / B2-iter per query, ignoring quality (evaluation reference only) |

Data: Qdrant's public LAION/CLIP filtered-ANN benchmark (100,000 × 512-d, cosine), 300 query vectors ×
9 selectivities (0.1%–90%) = 2,700 filtered queries, k = 10, exact ground truth computed with NumPy.
Definitions, design decisions and threats to validity are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); the
generated results are in `results/full/report.md`.

## Requirements

- Docker with Compose (the database runs in a container; give Docker ≥ 4 GB RAM)
- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and all dependencies)
- ~1 GB free disk (120 MB archive, extracted data, database volume)
- Internet access for the one-time dataset download

## Exact commands

```bash
# 1. Environment
uv sync                                   # creates .venv (Python 3.12, numpy, psycopg, pgvector, pandas, ...)

# 2. PostgreSQL 17 + pgvector 0.8.7
docker compose up -d                      # container fvs-postgres on 127.0.0.1:55432 (user/pass/db = fvs)
docker compose ps                         # wait for "healthy"

# 3. Download + validate the dataset (sha256-pinned)
uv run fvs download  --config configs/full.yaml

# 4. Queries, quantile filters, 70/30 split, exact NumPy ground truth (+ gate against Qdrant's own ids)
uv run fvs prepare   --config configs/full.yaml

# 5. Load table, build B-tree + HNSW indexes, VACUUM (FREEZE, ANALYZE), snapshot statistics
uv run fvs db-setup  --config configs/full.yaml

# 6. Verify: PG vs NumPy cosine, per-filter counts, predicate equivalence, B1 == ground truth, estimator vs planner
uv run fvs verify    --config configs/full.yaml

# 7. Baseline benchmark on the calibration split (B1, B2-plain, B2-iter, B3 + HNSW sensitivity variants)
uv run fvs bench     --config configs/full.yaml --phase calibration

# 8. Calibrate and freeze threshold T (calibration split only)
uv run fvs calibrate --config configs/full.yaml

# 9. Held-out evaluation, once (B1, B2-plain, B2-iter, B3, P1, supplementary B3+iter)
uv run fvs bench     --config configs/full.yaml --phase test

# 10. Figures, tables and report
uv run fvs report    --config configs/full.yaml        # -> results/full/{report.md, figures/, tables/}
```

`uv run fvs all --config configs/full.yaml` runs steps 3–10 in order. Every step reads its inputs from disk, so
any step can be re-run on its own. Use `--run-id <name>` to write a separate rerun to `results/<name>/` (for
example to measure run-to-run variance).

### Smoke test and automated tests

```bash
uv run pytest -m "not db"                             # unit tests (no database needed)
uv run fvs all --config configs/smoke.yaml            # whole pipeline on a 5,000-row prefix, 20 queries
uv run pytest -m db                                   # integration + end-to-end tests (needs Docker + data)
```

The smoke configuration uses its own database (`fvs_smoke`) and results directory (`results/smoke/`). The
end-to-end test runs the complete pipeline in a temporary results directory and database (`fvs_pytest`).

> Do not run tests or other heavy work while `fvs bench` is timing queries: it shares the machine with the
> measurement.

## What each step produces

| Step | Output | Content |
|---|---|---|
| download | `data/raw/laion-small-clip/` | `vectors.npy`, `payloads.jsonl`, `tests.jsonl` (sha256-checked archive) |
| prepare | `data/processed/<run>/` | `workload.parquet` (2,700 filters with actual selectivity and split), `groundtruth.parquet`, `queries.npy`, `manifest.json` (validation report, per-level selectivity), `gt_vs_qdrant.json` (gate) |
| db-setup | `results/<run>/env/` | `db_setup.json` (load/index times, relation sizes), `stats_snapshot.json` (cached `pg_stats` for P1), `server_env.json`, `host_env.json` |
| verify | `results/<run>/verify/` | `consistency.json` (all checks), `estimation.parquet` (ours vs planner vs actual rows), `b1_vs_gt.parquet` |
| bench | `results/<run>/<phase>/` | `raw.parquet` (every timed execution with returned ids), `plans.parquet` (EXPLAIN of every config × method), `explain_analyze/` (sampled EXPLAIN ANALYZE BUFFERS JSON), `settings_audit.json`, `phase_meta.json` (durations, RTT, errors) |
| calibrate | `results/<run>/calibration/` | `frozen_router.json` (T*, HNSW config, guard decision, hashes), `threshold_curve.parquet` |
| report | `results/<run>/` | `report.md`, `figures/*.png|pdf`, `tables/*.csv` |

## Repository layout

```
configs/            full.yaml (primary experiment), smoke.yaml
docker-compose.yml  PostgreSQL 17 + pgvector 0.8.7 (settings documented inline)
sql/                schema and index DDL
src/fvsroute/       dataset, workload, groundtruth, db, strategies, plans, estimator, router,
                    metrics, runner, calibrate, oracle, report, pipeline, cli
tests/              unit tests, DB integration tests, end-to-end smoke test
docs/METHODOLOGY.md definitions, decisions, ambiguity resolutions, threats to validity
```

## Results (run `full`, measured 2026-11-07)

Full generated report: `results/full/report.md` (all tables in `results/full/tables/`, figures in
`results/full/figures/`). Environment: Apple M4 (10 cores, 16 GB), Docker 29.2, PostgreSQL 17.11, pgvector 0.8.7.
Latency is client-side end-to-end, including a ~0.16–0.18 ms `SELECT 1` round trip.

**Held-out test split** (90 query vectors × 9 levels = 810 configs; per-config median of 5 timed runs; Recall@10
is strict ID overlap):

| Method | p50 (ms) | p95 (ms) | mean (ms) | Recall@10 | Complete | Meets quality? |
|---|---|---|---|---|---|---|
| B1 Exact | 4.159 | 16.344 | 6.688 | 1.000 | 100.0% | yes |
| B2-plain | 1.414 | 1.901 | 1.411 | 0.510 | 39.6% | no |
| B2-iter | 2.716 | 25.758 | 7.049 | 0.968 | 99.5% | yes |
| B3 Planner | 1.233 | 1.818 | 1.204 | 0.828 | 73.0% | no |
| **P1 Router** | **1.332** | **3.297** | **1.511** | **0.991** | **100.0%** | **yes** |
| Oracle (latency floor, ignores quality) | 1.313 | 3.247 | 1.469 | 0.991 | 100.0% | n/a |
| *B3+iter (supplementary)* | 1.359 | 5.415 | 1.873 | 0.989 | 100.0% | yes |

Pre-registered success criteria (all PASS overall):
1. P1 quality: mean strict Recall@10 0.991 (≥ 0.95), 100% complete (≥ 99%).
2. P1 p50 1.332 ms vs the best qualifying baseline B2-iter at 2.716 ms; P1 p95 3.297 ms vs the best qualifying
   baseline B1 at 16.344 ms. The paired cluster-bootstrap CIs exclude 0 for both.
3. Gain captured relative to the latency-floor Oracle: 98.7% (p50), 99.6% (p95), 99.2% (mean). P1's routing choice
   agrees with the Oracle on 98.1% of configs.

Main findings:
- **RQ1 (baselines vs selectivity).**
  - **B1** is exact everywhere but grows from 0.41 ms at 0.1% to 13–16 ms at ≥ 25% selectivity. The planner uses a
    non-parallel bitmap scan up to 25% and a parallel seq scan from 50%, which is why 25% is B1's slowest level.
  - **B2-iter** is fast for broad filters (~1.4 ms) but slow and imperfect for selective ones: at 0.1% the p50 is
    25.5 ms, recall 0.84, 95.6% complete.
  - **B2-plain** returns too few rows below 25%: 0% complete at ≤ 2%.
  - **B3** (PostgreSQL's default plan choice under the shared HNSW configuration) has the lowest raw latency but
    switches to non-iterative HNSW from the 2% level up. Its results are incomplete there: 0% complete at 2%, 10% at
    5%, 50% at 10%. Overall it fails the quality criterion.
- **RQ2.** P1 meets the quality targets and beats every baseline that also meets them, by a wide margin, mainly by
  avoiding B1 for broad filters and B2-iter for selective ones. Simply enabling iterative scan and trusting the
  planner (B3+iter) also meets quality, but its p95 is 64% higher and its mean 24% higher than P1's.
- **RQ3 (where P1 loses; negative results).** Against the best baseline *per level*, most differences are within
  the bootstrap noise. Three are statistically distinguishable:
  - **0.5% and 1% p50.** P1 runs the same exact plan as B1 at these levels plus about 9–10 µs of live routing
    overhead, which makes it ~1.5% slower.
  - **5% p95.** It is 5.58 ms vs B1's 4.31 ms. The calibration objective is nearly flat for T between 2,129 and 4,842
    estimated rows, and the frozen T\* = 4,696 lies inside the 5% level's estimate range. The 5% filters that went to
    B2-iter include its slow tail.

  Under the strict pre-registered per-level rule (P1 ≤ the best qualifying baseline at that level), P1 passes
  criterion 2 for both p50 and p95 at no level; it passes one of the two only at 1% (p95) and 10% (p50). Its overall
  advantage comes from staying close to the best method at every level, while every fixed method is bad somewhere.
- **Selectivity estimation.** The cached `pg_stats` estimator reproduces PostgreSQL's own planner estimate (median
  relative difference 0.02%). Its error against the true counts is 11% at 0.1% (≈ 11 rows) and falls to 0.2% at 90%.
  Only at the 5% level did estimates fall on both sides of T\*; at every other level all configs were routed to the same arm.
- **HNSW sensitivity (calibration split, report-only).** ef_search 40 → 200 trades latency for recall at broad
  filters: at 90%, p50 0.83 → 1.92 ms and recall 0.986 → 0.999. `max_scan_tuples = 100000` raises completeness at
  0.1% from 96.7% to 98.6% at a higher latency. `strict_order` lowers recall slightly. Iterative scanning never
  guaranteed complete results at 0.1%.
- **Verification.**
  - Qdrant's own exact ids are reproduced: 297/300 queries match exactly, and the rest differ only at exact-distance
    ties.
  - PostgreSQL and NumPy cosine distances differ by at most 4.5e-7.
  - B1 equals the ground truth on all 2,700 configs, apart from 2 documented near-ties.
  - Plan checks found 0 violations across 19,980 audited plans, and the sampled EXPLAIN ANALYZE runs read 0 blocks
    from disk.
  - There were 0 query errors.

Limitations: one host, warm cache, sequential single client, one filter attribute, Docker on macOS. Absolute
latencies are host-specific, and the crossover threshold depends on the HNSW parameters and the hardware. See
[docs/METHODOLOGY.md §15](docs/METHODOLOGY.md).
Measured runtime: index build 10.4 s; calibration benchmark 8.4 min; held-out benchmark 1.7 min.
