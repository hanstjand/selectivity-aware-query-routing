# Selectivity-Aware Query Routing for Filtered Vector Search

Course project for Advanced Database Systems (NTUST). For each filtered top-10 vector query in PostgreSQL +
pgvector, a router (P1) chooses between exact search and iterative HNSW search. It estimates the number of
matching rows from cached `pg_stats` histograms, with no per-query `COUNT(*)`:

```python
estimated_rows = estimate_rows(stats, lo, hi)
method = "B1" if estimated_rows <= threshold else "B2-iter"
```

The threshold is calibrated on 70% of the query vectors and evaluated once on the held-out 30%.

## Repository contents

| Path | Contents |
|---|---|
| `src/fvsroute/` | Implementation (seven modules, see below) |
| `configs/full.yaml` | The experiment: 100K vectors, 300 query vectors × 9 selectivities |
| `configs/smoke.yaml` | Small development run on a 5,000-row prefix |
| `sql/`, `docker-compose.yml` | Table schema, indexes, PostgreSQL 17 + pgvector 0.8.7 container |
| `results/full/` | Measured results reported in the paper: `report.md`, tables, figures, raw per-query summaries |
| `report/` | IEEE conference paper (`main.pdf`) and its LaTeX source |
| `docs/METHODOLOGY.md` | Every method, metric and decision |
| `tests/` | Unit tests, saved-result regression tests, database integration tests |

## Methods

| Method | What it runs |
|---|---|
| B1 Exact | Filter (B-tree bitmap scan or sequential scan), then exact distance ranking; HNSW disabled |
| B2-plain | Forced HNSW scan, post-filter, iterative scan off |
| B2-iter | Forced HNSW scan with `hnsw.iterative_scan = relaxed_order` |
| B3 Planner | PostgreSQL's own plan choice |
| **P1 Router** | Estimated rows ≤ T → B1, otherwise B2-iter |
| Oracle | Faster of B1 and B2-iter per query, in hindsight (latency floor, not deployable) |

All HNSW sessions share `m=16`, `ef_construction=64`, `ef_search=100`, `max_scan_tuples=20000`. Only T is
calibrated.

## Results (held-out split, 810 configurations)

| Method | p50 (ms) | p95 (ms) | Recall@10 | Complete | Meets quality |
|---|---|---|---|---|---|
| B1 | 6.579 | 26.281 | 1.000 | 100.0% | yes |
| B2-plain | 2.517 | 3.130 | 0.509 | 39.5% | no |
| B2-iter | 4.569 | 35.489 | 0.968 | 99.6% | yes |
| B3 | 2.317 | 3.096 | 0.828 | 72.8% | no |
| **P1** | **2.500** | **5.327** | **0.990** | **100.0%** | **yes** |
| Oracle | 2.504 | 5.317 | 0.991 | 100.0% | – |

P1 meets the quality targets (Recall@10 ≥ 0.95, ≥ 99% complete) and is faster than the best qualifying fixed
method at both p50 and p95. Per-level results, including the levels where P1 loses, are in
[`results/full/report.md`](results/full/report.md).

## Code

| File | What to look for |
|---|---|
| `main.py` | Step order and command-line arguments |
| `config.py` | YAML settings and paths |
| `data.py` | Dataset reading, seeded filters, exact NumPy ground truth |
| `prepare.py` | Load data, create indexes, verify correctness |
| `search.py` | Method SQL and settings, histogram estimator, P1 routing rule |
| `benchmark.py` | Timing loop, metrics, calibration, Oracle |
| `report.py` | Tables, success criteria and figures |

The main path is `main.py → prepare.py → benchmark.py → report.py`.

## Run

Requirements: Docker with at least 4 GB RAM, [`uv`](https://docs.astral.sh/uv/), and internet access for the
first dataset download (Qdrant's public `laion-small-clip`, checked by SHA-256).

```bash
uv sync
docker compose up -d --wait

# Small trial: 5,000 stored vectors, 20 query vectors (writes results/smoke/)
uv run python main.py --config configs/smoke.yaml

# Full experiment, in a new output folder so results/full/ is kept
uv run python main.py --config configs/full.yaml --run-id full-rerun
```

The pipeline runs eight steps in order. They can also be run one at a time; use the same `--run-id` for every
step:

```bash
uv run python main.py download  --run-id full-rerun   # download or verify the dataset
uv run python main.py prepare   --run-id full-rerun   # filters, split, exact ground truth
uv run python main.py db-setup  --run-id full-rerun   # load 100K vectors, build B-tree and HNSW
uv run python main.py verify    --run-id full-rerun   # check data, distances, filters, exact search
uv run python main.py bench     --run-id full-rerun --phase calibration
uv run python main.py calibrate --run-id full-rerun   # choose and freeze T on calibration data only
uv run python main.py bench     --run-id full-rerun --phase test
uv run python main.py report    --run-id full-rerun
```

`uv run fvs <step>` is equivalent to `uv run python main.py <step>`. A finished benchmark phase is never
overwritten. `db-setup` always recreates the `items` table, so run one experiment at a time and do not run
anything else while queries are being timed. Timings and the calibrated threshold depend on the machine, so a
rerun will not reproduce `results/full/` exactly.

To regenerate `results/full/report.md`, its tables and figures from the committed measurements (no database
needed):

```bash
uv run python main.py report
```

## Tests

```bash
uv run pytest -m "not db"   # no database needed
uv run pytest               # after the smoke pipeline has completed
```

Tests cover data integrity, ground truth, estimates, routing boundaries, SQL plans, calibration, metrics,
pipeline order, and a real database smoke run. The saved-result tests check that the committed measurements in
`results/full/` reproduce the frozen threshold, all 810 held-out routing decisions, and every reported table.

To stop PostgreSQL: `docker compose stop`.
