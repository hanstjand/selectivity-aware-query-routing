# Selectivity-Aware Query Routing

A final-project implementation of Group 13's proposal: choose exact search or iterative HNSW for
each filtered vector query in PostgreSQL. The router estimates matching rows from cached `pg_stats`:

```python
estimated_rows = estimate_rows(stats, lo, hi)
method = "B1" if estimated_rows <= threshold else "B2-iter"
```

## Run

Requirements: Docker Desktop running with at least 4 GB RAM, `uv`, and internet access for the first
dataset download. Run these commands from this repository:

```powershell
uv sync
docker compose up -d --wait

# Small trial: 5,000 stored vectors, 20 query vectors
uv run python main.py --config configs/smoke.yaml

# Proposal experiment: 100,000 stored vectors, 300 query vectors
uv run python main.py
```

The full run writes `results/proposal/report.md`, tables, and six figures, including the calibration
curve in PNG and PDF formats. The smoke run writes
`results/smoke/`. Fresh runs measure the current machine; timings and the calibrated threshold can vary.

To show the required results from the historical experiment without running new queries:

```powershell
uv run python main.py report --from-run full --run-id proposal-preview
```

This reads the committed measurements in `results/full/` and writes a shorter report to
`results/proposal-preview/report.md`. It does **not** perform a new benchmark. Required numerical
results are preserved; the historical router name `P1` is displayed as `B4`, as in the proposal.
The original measurements and paper under `results/full/` and `report/` remain historical artifacts.

## Read the code

There are seven implementation files under `src/fvsroute/`:

| File | What to look for |
|---|---|
| `main.py` | Step order and command-line arguments |
| `config.py` | YAML settings and paths |
| `data.py` | Dataset reading, seeded filters, exact NumPy answers |
| `prepare.py` | Load data, create indexes, verify correctness |
| `search.py` | Baseline SQL, histogram estimator, B4 routing rule |
| `benchmark.py` | Timing loop, metrics, calibration, Oracle |
| `report.py` | Required tables and plots |

The main path is `main.py → prepare.py → benchmark.py → report.py`.
There is no session-pool framework or plugin system. Search methods use ordinary connections and cursors.

## Proposal scope

- PostgreSQL + pgvector, 100K LAION/CLIP vectors, 512 dimensions, cosine distance, k = 10.
- 300 query vectors × nine selectivities: 0.1%, 0.5%, 1%, 2%, 5%, 10%, 25%, 50%, 90%.
- Split by query vector: 70% calibration, 30% held-out test.
- B1 exact, B2 plain HNSW, B2 iterative HNSW, B3 PostgreSQL planner, B4 router, and hindsight Oracle.
- Cached histogram estimates; no per-query `COUNT(*)` in the router.
- One warm-up pass and five timed repetitions in randomized order for the full experiment.
- Frozen threshold, EXPLAIN checks, logged software versions and query costs.
- p50/p95 latency, Recall@10, completeness, row-estimate error, and Oracle gain, overall and per level.
- Report negative results as well as successes.

HNSW sensitivity variants, B3+iter, Oracle-V, bootstrap analysis, and automatic `ef_search` adjustment
are not part of this implementation. HNSW settings are fixed in the YAML.

B1 retains the original exact baseline: the planner may use a B-tree bitmap scan or a sequential scan,
but cannot use HNSW. This preserves comparability with the historical experiment. See
[methodology](docs/METHODOLOGY.md) for the details and the proposal's simplified B-tree description.

## Individual steps

The existing `fvs` command also works:

```powershell
uv run fvs download --config configs/full.yaml
uv run fvs prepare --config configs/full.yaml
uv run fvs db-setup --config configs/full.yaml
uv run fvs verify --config configs/full.yaml
uv run fvs bench --config configs/full.yaml --phase calibration
uv run fvs calibrate --config configs/full.yaml
uv run fvs bench --config configs/full.yaml --phase test
uv run fvs report --config configs/full.yaml
```

`uv run fvs all --config configs/full.yaml` runs all of them. A completed benchmark phase is not
overwritten. For another experiment, use the same new `--run-id <name>` for every step, or pass it to
`all`. A run ID separates output files; `db-setup` still recreates the configured database table.
Do not run multiple experiments or tests while timing queries.

### Run it again in Windows PowerShell

Open Docker Desktop and wait until its engine is running. Open PowerShell in this repository, then
prepare the environment and choose a fresh output name:

```powershell
uv sync
docker compose up -d --wait
docker compose ps
$runId = "full-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
```

Run each command below in order. Wait for a command to finish successfully before starting the next.
Keep the same PowerShell window so `$runId` stays the same for all eight steps:

```powershell
# 1. Download the real dataset, or verify the cached archive's checksum
uv run python main.py download --config configs/full.yaml --run-id $runId
# 2. Generate filters, split queries, and compute exact reference answers
uv run python main.py prepare --config configs/full.yaml --run-id $runId
# 3. Load 100,000 vectors and build the B-tree and HNSW indexes
uv run python main.py db-setup --config configs/full.yaml --run-id $runId
# 4. Check stored data, distances, filters, and exact search
uv run python main.py verify --config configs/full.yaml --run-id $runId
# 5. Measure the four baselines on the calibration split
uv run python main.py bench --config configs/full.yaml --run-id $runId --phase calibration
# 6. Choose and freeze the routing threshold using calibration data only
uv run python main.py calibrate --config configs/full.yaml --run-id $runId
# 7. Measure the baselines and B4 on the held-out test split
uv run python main.py bench --config configs/full.yaml --run-id $runId --phase test
# 8. Generate the report, tables, and figures from those fresh measurements
uv run python main.py report --config configs/full.yaml --run-id $runId
```

To run all eight steps automatically, choose a new `$runId` and use:

```powershell
uv run python main.py --config configs/full.yaml --run-id $runId
```

For a small trial, use `configs/smoke.yaml` in the automatic command and a new output name, for example
`--run-id "smoke-$(Get-Date -Format 'yyyyMMdd-HHmmss')"`. Smoke timings do not validate the 100K experiment.

Open `results/<your-run-id>/report.md`. In `tables/success_criteria.csv`, check `scope=overall` for
quality, p50, and p95 PASS/FAIL results; then inspect each selectivity level for weaknesses.
`tables/summary_test_overall.csv` has the six method rows, and `calibration/frozen_router.json` has
the threshold. A new benchmark can have different timings and a different threshold from the paper.

If a command fails, inspect `results/<your-run-id>/logs/` and resolve the reported error before
continuing. A completed benchmark phase refuses to overwrite its measurements. Report generation
can be repeated after the test phase finishes. Database setup recreates `items` in the configured
database even when you choose a new run ID, so finish one experiment before starting another.

## Tests

```powershell
uv run pytest -m "not db"

# After the smoke pipeline has completed:
uv run pytest
```

Tests cover data integrity, ground truth, estimates, routing boundaries, SQL plans, calibration,
retrieval metrics, pipeline order, and a real database smoke run. Saved-result tests reproduce the
historical threshold, all 810 test routes, and required summary values. Tests for removed supplementary
features have been removed with those features.

To stop PostgreSQL:

```powershell
docker compose stop
```
