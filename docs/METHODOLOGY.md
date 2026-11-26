# Proposal experiment

The experiment follows the Group 13 proposal, pages 4–7. Its question is whether a cached row estimate
and one calibrated threshold can choose between exact search and iterative HNSW while meeting the
quality targets. It builds no new index.

## Data and workload

Download Qdrant's public `laion-small-clip.tgz` and verify its configured SHA-256. The full experiment
stores 100,000 vectors with 512 dimensions. Source float16 values are converted losslessly to float32.
Point IDs remain row indices. The metadata filter is `similarity BETWEEN lo AND hi`; this attribute
is distinct from the cosine distance to the query vector.

Deduplicate test query vectors and select 300 with the configured random seed. Split query IDs into
210 calibration and 90 test vectors. All nine filters of a query stay together. Generate quantile
range windows at 0.1%, 0.5%, 1%, 2%, 5%, 10%, 25%, 50%, and 90% target selectivity. Count boundary ties
when recording actual selectivity. Zero-imputed metadata remains stored but is excluded from windows.

NumPy computes exact cosine top-k results in float64 from the same float32 inputs as PostgreSQL.
Break equal-distance ties by ID. Before accepting new ground truth, reproduce the public benchmark's
IDs under its original filters, requiring mean overlap of at least 0.99. Small tie differences are logged.

## Database and methods

Docker runs PostgreSQL 17 with pgvector 0.8.7, using the same database settings as the historical run.
Load with binary COPY, create a metadata B-tree and cosine HNSW index, then VACUUM FREEZE ANALYZE.
Vectors are stored inline, and table autovacuum is disabled during the static experiment.

| Method | Behavior |
|---|---|
| B1 | Exact filtering and distance ranking; `enable_indexscan=off` excludes HNSW |
| B2-plain | Forced HNSW; iterative scanning off |
| B2-iter | Forced HNSW; `hnsw.iterative_scan=relaxed_order` |
| B3 | PostgreSQL planner's choice, with the shared fixed HNSW settings |
| B4 | Estimated rows ≤ T selects B1; otherwise selects B2-iter |
| Oracle | Faster measured B1/B2-iter result per configuration, without enforcing quality |

The proposal describes B1 as B-tree filtering followed by exact ranking. With the user's explicit
choice to preserve comparability, B1 keeps the historical behavior: PostgreSQL may use a bitmap B-tree
or sequential scan. Plans are recorded. It remains exact in both cases.

For B2, adding `0.0::float8` to the metadata expression prevents the B-tree from taking over the
query. Finite values are unchanged; verification checks predicate equivalence. All methods share fixed
HNSW settings: m=16, ef_construction=64, ef_search=100, max_scan_tuples=20000, scan_mem_multiplier=1.
Iterative scanning can stop before returning k valid rows; completeness is measured, not assumed.

Each method keeps one connection. Prepared statements are disabled so queries use their actual filter
parameters. Read back session settings and check every method/configuration with EXPLAIN. Abort on an
incorrect forced plan or changed statistics. Prewarm the table and indexes before timing.

## Estimation and calibration

Read `pg_class` and `pg_stats` once. Cache table row count, histogram bounds, most-common-value
frequencies, and null fraction. Interpolate linearly inside histogram buckets:

```text
estimated rows = table rows × [P(value ≤ hi) − P(value < lo)]
```

Routing performs no database queries. Explicit counts are used only during correctness checks.

Run B1, B2-plain, B2-iter, and B3 on the calibration split. For candidate thresholds 0, infinity, and
midpoints between distinct row estimates, simulate choosing the measured B1 or B2-iter result.
Add the measured routing overhead. Choose the lowest mean latency subject to mean strict Recall@10 ≥
0.95 and completeness ≥ 99%. Break ties by lower p95, then larger T. Freeze T, settings, and the
configuration/statistics hashes before the held-out phase. Reject test rows during calibration.

## Measurements and report

Run one discarded warm-up and five timed repetitions for the full experiment. Randomly interleave
query/method pairs within each pass. The smoke configuration uses fewer vectors and repetitions only
for development. Experiments are sequential on one host with a warm cache.

Latency surrounds execute plus fetch; B4 also includes estimation and the routing decision. Store
returned IDs, errors, routing decisions, and overhead. Aggregate each configuration's latency by its
median repetition, then report p50, p95, and mean over configurations.

Strict Recall@10 is ID overlap with exact ground truth divided by min(10, eligible rows). An empty
eligible set has undefined recall and is excluded from recall means. Completeness requires returning
min(10, eligible rows). Errors are incomplete and have zero recall for nonempty eligible sets.
Tie-tolerant recall is retained internally to diagnose float-rounding differences; success criteria
always use strict recall.

Evaluate the held-out split against the proposal's targets: recall ≥ 0.95, completeness ≥ 99%, and
p50/p95 no slower than the best qualifying fixed baseline. Report results overall and at every
selectivity level, including failures. Oracle gain is `(best − B4) / (best − Oracle)` for each latency
statistic. Mark it unavailable when no baseline qualifies or the denominator offers negligible
headroom, as in the historical calculation.

The report contains the six required method rows, success criteria, estimation errors, and six
plots. No sensitivity variants, supplementary methods, automatic HNSW tuning, or bootstrap analyses
are run. Historical artifacts in `results/full/` and `report/` retain their original contents.
Replaying saved measurements produces a clearly labeled report; it does not claim a new experiment.

Absolute timings depend on the host, cache state, and database build. The saved-results regression
tests establish numerical agreement on the same measurements; new measurements cannot be identical
by construction. Success targets are evaluated honestly rather than forced to pass.
