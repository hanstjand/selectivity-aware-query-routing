# Methodology

This document defines every method, metric and decision used by the implementation, so the results in
`results/<run_id>/report.md` can be interpreted (and criticised) without reading the code. Code references are
given as `module.function`.

## 1. Research question

Can a simple, selectivity-aware routing rule choose between exact filtered search and HNSW search **per query**
and improve end-to-end latency while keeping mean Recall@10 ≥ 0.95 and completeness ≥ 99%?

- **RQ1** How does filter selectivity affect latency, Recall@10 and completeness of exact and HNSW search?
- **RQ2** Does a threshold router (P1) reduce latency relative to fixed exact, fixed HNSW and PostgreSQL's
  default planner on held-out queries?
- **RQ3** In which selectivity ranges does the router fail, or does its overhead outweigh its savings?

The hypothesis (exact search wins for few matching rows, HNSW for many) is tested, not assumed. The proposal's
"B4 router" is called **P1** here, as in the latest slides.

## 2. Dataset (verified, not assumed)

Source: Qdrant's public filtered-ANN benchmark collection, `laion-small-clip.tgz`
(SHA-256 `6c88614e…d12c`, checked on every download, `dataset.download`).

| Fact | Value | How it was verified |
|---|---|---|
| Vectors | 100,000 × 512, **float16**, ≈ unit norm (0.99945–1.00056) | file size and `np.load`; float16→float32 conversion checked to be lossless (`dataset.validate_raw`) |
| IDs | original id = row index in `vectors.npy` | Qdrant's generator uploads with `ids = range(N)`; our NumPy exact search reproduces Qdrant's `closest_ids` (§4.4) |
| Filter attribute | `similarity` (LAION CLIP image–text score), 97,798 distinct values in [0.295, 0.501] | streamed and profiled |
| Imputed nulls | 152 rows have `similarity = 0.0` | the generator runs `df.fillna(0)`; these rows stay in the table but are never inside a filter window |
| Queries | `tests.jsonl`: 5,000 rows, 4,985 distinct vectors, taken from embeddings *outside* the 100K base | hashing the query vectors |
| Original filters | `similarity > x`, x ∈ [0.30, 0.40] (~3%–99.5% selectivity) | cannot produce 0.1–2% selectivity, so new filters and new ground truth are required |
| License | Qdrant repo has no license file; LAION-400M metadata is CC-BY 4.0 | we only download, never redistribute |

Other payload fields are unsuitable range attributes (`width`/`height` constant 256; `original_*` heavily tied).
No synthetic data is used anywhere in the experiment (unit tests use tiny synthetic fixtures, clearly separated).

## 3. Workload (`workload.py`)

- **Queries**: deduplicate the 5,000 test vectors, then draw 300 with a seeded RNG (`select_queries`). The query
  id is the original `tests.jsonl` line index.
- **Split**: seeded permutation of query ids → 210 calibration / 90 test (`split_queries`). The split is by
  **query vector**, so all nine filters of a query share a split (no leakage through shared vectors).
- **Filters**: for each (query, level s) a closed range `similarity BETWEEN lo AND hi` covering
  m = round(s·N) consecutive values of the sorted attribute, placed at a seeded random offset over the non-imputed
  values (`generate_filters`). The RNG is keyed by (seed, query id, level), so filters do not depend on how many
  queries exist. The actual eligible count |E| is recomputed exactly; boundary ties can only add rows (at most 3
  extra rows observed at full scale). All plots use the **actual** selectivity.
- Levels: 0.1%, 0.5%, 1%, 2%, 5%, 10%, 25%, 50%, 90% → 300 × 9 = **2,700 configurations**, k = 10.

## 4. Ground truth (`groundtruth.py`)

### 4.1 Distance
pgvector's `<=>` is cosine distance `1 − x·q / sqrt(‖x‖²‖q‖²)`, with similarity clipped to [−1, 1].
NumPy computes the same formula in float64 from the same float32 inputs. pgvector accumulates in float32, so the
two differ only by rounding. Measured maximum |difference| on 50 queries × 200 rows: 4.5e-7 (`verify`).

### 4.2 Exact filtered top-k
G(q, F, k) = the k rows of E(F) with the smallest distance, ordered by **(distance, id)**. The id is a
deterministic tie-break. We also store the k-th distance and a `tie_at_k` flag (the (k+1)-th distance is within
1e-6 of the k-th).

### 4.3 Edge cases
|E| < k gives all |E| rows; |E| = 0 gives an empty set. Neither occurs at full scale (the 0.1% level has ≥ 100
rows). The 5,000-row smoke configuration exercises |E| = 5 < k.

### 4.4 Gate: reproducing the benchmark's own ids
Before any new ground truth is accepted, the selected 300 queries must reproduce Qdrant's exact `closest_ids`
under their **original** conditions, with mean overlap ≥ 0.99. Otherwise `prepare` aborts. This validates the id
mapping, the vector loading and the cosine formula.

Result: mean overlap 0.9987, and 297/300 queries match exactly. In all three mismatching queries, the differing
ids sit at **exactly the same distance** (duplicate LAION images), and Qdrant breaks such ties differently. With
tie-aware matching the overlap is 1.0. The maximum score difference is 2.3e-7.

## 5. Database (`sql/`, `db.py`, `docker-compose.yml`)

- **Software**: PostgreSQL 17.11 + pgvector 0.8.7 (`pgvector/pgvector:0.8.7-pg17-bookworm`) in Docker.
- **Table**: `items(id int PK, similarity float8, embedding vector(512))`. The float16 source values are stored
  losslessly as float32. Loaded with binary COPY.
- **`SET STORAGE PLAIN`**: a 512-d float32 row (~2.1 KB) would otherwise be TOASTed. The pgvector README notes
  that the planner ignores TOAST in its cost estimates, so exact scans would pay an invisible detoast cost and the
  planner's B3 decisions would be based on wrong costs. With PLAIN storage the TOAST size is 0 B.
- **Indexes**: B-tree on `similarity`; HNSW `vector_cosine_ops` with m = 16 and ef_construction = 64 (the pgvector
  defaults).
- **Statistics**: `VACUUM (FREEZE, ANALYZE)` runs exactly once after loading, with `default_statistics_target = 100`
  (the default). Autovacuum is disabled on the table. A hash of the statistics is checked at the start and end of
  every benchmark phase, so a stray ANALYZE would abort the run.
- **Server settings**: PostgreSQL defaults except `shared_buffers = 2GB` (the ~0.55 GB heap + indexes must stay
  cached), `maintenance_work_mem = 1GB` and `max_parallel_maintenance_workers = 4` (index build only), and
  `jit = off` (removes JIT-compilation noise for every method). Planner cost constants are untouched, so B3 is the
  default planner. All non-default settings are logged in `env/server_env.json`.
- **Custom plans**: psycopg runs with `prepare_threshold = None`, so every execution is planned with its actual
  `lo/hi`. PostgreSQL never switches to a generic plan, and every method pays its own planning time.

## 6. Methods (`strategies.py`, `router.py`)

All methods run `SELECT id FROM items WHERE <predicate> ORDER BY embedding <=> $q LIMIT 10` on their own
persistent session, whose settings are fixed for the whole run.

| Method | Predicate | Session settings | What it measures |
|---|---|---|---|
| **B1 Exact** | `similarity BETWEEN lo AND hi` | `enable_indexscan = off` | Filter (the planner picks a B-tree bitmap scan or a (parallel) seq scan), then exact distance sort. HNSW is reachable only through an index scan, so it cannot be used. This is pgvector's documented way to get exact search. |
| **B2-plain** | `(similarity + 0.0::float8) BETWEEN lo AND hi` | `enable_seqscan = off`, shared HNSW config, `hnsw.iterative_scan = off` | Forced HNSW scan with post-filtering. |
| **B2-iter** | same as B2-plain | shared HNSW config, `hnsw.iterative_scan = relaxed_order` | Forced HNSW scan with iterative scanning. |
| **B3 Planner** | `similarity BETWEEN lo AND hi` | none of the `enable_*` settings | **PostgreSQL's default planner decision under the shared HNSW configuration**: no SQL rewriting, no hints, the same `hnsw.ef_search` as every other method, iterative scan at its pgvector default (off). |
| **P1 Router** | the chosen arm's | reuses the B1 and B2-iter sessions | est_rows ≤ T → B1, otherwise B2-iter. |
| B3+iter *(supplementary)* | `similarity BETWEEN lo AND hi` | B3 + `relaxed_order` | Does enabling iterative scan and trusting the planner already match P1? It is excluded from the success criteria. |

**Why the B2 predicate is rewritten.** At low selectivity a B-tree path is cheaper than HNSW, so plain SQL cannot
force HNSW. Adding `+ 0.0` makes the predicate non-indexable for the B-tree, and `enable_seqscan = off` removes the
last alternative, so the only remaining path is the HNSW index scan. The two predicates are **semantically
identical**: for every finite float8 x, `x + 0.0 = x` exactly under IEEE 754 (−0.0 + 0.0 = +0.0, which compares
equal). Three checks enforce this: (a) the loader rejects non-finite values; (b)
`SELECT count(*) FROM items WHERE similarity + 0.0::float8 IS DISTINCT FROM similarity` returns 0, which proves
equality for every lo/hi; (c) the per-filter SQL counts equal the NumPy counts for all 2,700 filters. The rewrite
changes only the planner's *row estimate* for the B2 query, which cannot affect execution because the plan is
forced and verified.

**P1** (`router.Router`). est_rows = reltuples · ŝ(lo, hi), where ŝ comes from PostgreSQL's own statistics
(`pg_stats` histogram bounds, MCV list, null_frac, read once after ANALYZE and cached; `estimator.py`):

    P(x ≤ c) = Σ MCV freq(≤ c) + (1 − null_frac − Σ MCV) · hist_frac(c)
    ŝ = P(x ≤ hi) − P(x < lo)

hist_frac interpolates linearly inside the equi-depth bucket, mirroring PostgreSQL's `scalarineqsel`. There is no
COUNT(*) and no database call per query. The P1 timer starts before estimation, so the end-to-end latency includes
estimation, the routing decision and execution. The estimator agrees with PostgreSQL's own `EXPLAIN` row estimate
to a median relative difference of 0.02% (`verify`).

## 7. HNSW configuration

- **Primary configuration**, fixed in advance and shared by every method: `hnsw.ef_search = 100` (the README's
  example value; the default of 40 is documented to return few rows under filtering), `max_scan_tuples = 20000` and
  `scan_mem_multiplier = 1` (defaults), with m = 16 and ef_construction = 64. B2-plain and B2-iter differ only in
  `iterative_scan`.
- **Sensitivity check** (calibration split only, report-only, never used to choose T): B2-iter with
  ef_search ∈ {40, 200}, B2-iter with `strict_order`, and B2-iter with `max_scan_tuples = 100000`.
- **Pre-registered guard** (decided before T, calibration data only): if primary B2-iter's mean strict Recall@10 at
  the broadest level is below 0.95, ef_search is raised to 200 for all methods. ef_search and T are never optimized
  jointly.
- Iterative scanning does **not** guarantee complete or exact results. It stops at `max_scan_tuples` (which does
  not bound the initial scan) or at the `work_mem × scan_mem_multiplier` memory limit, and `relaxed_order` may
  return rows slightly out of distance order. Our metrics are set-based, so ordering does not affect them.

## 8. Verification of intended execution (`runner.py`, `plans.py`)

1. **Settings audit**: before each phase every session reads its settings back with `SHOW` and compares them with
   the intended values. A mismatch aborts the run.
2. **Plan-shape audit at 100% coverage**: plain `EXPLAIN (FORMAT JSON)` for **every** (config × method) of a phase,
   including the sensitivity variants. This costs planning time only. It is equivalent to the executed plan because
   plans are custom, the statistics are frozen and hash-checked, and the SQL, parameters and settings are identical
   to the timed execution. Assertions: B1 has no HNSW node; every B2 variant has an Index Scan on the HNSW index and
   no B-tree or seq scan. The timed run aborts on any violation. B3 and B3+iter plan classes are recorded.
3. **Runtime sample**: `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` on a seeded stratified sample (5 configs per level,
   primary methods + P1 + B3+iter). It confirms actual rows, rows removed by the filter, a warm cache
   (shared read blocks), and server-side execution time. The JSON is kept in `<phase>/explain_analyze/`.
4. **Behavioural checks**: B1 must reach strict recall 1 on all 2,700 configs, and any miss must be a documented
   near-tie (`verify`; 2 such near-ties, 0 unexplained). Every returned id of every method must satisfy the filter.
   Result determinism across repetitions is recorded.

## 9. Metrics (`metrics.py`)

- **Recall@10 (primary, strict)** = |R ∩ G| / min(k, |E|), where G is the exact top-k ordered by (distance, id).
  This is the only recall used in the success criteria, calibration, the guard and the figures. Configs with
  |E| = 0 are excluded (none exist at full scale).
- **Tie-tolerant recall (diagnostic only)**: a returned r counts if r ∈ E and d(q, r) ≤ d_k + 1e-6. It is reported
  in a separate column to show how much of any strict miss is numerical ties.
- **Completeness**: a config is complete iff |R| = min(k, |E|). With |E| = 0 it is complete iff 0 rows are
  returned. Errors and timeouts count as incomplete, with recall 0.
- **Latency**: client-side `perf_counter_ns` around execute + fetch. This is end-to-end, including the network
  round trip (reported separately as `SELECT 1` RTT) and, for P1, routing overhead. Per-config latency is the
  **median of the 5 timed repetitions**. p50, p95 and mean are taken over configs, so each level weighs 1/9.
- **Aggregation**: mean recall and completeness are means over configs. Results are reported overall and per level.
- **Uncertainty**: 95% percentile CIs from a cluster bootstrap that resamples query vectors (all nine configs of a
  query move together), with 2,000 resamples, including paired P1 − baseline differences.

## 10. Benchmark protocol

| Phase | Configs | Methods |
|---|---|---|
| Calibration | 210 queries × 9 = 1,890 | B1, B2-plain, B2-iter, B3 + 4 sensitivity variants |
| Held-out test | 90 queries × 9 = 810 | B1, B2-plain, B2-iter, B3, P1, B3+iter |

Each phase runs: settings audit → statistics check → `pg_prewarm` → RTT baseline → plan-shape audit → EXPLAIN ANALYZE
sample → 1 warm-up pass (discarded) → 5 timed passes. Each pass is a seeded random permutation of all
(config × method) pairs, so methods are interleaved and drift spreads evenly. GC is disabled inside a pass. The
warm-cache, single-client, sequential design is deliberate; cold-cache and concurrent workloads are out of scope.
No runtime is assumed in advance: each phase's wall-clock time is logged in `phase_meta.json`.

## 11. Threshold calibration (`calibrate.py`)

Only calibration-split rows are used; the code raises `LeakageError` on any test row. Candidate thresholds are 0,
∞, and the geometric midpoints between consecutive distinct estimates. For each T, P1 is simulated per config from
the measured medians (B1 if est ≤ T, otherwise B2-iter), plus the measured median routing overhead.

**Rule**: minimize **mean** latency subject to mean strict Recall@10 ≥ 0.95 and completeness ≥ 99%. Ties go to
lower p95, then larger T. T = ∞ (always B1) is always feasible.

T* is frozen in `calibration/frozen_router.json` together with hashes of the statistics, the calibration input and
the config. The held-out phase refuses to start without that file or if the hashes differ, and it runs once. The
p95-optimal feasible T is reported as well, because a mean-optimal T is not necessarily p95-optimal. In the held-out
run, P1's live latency is compared with its chosen arm's latency in the same session; the difference should equal
the overhead, which validates the simulation.

## 12. Oracle and gain captured (`oracle.py`)

- **Oracle (primary reference) = hindsight latency floor**: per held-out config, the faster of B1 and B2-iter (the
  same per-config medians, no routing overhead), with **no quality requirement**. It is an optimistic latency bound,
  not a quality-constrained or deployable method; its strict recall and completeness are reported separately.
  Because it is a per-config minimum, its p50, p95 and mean are each ≤ those of any router that picks one of the two
  arms per config.
- **Oracle-V (optional diagnostic)**: the faster arm among those that are complete with strict Recall@10 = 1 (B1 as
  the fallback). It is supplementary only.
- An aggregate-constrained oracle is **not** computed.
- **Gain captured** = (S(best) − S(P1)) / (S(best) − S(Oracle)), with S one statistic (p50, p95 or mean, each
  reported separately) over one config set (all 810 held-out configs, or the 90 of one level), all from the same
  held-out session. "best" is the best of B1, B2-plain, B2-iter, B3 on S among those meeting criterion 1 on the same
  set. It is N/A (with the reason) if no baseline qualifies or the denominator ≤ 0.

## 13. Success criteria (pre-registered from the slides; held-out)

1. P1 mean strict Recall@10 ≥ 0.95 and ≥ 99% of queries complete.
2. P1 p50 and p95 no slower than the best single baseline (B1, B2-plain, B2-iter, B3) meeting criterion 1, applied
   strictly, with bootstrap CIs to show whether a difference is distinguishable from noise.
3. Gain captured relative to the Oracle latency floor.

All criteria are evaluated overall and per level. `report.md` prints PASS/FAIL without softening and lists the
levels where P1 loses.

## 14. Ambiguities in the proposal and how they were resolved

| Ambiguity | Resolution |
|---|---|
| Router named "B4" (written proposal) vs "P1" (slides) | P1 everywhere |
| 50–100 queries and low/med/high bins (written) vs 300 × 9 levels (slides) | 300 × 9 = 2,700 (the slides win) |
| "Ground truth must match the benchmark's IDs" although the filters are new | Gate: reproduce Qdrant's ids under the original conditions; new filters get recomputed GT |
| Is B1 "B-tree then sort" or any exact plan? | Any exact plan (`enable_indexscan = off`); the access path is recorded per level |
| How to force HNSW when the B-tree is cheaper | Semantically identical non-indexable predicate + `enable_seqscan = off`, verified on every config |
| B3 vs P1 conflates routing with "iterative scan enabled" | B3 as specified; B3+iter added as a supplementary diagnostic |
| Oracle "best per query" vs quality | Latency floor (quality reported, not enforced) as the primary reference; Oracle-V as a diagnostic |
| Recall with < k eligible rows, or ties | Denominator min(k, |E|); strict ids primary; tie-tolerant diagnostic |
| Calibrating "T and HNSW settings" | Only T is calibrated; HNSW fixed in advance with a pre-registered guard and a report-only sensitivity check |

## 15. Threats to validity and limitations

- **Single host, Docker on macOS.** The Linux VM adds client–server latency and noise. Mitigations: randomized
  interleaving, medians of 5, a warm-up pass, a per-pass drift table, and a reported RTT baseline. Absolute numbers
  will differ on other hardware; the comparisons are the result.
- **Warm cache only.** The whole working set is in shared_buffers. With cold-cache or I/O-bound workloads, HNSW's
  random access would cost more.
- **One attribute, one distribution.** `similarity` may correlate with the embedding (it is itself a CLIP score), so
  a filter window can be correlated with vector neighbourhoods. A single row-count threshold cannot capture such
  correlation (proposal §7); the per-level results show where this matters.
- **Static data.** Statistics are exact for the loaded table; no updates, so no staleness is studied.
- **Histogram resolution.** 100 equi-depth buckets (1% each) bound the estimation accuracy at the most selective
  levels; this is measured and reported.
- **Selection bias.** T is chosen on 210 query vectors and evaluated once on 90 others; the per-level and bootstrap
  results show the remaining uncertainty.
- **HNSW parameters.** Index parameters are pgvector defaults; a different m, ef_construction or ef_search shifts the
  crossover point. The sensitivity panel shows the direction of the effect for ef_search and the scan limit.
