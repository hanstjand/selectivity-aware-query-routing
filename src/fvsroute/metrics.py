"""Retrieval-quality and latency metrics.

Definitions (also in docs/METHODOLOGY.md):
  * Recall@k (PRIMARY, strict) = |R ∩ G| / min(k, |E|), G = exact top-k ordered by (distance, id).
    Undefined (NaN, excluded from means) when |E| = 0.
  * Tie-tolerant recall (DIAGNOSTIC only): a returned r counts if r ∈ E and d(q, r) <= d_k + eps.
  * Complete  <=>  |R| == min(k, |E|). Errors / timeouts are incomplete (and have recall 0 when |E| > 0).
  * Per-config latency = median over timed repetitions; summaries (p50, p95, mean) are taken over configs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIE_EPS = 1e-6


def recall_strict(ids, gt_ids, n_eligible: int, k: int) -> float:
    denom = min(k, n_eligible)
    if denom == 0:
        return float("nan")
    return len(set(list(ids)[:k]) & set(gt_ids)) / denom


def recall_tolerant(in_filter: np.ndarray, dists: np.ndarray, kth_dist: float, n_eligible: int, k: int,
                    eps: float = TIE_EPS) -> float:
    denom = min(k, n_eligible)
    if denom == 0:
        return float("nan")
    hits = int(np.sum(in_filter[:k] & (dists[:k] <= kth_dist + eps)))
    return min(hits, denom) / denom


def is_complete(n_returned: int, n_eligible: int, k: int) -> bool:
    return n_returned == min(k, n_eligible)


def evaluate_executions(raw: pd.DataFrame, workload: pd.DataFrame, gt: pd.DataFrame, values: np.ndarray,
                        base_normed: np.ndarray, queries_normed: dict[int, np.ndarray], k: int) -> pd.DataFrame:
    """Add recall / recall_tol / complete / filter_ok to every execution row.

    Distinct (config, result) pairs are evaluated once (results are deterministic across repetitions).
    """
    wl = workload.set_index("config_id")
    g = gt.set_index("config_id")
    cache: dict[tuple, tuple] = {}
    rec, rec_tol, comp, fok = [], [], [], []
    for cid, ids, err in zip(raw["config_id"], raw["ids"], raw["error"]):
        ids_t = tuple(int(i) for i in (ids if ids is not None else ()))
        key = (cid, ids_t, err is None)
        if key not in cache:
            w = wl.loc[cid]
            n_el = int(w["n_eligible"])
            arr = np.asarray(ids_t, dtype=np.int64)
            if arr.size:
                in_f = (values[arr] >= w["lo"]) & (values[arr] <= w["hi"])
                d = 1.0 - base_normed[arr] @ queries_normed[int(w["query_id"])]
            else:
                in_f = np.zeros(0, bool)
                d = np.zeros(0)
            gt_ids = g.loc[cid, "gt_ids"]
            cache[key] = (
                recall_strict(ids_t, gt_ids, n_el, k) if err is None or n_el == 0 else (0.0 if n_el else float("nan")),
                recall_tolerant(in_f, d, float(g.loc[cid, "gt_kth_dist"]), n_el, k)
                if err is None or n_el == 0 else (0.0 if n_el else float("nan")),
                bool(err is None and is_complete(arr.size, n_el, k)),
                bool(in_f.all()),
            )
        r, rt, c, f = cache[key]
        rec.append(r); rec_tol.append(rt); comp.append(c); fok.append(f)
    out = raw.copy()
    out["recall"] = rec
    out["recall_tol"] = rec_tol
    out["complete"] = comp
    out["filter_ok"] = fok
    return out


def per_config(evaluated: pd.DataFrame, workload: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, method): median latency over timed reps and quality metrics."""
    ev = evaluated.copy()
    ev["latency_ms"] = ev["latency_ns"] / 1e6
    ev["ids_key"] = ev["ids"].map(lambda x: tuple(x) if x is not None else ())
    agg = ev.groupby(["config_id", "method"], sort=False).agg(
        latency_ms_median=("latency_ms", "median"),
        latency_ms_mean=("latency_ms", "mean"),
        reps=("latency_ms", "size"),
        recall=("recall", "mean"),
        recall_tol=("recall_tol", "mean"),
        complete=("complete", "mean"),
        filter_ok=("filter_ok", "all"),
        n_errors=("error", lambda s: int(s.notna().sum())),
        n_distinct_results=("ids_key", "nunique"),
        n_returned=("n_returned", "median"),
    ).reset_index()
    if "route" in ev.columns and ev["route"].notna().any():
        p1 = ev[ev["route"].notna()].groupby(["config_id", "method"]).agg(
            route=("route", lambda s: s.mode().iloc[0]),
            est_rows=("est_rows", "first"),
            overhead_ns=("overhead_ns", "median"),
        ).reset_index()
        agg = agg.merge(p1, on=["config_id", "method"], how="left")
    cols = ["config_id", "query_id", "level_idx", "target_sel", "actual_sel", "n_eligible", "lo", "hi", "split"]
    return agg.merge(workload[cols], on="config_id", how="left")


def latency_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {"p50_ms": float(np.percentile(x, 50)), "p95_ms": float(np.percentile(x, 95)), "mean_ms": float(x.mean())}


def summarize(pc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pc.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        lat = latency_stats(g["latency_ms_median"].to_numpy())
        rows.append({**dict(zip(by, key)), "n_configs": len(g), **lat,
                     "recall": float(np.nanmean(g["recall"])) if g["recall"].notna().any() else float("nan"),
                     "recall_tol": float(np.nanmean(g["recall_tol"])) if g["recall_tol"].notna().any() else float("nan"),
                     "completeness": float(g["complete"].mean()),
                     "n_errors": int(g["n_errors"].sum())})
    return pd.DataFrame(rows)


def meets_quality(summary_row, min_recall: float, min_completeness: float) -> bool:
    return bool(summary_row["recall"] >= min_recall and summary_row["completeness"] >= min_completeness)


def cluster_bootstrap(pc: pd.DataFrame, methods: list[str], reference: str | None, n_resamples: int,
                      seed: int) -> pd.DataFrame:
    """95% percentile CIs by resampling query vectors (all configs of a query move together).

    Returns rows per method (and per `reference - method` difference) for p50/p95/mean latency, recall,
    completeness. Methods must share the same config set.
    """
    lat = pc.pivot(index="config_id", columns="method", values="latency_ms_median")[methods]
    rec = pc.pivot(index="config_id", columns="method", values="recall")[methods]
    com = pc.pivot(index="config_id", columns="method", values="complete")[methods]
    qid = pc.drop_duplicates("config_id").set_index("config_id").loc[lat.index, "query_id"].to_numpy()
    L, R, C = lat.to_numpy(), rec.to_numpy(), com.to_numpy()
    groups = [np.flatnonzero(qid == q) for q in np.unique(qid)]
    rng = np.random.default_rng(seed)
    stats = {s: [] for s in ("p50_ms", "p95_ms", "mean_ms", "recall", "completeness")}
    for _ in range(n_resamples):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        stats["p50_ms"].append(np.percentile(L[idx], 50, axis=0))
        stats["p95_ms"].append(np.percentile(L[idx], 95, axis=0))
        stats["mean_ms"].append(L[idx].mean(axis=0))
        stats["recall"].append(np.nanmean(R[idx], axis=0))
        stats["completeness"].append(C[idx].mean(axis=0))
    rows = []
    ref_i = methods.index(reference) if reference in methods else None
    for s, vals in stats.items():
        a = np.vstack(vals)  # (B, M)
        for j, m in enumerate(methods):
            lo, hi = np.percentile(a[:, j], [2.5, 97.5])
            rows.append({"method": m, "stat": s, "ci_low": lo, "ci_high": hi, "kind": "value"})
            if ref_i is not None and j != ref_i:
                d = a[:, ref_i] - a[:, j]
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append({"method": f"{reference} - {m}", "stat": s, "ci_low": lo, "ci_high": hi,
                             "kind": "difference"})
    return pd.DataFrame(rows)
