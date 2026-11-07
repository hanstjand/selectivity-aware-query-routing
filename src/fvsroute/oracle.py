"""Hindsight references over P1's action space {B1, B2-iter}.

* Oracle (primary reference) = hindsight LATENCY FLOOR: per config, the faster of B1 and B2-iter, with no quality
  requirement and no routing overhead. It is an optimistic latency bound - not a quality-constrained method and
  not deployable. Because it is a per-config minimum, its p50, p95 and mean are each <= those of any router that
  picks one of the two arms per config. Its strict recall and completeness are reported separately.
* Oracle-V (optional diagnostic) = per config, the faster arm among those whose result is complete with strict
  Recall@10 = 1. B1 is the fallback when neither qualifies (only possible on numerical near-ties).
* Gain captured = (S(best) - S(P1)) / (S(best) - S(Oracle)) for ONE statistic S over ONE config set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ARMS = ("B1", "B2-iter")


def _arm_table(pc: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    sub = pc[pc["method"].isin(arms)]
    t = sub.pivot(index="config_id", columns="method", values=["latency_ms_median", "recall", "complete"])
    missing = [a for a in arms if a not in t["latency_ms_median"].columns]
    if missing:
        raise ValueError(f"oracle needs per-config results for {missing}")
    return t


def _assemble(pc: pd.DataFrame, t: pd.DataFrame, choice: pd.Series, name: str) -> pd.DataFrame:
    rows = []
    for cid, arm in choice.items():
        rows.append({"config_id": cid, "method": name, "choice": arm,
                     "latency_ms_median": t.loc[cid, ("latency_ms_median", arm)],
                     "recall": t.loc[cid, ("recall", arm)],
                     "complete": t.loc[cid, ("complete", arm)]})
    out = pd.DataFrame(rows)
    meta = pc.drop_duplicates("config_id")[["config_id", "query_id", "level_idx", "target_sel", "actual_sel",
                                            "n_eligible", "split"]]
    out = out.merge(meta, on="config_id", how="left")
    out["recall_tol"] = np.nan
    out["n_errors"] = 0
    return out


def oracle_floor(pc: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    t = _arm_table(pc, arms)
    lat = t["latency_ms_median"][list(arms)]
    choice = lat.idxmin(axis=1)  # ties -> first arm (B1)
    return _assemble(pc, t, choice, "Oracle")


def oracle_valid(pc: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    t = _arm_table(pc, arms)
    lat = t["latency_ms_median"][list(arms)].copy()
    valid = (t["complete"][list(arms)] >= 1.0) & (t["recall"][list(arms)].fillna(1.0) >= 1.0)
    lat_valid = lat.where(valid, np.inf)        # invalid arms can never be the fastest
    choice = lat_valid.idxmin(axis=1)
    choice = choice.where(valid.any(axis=1), arms[0])  # fall back to B1 (exact)
    out = _assemble(pc, t, choice, "Oracle-V")
    out["fallback"] = (~valid.any(axis=1)).reindex(out["config_id"]).to_numpy()
    return out


def gain_captured(s_best: float | None, s_p1: float, s_oracle: float, min_rel_headroom: float = 0.01,
                  min_abs_headroom_ms: float = 0.01) -> tuple[float | None, str]:
    """(S(best) - S(P1)) / (S(best) - S(Oracle)); all three must be the same statistic on the same configs.

    The ratio is undefined (N/A) when the oracle's headroom over the best baseline is below
    max(min_abs_headroom_ms, min_rel_headroom * S(best)): with (near-)zero headroom the ratio only amplifies noise.
    """
    if s_best is None or not np.isfinite(s_best):
        return None, "no single baseline meets criterion 1"
    denom = s_best - s_oracle
    if denom <= 0:
        return None, "no headroom: oracle not faster than the best baseline"
    if denom < max(min_abs_headroom_ms, min_rel_headroom * s_best):
        return None, f"headroom {denom * 1e3:.1f} us < 1% of best: ratio not meaningful"
    return (s_best - s_p1) / denom, ""
