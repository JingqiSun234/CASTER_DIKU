from __future__ import annotations
import numpy as np
import pandas as pd

def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("-inf")
    m = np.max(values)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.exp(values - m).sum()))

def compute_log_evidence(scored_rows: pd.DataFrame, *, model_id: str, inner_weights: dict[int, float] | None = None) -> float:
    rows = scored_rows[scored_rows["model_id"].astype(str) == str(model_id)].copy()
    rows = rows[rows["observed_mask"].astype(bool)]
    if rows.empty:
        return 0.0
    if "event_weight" not in rows:
        rows["event_weight"] = 1.0
    particle_scores, particle_ids = [], []
    for pid, group in rows.groupby("particle_id"):
        w = group["event_weight"].astype(float).to_numpy()
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        particle_ids.append(int(pid)); particle_scores.append(float(np.sum(w * group["log_score"].astype(float).to_numpy())))
    if inner_weights is None:
        inner_weights = {pid: 1.0 / len(particle_ids) for pid in particle_ids}
    log_terms = []
    for pid, ell in zip(particle_ids, particle_scores):
        w = float(inner_weights.get(pid, 0.0))
        if w > 0:
            log_terms.append(np.log(w) + ell)
    return logsumexp(np.array(log_terms, dtype=float))

def update_inner_particle_weights(scored_rows: pd.DataFrame, prev_weights: dict[int, float] | None = None) -> dict[int, float]:
    particle_ids = sorted(int(x) for x in scored_rows["particle_id"].unique())
    if prev_weights is None:
        prev_weights = {pid: 1.0 / len(particle_ids) for pid in particle_ids}
    log_terms = []
    for pid in particle_ids:
        pid_rows = scored_rows[(scored_rows["particle_id"] == pid) & (scored_rows["observed_mask"].astype(bool))]
        if pid_rows.empty:
            ell = 0.0
        else:
            w = pid_rows.get("event_weight", pd.Series(1.0, index=pid_rows.index)).astype(float).to_numpy()
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            ell = float(np.sum(w * pid_rows["log_score"].astype(float).to_numpy()))
        log_terms.append(np.log(max(float(prev_weights.get(pid, 0.0)), 1e-300)) + ell)
    norm = logsumexp(np.array(log_terms))
    return {pid: float(np.exp(term - norm)) for pid, term in zip(particle_ids, log_terms)}

def effective_sample_size(weights: dict[int | str, float] | pd.Series) -> float:
    arr = weights.astype(float).to_numpy() if isinstance(weights, pd.Series) else np.array(list(weights.values()), dtype=float)
    s = arr.sum()
    if s <= 0:
        return 0.0
    arr = arr / s
    return float(1.0 / np.square(arr).sum())
