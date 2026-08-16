from __future__ import annotations
import numpy as np
import pandas as pd

def _softmax(values: np.ndarray) -> np.ndarray:
    z = values - np.max(values)
    e = np.exp(z)
    return e / e.sum()

def compute_family_balanced_prior(registry: pd.DataFrame, family_prior: dict[str, float] | None = None, candidate_scores: dict[str, float] | None = None) -> pd.DataFrame:
    if "model_id" not in registry or "family" not in registry:
        raise ValueError("registry must include model_id and family")
    reg = registry[["model_id", "family"]].copy()
    reg["model_id"] = reg["model_id"].astype(str); reg["family"] = reg["family"].astype(str)
    families = sorted(reg["family"].unique())
    if family_prior is None:
        family_prior = {g: 1.0 / len(families) for g in families}
    total = float(sum(family_prior.get(g, 0.0) for g in families))
    if total <= 0:
        raise ValueError("family prior must have positive total mass")
    fam_prior_norm = {g: float(family_prior.get(g, 0.0)) / total for g in families}
    candidate_scores = candidate_scores or {}
    for g in families:
        idx = reg.index[reg["family"] == g].tolist()
        scores = np.array([float(candidate_scores.get(reg.loc[i, "model_id"], 0.0)) for i in idx], dtype=float)
        for i, w in zip(idx, _softmax(scores)):
            reg.loc[i, "prior_weight"] = fam_prior_norm[g] * float(w)
    reg["prior_weight"] = reg["prior_weight"].astype(float)
    return reg.sort_values("model_id").reset_index(drop=True)

def compute_model_uniform_prior(registry: pd.DataFrame) -> pd.DataFrame:
    if "model_id" not in registry:
        raise ValueError("registry must include model_id")
    reg = registry[["model_id"]].copy()
    reg["model_id"] = reg["model_id"].astype(str)
    if "family" in registry:
        reg["family"] = registry["family"].astype(str).to_numpy()
    else:
        reg["family"] = ""
    if reg.empty:
        raise ValueError("registry must contain at least one model")
    reg["prior_weight"] = 1.0 / float(len(reg))
    return reg.reset_index(drop=True)

def family_mass(weight_table: pd.DataFrame, weight_col: str = "prior_weight") -> pd.Series:
    return weight_table.groupby("family")[weight_col].sum().sort_index()
