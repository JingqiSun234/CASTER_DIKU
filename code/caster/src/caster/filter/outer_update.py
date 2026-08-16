from __future__ import annotations
import numpy as np
import pandas as pd
from .evidence import effective_sample_size, logsumexp




def update_outer_weights(
    previous_weights: pd.DataFrame,
    log_evidence: pd.DataFrame,
    *,
    rho: float = 1.0,
) -> pd.DataFrame:
    ""






    prev = previous_weights[["model_id", "family", "weight"]].copy()
    evidence_columns = ["model_id", "log_evidence"]
    if "evidence_available" in log_evidence.columns:
        evidence_columns.append("evidence_available")
    ev = log_evidence[evidence_columns].copy()
    df = prev.merge(ev, on="model_id", how="left")
    df["log_evidence"] = df["log_evidence"].fillna(0.0).astype(float)
    if "evidence_available" in df.columns:
        values = df["evidence_available"]
        if values.dtype == bool:
            available = values.fillna(False).astype(bool)
        else:
            available = values.astype(str).str.strip().str.lower().isin(
                {"true", "1", "t", "yes", "y"}
            )
    else:
        available = pd.Series(True, index=df.index, dtype=bool)
    df["evidence_available"] = available

    previous = df["weight"].astype(float).to_numpy()
    active = available.to_numpy(dtype=bool)
    if active.any():
        active_mass = float(previous[active].sum())
        logw = (
            np.log(np.maximum(previous[active], 1e-300))
            + float(rho)
            * df.loc[active, "log_evidence"].to_numpy(dtype=float)
        )
        norm = logsumexp(logw)
        updated = previous.copy()
        updated[active] = active_mass * np.exp(logw - norm)
        df["weight"] = updated
    else:
        df["weight"] = previous
    df["model_ess"] = effective_sample_size(df.set_index("model_id")["weight"])
    return df[
        [
            "model_id",
            "family",
            "weight",
            "log_evidence",
            "evidence_available",
            "model_ess",
        ]
    ]

def summarize_model_distribution(weights: pd.DataFrame) -> dict[str, object]:
    w = weights["weight"].astype(float).to_numpy()
    entropy = float(-np.sum(w * np.log(np.maximum(w, 1e-300))))
    return {"model_ess": effective_sample_size(weights.set_index("model_id")["weight"]), "structural_entropy": entropy, "family_mass": weights.groupby("family")["weight"].sum().sort_index().to_dict()}
