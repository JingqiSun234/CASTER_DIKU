from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")

FORMAL_RETRIEVAL_PROFILE = "embedding_validation_full_history_v1"


@dataclass(frozen=True)
class RetrievalConfig:
    dim: int = 128
    validation_weight: float = 0.25
    priority_weight: float = 0.05
    family_diversity_bonus: float = 0.0


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(str(text))]


def hashed_text_embedding(text: str, *, dim: int = 128) -> np.ndarray:
    if dim <= 0:
        raise ValueError("dim must be positive")
    vec = np.zeros(dim, dtype=float)
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.sha1(tok.encode()).hexdigest()
        idx = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_registry(registry: pd.DataFrame, *, dim: int = 128) -> pd.DataFrame:
    rows = []
    for _, row in registry.iterrows():
        text = str(row.get("description", "") or "").strip()
        if not text:
            raise ValueError("candidate scientific description must be non-empty")
        emb = hashed_text_embedding(text, dim=dim)
        payload = {f"emb_{i:03d}": float(v) for i, v in enumerate(emb)}
        payload.update({"model_id": row["model_id"], "family": row.get("family", "")})
        rows.append(payload)
    return pd.DataFrame(rows)


def cosine_score(query_embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=float)
    q = np.asarray(query_embedding, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros(matrix.shape[0], dtype=float)
    m = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(m, axis=1)
    denom = np.maximum(norms * q_norm, 1e-12)
    return (m @ q) / denom


def select_top_k_candidates(
    registry: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    query: str,
    top_k: int,
    config: RetrievalConfig | None = None,
) -> pd.DataFrame:
    config = config or RetrievalConfig()
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    enabled = registry.copy()
    if "enabled" in enabled.columns:
        enabled = enabled[enabled["enabled"].map(lambda x: bool(x) if isinstance(x, bool) else str(x).strip().lower() not in {"0", "false", "no", "disabled"})]
    if enabled.empty:
        raise ValueError("No enabled candidates in registry")
    emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
    emb = embeddings[["model_id"] + emb_cols].copy()
    df = enabled.merge(emb, on="model_id", how="left")
    if df[emb_cols].isna().any().any():
        missing = df.loc[df[emb_cols].isna().any(axis=1), "model_id"].astype(str).tolist()
        raise ValueError(f"Missing embeddings for candidates: {missing}")
    matrix = df[emb_cols].to_numpy(dtype=float)
    q = hashed_text_embedding(query, dim=len(emb_cols))
    sim = cosine_score(q, matrix)
    val = pd.to_numeric(df.get("validation_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if val.size and np.nanmax(np.abs(val)) > 0:
        val = (val - np.nanmean(val)) / max(float(np.nanstd(val)), 1e-6)
    priority = pd.to_numeric(df.get("priority", 100), errors="coerce").fillna(100.0).to_numpy(dtype=float)
                                                                                    
    priority_score = -priority / max(float(np.nanstd(priority)), 1.0)
    combined = sim + config.validation_weight * val + config.priority_weight * priority_score
    df = df.copy()
    df["embedding_score"] = sim
    df["validation_score_norm"] = val
    df["priority_score"] = priority_score
    df["combined_score"] = combined
    df["selection_reason"] = [
        f"embedding={s:.4f}; validation_norm={v:.4f}; priority_score={p:.4f}"
        for s, v, p in zip(sim, val, priority_score)
    ]
                                                 
    selected_indices: list[int] = []
    selected_bonuses: list[float] = []
    selected_step_scores: list[float] = []
    remaining = set(range(len(df)))
    selected_families: set[str] = set()
    while remaining and len(selected_indices) < min(top_k, len(df)):
        best_idx = None
        best_score = -math.inf
        best_bonus = 0.0
        for idx in remaining:
            fam = str(df.iloc[idx].get("family", ""))
            bonus = config.family_diversity_bonus if fam not in selected_families else 0.0
            score = float(df.iloc[idx]["combined_score"]) + bonus
            if score > best_score:
                best_score = score
                best_idx = idx
                best_bonus = bonus
        assert best_idx is not None
        selected_indices.append(best_idx)
        selected_bonuses.append(float(best_bonus))
        selected_step_scores.append(float(best_score))
        selected_families.add(str(df.iloc[best_idx].get("family", "")))
        remaining.remove(best_idx)
    out = df.iloc[selected_indices].copy().reset_index(drop=True)
    out["family_diversity_bonus_applied"] = selected_bonuses
    out["selection_step_score"] = selected_step_scores
    out["selection_reason"] = [
        f"{reason}; diversity_bonus={bonus:.4f}; step_score={score:.4f}"
        for reason, bonus, score in zip(out["selection_reason"], selected_bonuses, selected_step_scores)
    ]
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[[
        "rank", "model_id", "family", "candidate_type", "description", "adapter_path",
        "embedding_score", "validation_score", "priority", "combined_score",
        "family_diversity_bonus_applied", "selection_step_score", "selection_reason"
    ]]


def select_top_k_candidates_formal(
    registry: pd.DataFrame,
    embeddings: pd.DataFrame,
    validation_summary: pd.DataFrame,
    *,
    selection_text: str,
    task_embedding: np.ndarray | None = None,
    task_id: str,
    t_sel: str,
    task_spec_sha256: str,
    selection_context_sha256: str,
    candidate_validation_sha256: str,
    top_k: int,
) -> pd.DataFrame:
    ""





    if top_k <= 0:
        raise ValueError("top_k must be positive")
    enabled = registry.copy()
    if "enabled" in enabled.columns:
        enabled = enabled[
            enabled["enabled"].map(
                lambda value: bool(value) if isinstance(value, bool)
                else str(value).strip().lower() not in {"0", "false", "no", "disabled"}
            )
        ]
    required_validation = {"model_id", "validation_utility_robust_norm", "score_source"}
    if missing := sorted(required_validation - set(validation_summary.columns)):
        raise ValueError(f"validation summary missing columns {missing}")
    if set(validation_summary["score_source"].astype(str)) != {"task_specific_official_validation_folds"}:
        raise ValueError("formal validation score source must be official validation folds")
    emb_cols = sorted(c for c in embeddings.columns if c.startswith("emb_"))
    if not emb_cols:
        raise ValueError("embeddings contain no emb_ columns")
    df = enabled.merge(embeddings[["model_id", *emb_cols]], on="model_id", how="left", validate="one_to_one")
    df = df.merge(
        validation_summary[["model_id", "validation_utility_robust_norm", "score_source"]],
        on="model_id", how="left", validate="one_to_one",
    )
    required_values = emb_cols + ["validation_utility_robust_norm"]
    missing_rows = df[df[required_values].isna().any(axis=1)]["model_id"].astype(str).tolist()
    if missing_rows:
        raise ValueError(f"formal retrieval missing embedding/validation inputs for {missing_rows}")
    if task_embedding is None:
        query_embedding = hashed_text_embedding(selection_text, dim=len(emb_cols))
    else:
        query_embedding = np.asarray(task_embedding, dtype=float).reshape(-1)
        if query_embedding.shape != (len(emb_cols),):
            raise ValueError(
                "task embedding dimension does not match candidate embeddings: "
                f"{query_embedding.shape} versus {(len(emb_cols),)}"
            )
        if not np.isfinite(query_embedding).all():
            raise ValueError("task embedding contains a non-finite value")
    df["embedding_score"] = cosine_score(query_embedding, df[emb_cols].to_numpy(dtype=float))
    df["validation_utility_robust_norm"] = pd.to_numeric(df["validation_utility_robust_norm"], errors="raise")
    df["beta_val"] = 1.0
    df["validation_contribution"] = df["validation_utility_robust_norm"]
    df["runtime_score"] = 0.0
    df["beta_runtime"] = 0.0
    df["runtime_contribution"] = 0.0
    df["retrieval_score"] = df["embedding_score"] + df["validation_contribution"]
    df["retrieval_profile"] = FORMAL_RETRIEVAL_PROFILE
    df["priority_weight"] = 0.0
    df["family_diversity_bonus"] = 0.0
    df["greedy_selection"] = False
    df["runtime_functionality_implemented"] = False
    df["task_id"] = str(task_id)
    df["t_sel"] = str(t_sel)
    df["task_spec_sha256"] = str(task_spec_sha256)
    df["selection_context_sha256"] = str(selection_context_sha256)
    df["candidate_validation_sha256"] = str(candidate_validation_sha256)
    out = df.sort_values(["retrieval_score", "model_id"], ascending=[False, True], kind="mergesort").head(top_k).copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[[
        "rank", "model_id", "family", "embedding_score", "validation_utility_robust_norm",
        "beta_val", "validation_contribution", "runtime_score", "beta_runtime", "runtime_contribution",
        "retrieval_score", "score_source", "task_id", "t_sel", "task_spec_sha256",
        "selection_context_sha256", "candidate_validation_sha256", "retrieval_profile",
        "priority_weight", "family_diversity_bonus", "greedy_selection", "runtime_functionality_implemented",
    ]].reset_index(drop=True)
