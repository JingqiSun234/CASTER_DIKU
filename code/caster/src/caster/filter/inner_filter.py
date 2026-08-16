from __future__ import annotations
import numpy as np
from .evidence import effective_sample_size

def normalize_weights(weights: dict[int, float]) -> dict[int, float]:
    total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if total <= 0:
        return {int(k): 1.0 / len(weights) for k in weights}
    return {int(k): max(0.0, float(v)) / total for k, v in weights.items()}

def systematic_resample(weights: dict[int, float], seed: int = 0) -> list[int]:
    norm = normalize_weights(weights); keys = list(norm.keys()); probs = np.array([norm[k] for k in keys], dtype=float); n = len(keys)
    rng = np.random.default_rng(seed); positions = (rng.random() + np.arange(n)) / n; cumulative = np.cumsum(probs)
    out, j = [], 0
    for p in positions:
        while p >= cumulative[j]:
            j += 1
        out.append(keys[j])
    return out

def maybe_resample(weights: dict[int, float], threshold_fraction: float = 0.5, seed: int = 0) -> tuple[dict[int, float], list[int] | None, float]:
    norm = normalize_weights(weights); ess = effective_sample_size(norm)
    if ess < threshold_fraction * len(norm):
        ancestors = systematic_resample(norm, seed=seed)
        return {i: 1.0 / len(ancestors) for i in range(len(ancestors))}, ancestors, ess
    return norm, None, ess
