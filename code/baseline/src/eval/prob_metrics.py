from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import default_group_cols

EPS = 1e-8
Z50 = 0.67448975
Z90 = 1.64485363


def finite_arrays(*values: Iterable[float]) -> tuple[np.ndarray, ...]:
    arrays = [np.asarray(list(v), dtype=float) for v in values]
    if not arrays:
        return tuple()
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return tuple(arr[mask] for arr in arrays)


def gaussian_nll(y_true: Iterable[float], mean: Iterable[float], sigma: Iterable[float] | float) -> float:
    y = np.asarray(list(y_true), dtype=float)
    m = np.asarray(list(mean), dtype=float)
    if np.isscalar(sigma):
        s = np.full_like(y, float(sigma), dtype=float)
    else:
        s = np.asarray(list(sigma), dtype=float)
    s = np.maximum(s, EPS)
    y, m, s = finite_arrays(y, m, s)
    if len(y) == 0:
        return float("nan")
    z = (y - m) / s
    return float(np.mean(0.5 * math.log(2.0 * math.pi) + np.log(s) + 0.5 * z**2))


def coverage(y_true: Iterable[float], lower: Iterable[float], upper: Iterable[float]) -> float:
    y, lo, hi = finite_arrays(y_true, lower, upper)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((lo <= y) & (y <= hi)))


def interval_width(lower: Iterable[float], upper: Iterable[float]) -> float:
    lo, hi = finite_arrays(lower, upper)
    if len(lo) == 0:
        return float("nan")
    return float(np.mean(hi - lo))


def sigma_from_interval(lower: Iterable[float], upper: Iterable[float], z_value: float = Z90) -> np.ndarray:
    lo = np.asarray(list(lower), dtype=float)
    hi = np.asarray(list(upper), dtype=float)
    return np.maximum((hi - lo) / (2.0 * z_value), EPS)


def interval_score(y_true: Iterable[float], lower: Iterable[float], upper: Iterable[float], alpha: float) -> float:
    y, lo, hi = finite_arrays(y_true, lower, upper)
    if len(y) == 0:
        return float("nan")
    score = (hi - lo) + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(score))


def weighted_interval_score(
    y_true: Iterable[float],
    median: Iterable[float],
    intervals: Sequence[tuple[float, Iterable[float], Iterable[float]]],
) -> float:
    y = np.asarray(list(y_true), dtype=float)
    med = np.asarray(list(median), dtype=float)
    numerator = 0.5 * np.abs(y - med)
    denominator = 0.5
    mask = np.isfinite(y) & np.isfinite(med)
    for alpha, lower, upper in intervals:
        lo = np.asarray(list(lower), dtype=float)
        hi = np.asarray(list(upper), dtype=float)
        mask &= np.isfinite(lo) & np.isfinite(hi)
        interval = (hi - lo) + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
        numerator = numerator + (alpha / 2.0) * interval
        denominator += 1.0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(numerator[mask] / denominator))


def crps_gaussian(y_true: Iterable[float], mean: Iterable[float], sigma: Iterable[float] | float) -> float:
    y = np.asarray(list(y_true), dtype=float)
    m = np.asarray(list(mean), dtype=float)
    if np.isscalar(sigma):
        s = np.full_like(y, float(sigma), dtype=float)
    else:
        s = np.asarray(list(sigma), dtype=float)
    s = np.maximum(s, EPS)
    y, m, s = finite_arrays(y, m, s)
    if len(y) == 0:
        return float("nan")
    z = (y - m) / s
    density = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return float(np.mean(s * (z * (2.0 * cdf - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi))))


def group_prob_metrics(forecast_df: pd.DataFrame, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    required = {
        "y_true",
        "pred_mean",
        "pred_lower_50",
        "pred_upper_50",
        "pred_lower_90",
        "pred_upper_90",
    }
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"forecast_df missing required columns: {sorted(missing)}")
    group_cols = list(group_cols) if group_cols is not None else default_group_cols(forecast_df)
    rows = []
    for keys, g in forecast_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sigma = sigma_from_interval(g["pred_lower_90"], g["pred_upper_90"], Z90)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": int(len(g)),
            "gaussian_nll": gaussian_nll(g["y_true"], g["pred_mean"], sigma),
            "coverage_50": coverage(g["y_true"], g["pred_lower_50"], g["pred_upper_50"]),
            "coverage_90": coverage(g["y_true"], g["pred_lower_90"], g["pred_upper_90"]),
            "width_50": interval_width(g["pred_lower_50"], g["pred_upper_50"]),
            "width_90": interval_width(g["pred_lower_90"], g["pred_upper_90"]),
            "wis": weighted_interval_score(
                g["y_true"],
                g["pred_mean"],
                [
                    (0.50, g["pred_lower_50"], g["pred_upper_50"]),
                    (0.10, g["pred_lower_90"], g["pred_upper_90"]),
                ],
            ),
            "crps_gaussian": crps_gaussian(g["y_true"], g["pred_mean"], sigma),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
