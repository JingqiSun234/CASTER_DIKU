from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from eval.metrics import default_group_cols, group_point_metrics, macro_average, mae, rmse
from eval.prob_metrics import (
    Z90,
    coverage,
    crps_gaussian,
    gaussian_nll,
    group_prob_metrics,
    interval_width,
    sigma_from_interval,
    weighted_interval_score,
)


def summarize_forecasts(forecast_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "method", "component", "horizon", "split", "y_true", "pred_mean",
        "pred_lower_50", "pred_upper_50", "pred_lower_90", "pred_upper_90",
    }
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"forecast_df missing required columns: {sorted(missing)}")
    rows = []
    group_cols = default_group_cols(forecast_df)
    for keys, g in forecast_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sigma = sigma_from_interval(g["pred_lower_90"], g["pred_upper_90"], Z90)
        row = dict(zip(group_cols, keys))
        row["horizon"] = int(row["horizon"])
        row.update({
            "n": int(len(g)),
            "mae": mae(g["y_true"], g["pred_mean"]),
            "rmse": rmse(g["y_true"], g["pred_mean"]),
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
        rows.append({
            **row,
        })
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
