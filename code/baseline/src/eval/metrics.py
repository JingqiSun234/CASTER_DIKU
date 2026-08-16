from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

OPTIONAL_CONTEXT_COLS = ("dataset_key", "dataset", "mode", "mode_kind", "forecast_strategy")
BASE_GROUP_COLS = ("method", "component", "horizon", "split")


def finite_pair(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y, p = finite_pair(y_true, y_pred)
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.abs(y - p)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y, p = finite_pair(y_true, y_pred)
    if len(y) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def default_group_cols(df: pd.DataFrame, extra: Sequence[str] = ()) -> list[str]:
    wanted = list(OPTIONAL_CONTEXT_COLS) + list(BASE_GROUP_COLS) + list(extra)
    return [col for col in wanted if col in df.columns]


def group_point_metrics(forecast_df: pd.DataFrame, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    required = {"y_true", "pred_mean"}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"forecast_df missing required columns: {sorted(missing)}")
    group_cols = list(group_cols) if group_cols is not None else default_group_cols(forecast_df)
    if not group_cols:
        raise ValueError("at least one grouping column is required")
    rows = []
    for keys, g in forecast_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        y, p = finite_pair(g["y_true"], g["pred_mean"])
        row.update({
            "n": int(len(y)),
            "mae": mae(y, p),
            "rmse": rmse(y, p),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def macro_average(
    metrics_df: pd.DataFrame,
    value_cols: Sequence[str] = ("mae", "rmse"),
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    missing = set(value_cols) - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df missing value columns: {sorted(missing)}")
    if group_cols is None:
        group_cols = [col for col in list(OPTIONAL_CONTEXT_COLS) + ["method", "split", "horizon"] if col in metrics_df.columns]
    group_cols = list(group_cols)
    rows = []
    grouped = metrics_df.groupby(group_cols, dropna=False) if group_cols else [((), metrics_df)]
    for keys, g in grouped:
        if group_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
        else:
            row = {}
        row["macro_groups"] = int(len(g))
        for col in value_cols:
            values = pd.to_numeric(g[col], errors="coerce")
            if col == "rmse":
                row[col] = float(np.sqrt(np.square(values).mean()))
            else:
                row[col] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True) if group_cols else pd.DataFrame(rows)
