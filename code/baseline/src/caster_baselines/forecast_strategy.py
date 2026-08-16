""












from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


DIRECT = "direct"
RECURSIVE_ROLLOUT = "recursive_rollout"
KNOWN_STRATEGIES = {DIRECT, RECURSIVE_ROLLOUT}


def normalize_forecast_strategy(
    value: object = "",
    *,
    mode: object = "",
    mode_kind: object = "",
) -> str:
    ""

    strategy = str(value).strip().lower()
    aliases = {
        "direct": DIRECT,
        "native_direct": DIRECT,
        "native_multi_horizon": DIRECT,
        "recursive_rollout": RECURSIVE_ROLLOUT,
        "rollout": RECURSIVE_ROLLOUT,
        "recursive": RECURSIVE_ROLLOUT,
    }
    if strategy in aliases:
        return aliases[strategy]
    kind = str(mode_kind).strip().lower()
    if kind in aliases:
        return aliases[kind]
    mode_text = str(mode).strip().lower()
    if mode_text.startswith("rollout"):
        return RECURSIVE_ROLLOUT
    return DIRECT


def strategy_from_event(event: pd.Series | dict[str, Any]) -> str:
    getter = event.get
    return normalize_forecast_strategy(
        getter("forecast_strategy", ""),
        mode=getter("mode", ""),
        mode_kind=getter("mode_kind", ""),
    )


def strategy_group_columns(ledger: pd.DataFrame, base: list[str]) -> list[str]:
    ""

    out = list(base)
    for col in ("mode", "forecast_strategy"):
        if col in ledger.columns and col not in out:
            out.append(col)
    return out


def append_predicted_mean(
    times: np.ndarray,
    values: np.ndarray,
    pred_mean: float,
    *,
    cadence_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    ""

    values_out = np.append(np.asarray(values, dtype=float), float(pred_mean))
    times_arr = np.asarray(times, dtype="datetime64[ns]")
    if len(times_arr):
        next_time = times_arr[-1] + np.timedelta64(max(1, int(cadence_days)), "D")
    else:
        next_time = np.datetime64("2000-01-01", "ns")
    return np.append(times_arr, next_time), values_out


def recursive_mean_path(
    *,
    times: np.ndarray,
    values: np.ndarray,
    max_horizon: int,
    cadence_days: int,
    one_step: Callable[[np.ndarray, np.ndarray, int], tuple[float, Any]],
) -> tuple[dict[int, float], dict[int, Any]]:
    ""

    current_times = np.asarray(times, dtype="datetime64[ns]")
    current_values = np.asarray(values, dtype=float)
    means: dict[int, float] = {}
    payloads: dict[int, Any] = {}
    for step in range(1, int(max_horizon) + 1):
        mean, payload = one_step(current_times, current_values, step)
        mean = float(mean)
        if not np.isfinite(mean):
            raise ValueError(f"recursive one-step prediction is non-finite at step={step}")
        means[step] = mean
        payloads[step] = payload
        current_times, current_values = append_predicted_mean(
            current_times,
            current_values,
            mean,
            cadence_days=cadence_days,
        )
    return means, payloads
