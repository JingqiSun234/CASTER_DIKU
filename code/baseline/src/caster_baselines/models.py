from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class ForecastRequest:
    entity_id: str
    forecast_origin: pd.Timestamp
    target_time: pd.Timestamp
    component: str
    horizon: int
    split: str


class BaseBaseline:
    name = "base"

    def predict(self, history: pd.DataFrame, component: str, horizon: int) -> tuple[float, float]:
        raise NotImplementedError

    @staticmethod
    def _clean_values(history: pd.DataFrame, component: str) -> np.ndarray:
        vals = history[component].astype(float).to_numpy()
        return vals[np.isfinite(vals)]

    @staticmethod
    def _residual_sigma(values: np.ndarray, default: float = 1.0) -> float:
        if len(values) < 3:
            return float(default)
        diffs = np.diff(values)
        s = float(np.nanstd(diffs))
        return max(s, default * 0.05, 1e-6)


class LastValueBaseline(BaseBaseline):
    name = "last_value"

    def predict(self, history: pd.DataFrame, component: str, horizon: int) -> tuple[float, float]:
        vals = self._clean_values(history, component)
        if len(vals) == 0:
            return 0.0, 1.0
        return float(vals[-1]), self._residual_sigma(vals, default=max(float(vals[-1]) * 0.1, 1.0))


class SeasonalNaiveBaseline(BaseBaseline):
    name = "seasonal_naive"

    def __init__(self, season_length: int = 52):
        self.season_length = int(season_length)

    def predict(self, history: pd.DataFrame, component: str, horizon: int) -> tuple[float, float]:
        vals = self._clean_values(history, component)
        if len(vals) == 0:
            return 0.0, 1.0
        idx = len(vals) - self.season_length + horizon - 1
        if idx < 0 or idx >= len(vals):
            pred = vals[-1]
        else:
            pred = vals[idx]
        return float(pred), self._residual_sigma(vals, default=max(float(vals[-1]) * 0.1, 1.0))




def get_baselines(names: Iterable[str] | None = None) -> list[BaseBaseline]:
    registry = {
        "last_value": LastValueBaseline(),
        "seasonal_naive": SeasonalNaiveBaseline(),
    }
    if names is None:
        return list(registry.values())
    missing = [n for n in names if n not in registry]
    if missing:
        raise ValueError(f"unknown baseline names: {missing}; available={sorted(registry)}")
    return [registry[n] for n in names]
