""




from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SUPPORTED_GROUPS = ("ed", "wastewater", "calendar", "autoregressive", "mobility")


@dataclass(frozen=True)
class CausalCovariateSignal:
    value: float
    feature_count: int
    groups: tuple[str, ...]


def _time_column(panel: pd.DataFrame) -> str:
    for column in ("__time__", "week_end", "date", "time", "ds", "target_time"):
        if column in panel.columns:
            return column
    raise ValueError("covariate panel lacks a time column")


def _entity_column(panel: pd.DataFrame) -> str:
    for column in ("entity_id", "jurisdiction", "region", "unit", "unique_id"):
        if column in panel.columns:
            return column
    raise ValueError("covariate panel lacks an entity column")


def _feature_group(column: str) -> str | None:
    if column.startswith("mobility_"):
        return "mobility"
    if column.startswith("nssp_"):
        return "ed"
    if column.startswith(("flua_ww_", "rsv_ww_")):
        return "wastewater"
    if column in {"weekofyear", "weekofyear_sin", "weekofyear_cos"}:
        return "calendar"
    if "_lag" in column or "_roll" in column:
        return "autoregressive"
    return None


def _relevant(column: str, group: str, component: str) -> bool:
    if group in {"mobility", "calendar"}:
        return True
    name = column.lower()
    target = component.lower()
    if "covid" in target:
        return "covid" in name
    if "flu" in target or "influenza" in target:
        return any(token in name for token in ("flu", "influenza", "flua"))
    return False


def _robust_momentum(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return None
    diffs = np.diff(finite[-min(len(finite), 17):])
    median = float(np.median(diffs))
    scale = float(1.4826 * np.median(np.abs(diffs - median)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(diffs))
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    return float(np.clip(diffs[-1] / scale, -3.0, 3.0))


class CausalCovariateIndex:
    ""

    def __init__(self, panel: pd.DataFrame):
        source = panel.copy()
        entity_col, time_col = _entity_column(source), _time_column(source)
        source["__cov_entity__"] = source[entity_col].astype(str)
        source["__cov_time__"] = pd.to_datetime(source[time_col], errors="raise")
                                                                            
                                                             
        source = source.sort_values(["__cov_entity__", "__cov_time__"]).drop_duplicates(
            ["__cov_entity__", "__cov_time__"], keep="first"
        )
        self._series: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._columns_by_group: dict[str, tuple[str, ...]] = {}
        for group in SUPPORTED_GROUPS:
            columns = tuple(
                sorted(column for column in source.columns if _feature_group(str(column)) == group)
            )
            release_col = f"__release_time__{group}"
            if not columns or release_col not in source.columns:
                continue
            releases_all = pd.to_datetime(source[release_col], errors="raise")
            self._columns_by_group[group] = columns
            for entity, frame in source.groupby("__cov_entity__", sort=False):
                times = frame["__cov_time__"].to_numpy(dtype="datetime64[ns]")
                releases = releases_all.loc[frame.index].to_numpy(dtype="datetime64[ns]")
                for column in columns:
                    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                    self._series[(str(entity), group, column)] = (times, releases, values)
        self._cache: dict[tuple[str, str, int], CausalCovariateSignal] = {}

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(group for group in SUPPORTED_GROUPS if group in self._columns_by_group)

    def signal(
        self, entity_id: object, component: object, forecast_origin: object
    ) -> CausalCovariateSignal:
        origin = pd.Timestamp(forecast_origin)
        key = (str(entity_id), str(component), int(origin.value))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        cutoff = np.datetime64(origin.to_datetime64())
        group_values: list[float] = []
        used_groups: list[str] = []
        feature_count = 0
        for group, columns in self._columns_by_group.items():
            momenta: list[float] = []
            for column in columns:
                if not _relevant(column, group, str(component)):
                    continue
                series = self._series.get((str(entity_id), group, column))
                if series is None:
                    continue
                times, releases, values = series
                visible = (times <= cutoff) & (releases <= cutoff) & np.isfinite(values)
                momentum = _robust_momentum(values[visible])
                if momentum is not None:
                    momenta.append(momentum)
            if momenta:
                group_values.append(float(np.median(momenta)))
                feature_count += len(momenta)
                used_groups.append(group)
        result = CausalCovariateSignal(
            value=float(np.clip(np.median(group_values), -3.0, 3.0)) if group_values else 0.0,
            feature_count=int(feature_count),
            groups=tuple(used_groups),
        )
        self._cache[key] = result
        return result


def adjust_forecast(
    mean: float,
    variance: float,
    target_history: np.ndarray,
    horizon: int,
    signal: CausalCovariateSignal,
    *,
    gain: float,
    damping: float = 0.90,
) -> tuple[float, float]:
    ""

    if signal.feature_count == 0 or gain == 0.0:
        return float(max(mean, 0.0)), float(max(variance, 0.0))
    values = np.asarray(target_history, dtype=float)
    values = values[np.isfinite(values)]
    diffs = np.diff(values[-min(len(values), 17):]) if len(values) >= 2 else np.asarray([])
    scale = float(1.4826 * np.median(np.abs(diffs - np.median(diffs)))) if len(diffs) else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(diffs)) if len(diffs) else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.sqrt(max(float(variance) / max(int(horizon), 1), 1e-6)))
    path_weight = sum(float(damping) ** step for step in range(1, max(int(horizon), 1) + 1))
    delta = float(gain) * float(signal.value) * scale * path_weight
    adjusted_mean = float(max(float(mean) + delta, 0.0))
    adjusted_variance = float(
        max(float(variance) + (float(gain) * scale) ** 2 * path_weight, 0.0)
    )
    return adjusted_mean, adjusted_variance
