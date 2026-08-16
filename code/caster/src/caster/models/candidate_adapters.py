""














from __future__ import annotations

import os
import warnings
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from caster.forecast import FORECAST_ARCHIVE_COLUMNS, attach_ledger_context
from .base_adapter import BaseCandidateAdapter
from .causal_covariates import CausalCovariateIndex, adjust_forecast


PANEL_RELEASE_TIME_COLUMN = "__release_time__"


def detect_device() -> str:
    ""
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


@dataclass
class ReferenceState:
    panel: pd.DataFrame
    seed: int
    meta: dict[str, Any]
    covariate_index: CausalCovariateIndex | None = None


def _time_col(panel: pd.DataFrame) -> str:
    for col in ("week_end", "date", "ds", "time", "target_time"):
        if col in panel.columns:
            return col
    raise ValueError("panel must contain one of week_end/date/ds/time/target_time")


def _entity_col(panel: pd.DataFrame) -> str:
    for col in ("entity_id", "jurisdiction", "region", "unit", "unique_id"):
        if col in panel.columns:
            return col
    return "__entity_id__"


def _normalise_panel(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    e_col = _entity_col(p)
    if e_col == "__entity_id__":
        p[e_col] = "global"
    p["entity_id"] = p[e_col].astype(str)
    t_col = _time_col(p)
    p["__time__"] = pd.to_datetime(p[t_col], errors="coerce", utc=True).dt.tz_convert(None)
    release_col = PANEL_RELEASE_TIME_COLUMN if PANEL_RELEASE_TIME_COLUMN in p.columns else "release_time" if "release_time" in p.columns else ""
    p[PANEL_RELEASE_TIME_COLUMN] = (
        pd.to_datetime(p[release_col], errors="coerce", utc=True).dt.tz_convert(None)
        if release_col else p["__time__"]
    )
    p = p.dropna(subset=["__time__", PANEL_RELEASE_TIME_COLUMN])
    if (p[PANEL_RELEASE_TIME_COLUMN] < p["__time__"]).any():
        raise ValueError("panel release time cannot precede panel observation time")
    return p.sort_values(["entity_id", "__time__"]).reset_index(drop=True)


def _ledger_entity(row: pd.Series) -> str:
    for col in ("entity_id", "jurisdiction", "region", "unit", "unique_id"):
        if col in row.index:
            return str(row[col])
    return "global"


def _series(panel: pd.DataFrame, entity_id: str, component: str, origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    cutoff = pd.Timestamp(origin)
    subset = panel[
        (panel["entity_id"].astype(str) == str(entity_id))
        & (panel["__time__"] <= cutoff)
        & (panel[PANEL_RELEASE_TIME_COLUMN] <= cutoff)
    ].copy()
    if component in subset.columns:
        values = pd.to_numeric(subset[component], errors="coerce")
        times = subset["__time__"]
    elif "component" in subset.columns and ({"value", "observed_value"} & set(subset.columns)):
        subset = subset[subset["component"].astype(str) == str(component)].copy()
        value_col = "observed_value" if "observed_value" in subset.columns else "value"
        values = pd.to_numeric(subset[value_col], errors="coerce")
        times = subset["__time__"]
    else:
        return np.asarray([], dtype="datetime64[ns]"), np.asarray([], dtype=float)
    mask = np.isfinite(values.to_numpy(dtype=float))
    return times.to_numpy(dtype="datetime64[ns]")[mask], values.to_numpy(dtype=float)[mask]


def _build_series_index(panel: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    index: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if "component" in panel.columns and ({"value", "observed_value"} & set(panel.columns)):
        value_col = "observed_value" if "observed_value" in panel.columns else "value"
        for (entity, component), group in panel.groupby(["entity_id", "component"], sort=False):
            group = group.sort_values("__time__")
            values = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
            times = group["__time__"].to_numpy(dtype="datetime64[ns]")
            releases = group[PANEL_RELEASE_TIME_COLUMN].to_numpy(dtype="datetime64[ns]")
            mask = np.isfinite(values)
            index[(str(entity), str(component))] = (times[mask], releases[mask], values[mask])
        return index
    target_cols = [c for c in panel.columns if c not in {"entity_id", "__time__", "week_end", "date", "ds", "time", "target_time"}]
    for entity, group in panel.groupby("entity_id", sort=False):
        group = group.sort_values("__time__")
        times = group["__time__"].to_numpy(dtype="datetime64[ns]")
        releases = group[PANEL_RELEASE_TIME_COLUMN].to_numpy(dtype="datetime64[ns]")
        for component in target_cols:
            values = pd.to_numeric(group[component], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values)
            if mask.any():
                index[(str(entity), str(component))] = (times[mask], releases[mask], values[mask])
    return index


def _series_from_index(
    index: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    entity_id: str,
    component: str,
    origin: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    times, releases, values = index.get(
        (str(entity_id), str(component)),
        (np.asarray([], dtype="datetime64[ns]"), np.asarray([], dtype="datetime64[ns]"), np.asarray([], dtype=float)),
    )
    if len(times) == 0:
        return times, values
    cutoff = np.datetime64(pd.Timestamp(origin).to_datetime64())
    visible = (times <= cutoff) & (releases <= cutoff)
    return times[visible], values[visible]


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))


def _residual_var(values: np.ndarray, pred: float = 0.0, floor: float = 1.0) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) >= 3:
        diffs = np.diff(y)
        return float(max(np.nanvar(diffs), 0.05 * (1.0 + abs(pred)), floor, 1e-6))
    return float(max(floor, 0.10 * (1.0 + abs(pred)), 1e-6))


def _nonneg(x: float) -> float:
    return float(max(0.0, x))


def _season_length_from_history(times: np.ndarray, default: int = 52) -> int:
    if len(times) < 3:
        return default
    dt = pd.Series(pd.to_datetime(times)).diff().dt.days.dropna()
    if dt.empty:
        return default
    cadence = float(dt.median())
    if cadence <= 1.5:
        return 7
    if 6 <= cadence <= 8:
        return 52
    return max(1, int(round(365.25 / max(cadence, 1.0))))


def _naive_season_length_from_history(times: np.ndarray, default: int = 8) -> int:
    if len(times) < 2:
        return default
    dt = pd.Series(pd.to_datetime(times)).diff().dt.days.dropna()
    if dt.empty:
        return default
    cadence = float(dt.median())
    if cadence <= 1.5:
        return 7
    if 6 <= cadence <= 8:
        return 8
    return max(1, int(round(56.0 / max(cadence, 1.0))))


def _baseline_residual_sigma(values: np.ndarray, default: float = 1.0) -> float:
    ""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return float(default)
    s = float(np.nanstd(np.diff(y)))
    return max(s, default * 0.05, 1e-6)


def _baseline_residual_var(values: np.ndarray, default: float = 1.0) -> float:
    sigma = _baseline_residual_sigma(values, default=default)
    return float(sigma * sigma)


def _statsforecast_seasonality(values: np.ndarray, max_horizon: int, season_length: int) -> tuple[bool, int]:
    ""
    seasonal = bool(
        season_length > 1
        and len(values) >= max(2 * int(season_length), int(max_horizon) + int(season_length) + 2)
    )
    return seasonal, int(season_length) if seasonal else 1


def _freq_from_times(times: np.ndarray | None) -> str:
    if times is None or len(times) < 2:
        return "D"
    dt = pd.Series(pd.to_datetime(times)).diff().dt.days.dropna()
    if dt.empty:
        return "D"
    cadence_days = max(1, int(round(float(dt.median()))))
    if cadence_days <= 1:
        return "D"
    return f"{cadence_days}D"


def _cadence_days_from_times(times: np.ndarray | None, default: int = 1) -> int:
    if times is None or len(times) < 2:
        return int(default)
    dt = pd.Series(pd.to_datetime(times)).diff().dt.days.dropna()
    if dt.empty:
        return int(default)
    return max(1, int(round(float(dt.median()))))


def _target_time_from_horizon(times: np.ndarray | None, horizon: int) -> pd.Timestamp:
    if times is None or len(times) == 0:
        start = pd.Timestamp("2000-01-01")
    else:
        start = pd.Timestamp(pd.to_datetime(times[-1]))
    return start + pd.Timedelta(days=_cadence_days_from_times(times) * int(horizon))


def _forecast_strategy(event: pd.Series) -> str:
    declared = str(event.get("forecast_strategy", "")).strip().lower()
    if declared in {"recursive_rollout", "rollout", "recursive"}:
        return "recursive_rollout"
    if declared in {"direct", "native_direct", "native_multi_horizon"}:
        return "direct"
    if str(event.get("mode_kind", "")).strip().lower() == "rollout":
        return "recursive_rollout"
    return "recursive_rollout" if str(event.get("mode", "")).strip().lower().startswith("rollout") else "direct"


def _append_recursive_mean(
    times: np.ndarray,
    values: np.ndarray,
    mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    cadence_days = _cadence_days_from_times(times, default=1)
    if len(times):
        next_time = np.asarray(times, dtype="datetime64[ns]")[-1] + np.timedelta64(cadence_days, "D")
    else:
        next_time = np.datetime64("2000-01-01", "ns")
    return (
        np.append(np.asarray(times, dtype="datetime64[ns]"), next_time),
        np.append(np.asarray(values, dtype=float), float(mean)),
    )


def _strategy_group_columns(ledger: pd.DataFrame) -> list[str]:
    cols = [next(col for col in ("entity_id", "jurisdiction", "region", "unit", "unique_id") if col in ledger.columns), "component", "forecast_origin"]
    cols.extend(col for col in ("mode", "forecast_strategy") if col in ledger.columns)
    return cols


def _nf_prediction_col(frame: pd.DataFrame, model_name: str) -> str:
    candidates = [model_name, model_name.upper(), model_name.capitalize()]
    for col in candidates:
        if col in frame.columns:
            return col
    skip = {"unique_id", "ds", "cutoff"}
    numeric = [
        col
        for col in frame.columns
        if col not in skip and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not numeric:
        raise ValueError(f"no prediction column found in NeuralForecast output columns={list(frame.columns)}")
    return numeric[0]


def _nf_choose_input_size(n_rows: int, h: int) -> int:
    ""
    min_len = int(n_rows)
    if min_len <= h + 1:
        return max(1, min_len - 1)
    target = max(2 * int(h), 8)
                                                                            
                                                                       
    return int(max(2, min(target, max(2, min_len - int(h) - 1))))


def _nf_train_df(values: np.ndarray, times: np.ndarray | None, freq: str) -> pd.DataFrame:
    original = np.asarray(values, dtype=float)
    mask = np.isfinite(original)
    y = original[mask]
    if times is not None and len(times) == len(original):
        ds = pd.to_datetime(np.asarray(times)[mask], errors="coerce")
    else:
        ds = pd.date_range("2000-01-01", periods=len(y), freq=freq)
    frame = pd.DataFrame({"unique_id": "series", "ds": ds, "y": y})
    return frame.dropna(subset=["ds", "y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _disable_cuda_for_cpu(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _build_baseline_neuralforecast_model(
    model_name: str,
    *,
    h: int,
    input_size: int,
    max_steps: int,
    seed: int,
    device: str,
):
    ""
    _disable_cuda_for_cpu(device)
    from neuralforecast.models import DeepAR, LSTM, NBEATS, NHITS, PatchTST, TFT
    from neuralforecast.losses.pytorch import MAE
    try:
        from neuralforecast.losses.pytorch import DistributionLoss
        probabilistic_loss = DistributionLoss(distribution="Normal", level=[50, 90])
    except Exception:
        probabilistic_loss = MAE()

    common = {
        "h": h,
        "input_size": input_size,
        "max_steps": max_steps,
        "early_stop_patience_steps": -1,
        "val_check_steps": max(1, max_steps),
        "batch_size": 32,
        "valid_batch_size": 32,
        "windows_batch_size": 256,
        "inference_windows_batch_size": 256,
        "random_seed": seed,
        "scaler_type": "robust",
        "alias": model_name,
        "enable_checkpointing": False,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "accelerator": device,
        "devices": 1,
    }
    if model_name == "nbeats":
        return NBEATS(
            stack_types=["identity", "trend", "seasonality"],
            n_blocks=[2, 2, 2],
            mlp_units=[[128, 128], [128, 128], [128, 128]],
            loss=MAE(),
            **common,
        )
    if model_name == "nhits":
        return NHITS(
            stack_types=["identity", "identity", "identity"],
            n_blocks=[1, 1, 1],
            mlp_units=[[128, 128], [128, 128], [128, 128]],
            n_pool_kernel_size=[2, 2, 1],
            n_freq_downsample=[4, 2, 1],
            loss=MAE(),
            **common,
        )
    if model_name == "deepar":
        return DeepAR(
            h_train=h,
            lstm_n_layers=2,
            lstm_hidden_size=40,
            lstm_dropout=0.10,
            decoder_hidden_layers=1,
            decoder_hidden_size=40,
            trajectory_samples=100,
            loss=probabilistic_loss,
            valid_loss=probabilistic_loss,
            **common,
        )
    if model_name == "lstm":
        return LSTM(
            h_train=h,
            encoder_n_layers=2,
            encoder_hidden_size=64,
            encoder_dropout=0.10,
            decoder_hidden_size=64,
            decoder_layers=1,
            loss=MAE(),
            **common,
        )
    if model_name == "patchtst":
        patch_len = max(2, min(16, input_size))
        return PatchTST(
            encoder_layers=3,
            n_heads=4,
            hidden_size=64,
            linear_hidden_size=128,
            dropout=0.10,
            fc_dropout=0.10,
            head_dropout=0.10,
            attn_dropout=0.10,
            patch_len=patch_len,
            stride=max(1, patch_len // 2),
            loss=MAE(),
            **common,
        )
    if model_name == "tft":
        return TFT(
            hidden_size=64,
            n_head=4,
            n_rnn_layers=2,
            dropout=0.10,
            loss=MAE(),
            **common,
        )
    raise ValueError(f"unsupported NeuralForecast model {model_name}")


def _baseline_neuralforecast_predict_one(
    model_name: str,
    values: np.ndarray,
    horizon: int,
    times: np.ndarray | None,
    *,
    max_steps: int,
    seed: int,
    device: str,
) -> float:
    from neuralforecast import NeuralForecast

    h = max(1, int(horizon))
    freq = _freq_from_times(times)
    train_df = _nf_train_df(values, times, freq)
    input_size = _nf_choose_input_size(len(train_df), h)
    if len(train_df) <= input_size or len(train_df) <= h:
        raise ValueError(
            f"insufficient training history for NeuralForecast: "
            f"n={len(train_df)} input_size={input_size} h={h}"
        )
    model = _build_baseline_neuralforecast_model(
        model_name,
        h=h,
        input_size=input_size,
        max_steps=int(max_steps),
        seed=int(seed),
        device=str(device),
    )
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=train_df, val_size=0, verbose=False)
    forecast = nf.predict(df=train_df, verbose=False)
    col = model_name if model_name in forecast.columns else _nf_prediction_col(forecast, model_name)
    forecast = forecast.copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
    forecast = forecast.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    if len(forecast) < h:
        raise ValueError(f"NeuralForecast output shorter than requested horizon: {len(forecast)} < {h}")
    value = pd.to_numeric(pd.Series([forecast.iloc[h - 1][col]]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        raise ValueError("NeuralForecast prediction is non-finite")
    return float(value)


class SeriesForecastAdapter(BaseCandidateAdapter):
    model_id = "series_forecaster"
    family = "statistical"

    def __init__(self, model_id: str | None = None, family: str | None = None, **kwargs: Any) -> None:
        if model_id is not None:
            self.model_id = str(model_id)
        if family is not None:
            self.family = str(family)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> ReferenceState:
        return ReferenceState(_normalise_panel(panel), int(seed), {"adapter": self.__class__.__name__})

    def transition(self, state: ReferenceState, forecast_origin: pd.Timestamp) -> ReferenceState:
        return state

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        pred = float(values[-1])
        return _nonneg(pred), _residual_var(values, pred)

    def forecast_one_for_event(
        self,
        state: ReferenceState,
        event: pd.Series,
        values: np.ndarray,
        horizon: int,
        times: np.ndarray | None = None,
        *,
        recursive_step: int | None = None,
    ) -> tuple[float, float]:
        return self.forecast_one(values, horizon, times)

    def forecast_ledger(self, state: ReferenceState, ledger: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        series_index = _build_series_index(state.panel)
        for _, group in ledger.groupby(_strategy_group_columns(ledger), dropna=False, sort=False):
            representative = group.iloc[0]
            origin = pd.Timestamp(representative["forecast_origin"])
            entity = _ledger_entity(representative)
            component = str(representative["component"])
            times, values = _series_from_index(series_index, entity, component, origin)
            strategy = _forecast_strategy(representative)
            requested = sorted(pd.to_numeric(group["horizon"], errors="raise").astype(int).unique())
            path: dict[int, tuple[float, float]] = {}
            if strategy == "recursive_rollout":
                recursive_times = np.asarray(times, dtype="datetime64[ns]")
                recursive_values = np.asarray(values, dtype=float)
                event_by_horizon = {
                    int(pd.to_numeric(event["horizon"], errors="raise")): event
                    for _, event in group.iterrows()
                }
                for step in range(1, max(requested) + 1):
                    step_event = event_by_horizon.get(step, representative)
                    mean, var = self.forecast_one_for_event(
                        state,
                        step_event,
                        recursive_values,
                        1,
                        recursive_times,
                        recursive_step=step,
                    )
                    path[step] = (_nonneg(mean), float(max(var, 0.0)))
                    recursive_times, recursive_values = _append_recursive_mean(recursive_times, recursive_values, mean)
            else:
                for _, event in group.iterrows():
                    horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
                    path[horizon] = self.forecast_one_for_event(
                        state,
                        event,
                        values,
                        horizon,
                        times,
                    )
            for _, e in group.iterrows():
                horizon = int(pd.to_numeric(e["horizon"], errors="raise"))
                mean, var = path[horizon]
                row = {
                    "dataset": str(e.get("dataset", "dataset")),
                    "model_id": self.model_id,
                    "family": self.family,
                    "particle_id": 0,
                    "entity_id": entity,
                    "forecast_origin": e["forecast_origin"],
                    "target_time": e["target_time"],
                    "component": component,
                    "horizon": horizon,
                    "forecast_id": e["forecast_id"],
                    "pred_mean": _nonneg(mean),
                    "pred_var": float(max(var, 0.0)),
                    "generated_at": e["forecast_origin"],
                    "features_available_until": e["forecast_origin"],
                }
                if state.covariate_index is not None:
                    signal = state.covariate_index.signal(
                        entity, component, e["forecast_origin"]
                    )
                    row.update(
                        {
                            "causal_covariate_signal": signal.value,
                            "causal_covariate_feature_count": signal.feature_count,
                            "causal_covariate_groups": "|".join(signal.groups),
                            "causal_covariate_adjustment_applied": signal.feature_count > 0,
                        }
                    )
                rows.append(row)
        return attach_ledger_context(pd.DataFrame(rows), ledger)

    def forecast_draws(self, state: ReferenceState, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        archive = self.forecast_ledger(state, ledger)
        rng = np.random.default_rng(seed)
        rows: list[dict[str, Any]] = []
        for _, r in archive.iterrows():
            scale = float(np.sqrt(max(float(r["pred_var"]), 1e-6)))
            draws = rng.normal(float(r["pred_mean"]), scale, size=int(n_draws))
            for draw_id, draw in enumerate(draws):
                rows.append({
                    "forecast_id": r["forecast_id"],
                    "model_id": self.model_id,
                    "particle_id": int(r["particle_id"]),
                    "draw_id": int(draw_id),
                    "draw": _nonneg(float(draw)),
                })
        return pd.DataFrame(rows)

    def serialize_state(self, state: ReferenceState) -> dict[str, Any]:
        return {"seed": state.seed, "rows": int(len(state.panel)), **state.meta}


class _CausalCovariateAdjustmentMixin:
    ""






    covariate_gain: float
    covariate_damping: float

    def __init__(
        self,
        covariate_gain: float = 0.15,
        covariate_damping: float = 0.90,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.covariate_gain = float(covariate_gain)
        self.covariate_damping = float(covariate_damping)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> ReferenceState:
        state = super().initialize(panel, seed=seed)
        state.covariate_index = CausalCovariateIndex(state.panel)
        state.meta.update(
            {
                "causal_covariate_variant": True,
                "causal_covariate_groups": list(state.covariate_index.groups),
                "causal_covariate_gain": self.covariate_gain,
                "causal_covariate_damping": self.covariate_damping,
                "causal_covariate_cutoff": "feature_release_time_lte_forecast_origin",
                "algorithm_learning_policy": "no_learning_behavior_preserved",
            }
        )
        return state

    def forecast_one_for_event(
        self,
        state: ReferenceState,
        event: pd.Series,
        values: np.ndarray,
        horizon: int,
        times: np.ndarray | None = None,
        *,
        recursive_step: int | None = None,
    ) -> tuple[float, float]:
        mean, variance = super().forecast_one_for_event(
            state,
            event,
            values,
            horizon,
            times,
            recursive_step=recursive_step,
        )
        if state.covariate_index is None:
            return mean, variance
        signal = state.covariate_index.signal(
            _ledger_entity(event),
            str(event["component"]),
            event["forecast_origin"],
        )
        return adjust_forecast(
            mean,
            variance,
            values,
            horizon,
            signal,
            gain=self.covariate_gain,
            damping=self.covariate_damping,
        )


                                                                             
                              
                                                                             
class LastValueReferenceAdapter(SeriesForecastAdapter):
    model_id = "last_value"
    family = "statistical"

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return 0.0, 1.0
        pred = float(vals[-1])
        var = _baseline_residual_var(vals, default=max(pred * 0.1, 1.0))
        return _nonneg(pred), var


class SeasonalNaiveReferenceAdapter(SeriesForecastAdapter):
    model_id = "seasonal_naive"
    family = "statistical"

    def __init__(
        self,
        season_length: int | None = None,
        daily_season_length: int = 7,
        weekly_season_length: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.season_length = season_length
        self.daily_season_length = int(daily_season_length)
        self.weekly_season_length = int(weekly_season_length)

    def _season(self, times: np.ndarray | None) -> int:
        if self.season_length is not None:
            return int(self.season_length)
        inferred = _naive_season_length_from_history(
            times if times is not None else np.asarray([])
        )
        if inferred == 7:
            return self.daily_season_length
        if inferred == 8:
            return self.weekly_season_length
        return inferred

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        season = self._season(times)
        idx = len(values) - season + int(horizon) - 1
        if not 0 <= idx < len(values):
            raise ValueError(
                f"SeasonalNaive-{season} has no native previous-season observation"
            )
        pred = float(values[idx])
        var = _baseline_residual_var(values, default=max(float(values[-1]) * 0.1, 1.0))
        return _nonneg(pred), var

    def forecast_one_for_event(
        self,
        state: ReferenceState,
        event: pd.Series,
        values: np.ndarray,
        horizon: int,
        times: np.ndarray | None = None,
        *,
        recursive_step: int | None = None,
    ) -> tuple[float, float]:
        if times is None or len(times) == 0 or len(values) == 0:
            raise ValueError("SeasonalNaive requires timestamped history")
        season = self._season(times)
        ordered_times = pd.to_datetime(np.asarray(times), errors="coerce")
        valid_times = ordered_times[~pd.isna(ordered_times)]
        if len(valid_times) < 2:
            raise ValueError("SeasonalNaive requires cadence-identifiable history")
        cadence_days = int(
            round(float(pd.Series(valid_times).diff().dt.days.dropna().median()))
        )
        target = pd.to_datetime(event.get("target_time"), errors="coerce")
        if pd.isna(target):
            raise ValueError("SeasonalNaive requires a valid target_time")
        seasonal_time = pd.Timestamp(target) - pd.Timedelta(
            days=season * cadence_days
        )
        matches = ordered_times == seasonal_time
        if not bool(np.any(matches)):
            raise ValueError(
                f"SeasonalNaive-{season} has no observation at {seasonal_time.date()}"
            )
        pred = float(np.asarray(values, dtype=float)[np.flatnonzero(matches)[-1]])
        var = _baseline_residual_var(
            np.asarray(values, dtype=float),
            default=max(float(values[-1]) * 0.1, 1.0),
        )
        return _nonneg(pred), var


class DriftReferenceAdapter(SeriesForecastAdapter):
    model_id = "drift"
    family = "state_space"

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        if len(values) < 2:
            pred = float(values[-1])
        else:
                                                                                      
            drift = float((values[-1] - values[0]) / max(len(values) - 1, 1))
            pred = float(values[-1] + int(horizon) * drift)
        return _nonneg(pred), _residual_var(values, pred) * max(1, int(horizon))


class CovariateDriftReferenceAdapter(_CausalCovariateAdjustmentMixin, DriftReferenceAdapter):
    ""

    model_id = "covariate_drift"
    family = "state_space"


                                                                             
                                 
                                                                             
def _stable_uint32(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def _binomial_draw(rng: np.random.Generator, n: float, p: float) -> float:
    trials = int(round(max(float(n), 0.0)))
    prob = float(np.clip(float(p), 0.0, 1.0))
    if trials <= 0 or prob <= 0.0:
        return 0.0
    return float(rng.binomial(trials, prob))


DETERMINISTIC_TRANSITION_MODES = {"", "deterministic", "deterministic_mean_field", "mean_field"}
STOCHASTIC_TRANSITION_MODES = {"stochastic", "stochastic_binomial", "binomial", "stochastic_tau_leap"}


def _mode_is_stochastic(mode: object) -> bool:
    ""

    return str(mode).strip().lower() in STOCHASTIC_TRANSITION_MODES


class SIRReferenceAdapter(SeriesForecastAdapter):
    model_id = "sir_tau"
    family = "compartmental"

    def __init__(
        self,
        population: float = 1_000_000.0,
        gamma: float = 1 / 5.0,
        reporting_rate: float = 0.04,
        transition_mode: str = "stochastic_binomial",
        n_simulations: int = 64,
        log_beta_rw_scale: float = 0.0,
        log_gamma_rw_scale: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.population = float(population)
        self.gamma = float(gamma)
        self.reporting_rate = float(reporting_rate)
        self.transition_mode = str(transition_mode)
        self.n_simulations = int(n_simulations)
        self.log_beta_rw_scale = max(float(log_beta_rw_scale), 0.0)
        self.log_gamma_rw_scale = max(float(log_gamma_rw_scale), 0.0)

    def _estimate_rt(self, values: np.ndarray) -> float:
        y = np.maximum(np.asarray(values, dtype=float), 1.0)
        if len(y) < 3:
            return 1.0
        growth = np.diff(np.log(y[-min(len(y), 8):]))
        return float(np.clip(np.exp(np.nanmean(growth)) / max(1.0 - self.gamma, 1e-6), 0.2, 3.5))

    def _initial_sir(self, values: np.ndarray) -> tuple[float, float, float]:
        last_obs = float(max(values[-1], 0.0))
        pop = max(self.population, 1.0)
        i = min(max(last_obs / max(self.reporting_rate, 1e-6), 1.0), 0.25 * pop)
        r = min(0.05 * pop + max(np.nansum(values[-min(len(values), 21):]), 0.0) / max(self.reporting_rate, 1e-6), 0.80 * pop)
        s = max(pop - i - r, 0.0)
        return s, i, r

    def _initial_rates(self, values: np.ndarray) -> tuple[float, float]:
        ""

        gamma = max(float(self.gamma), np.finfo(float).tiny)
        beta = max(float(self._estimate_rt(values)) * gamma, np.finfo(float).tiny)
        return beta, gamma

    def _advance_log_rates(
        self,
        beta: np.ndarray | float,
        gamma: np.ndarray | float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        ""







        beta_next: np.ndarray | float = beta
        gamma_next: np.ndarray | float = gamma
        if self.log_beta_rw_scale > 0.0:
            beta_arr = np.asarray(beta, dtype=float)
            beta_next = beta_arr * np.exp(
                rng.normal(0.0, self.log_beta_rw_scale, size=beta_arr.shape or None)
            )
        if self.log_gamma_rw_scale > 0.0:
            gamma_arr = np.asarray(gamma, dtype=float)
            gamma_next = gamma_arr * np.exp(
                rng.normal(0.0, self.log_gamma_rw_scale, size=gamma_arr.shape or None)
            )
        return beta_next, gamma_next

    def _simulate_sir_mean_field(self, values: np.ndarray, horizon: int) -> float:
        s, i, r = self._initial_sir(values)
        pop = max(self.population, 1.0)
        beta, gamma = self._initial_rates(values)
        incidence = 0.0
        for _ in range(max(1, int(horizon))):
            new_inf = min(s, s * (1.0 - np.exp(-beta * i / pop)))
            new_rec = min(i, i * (1.0 - np.exp(-gamma)))
            s, i, r = max(s - new_inf, 0.0), max(i + new_inf - new_rec, 0.0), max(r + new_rec, 0.0)
            incidence = new_inf
        return self.reporting_rate * incidence

    def _simulate_sir_binomial(self, values: np.ndarray, horizon: int, rng: np.random.Generator) -> float:
        s, i, r = self._initial_sir(values)
        pop = max(self.population, 1.0)
        beta, gamma = self._initial_rates(values)
        incidence = 0.0
        n_steps = max(1, int(horizon))
        for step in range(n_steps):
            p_inf = 1.0 - np.exp(-beta * max(i, 0.0) / pop)
            p_rec = 1.0 - np.exp(-gamma)
            new_inf = min(s, _binomial_draw(rng, s, p_inf))
            new_rec = min(i, _binomial_draw(rng, i, p_rec))
            s, i, r = max(s - new_inf, 0.0), max(i + new_inf - new_rec, 0.0), max(r + new_rec, 0.0)
            incidence = new_inf
            if step + 1 < n_steps:
                beta, gamma = self._advance_log_rates(beta, gamma, rng)
        return max(0.0, self.reporting_rate * incidence)

    def _simulate_sir_binomial_draws(self, values: np.ndarray, horizon: int, rng: np.random.Generator, n_sim: int) -> np.ndarray:
        s0, i0, r0 = self._initial_sir(values)
        s = np.full(int(n_sim), s0, dtype=float)
        i = np.full(int(n_sim), i0, dtype=float)
        r = np.full(int(n_sim), r0, dtype=float)
        pop = max(self.population, 1.0)
        beta0, gamma0 = self._initial_rates(values)
        beta = np.full(int(n_sim), beta0, dtype=float)
        gamma = np.full(int(n_sim), gamma0, dtype=float)
        incidence = np.zeros(int(n_sim), dtype=float)
        n_steps = max(1, int(horizon))
        for step in range(n_steps):
            p_inf = np.clip(1.0 - np.exp(-beta * np.maximum(i, 0.0) / pop), 0.0, 1.0)
            p_rec = np.clip(1.0 - np.exp(-gamma), 0.0, 1.0)
            new_inf = rng.binomial(np.maximum(np.rint(s).astype(np.int64), 0), p_inf).astype(float)
            new_rec = rng.binomial(np.maximum(np.rint(i).astype(np.int64), 0), p_rec).astype(float)
            new_inf = np.minimum(s, new_inf)
            new_rec = np.minimum(i, new_rec)
            s = np.maximum(s - new_inf, 0.0)
            i = np.maximum(i + new_inf - new_rec, 0.0)
            r = np.maximum(r + new_rec, 0.0)
            incidence = new_inf
            if step + 1 < n_steps:
                beta, gamma = self._advance_log_rates(beta, gamma, rng)
        return np.maximum(0.0, self.reporting_rate * incidence)

    def _stochastic_summary(self, values: np.ndarray, horizon: int, *, seed: int, row_key: tuple[object, ...]) -> tuple[float, float]:
        n_sim = max(1, int(self.n_simulations))
        rng = np.random.default_rng(_stable_uint32(seed, self.model_id, *row_key))
        draws = self._simulate_sir_binomial_draws(values, horizon, rng, n_sim)
        mean = float(np.mean(draws)) if draws.size else 0.0
        latent_var = float(np.var(draws, ddof=1)) if draws.size > 1 else 0.0
        return _nonneg(mean), max(latent_var + max(mean, 1.0), 1.0)

    def _forecast_with_key(self, values: np.ndarray, horizon: int, times: np.ndarray | None, *, seed: int, row_key: tuple[object, ...]) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        if _mode_is_stochastic(self.transition_mode):
            return self._stochastic_summary(values, horizon, seed=seed, row_key=row_key)
        mean = self._simulate_sir_mean_field(values, horizon)
        return _nonneg(mean), max(mean, 1.0) + 0.10 * mean * mean

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        return self._forecast_with_key(values, horizon, times, seed=0, row_key=("direct", int(horizon), len(values)))

    def forecast_one_for_event(
        self,
        state: ReferenceState,
        event: pd.Series,
        values: np.ndarray,
        horizon: int,
        times: np.ndarray | None = None,
        *,
        recursive_step: int | None = None,
    ) -> tuple[float, float]:
        row_key = (
            _forecast_strategy(event),
            _ledger_entity(event),
            str(event.get("component", "")),
            event.get("forecast_origin", ""),
            int(recursive_step or horizon),
            event.get("forecast_id", ""),
        )
        return self._forecast_with_key(values, horizon, times, seed=state.seed, row_key=row_key)


class SEIRReferenceAdapter(SIRReferenceAdapter):
    model_id = "seir_tau"

    def __init__(self, sigma: float = 1 / 3.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sigma = float(sigma)

    def _initial_seir(self, values: np.ndarray) -> tuple[float, float, float, float]:
        last_obs = float(max(values[-1], 0.0))
        pop = max(self.population, 1.0)
        i = min(max(last_obs / max(self.reporting_rate, 1e-6), 1.0), 0.20 * pop)
        e = min(0.7 * i, 0.20 * pop)
        r = min(0.05 * pop + max(np.nansum(values[-min(len(values), 21):]), 0.0) / max(self.reporting_rate, 1e-6), 0.80 * pop)
        s = max(pop - e - i - r, 0.0)
        return s, e, i, r

    def _step_rt(self, values: np.ndarray, step: int) -> float:
        return self._estimate_rt(values)

    def _simulate_seir_mean_field(self, values: np.ndarray, horizon: int, *, seirs: bool = False) -> float:
        s, e, i, r = self._initial_seir(values)
        pop = max(self.population, 1.0)
        beta, gamma = self._initial_rates(values)
        incidence = 0.0
        for _ in range(max(1, int(horizon))):
            waned = min(r, r * (1.0 - np.exp(-float(getattr(self, "waning_rate", 0.0))))) if seirs else 0.0
            new_exp = min(s, s * (1.0 - np.exp(-beta * i / pop)))
            new_inf = min(e, e * (1.0 - np.exp(-self.sigma)))
            new_rec = min(i, i * (1.0 - np.exp(-gamma)))
            s = max(s + waned - new_exp, 0.0)
            e = max(e + new_exp - new_inf, 0.0)
            i = max(i + new_inf - new_rec, 0.0)
            r = max(r + new_rec - waned, 0.0)
            incidence = new_inf
        return self.reporting_rate * incidence

    def _simulate_seir_binomial(self, values: np.ndarray, horizon: int, rng: np.random.Generator, *, seirs: bool = False) -> float:
        s, e, i, r = self._initial_seir(values)
        pop = max(self.population, 1.0)
        beta, gamma = self._initial_rates(values)
        incidence = 0.0
        n_steps = max(1, int(horizon))
        for step in range(n_steps):
            p_inf = 1.0 - np.exp(-beta * max(i, 0.0) / pop)
            p_inc = 1.0 - np.exp(-self.sigma)
            p_rec = 1.0 - np.exp(-gamma)
            waned = _binomial_draw(rng, r, 1.0 - np.exp(-float(getattr(self, "waning_rate", 0.0)))) if seirs else 0.0
            new_exp = min(s + waned, _binomial_draw(rng, s + waned, p_inf))
            new_inf = min(e, _binomial_draw(rng, e, p_inc))
            new_rec = min(i, _binomial_draw(rng, i, p_rec))
            s = max(s + waned - new_exp, 0.0)
            e = max(e + new_exp - new_inf, 0.0)
            i = max(i + new_inf - new_rec, 0.0)
            r = max(r + new_rec - waned, 0.0)
            incidence = new_inf
            if step + 1 < n_steps:
                beta, gamma = self._advance_log_rates(beta, gamma, rng)
        return max(0.0, self.reporting_rate * incidence)

    def _simulate_seir_binomial_draws(
        self,
        values: np.ndarray,
        horizon: int,
        rng: np.random.Generator,
        n_sim: int,
        *,
        seirs: bool = False,
    ) -> np.ndarray:
        s0, e0, i0, r0 = self._initial_seir(values)
        n = int(n_sim)
        s = np.full(n, s0, dtype=float)
        e = np.full(n, e0, dtype=float)
        i = np.full(n, i0, dtype=float)
        r = np.full(n, r0, dtype=float)
        pop = max(self.population, 1.0)
        p_inc = float(np.clip(1.0 - np.exp(-self.sigma), 0.0, 1.0))
        p_wane = float(np.clip(1.0 - np.exp(-float(getattr(self, "waning_rate", 0.0))), 0.0, 1.0))
        beta0, gamma0 = self._initial_rates(values)
        beta = np.full(n, beta0, dtype=float)
        gamma = np.full(n, gamma0, dtype=float)
        incidence = np.zeros(n, dtype=float)
        n_steps = max(1, int(horizon))
        for step in range(n_steps):
            p_inf = np.clip(1.0 - np.exp(-beta * np.maximum(i, 0.0) / pop), 0.0, 1.0)
            p_rec = np.clip(1.0 - np.exp(-gamma), 0.0, 1.0)
            waned = (
                rng.binomial(np.maximum(np.rint(r).astype(np.int64), 0), p_wane).astype(float)
                if seirs
                else np.zeros(n, dtype=float)
            )
            susceptible = s + waned
            new_exp = rng.binomial(np.maximum(np.rint(susceptible).astype(np.int64), 0), p_inf).astype(float)
            new_inf = rng.binomial(np.maximum(np.rint(e).astype(np.int64), 0), p_inc).astype(float)
            new_rec = rng.binomial(np.maximum(np.rint(i).astype(np.int64), 0), p_rec).astype(float)
            new_exp = np.minimum(susceptible, new_exp)
            new_inf = np.minimum(e, new_inf)
            new_rec = np.minimum(i, new_rec)
            s = np.maximum(susceptible - new_exp, 0.0)
            e = np.maximum(e + new_exp - new_inf, 0.0)
            i = np.maximum(i + new_inf - new_rec, 0.0)
            r = np.maximum(r + new_rec - waned, 0.0)
            incidence = new_inf
            if step + 1 < n_steps:
                beta, gamma = self._advance_log_rates(beta, gamma, rng)
        return np.maximum(0.0, self.reporting_rate * incidence)

    def _stochastic_summary(self, values: np.ndarray, horizon: int, *, seed: int, row_key: tuple[object, ...]) -> tuple[float, float]:
        n_sim = max(1, int(self.n_simulations))
        rng = np.random.default_rng(_stable_uint32(seed, self.model_id, *row_key))
        seirs = isinstance(self, SEIRSReferenceAdapter)
        draws = self._simulate_seir_binomial_draws(values, horizon, rng, n_sim, seirs=seirs)
        mean = float(np.mean(draws)) if draws.size else 0.0
        latent_var = float(np.var(draws, ddof=1)) if draws.size > 1 else 0.0
        return _nonneg(mean), max(latent_var + max(mean, 1.0), 1.0)

    def _forecast_with_key(self, values: np.ndarray, horizon: int, times: np.ndarray | None, *, seed: int, row_key: tuple[object, ...]) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        if _mode_is_stochastic(self.transition_mode):
            return self._stochastic_summary(values, horizon, seed=seed, row_key=row_key)
        mean = self._simulate_seir_mean_field(values, horizon, seirs=isinstance(self, SEIRSReferenceAdapter))
        return _nonneg(mean), max(mean, 1.0) + 0.12 * mean * mean


class SEIRSReferenceAdapter(SEIRReferenceAdapter):
    model_id = "seirs_tau"

    def __init__(self, waning_rate: float = 1 / 180.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.waning_rate = float(waning_rate)

    def _initial_seir(self, values: np.ndarray) -> tuple[float, float, float, float]:
        last_obs = float(max(values[-1], 0.0))
        pop = max(self.population, 1.0)
        i = min(max(last_obs / max(self.reporting_rate, 1e-6), 1.0), 0.20 * pop)
        e = min(0.7 * i, 0.20 * pop)
        r = min(0.08 * pop, 0.80 * pop)
        s = max(pop - e - i - r, 0.0)
        return s, e, i, r


class TimeVaryingSEIRAdapter(SEIRReferenceAdapter):
    model_id = "tv_seir_rt"

    def __init__(
        self,
        log_beta_rw_scale: float = 0.05,
        log_gamma_rw_scale: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            log_beta_rw_scale=log_beta_rw_scale,
            log_gamma_rw_scale=log_gamma_rw_scale,
            **kwargs,
        )

    def _initial_seir(self, values: np.ndarray) -> tuple[float, float, float, float]:
        last_obs = float(max(values[-1], 0.0))
        pop = max(self.population, 1.0)
        i = min(max(last_obs / max(self.reporting_rate, 1e-6), 1.0), 0.20 * pop)
        e = min(0.7 * i, 0.20 * pop)
        r = min(0.05 * pop, 0.80 * pop)
        s = max(pop - e - i - r, 0.0)
        return s, e, i, r

    def _forecast_with_key(self, values: np.ndarray, horizon: int, times: np.ndarray | None, *, seed: int, row_key: tuple[object, ...]) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        if _mode_is_stochastic(self.transition_mode):
            return self._stochastic_summary(values, horizon, seed=seed, row_key=row_key)
        mean = self._simulate_seir_mean_field(values, horizon, seirs=False)
        return _nonneg(mean), max(mean, 1.0) + 0.15 * mean * mean


                                                                             
                       
                                                                             
class RenewalRtReferenceAdapter(SeriesForecastAdapter):
    ""






    model_id = "renewal_rt"
    family = "renewal"

    def __init__(self, serial_interval: tuple[float, ...] = (0.05, 0.15, 0.30, 0.25, 0.15, 0.07, 0.03), rt_shrink: float = 0.85, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        w = np.asarray(serial_interval, dtype=float)
        self.serial_interval = tuple((w / max(w.sum(), 1e-12)).tolist())
        self.rt_shrink = float(rt_shrink)

    def _rt(self, y: np.ndarray) -> float:
        if len(y) < 3:
            return 1.0
        w = np.asarray(self.serial_interval, dtype=float)
        hist = np.maximum(y.astype(float), 0.0)
        lam = []
        ratios = []
        for t in range(1, len(hist)):
            tail = hist[max(0, t - len(w)):t][::-1]
            ww = w[: len(tail)]
            if ww.sum() > 0:
                denom = float(np.dot(tail, ww / ww.sum()))
                if denom > 0:
                    ratios.append(float(hist[t] / denom))
        if not ratios:
            return 1.0
        return float(np.clip(np.nanmedian(ratios[-min(6, len(ratios)):]), 0.2, 3.5))

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        hist = list(np.maximum(np.asarray(values, dtype=float), 0.0))
        if not hist:
            return 0.0, 1.0
        w = np.asarray(self.serial_interval, dtype=float)
        rt0 = self._rt(np.asarray(hist))
        mean = hist[-1]
        for step in range(max(1, int(horizon))):
            tail = np.asarray(hist[-len(w):][::-1], dtype=float)
            ww = w[: len(tail)]
            ww = ww / max(ww.sum(), 1e-12)
            rt = 1.0 + (rt0 - 1.0) * (self.rt_shrink ** step)
            mean = _nonneg(rt * float(np.dot(ww, tail)))
            hist.append(mean)
        return _nonneg(mean), max(mean, 1.0) + float(np.var(np.diff(hist[-min(len(hist), 8):])) if len(hist) >= 3 else 0.0)


                                                                             
                                  
                                                                             
class LocalLevelDLMAdapter(SeriesForecastAdapter):
    ""





    model_id = "local_level"
    family = "state_space"

    def __init__(self, alpha: float = 0.35, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        level = float(values[0])
        residuals: list[float] = []
        for val in values[1:]:
            residuals.append(float(val) - level)
            level = self.alpha * float(val) + (1.0 - self.alpha) * level
        var = max(float(np.var(residuals)) if residuals else 1.0, 1.0) * max(1, int(horizon))
        return _nonneg(level), var


class _DampedTrendCoreAdapter(LocalLevelDLMAdapter):
    ""

    model_id = "damped_trend_core"

    def __init__(self, alpha: float = 0.35, beta: float = 0.10, damping: float = 0.90, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, **kwargs)
        self.beta = float(beta)
        self.damping = float(damping)

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 1.0
        level = float(values[0])
        trend = 0.0
        residuals: list[float] = []
        for val in values[1:]:
            pred = level + trend
            residuals.append(float(val) - pred)
            new_level = self.alpha * float(val) + (1.0 - self.alpha) * pred
            trend = self.beta * (new_level - level) + (1.0 - self.beta) * trend
            level = new_level
        pred = level + sum((self.damping ** i) * trend for i in range(1, int(horizon) + 1))
        var = max(float(np.var(residuals)) if residuals else 1.0, 1.0) * max(1, int(horizon))
        return _nonneg(pred), var


class CovariateDynamicLinearTrendDLMAdapter(
    _CausalCovariateAdjustmentMixin, _DampedTrendCoreAdapter
):
    ""




    model_id = "covariate_dynamic_linear_trend"
    family = "state_space"


class ParticleFilteredLocalLevelAdapter(SeriesForecastAdapter):
    ""






    model_id = "particle_local_level"
    family = "state_space"

                                                                        
                                                                            
                                                                             
                                                                    
    _alternate_PROTOCOL_SEED = 42

    def __init__(self, n_particles: int = 128, process_scale: float = 0.20, obs_scale: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_particles = int(n_particles)
        self.process_scale = float(process_scale)
        self.obs_scale = float(obs_scale)

    @staticmethod
    def _alternate_rng_seed(values: np.ndarray, horizon: int) -> int:
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        return 17 + len(finite_values) + int(horizon)

    def _event_rng_seed(
        self,
        state: ReferenceState,
        event: pd.Series,
        values: np.ndarray,
        horizon: int,
        *,
        recursive_step: int | None,
    ) -> int:
        if int(state.seed) == self._alternate_PROTOCOL_SEED:
            return self._alternate_rng_seed(values, horizon)
        return _stable_uint32(
            "particle_local_level_event_rng_v1",
            int(state.seed),
            self.model_id,
            str(event.get("forecast_id", "")),
            _ledger_entity(event),
            str(event.get("component", "")),
            str(event.get("forecast_origin", "")),
            str(event.get("target_time", "")),
            int(pd.to_numeric(event.get("horizon", horizon), errors="raise")),
            "" if recursive_step is None else int(recursive_step),
        )

    def _forecast_one_with_rng_seed(
        self,
        values: np.ndarray,
        horizon: int,
        *,
        rng_seed: int,
    ) -> tuple[float, float]:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            return 0.0, 1.0
        rng = np.random.default_rng(int(rng_seed))
        base_sigma = np.sqrt(_residual_var(y, y[-1]))
        particles = rng.normal(float(y[0]), base_sigma, size=self.n_particles)
        weights = np.ones(self.n_particles) / self.n_particles
        for obs in y[1:]:
            particles = particles + rng.normal(0.0, self.process_scale * base_sigma, size=self.n_particles)
            ll = -0.5 * ((obs - particles) / max(self.obs_scale * base_sigma, 1e-6)) ** 2
            ll -= ll.max()
            weights = np.exp(ll)
            weights = weights / max(weights.sum(), 1e-12)
            ess = 1.0 / max(float(np.sum(weights * weights)), 1e-12)
            if ess < self.n_particles / 2:
                idx = rng.choice(self.n_particles, size=self.n_particles, replace=True, p=weights)
                particles = particles[idx]
                weights = np.ones(self.n_particles) / self.n_particles
        for _ in range(max(1, int(horizon))):
            particles = particles + rng.normal(0.0, self.process_scale * base_sigma, size=self.n_particles)
        mean = float(np.average(particles, weights=weights))
        var = float(np.average((particles - mean) ** 2, weights=weights) + base_sigma ** 2)
        return _nonneg(mean), max(var, 1.0)

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        ""

        return self._forecast_one_with_rng_seed(
            values,
            horizon,
            rng_seed=self._alternate_rng_seed(values, horizon),
        )

    def forecast_one_for_event(
        self,
        state: ReferenceState,
        event: pd.Series,
        values: np.ndarray,
        horizon: int,
        times: np.ndarray | None = None,
        *,
        recursive_step: int | None = None,
    ) -> tuple[float, float]:
        return self._forecast_one_with_rng_seed(
            values,
            horizon,
            rng_seed=self._event_rng_seed(
                state,
                event,
                values,
                horizon,
                recursive_step=recursive_step,
            ),
        )


                                                                             
                                                                             
                                                                             
class StatsForecastReferenceAdapter(SeriesForecastAdapter):
    family = "statistical"
    sf_model = "autoarima"

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 3:
            raise RuntimeError(
                "structural_history_unavailable: StatsForecast requires at least 3 finite observations"
            )
        try:
            from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta
            season = _season_length_from_history(times if times is not None else np.asarray([]))
            seasonal, s = _statsforecast_seasonality(y, int(horizon), int(season))
            if self.sf_model == "autoarima":
                                                                         
                                                                                
                                                                                
                fit_values = y[-min(len(y), 104):]
                model = AutoARIMA(
                    season_length=1,
                    seasonal=False,
                    stepwise=True,
                    approximation=True,
                    max_p=1,
                    max_q=1,
                    max_P=0,
                    max_Q=0,
                    max_order=2,
                    max_d=1,
                    max_D=0,
                    start_p=0,
                    start_q=0,
                    start_P=0,
                    start_Q=0,
                    nmodels=8,
                    allowdrift=True,
                    allowmean=True,
                )
            elif self.sf_model == "autoets":
                fit_values = y
                model = AutoETS(season_length=s, model="ZZZ")
            elif self.sf_model == "autotheta":
                fit_values = y
                model = AutoTheta(season_length=s, decomposition_type="additive")
            elif self.sf_model == "autoces":
                fit_values = y
                model = AutoCES(season_length=s)
            else:
                raise ValueError(self.sf_model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="possible convergence problem.*")
                warnings.filterwarnings("ignore", message="Stepwise search was stopped early.*")
                model.fit(fit_values)
                pred = model.predict(h=int(horizon))["mean"]
            mean_vec = np.asarray(pred, dtype=float)
            if len(mean_vec) == 0:
                raise ValueError("StatsForecast returned empty mean vector")
            mean = float(mean_vec[min(max(int(horizon) - 1, 0), len(mean_vec) - 1)])
            return _nonneg(mean), _residual_var(y, mean)
        except Exception as exc:
            raise RuntimeError(
                f"native StatsForecast {self.sf_model} forecast failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


class AutoARIMAReferenceAdapter(StatsForecastReferenceAdapter):
    model_id = "statsforecast_autoarima"
    sf_model = "autoarima"


class AutoETSReferenceAdapter(StatsForecastReferenceAdapter):
    model_id = "statsforecast_autoets"
    sf_model = "autoets"


class AutoThetaReferenceAdapter(StatsForecastReferenceAdapter):
    model_id = "statsforecast_autotheta"
    sf_model = "autotheta"


class AutoCESReferenceAdapter(StatsForecastReferenceAdapter):
    model_id = "statsforecast_autoces"
    sf_model = "autoces"


class ProphetReferenceAdapter(SeriesForecastAdapter):
    model_id = "prophet"
    family = "statistical"

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 2 or times is None or len(times) != len(y):
            raise RuntimeError(
                "structural_history_unavailable: Prophet requires at least 2 dated finite observations"
            )
        try:
            from prophet import Prophet
            train = pd.DataFrame({"ds": pd.to_datetime(times), "y": y}).dropna().sort_values("ds")
            season = _season_length_from_history(times)
            yearly = bool(season >= 52 and len(train) >= 2 * season)
            median_days = train["ds"].diff().dt.days.dropna().median()
            weekly = bool(season == 7 or (len(train) >= 21 and pd.notna(median_days) and median_days <= 1.5))
            model = Prophet(
                growth="linear",
                n_changepoints=max(0, min(25, len(train) // 2 - 1)),
                changepoint_prior_scale=0.05,
                seasonality_mode="additive",
                daily_seasonality=False,
                weekly_seasonality=weekly,
                yearly_seasonality=yearly,
                uncertainty_samples=1000,
            )
            if season > 1 and not weekly and not yearly and len(train) >= 2 * int(season):
                model.add_seasonality(
                    name=f"season_{int(season)}",
                    period=float(season),
                    fourier_order=min(10, max(3, int(season) // 2)),
                )
            model.fit(train)
            future = pd.DataFrame({"ds": [_target_time_from_horizon(times, int(horizon))]})
            fcst = model.predict(future)
            mean = float(fcst["yhat"].iloc[0])
            return _nonneg(mean), _residual_var(y, mean)
        except Exception as exc:
            raise RuntimeError(
                f"native Prophet forecast failed: {type(exc).__name__}: {exc}"
            ) from exc


                                                                             
                                                    
                                                                             


class RNNReferenceAdapter(SeriesForecastAdapter):
    ""

    model_id = "rnn_simple"
    family = "neural"

    def __init__(self, alpha: float = 0.55, gain: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.gain = float(gain)

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        y = np.maximum(np.asarray(values, dtype=float), 0.0)
        if len(y) == 0:
            return 0.0, 1.0
        h = np.log1p(y[0])
        resid = []
        for val in y[1:]:
            resid.append(val - np.expm1(h))
            h = self.alpha * np.log1p(val) + (1.0 - self.alpha) * h
        for _ in range(max(1, int(horizon))):
            h = self.gain * h
        pred = _nonneg(np.expm1(h))
        return pred, max(float(np.var(resid)) if resid else 1.0, 1.0)


class BaselineNeuralForecastReferenceAdapter(SeriesForecastAdapter):
    ""

    family = "neural"
    nf_model_name = "lstm"
    max_steps = 5
    seed = 1
    device = "cpu"

    def __init__(
        self,
        max_steps: int = 5,
        seed: int = 1,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.device = detect_device() if device is None else str(device)

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            return 0.0, 1.0
        try:
            pred = _baseline_neuralforecast_predict_one(
                self.nf_model_name,
                y,
                int(horizon),
                times,
                max_steps=self.max_steps,
                seed=self.seed,
                device=self.device,
            )
            return _nonneg(pred), _residual_var(y, pred)
        except Exception as exc:
            raise RuntimeError(
                f"native NeuralForecast {self.nf_model_name} forecast failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


class LSTMReferenceAdapter(BaselineNeuralForecastReferenceAdapter):
    model_id = "lstm_style"
    nf_model_name = "lstm"


class GRUReferenceAdapter(RNNReferenceAdapter):
    ""

    model_id = "gru_style"

    def forecast_one(self, values: np.ndarray, horizon: int, times: np.ndarray | None = None) -> tuple[float, float]:
        y = np.maximum(np.asarray(values, dtype=float), 0.0)
        if len(y) == 0:
            return 0.0, 1.0
        h = np.log1p(y[0])
        resid = []
        for val in y[1:]:
            x = np.log1p(val)
            resid.append(val - np.expm1(h))
            update = 1.0 / (1.0 + np.exp(-0.6 * (x - h)))
            reset = 1.0 / (1.0 + np.exp(-0.5 * h))
            candidate = np.tanh(x + reset * h)
            h = (1.0 - update) * h + update * candidate
        for _ in range(max(1, int(horizon))):
            h = 0.95 * h
        pred = _nonneg(np.expm1(h))
        return pred, max(float(np.var(resid)) if resid else 1.0, 1.0)


class DeepARStyleAdapter(BaselineNeuralForecastReferenceAdapter):
    model_id = "deepar_style"
    nf_model_name = "deepar"


class NBEATSReferenceAdapter(BaselineNeuralForecastReferenceAdapter):
    model_id = "nbeats_basis"
    nf_model_name = "nbeats"


class NHITSReferenceAdapter(NBEATSReferenceAdapter):
    model_id = "nhits_hinterp"
    nf_model_name = "nhits"


class PatchTSTReferenceAdapter(BaselineNeuralForecastReferenceAdapter):
    model_id = "patchtst_patched"
    nf_model_name = "patchtst"


class TFTReferenceAdapter(BaselineNeuralForecastReferenceAdapter):
    model_id = "tft_gated"
    nf_model_name = "tft"
