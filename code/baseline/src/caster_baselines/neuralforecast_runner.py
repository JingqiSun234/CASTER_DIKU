from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .data_validation import baseline_root, caster_root_from_baseline, resolve_manifest_path, sha256_file
from .ledger_runner import (
    Z50,
    Z90,
    build_history_index,
    choose_ledger_entity_col,
    context_columns,
    format_date,
    forecast_strategy_manifest_fields,
    parse_bool,
    residual_sigma,
    split_semicolon,
    values_until_origin,
    write_blocker_report,
)
from .metrics import summarize_forecasts
from .neuralforecast_dependencies import NeuralDependencyError, require_neuralforecast_dependencies
from .forecast_strategy import RECURSIVE_ROLLOUT, normalize_forecast_strategy


NEURAL_MODELS = {"nbeats", "nhits", "deepar", "patchtst", "tft"}
MODEL_OUTPUT_DIRS = {
    "nbeats": "nbeats",
    "nhits": "nhits",
    "deepar": "deepar",
    "patchtst": "patchtst",
    "tft": "tft",
}
TRAIN_SPLIT = "train"
SCORING_SPLITS = {"train", "val", "embargo", "test"}
RELEASE_TIME_COL = "__release_time__"
RELEASE_TIME_CANDIDATES = ("__release_time__target", "__release_time__", "release_time")


@dataclass(frozen=True)
class NeuralScope:
    dataset_key: str
    dataset: str
    component: str
    h: int
    input_size: int
    freq: str
    cadence_days: int
    train_cutoff: pd.Timestamp
    model_name: str
    seed: int
    max_steps: int
    device: str
    forecast_strategy: str = "direct"
    mode: str = ""


class NeuralBackend(Protocol):
    def fit_predict(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        full_df: pd.DataFrame,
        scoring_ledger: pd.DataFrame,
        ledger_entity_col: str,
        scope: NeuralScope,
    ) -> tuple[dict[tuple[str, str, int], float], list[dict[str, object]]]:
        ...


def canonical_neural_model(model: str) -> str:
    key = model.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "nbeats": "nbeats",
        "blnfnbeats": "nbeats",
        "nhits": "nhits",
        "blnfnhits": "nhits",
        "deepar": "deepar",
        "blnfdeepar": "deepar",
"patchtst": "patchtst",
        "blnfpatchtst": "patchtst",
        "tft": "tft",
        "blnftft": "tft",
    }
    if key not in aliases:
        raise ValueError(f"unknown NeuralForecast model {model!r}; available={sorted(NEURAL_MODELS)}")
    return aliases[key]


def cadence_to_freq(cadence_days: int) -> str:
    if cadence_days <= 1:
        return "D"
    return f"{int(cadence_days)}D"


def _series_length_summary(df: pd.DataFrame) -> tuple[int, int]:
    counts = df.groupby("unique_id", dropna=False)["y"].size()
    if counts.empty:
        return 0, 0
    return int(counts.min()), int(counts.max())


def choose_input_size(train_df: pd.DataFrame, h: int) -> int:
    min_len, _ = _series_length_summary(train_df)
    if min_len <= h + 1:
        return max(1, min_len - 1)
    target = max(2 * int(h), 8)
                                                                          
                                                                             
                                                                           
    return int(max(2, min(target, max(2, min_len - int(h) - 1))))


def _panel_release_time_col(panel: pd.DataFrame) -> str | None:
    return next((col for col in RELEASE_TIME_CANDIDATES if col in panel.columns), None)


def _available_nf_df(
    full_df: pd.DataFrame,
    origin: pd.Timestamp,
    entities: set[str],
) -> pd.DataFrame:
    """Return only observations that were available as of ``origin``."""
    origin = pd.Timestamp(origin)
    mask = (
        full_df["ds"].le(origin)
        & full_df["unique_id"].astype(str).isin({str(entity) for entity in entities})
    )
    if RELEASE_TIME_COL in full_df.columns:
        releases = pd.to_datetime(full_df[RELEASE_TIME_COL], errors="coerce")
        mask &= releases.notna() & releases.le(origin)
    out = full_df.loc[mask, ["unique_id", "ds", "y"]].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["ds", "y"])
    out = out[np.isfinite(out["y"])].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return out


def _native_target_requests(
    origin_group: pd.DataFrame,
    pred_input: pd.DataFrame,
    ledger_entity_col: str,
    cadence_days: int,
) -> tuple[dict[tuple[str, str], tuple[int, int]], int]:
    """Map requested target dates to (nominal horizon, native step)."""
    if int(cadence_days) <= 0:
        raise ValueError(f"cadence_days must be positive, got {cadence_days}")
    last_ds = pred_input.groupby("unique_id", dropna=False)["ds"].max().to_dict()
    requests: dict[tuple[str, str], tuple[int, int]] = {}
    max_step = 0
    for _, event in origin_group.iterrows():
        entity = str(event[ledger_entity_col])
        if entity not in last_ds or pd.isna(last_ds[entity]):
            raise ValueError(f"no released NeuralForecast history for entity={entity}")
        target = pd.to_datetime(event["target_time"], errors="coerce")
        if pd.isna(target):
            raise ValueError(f"target_time parse failed: {event['target_time']}")
        step_float = (pd.Timestamp(target) - pd.Timestamp(last_ds[entity])) / pd.Timedelta(
            days=int(cadence_days)
        )
        native_step = int(round(float(step_float)))
        if native_step < 1 or not math.isclose(float(step_float), native_step, abs_tol=1e-9):
            raise ValueError(
                "target_time is not a positive native-cadence step after the last released "
                f"observation: entity={entity} last_ds={format_date(pd.Timestamp(last_ds[entity]))} "
                f"target={format_date(pd.Timestamp(target))} cadence_days={cadence_days}"
            )
        nominal_horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
        key = (entity, format_date(pd.Timestamp(target)))
        previous = requests.get(key)
        request = (nominal_horizon, native_step)
        if previous is not None and previous != request:
            raise ValueError(f"conflicting ledger requests for entity/target={key}: {previous} vs {request}")
        requests[key] = request
        max_step = max(max_step, native_step)
    if not requests:
        raise ValueError("empty NeuralForecast target request group")
    return requests, max_step


def _maximum_native_horizon(
    full_df: pd.DataFrame,
    scoring_ledger: pd.DataFrame,
    ledger_entity_col: str,
    cadence_days: int,
) -> int:
    max_step = 0
    for origin_text, origin_group in scoring_ledger.groupby(
        "forecast_origin", dropna=False, sort=True
    ):
        origin = pd.to_datetime(origin_text, errors="coerce")
        if pd.isna(origin):
            raise ValueError(f"forecast_origin parse failed: {origin_text}")
        entities = set(origin_group[ledger_entity_col].astype(str))
        pred_input = _available_nf_df(full_df, pd.Timestamp(origin), entities)
        _, group_max = _native_target_requests(
            origin_group, pred_input, ledger_entity_col, cadence_days
        )
        max_step = max(max_step, group_max)
    if max_step < 1:
        raise ValueError("no positive native NeuralForecast horizon found")
    return max_step


def panel_component_to_nf_df(
    panel: pd.DataFrame,
    manifest_row: pd.Series,
    ledger: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    entity_col = str(manifest_row["panel_entity_col"])
    time_col = str(manifest_row["panel_time_col"])
    panel_format = str(manifest_row["panel_format"])
    panel = panel.copy()
    panel[entity_col] = panel[entity_col].astype(str)
    if panel_format == "long":
        component_col = str(manifest_row["panel_component_col"])
        value_col = str(manifest_row["panel_value_col"])
        subset = panel[panel[component_col].astype(str) == str(component)].copy()
        source_value_col = value_col
    else:
        targets = split_semicolon(manifest_row["panel_target_cols"])
        if component not in targets and component not in panel.columns:
            raise ValueError(f"wide panel component {component!r} is not a target column")
        subset = panel.copy()
        source_value_col = component
    release_col = _panel_release_time_col(subset)
    out_data: dict[str, object] = {
        "unique_id": subset[entity_col].astype(str),
        "ds": pd.to_datetime(subset[time_col], errors="coerce"),
        "y": pd.to_numeric(subset[source_value_col], errors="coerce"),
    }
    if release_col is not None:
        out_data[RELEASE_TIME_COL] = pd.to_datetime(subset[release_col], errors="coerce")
    out = pd.DataFrame(out_data)
    out = out.dropna(subset=["ds"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)
                                                                          
                                                                             
                                                                          
                                                                         
                                                                          
    out["y"] = out.groupby("unique_id", sort=False)["y"].ffill()
    out = out.dropna(subset=["y"]).reset_index(drop=True)
    return out


def _prediction_col(frame: pd.DataFrame, model_name: str) -> str:
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


def _disable_cuda_for_cpu(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


class RealNeuralForecastBackend:
    def __init__(self, max_steps: int = 5, seed: int = 1, device: str = "cpu") -> None:
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.device = device

    def _model(self, model_name: str, scope: NeuralScope):
        ""






        _disable_cuda_for_cpu(scope.device)
        from neuralforecast.models import DeepAR, NBEATS, NHITS, PatchTST, TFT
        from neuralforecast.losses.pytorch import MAE
        try:
            from neuralforecast.losses.pytorch import DistributionLoss
            probabilistic_loss = DistributionLoss(distribution="Normal", level=[50, 90])
        except Exception:
            probabilistic_loss = MAE()

                                                                                
                                                                                 
        common = {
            "h": scope.h,
            "input_size": scope.input_size,
            "max_steps": scope.max_steps,
            "early_stop_patience_steps": -1,
            "val_check_steps": max(1, scope.max_steps),
            "batch_size": 32,
            "valid_batch_size": 32,
            "windows_batch_size": 256,
            "inference_windows_batch_size": 256,
            "random_seed": scope.seed,
            "scaler_type": "robust",
            "alias": model_name,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "accelerator": scope.device,
            "devices": 1,
        }
        if model_name == "nbeats":
            if int(scope.h) == 1:
                                                                         
                                                                             
                                                                             
                return NBEATS(
                    stack_types=["identity"],
                    n_blocks=[6],
                    mlp_units=[[128, 128]],
                    loss=MAE(),
                    **common,
                )
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
                h_train=scope.h,
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
        if model_name == "patchtst":
            patch_len = max(2, min(16, scope.input_size))
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


    def fit_predict(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        full_df: pd.DataFrame,
        scoring_ledger: pd.DataFrame,
        ledger_entity_col: str,
        scope: NeuralScope,
    ) -> tuple[dict[tuple[str, str, int], float], list[dict[str, object]]]:
        _disable_cuda_for_cpu(scope.device)
        from neuralforecast import NeuralForecast

        fit_df = train_df
        if RELEASE_TIME_COL in full_df.columns:
            fit_entities = set(train_df["unique_id"].astype(str))
            fit_df = _available_nf_df(full_df, scope.train_cutoff, fit_entities)
            native_h = (
                1
                if scope.forecast_strategy == RECURSIVE_ROLLOUT
                else _maximum_native_horizon(
                    full_df,
                    scoring_ledger,
                    ledger_entity_col,
                    scope.cadence_days,
                )
            )
            scope = replace(
                scope,
                h=native_h,
                input_size=choose_input_size(fit_df, native_h),
            )
        if fit_df.empty:
            raise ValueError("no released NeuralForecast training observations at train_cutoff")
        model = self._model(model_name, scope)
        nf = NeuralForecast(models=[model], freq=scope.freq)
        fit_start = time.time()
        nf.fit(df=fit_df, val_size=0, verbose=False)
        fit_seconds = round(time.time() - fit_start, 6)
        pred_col = model_name
        predictions: dict[tuple[str, str, int], float] = {}
        logs: list[dict[str, object]] = []
        train_min, train_max = _series_length_summary(fit_df)
        for origin_text, origin_group in scoring_ledger.groupby("forecast_origin", dropna=False, sort=True):
            origin = pd.to_datetime(origin_text, errors="coerce")
            if pd.isna(origin):
                raise ValueError(f"forecast_origin parse failed: {origin_text}")
                                                                             
                                                                            
                                                                              
                                                            
            origin_entities = set(origin_group[ledger_entity_col].astype(str))
            pred_input = _available_nf_df(full_df, pd.Timestamp(origin), origin_entities)
            max_input = pred_input["ds"].max()
            if pd.notna(max_input) and pd.Timestamp(max_input) > origin:
                raise ValueError("origin-truncated NeuralForecast input contains future y")
            requests, native_max_step = _native_target_requests(
                origin_group,
                pred_input,
                ledger_entity_col,
                scope.cadence_days,
            )
            pred_start = time.time()
            missing = 0
            if scope.forecast_strategy == RECURSIVE_ROLLOUT:
                recursive_input = pred_input.copy()
                for step in range(1, native_max_step + 1):
                    forecast = nf.predict(df=recursive_input, verbose=False).copy()
                    col = pred_col if pred_col in forecast.columns else _prediction_col(forecast, model_name)
                    forecast["unique_id"] = forecast["unique_id"].astype(str)
                    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
                    append_rows: list[dict[str, object]] = []
                    for _, pred_row in forecast.iterrows():
                        value = pd.to_numeric(pd.Series([pred_row[col]]), errors="coerce").iloc[0]
                        if pd.isna(pred_row["ds"]) or not np.isfinite(value):
                            continue
                        entity = str(pred_row["unique_id"])
                        target_text = format_date(pd.Timestamp(pred_row["ds"]))
                        request = requests.get((entity, target_text))
                        if request is not None:
                            nominal_horizon, expected_native_step = request
                            if expected_native_step != step:
                                raise ValueError(
                                    "recursive NeuralForecast target-date/native-step mismatch: "
                                    f"entity={entity} target={target_text} expected_step="
                                    f"{expected_native_step} actual_step={step}"
                                )
                            predictions[(entity, str(origin_text), nominal_horizon)] = float(value)
                        append_rows.append({"unique_id": entity, "ds": pred_row["ds"], "y": float(value)})
                    if not append_rows:
                        raise ValueError(f"recursive NeuralForecast returned no finite h=1 predictions at step={step}")
                    recursive_input = pd.concat(
                        [recursive_input, pd.DataFrame(append_rows)],
                        ignore_index=True,
                    ).sort_values(["unique_id", "ds"]).reset_index(drop=True)
                missing = sum(
                    (entity, str(origin_text), nominal_horizon) not in predictions
                    for (entity, _), (nominal_horizon, _) in requests.items()
                )
            else:
                forecast = nf.predict(df=pred_input, verbose=False)
                col = pred_col if pred_col in forecast.columns else _prediction_col(forecast, model_name)
                forecast = forecast.copy()
                forecast["unique_id"] = forecast["unique_id"].astype(str)
                forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
                by_key = {
                    (str(row["unique_id"]), format_date(pd.Timestamp(row["ds"]))): float(row[col])
                    for _, row in forecast.iterrows()
                    if pd.notna(row["ds"]) and np.isfinite(pd.to_numeric(row[col], errors="coerce"))
                }
                for _, event in origin_group.iterrows():
                    target = pd.to_datetime(event["target_time"], errors="coerce")
                    horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
                    entity = str(event[ledger_entity_col])
                    key = (entity, format_date(target))
                    if key not in by_key:
                        missing += 1
                        continue
                    predictions[(entity, str(origin_text), horizon)] = by_key[key]
            logs.append({
                "dataset_key": scope.dataset_key,
                "dataset": scope.dataset,
                "component": scope.component,
                "split": ";".join(sorted(origin_group["split"].astype(str).unique())),
                "forecast_origin": str(origin_text),
                "train_cutoff": format_date(scope.train_cutoff),
                "train_rows": int(len(fit_df)),
                "train_series_min_rows": train_min,
                "train_series_max_rows": train_max,
                "h": scope.h,
                "input_size": scope.input_size,
                "status": "model_ok" if missing == 0 else "missing_predictions",
                "runtime_seconds": round(time.time() - pred_start, 6),
                "fit_seconds": fit_seconds,
                "failure_reason": "" if missing == 0 else f"missing_predictions={missing}",
                "fallback_used": False,
                "seed": scope.seed,
                "device": scope.device,
                "config_summary": f"max_steps={scope.max_steps};freq={scope.freq}",
                "forecast_strategy": scope.forecast_strategy,
                "mode": scope.mode,
                "release_gated": RELEASE_TIME_COL in full_df.columns,
                "native_max_step": native_max_step,
            })
            if missing:
                raise ValueError(f"NeuralForecast output missing {missing} predictions for origin {origin_text}")
        return predictions, logs


def finite_metric_check(metrics: pd.DataFrame) -> list[str]:
    numeric_cols = ["mae", "rmse", "gaussian_nll", "coverage_50", "coverage_90", "width_50", "width_90"]
    return [
        col
        for col in numeric_cols
        if col in metrics.columns and not np.isfinite(pd.to_numeric(metrics[col], errors="coerce")).all()
    ]


def _last_value_row(
    model_name: str,
    dataset_key: str,
    manifest_row: pd.Series,
    event: pd.Series,
    entity_id: str,
    component: str,
    origin: pd.Timestamp,
    target: pd.Timestamp,
    horizon: int,
    values: np.ndarray,
    context_cols: list[str],
    status: str,
    failure_reason: str,
) -> dict[str, object]:
    pred = float(values[-1])
    sigma = max(residual_sigma(values, pred), 1e-6)
    y_true = pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0]
    row = {
        "dataset_key": dataset_key,
        "dataset": str(event["dataset"]) if "dataset" in event.index else str(manifest_row["dataset"]),
        "method": model_name,
        "entity_id": entity_id,
        "forecast_origin": format_date(origin),
        "target_time": format_date(target),
        "component": component,
        "horizon": horizon,
        "y_true": float(y_true) if np.isfinite(y_true) else np.nan,
        "pred_mean": pred,
        "pred_lower_50": float(pred - Z50 * sigma),
        "pred_upper_50": float(pred + Z50 * sigma),
        "pred_lower_90": float(pred - Z90 * sigma),
        "pred_upper_90": float(pred + Z90 * sigma),
        "split": str(event["split"]) if "split" in event.index else "NA",
        "model_status": status,
        "failure_reason": failure_reason,
        "fallback_method": "last_value",
        "forecast_status": status,
        "forecast_fallback_used": True,
        "forecast_failure_reason": failure_reason,
        "forecast_fallback_method": "last_value",
        "proxy_fallback_used": False,
        "unsafe_native_proxy_executed": False,
    }
    for col in context_cols:
        row[col] = event[col]
    return row


def _model_row(
    model_name: str,
    dataset_key: str,
    manifest_row: pd.Series,
    event: pd.Series,
    entity_id: str,
    component: str,
    origin: pd.Timestamp,
    target: pd.Timestamp,
    horizon: int,
    pred: float,
    sigma_values: np.ndarray,
    context_cols: list[str],
) -> dict[str, object]:
    sigma = max(residual_sigma(sigma_values, float(pred)), 1e-6)
    y_true = pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0]
    row = {
        "dataset_key": dataset_key,
        "dataset": str(event["dataset"]) if "dataset" in event.index else str(manifest_row["dataset"]),
        "method": model_name,
        "entity_id": entity_id,
        "forecast_origin": format_date(origin),
        "target_time": format_date(target),
        "component": component,
        "horizon": horizon,
        "y_true": float(y_true) if np.isfinite(y_true) else np.nan,
        "pred_mean": float(pred),
        "pred_lower_50": float(pred - Z50 * sigma),
        "pred_upper_50": float(pred + Z50 * sigma),
        "pred_lower_90": float(pred - Z90 * sigma),
        "pred_upper_90": float(pred + Z90 * sigma),
        "split": str(event["split"]) if "split" in event.index else "NA",
        "model_status": "model_ok",
        "failure_reason": "",
        "fallback_method": "",
        "forecast_status": "model_ok",
        "forecast_fallback_used": False,
        "forecast_failure_reason": "",
        "forecast_fallback_method": "",
        "proxy_fallback_used": False,
        "unsafe_native_proxy_executed": False,
    }
    for col in context_cols:
        row[col] = event[col]
    return row


def run_external_forecast_csv(
    manifest: pd.DataFrame,
    manifest_path: Path,
    external_csv: Path,
    out_dir: Path,
    model_name: str,
    root: Path,
    caster_root: Path,
) -> Path:
    forecast = pd.read_csv(external_csv, keep_default_na=False)
    required = {
        "dataset_key",
        "dataset",
        "entity_id",
        "forecast_origin",
        "target_time",
        "component",
        "horizon",
        "pred_mean",
        "split",
        "mode",
    }
    missing = required - set(forecast.columns)
    if missing:
        write_blocker_report(out_dir, [{
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"external forecast csv missing columns: {sorted(missing)}",
        }])
        raise RuntimeError(f"external forecast csv missing columns; see {out_dir / 'blocker_report.csv'}")

    ledger_keys = []
    for _, manifest_row in manifest.iterrows():
        ledger = pd.read_csv(resolve_manifest_path(manifest_row["ledger_path"], caster_root, root), keep_default_na=False)
        entity_col = choose_ledger_entity_col(ledger, str(manifest_row["panel_entity_col"]))
        for _, event in ledger.iterrows():
            ledger_keys.append((
                str(manifest_row["dataset_key"]),
                str(event["dataset"]) if "dataset" in event.index else str(manifest_row["dataset"]),
                str(event[entity_col]),
                str(event["forecast_origin"]),
                str(event["target_time"]),
                str(event["component"]),
                int(pd.to_numeric(event["horizon"], errors="raise")),
                str(event["split"]) if "split" in event.index else "NA",
                str(event["mode"]) if "mode" in event.index else "NA",
            ))
    expected = pd.DataFrame(
        ledger_keys,
        columns=[
            "dataset_key",
            "dataset",
            "entity_id",
            "forecast_origin",
            "target_time",
            "component",
            "horizon",
            "split",
            "mode",
        ],
    )
    actual = forecast.copy()
    actual["horizon"] = pd.to_numeric(actual["horizon"], errors="raise").astype(int)
    merged = expected.merge(actual, on=list(expected.columns), how="left", indicator=True)
    if (merged["_merge"] != "both").any():
        missing_count = int((merged["_merge"] != "both").sum())
        write_blocker_report(out_dir, [{
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"external forecast csv missing {missing_count} ledger rows",
        }])
        raise RuntimeError(f"external forecast csv incomplete; see {out_dir / 'blocker_report.csv'}")
    if "method" not in forecast.columns:
        forecast["method"] = model_name
    if "split" not in forecast.columns:
        forecast["split"] = "NA"
    for col in ["pred_lower_50", "pred_upper_50", "pred_lower_90", "pred_upper_90"]:
        if col not in forecast.columns:
            forecast[col] = pd.to_numeric(forecast["pred_mean"], errors="coerce")
    if "y_true" not in forecast.columns:
        forecast["y_true"] = np.nan
    forecast.to_csv(out_dir / "forecast.csv", index=False)
    metrics = summarize_forecasts(forecast)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    timing = {
        "total_seconds": 0.0,
        "forecast_rows": int(len(forecast)),
        "metric_rows": int(len(metrics)),
        "expected_rows": int(pd.to_numeric(manifest["ledger_rows"], errors="raise").sum()),
        "fallback_group_count": 0,
        "nonfallback_scoring_rows": int(len(forecast)),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8")
    run_manifest = {
        "baseline_names": [model_name],
        "model": model_name,
        "backend": "external_forecast_csv",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "external_forecast_csv": str(external_csv),
        "expected_rows": timing["expected_rows"],
        "forecast_rows": int(len(forecast)),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([{"status": "external_forecast_csv", "path": str(external_csv)}]).to_csv(out_dir / "training_log.csv", index=False)
    return out_dir


def run_neuralforecast_from_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    model: str,
    external_forecast_csv: str | Path | None = None,
    backend: NeuralBackend | None = None,
    max_steps: int = 5,
    seed: int = 1,
    device: str = "cpu",
    require_dependencies: bool = True,
    require_test_rows: bool = True,
    fail_on_fallback: bool = False,
    root: Path | None = None,
    caster_root: Path | None = None,
) -> Path:
    start = time.time()
    model_name = canonical_neural_model(model)
    root = root or baseline_root()
    caster_root = caster_root or caster_root_from_baseline(root)
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("failed_series.csv", "sleeping_prefix_rows.csv", "blocker_report.csv", "blocker_report.md"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    blockers: list[dict[str, object]] = []

    if device == "cpu":
        _disable_cuda_for_cpu(device)
    if require_dependencies:
        try:
            dependency_report = require_neuralforecast_dependencies(out_dir / "dependency_report.json", seed=seed, selected_device=device)
        except NeuralDependencyError as exc:
            blockers.append({
                "dataset_key": "ALL",
                "ledger_row_index": "",
                "entity_id": "",
                "component": "",
                "forecast_origin": "",
                "reason": str(exc),
            })
            write_blocker_report(out_dir, blockers)
            raise
    else:
        dependency_report = {}

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    expected_rows = int(pd.to_numeric(manifest["ledger_rows"], errors="raise").sum())
    if external_forecast_csv:
        return run_external_forecast_csv(
            manifest=manifest,
            manifest_path=manifest_path,
            external_csv=Path(external_forecast_csv),
            out_dir=out_dir,
            model_name=model_name,
            root=root,
            caster_root=caster_root,
        )

    backend = backend or RealNeuralForecastBackend(max_steps=max_steps, seed=seed, device=device)
    rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    sleeping_prefix_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    nonfallback_scoring_rows = 0
    fallback_group_count = 0
    train_fallback_group_count = 0
    scope_summaries: list[dict[str, object]] = []
    activation_origins: dict[tuple[str, str, str], pd.Timestamp] = {}

    for _, manifest_row in manifest.iterrows():
        dataset_key = str(manifest_row["dataset_key"])
        panel_path = resolve_manifest_path(manifest_row["panel_path"], caster_root, root)
        ledger_path = resolve_manifest_path(manifest_row["ledger_path"], caster_root, root)
        panel = pd.read_csv(panel_path)
        ledger = pd.read_csv(ledger_path, keep_default_na=False)
        declared_rows = int(manifest_row["ledger_rows"])
        if len(ledger) != declared_rows:
            blockers.append({
                "dataset_key": dataset_key,
                "ledger_row_index": "",
                "entity_id": "",
                "component": "",
                "forecast_origin": "",
                "reason": f"ledger row count mismatch: declared={declared_rows} actual={len(ledger)}",
            })
            continue
        ledger_entity_col = choose_ledger_entity_col(ledger, str(manifest_row["panel_entity_col"]))
        history_index = build_history_index(panel, manifest_row, ledger)
        context_cols = context_columns(ledger)
        dataset_rows_before = len(rows)
                                                                               
                                                                              
                                                                                
                                                                       
        scoring = ledger[ledger["split"].astype(str).isin(SCORING_SPLITS)].copy()

        predictions: dict[tuple[str, str, str, int], float] = {}
        scoring_group_cols = ["component"]
        scoring_group_cols.extend(col for col in ("mode", "forecast_strategy") if col in scoring.columns)
        for _, comp_scoring in scoring.groupby(scoring_group_cols, dropna=False, sort=False):
            first_scoring = comp_scoring.iloc[0]
            component = str(first_scoring["component"])
            mode = str(first_scoring.get("mode", ""))
            strategy = normalize_forecast_strategy(
                first_scoring.get("forecast_strategy", ""),
                mode=mode,
                mode_kind=first_scoring.get("mode_kind", ""),
            )
            requested_h = int(pd.to_numeric(comp_scoring["horizon"], errors="raise").max())
            full_df = panel_component_to_nf_df(panel, manifest_row, ledger, component)
            h = 1 if strategy == RECURSIVE_ROLLOUT else requested_h
            if RELEASE_TIME_COL in full_df.columns and strategy != RECURSIVE_ROLLOUT:
                h = _maximum_native_horizon(
                    full_df,
                    comp_scoring,
                    ledger_entity_col,
                    int(manifest_row["cadence_days"]),
                )
            parsed_origins = pd.to_datetime(comp_scoring["forecast_origin"], errors="coerce")
            if parsed_origins.dropna().empty:
                blockers.append({
                    "dataset_key": dataset_key,
                    "ledger_row_index": "",
                    "entity_id": "",
                    "component": component,
                    "forecast_origin": "",
                    "reason": "scoring forecast_origin parse failed",
                })
                continue
            first_origin = pd.Timestamp(parsed_origins.dropna().min())
            first_entities = set(
                comp_scoring.loc[parsed_origins.eq(first_origin), ledger_entity_col].astype(str)
            )
            train_df = _available_nf_df(full_df, first_origin, first_entities)
            input_size = choose_input_size(train_df, h)
            min_len, max_len = _series_length_summary(train_df)
            for candidate_origin in sorted(pd.Timestamp(value) for value in parsed_origins.dropna().unique()):
                candidate_entities = set(
                    comp_scoring.loc[parsed_origins.eq(candidate_origin), ledger_entity_col].astype(str)
                )
                candidate_train = _available_nf_df(
                    full_df, candidate_origin, candidate_entities
                )
                candidate_input = choose_input_size(candidate_train, h)
                candidate_min, candidate_max = _series_length_summary(candidate_train)
                if candidate_min > candidate_input and candidate_min > h:
                    first_origin = candidate_origin
                    train_df = candidate_train
                    input_size = candidate_input
                    min_len, max_len = candidate_min, candidate_max
                    break
            model_scoring = comp_scoring[parsed_origins >= first_origin].copy()
            scope = NeuralScope(
                dataset_key=dataset_key,
                dataset=str(manifest_row["dataset"]),
                component=component,
                h=h,
                input_size=input_size,
                freq=cadence_to_freq(int(manifest_row["cadence_days"])),
                cadence_days=int(manifest_row["cadence_days"]),
                train_cutoff=first_origin,
                model_name=model_name,
                seed=seed,
                max_steps=max_steps,
                device=device,
                forecast_strategy=strategy,
                mode=mode,
            )
            scope_summaries.append({
                "dataset_key": dataset_key,
                "component": component,
                "h": h,
                "input_size": input_size,
                "train_cutoff": format_date(first_origin),
                "train_series_min_rows": min_len,
                "train_series_max_rows": max_len,
                "scoring_rows": int(len(model_scoring)),
                "pre_fit_insufficient_rows": int(len(comp_scoring) - len(model_scoring)),
                "forecast_strategy": strategy,
                "mode": mode,
                "requested_max_horizon": requested_h,
                "native_max_horizon": h,
                "release_gated": RELEASE_TIME_COL in full_df.columns,
            })
            if min_len <= input_size or min_len <= h:
                reason = f"insufficient training history for NeuralForecast: min_len={min_len} input_size={input_size} h={h}"
                for origin_text in sorted(comp_scoring["forecast_origin"].astype(str).unique()):
                    fallback_group_count += 1
                    training_rows.append({
                        "dataset_key": dataset_key,
                        "dataset": str(manifest_row["dataset"]),
                        "component": component,
                        "split": ";".join(sorted(comp_scoring["split"].astype(str).unique())),
                        "forecast_origin": origin_text,
                        "train_cutoff": format_date(first_origin),
                        "train_rows": int(len(train_df)),
                        "train_series_min_rows": min_len,
                        "train_series_max_rows": max_len,
                        "h": h,
                        "input_size": input_size,
                        "status": "fallback",
                        "runtime_seconds": 0.0,
                        "fit_seconds": 0.0,
                        "failure_reason": reason,
                        "fallback_used": True,
                        "seed": seed,
                        "device": device,
                        "config_summary": f"max_steps={max_steps};freq={scope.freq}",
                        "forecast_strategy": strategy,
                        "mode": mode,
                    })
                continue
            activation_origins[(dataset_key, component, mode)] = first_origin
            try:
                comp_predictions, comp_logs = backend.fit_predict(
                    model_name=model_name,
                    train_df=train_df,
                    full_df=full_df,
                    scoring_ledger=model_scoring,
                    ledger_entity_col=ledger_entity_col,
                    scope=scope,
                )
                for key, value in comp_predictions.items():
                    predictions[(component, mode, *key)] = value
                training_rows.extend(comp_logs)
            except Exception as exc:
                blockers.append({
                    "dataset_key": dataset_key,
                    "ledger_row_index": "",
                    "entity_id": "",
                    "component": component,
                    "forecast_origin": "",
                    "reason": f"NeuralForecast bounded training/prediction failed for {component}: {type(exc).__name__}: {exc}",
                })

        for ledger_idx, event in ledger.iterrows():
            entity_id = str(event[ledger_entity_col])
            component = str(event["component"])
            origin = pd.to_datetime(event["forecast_origin"], errors="coerce")
            target = pd.to_datetime(event["target_time"], errors="coerce")
            horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
            series = history_index.get((entity_id, component))
            values = values_until_origin(series, origin) if series is not None else np.asarray([], dtype=float)
            reason = ""
            if pd.isna(origin):
                reason = "forecast_origin parse failed"
            elif pd.isna(target):
                reason = "target_time parse failed"
            elif series is None:
                reason = "no panel series for entity/component"
            elif len(values) == 0:
                reason = "no finite history with panel_time <= forecast_origin"
            y_true = pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0]
            if parse_bool(event.get("observed_mask", True)) and not np.isfinite(y_true):
                reason = "observed_mask true but observed_value missing/non-finite"
            if reason:
                blockers.append({
                    "dataset_key": dataset_key,
                    "ledger_row_index": int(ledger_idx),
                    "entity_id": entity_id,
                    "component": component,
                    "forecast_origin": str(event.get("forecast_origin", "")),
                    "reason": reason,
                })
                continue
            split = str(event["split"]) if "split" in ledger.columns else "NA"
            mode = str(event.get("mode", ""))
            pred_key = (component, mode, entity_id, str(event["forecast_origin"]), horizon)
            if split in SCORING_SPLITS and pred_key in predictions:
                rows.append(_model_row(
                    model_name,
                    dataset_key,
                    manifest_row,
                    event,
                    entity_id,
                    component,
                    origin,
                    target,
                    horizon,
                    predictions[pred_key],
                    values,
                    context_cols,
                ))
                nonfallback_scoring_rows += 1
            else:
                activation_origin = activation_origins.get((dataset_key, component, mode))
                structural_prefix = bool(
                    split == TRAIN_SPLIT
                    and activation_origin is not None
                    and pd.notna(origin)
                    and pd.Timestamp(origin) < pd.Timestamp(activation_origin)
                )
                if structural_prefix:
                    status = "structural_unavailable"
                    failure_reason = (
                        "structural_history_unavailable: neural model activates at "
                        f"{format_date(pd.Timestamp(activation_origin))}"
                    )
                else:
                    status = "runtime_forecast_missing"
                    failure_reason = "missing model-generated forecast after activation"
                rows.append(_last_value_row(
                    model_name,
                    dataset_key,
                    manifest_row,
                    event,
                    entity_id,
                    component,
                    origin,
                    target,
                    horizon,
                    values,
                    context_cols,
                    status,
                    failure_reason,
                ))
                validation_row = {
                        "dataset_key": dataset_key,
                        "dataset": str(manifest_row["dataset"]),
                        "entity_id": entity_id,
                        "component": component,
                        "forecast_origin": str(event["forecast_origin"]),
                        "split": split,
                        "horizon": horizon,
                        "failure_reason": failure_reason,
                        "fallback_method": "last_value",
                        "forecast_status": status,
                    }
                if structural_prefix:
                    sleeping_prefix_rows.append(validation_row)
                    train_fallback_group_count += 1
                elif split in SCORING_SPLITS:
                    failed_rows.append(validation_row)

        dataset_summaries.append({
            "dataset_key": dataset_key,
            "dataset": str(manifest_row["dataset"]),
            "ledger_rows": declared_rows,
            "forecast_rows": int(len(rows) - dataset_rows_before),
            "panel_path": str(manifest_row["panel_path"]),
            "ledger_path": str(manifest_row["ledger_path"]),
            "cadence_days": int(manifest_row["cadence_days"]),
        })

    pd.DataFrame(training_rows).to_csv(out_dir / "training_log.csv", index=False)
    if sleeping_prefix_rows:
        pd.DataFrame(sleeping_prefix_rows).to_csv(out_dir / "sleeping_prefix_rows.csv", index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(out_dir / "failed_series.csv", index=False)
    if blockers:
        write_blocker_report(out_dir, blockers)
        raise RuntimeError(f"NeuralForecast run blocked; see {out_dir / 'blocker_report.csv'}")
    if fail_on_fallback and failed_rows:
        raise RuntimeError(
            f"NeuralForecast produced {len(failed_rows)} non-sleeping fallback rows; "
            "formal validation/embargo/test and post-activation runs require native forecasts; see "
            f"{out_dir / 'failed_series.csv'}"
        )

    forecast = pd.DataFrame(rows)
    if len(forecast) != expected_rows:
        write_blocker_report(out_dir, [{
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"forecast row count mismatch: expected={expected_rows} actual={len(forecast)}",
        }])
        raise RuntimeError(f"row-count mismatch; see {out_dir / 'blocker_report.csv'}")
    scoring_rows = forecast[forecast["split"].astype(str).isin(SCORING_SPLITS)]
    test_rows = forecast[forecast["split"].astype(str) == "test"]
    nonfallback_test_rows = int((test_rows["model_status"] == "model_ok").sum())
    if nonfallback_scoring_rows <= 0 or (require_test_rows and nonfallback_test_rows <= 0):
        write_blocker_report(out_dir, [{
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": "no NeuralForecast model-generated scoring forecasts or required test forecasts",
        }])
        raise RuntimeError(f"no non-fallback scoring forecasts; see {out_dir / 'blocker_report.csv'}")

    forecast.to_csv(out_dir / "forecast.csv", index=False)
    metrics = summarize_forecasts(forecast)
    finite_failures = finite_metric_check(metrics)
    if finite_failures:
        write_blocker_report(out_dir, [{
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"metrics contain non-finite values: {finite_failures}",
        }])
        raise RuntimeError(f"non-finite metrics; see {out_dir / 'blocker_report.csv'}")
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    fallback_rows = int(forecast["model_status"].astype(str).ne("model_ok").sum())
    timing = {
        "total_seconds": round(time.time() - start, 6),
        "forecast_rows": int(len(forecast)),
        "metric_rows": int(len(metrics)),
        "expected_rows": expected_rows,
        "fallback_rows": fallback_rows,
        "structural_sleeping_prefix_rows": int(len(sleeping_prefix_rows)),
        "disallowed_fallback_rows": int(len(failed_rows)),
        "nonfallback_scoring_rows": int(nonfallback_scoring_rows),
        "nonfallback_test_rows": nonfallback_test_rows,
        "require_test_rows": bool(require_test_rows),
        "formal_fail_on_fallback": bool(fail_on_fallback),
        "fallback_group_count": int(fallback_group_count),
        "train_fallback_group_count": int(train_fallback_group_count),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8")
    run_manifest = {
        "baseline_names": [model_name],
        "model": model_name,
        "backend": "neuralforecast",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_rows": expected_rows,
        "forecast_rows": int(len(forecast)),
        "dataset_summaries": dataset_summaries,
        "scope_summaries": scope_summaries,
        "dependency_report": dependency_report,
        "seed": seed,
        "device": device,
        "max_steps": max_steps,
        "nonfallback_scoring_rows": int(nonfallback_scoring_rows),
        "nonfallback_test_rows": nonfallback_test_rows,
        "fallback_rows": fallback_rows,
        "structural_sleeping_prefix_rows": int(len(sleeping_prefix_rows)),
        "disallowed_fallback_rows": int(len(failed_rows)),
        "train_fallback_group_count": int(train_fallback_group_count),
        "scoring_fallback_group_count": int(fallback_group_count),
        "failed_series_path": "failed_series.csv" if failed_rows else "",
        "sleeping_prefix_rows_path": "sleeping_prefix_rows.csv" if sleeping_prefix_rows else "",
        "no_leakage_rule": (
            "prediction and fit inputs are truncated to panel_time <= forecast_origin and, when "
            "present, target_release_time <= forecast_origin"
        ),
        "release_time_column_precedence": list(RELEASE_TIME_CANDIDATES),
        "release_availability_rule": (
            "a target row with release metadata is usable only when its parsed release time is no "
            "later than the as-of forecast origin"
        ),
        "native_horizon_alignment_rule": (
            "native step is the target-date distance from each entity's last released observation "
            "on the declared cadence; output is mapped back to the ledger's nominal horizon by "
            "entity and target date"
        ),
        "forecast_strategy_rule": (
            "direct trains enough native multi-output steps to reach requested target dates from the "
            "last released observation; recursive_rollout trains h=1, executes the actual native "
            "steps, and repeatedly appends only predicted means during inference"
        ),
        "forecast_availability_policy": (
            "only provenance-marked train rows before the first leakage-safe fit origin are sleeping; "
            "validation/embargo/test and all post-activation missing forecasts fail closed"
        ),
        "missing_value_policy": (
            "past-only within-series forward fill before as-of NeuralForecast input; leading missing "
            "observations are dropped and release-dated rows remain unavailable until release"
        ),
        **forecast_strategy_manifest_fields(forecast),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir
