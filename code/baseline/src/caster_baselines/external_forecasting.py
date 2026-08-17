from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_validation import baseline_root, caster_root_from_baseline, resolve_manifest_path, sha256_file
from .forecasting_dependencies import DependencyError, require_forecasting_dependencies
from .ledger_runner import (
    Z50,
    Z90,
    build_history_index,
    choose_ledger_entity_col,
    context_columns,
    forecast_strategy_manifest_fields,
    format_date,
    infer_season_length,
    parse_bool,
    residual_sigma,
    write_blocker_report,
)
from .metrics import summarize_forecasts
from .forecast_strategy import (
    RECURSIVE_ROLLOUT,
    recursive_mean_path,
    strategy_from_event,
    strategy_group_columns,
)
from .native_sidecar import (
    NativeAvailabilityRecord,
    NativeSidecarStorageValidationRow,
    PredictionBundle,
    default_native_sidecar_root,
    native_rows_from_prediction,
    status_availability,
    write_availability_report,
    write_dependency_unavailable,
    write_storage_validation,
    STATUS_UNAVAILABLE,
)


PredictionFn = Callable[
    [str, np.ndarray, np.ndarray, int, dict[int, pd.Timestamp], int],
    dict[int, float] | PredictionBundle,
]

STATSFORECAST_MODELS = {"autoarima", "autoets", "autotheta", "autoces"}

                                                                        
                                                                           
                                                        
CSV_NULL_RESTORE_COMPATIBLE_RUNNER_SHA256 = {
    "b4216a39d22bc0ed8c7c012868e0a6d5d143b721c52f9cca6ba1fbcd727d958a",
}


@dataclass(frozen=True)
class HistorySlice:
    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class NativeTargetRequest:
    nominal_horizon: int
    native_horizon_steps: int
    target_time: pd.Timestamp
    forecasted_native_target_time: pd.Timestamp


def history_until_origin(series, origin: pd.Timestamp) -> HistorySlice:
    if pd.isna(origin):
        return HistorySlice(np.asarray([], dtype="datetime64[ns]"), np.asarray([], dtype=float))
    origin_np = np.datetime64(pd.Timestamp(origin), "ns")
    visible = (series.times <= origin_np) & (series.releases <= origin_np)
    times = series.times[visible]
    values = series.values[visible]
    mask = np.isfinite(values)
    return HistorySlice(times=times[mask], values=values[mask])


def native_target_requests(
    group: pd.DataFrame,
    history: HistorySlice,
    cadence_days: int,
) -> tuple[pd.Timestamp, dict[int, NativeTargetRequest], int]:
    """Resolve ledger targets against the last finite, released target value.

    Ledger horizons are measured from the forecast origin.  A model's native
    horizon is instead measured from the final target observation admitted by
    release gating.  These coincide for no-lag panels but need not coincide for
    release-lagged or intermittently missing targets.
    """

    cadence_days = int(cadence_days)
    if cadence_days <= 0:
        raise ValueError(f"cadence_days must be positive, got {cadence_days}")
    if len(history.times) == 0 or len(history.values) == 0:
        raise ValueError("cannot resolve native horizons from empty released history")
    last_released_target_time = pd.Timestamp(history.times[-1])
    cadence = pd.Timedelta(days=cadence_days)
    requests: dict[int, NativeTargetRequest] = {}
    max_native_horizon = 0
    for _, event in group.iterrows():
        nominal_horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
        target = pd.to_datetime(event["target_time"], errors="coerce")
        if pd.isna(target):
            raise ValueError(f"target_time parse failed: {event['target_time']}")
        native_float = (pd.Timestamp(target) - last_released_target_time) / cadence
        native_horizon = int(round(float(native_float)))
        if native_horizon < 1 or not math.isclose(
            float(native_float), float(native_horizon), abs_tol=1e-9
        ):
            raise ValueError(
                "target_time is not a positive native-cadence step after the last "
                "finite released target: "
                f"last_released_target_time={format_date(last_released_target_time)} "
                f"target_time={format_date(pd.Timestamp(target))} "
                f"cadence_days={cadence_days}"
            )
        forecasted_target = last_released_target_time + native_horizon * cadence
        if pd.Timestamp(forecasted_target) != pd.Timestamp(target):
            raise ValueError(
                "native target-date assertion failed: "
                f"forecasted_native_target_time={format_date(forecasted_target)} "
                f"ledger_target_time={format_date(pd.Timestamp(target))}"
            )
        request = NativeTargetRequest(
            nominal_horizon=nominal_horizon,
            native_horizon_steps=native_horizon,
            target_time=pd.Timestamp(target),
            forecasted_native_target_time=pd.Timestamp(forecasted_target),
        )
        previous = requests.get(nominal_horizon)
        if previous is not None and previous != request:
            raise ValueError(
                f"conflicting target requests for nominal horizon {nominal_horizon}: "
                f"{previous} vs {request}"
            )
        requests[nominal_horizon] = request
        max_native_horizon = max(max_native_horizon, native_horizon)
    if not requests:
        raise ValueError("empty target request group")
    return last_released_target_time, requests, max_native_horizon


def fallback_predictions(values: np.ndarray, horizons: list[int]) -> dict[int, float]:
    if len(values) == 0:
        raise ValueError("empty history")
    last = float(values[-1])
    return {int(h): last for h in horizons}


def prophet_native_min_train_rows(cadence_days: int, season_length: int) -> int:
    ""









    _ = max(1, int(cadence_days)), max(1, int(season_length))
    return 2


def cadence_aware_min_train_rows(cadence_days: int, season_length: int) -> int:
    ""

    return prophet_native_min_train_rows(cadence_days, season_length)


def make_statsforecast_predictor() -> PredictionFn:
    ""







    from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta

    def predict(
        model_name: str,
        history_times: np.ndarray,
        values: np.ndarray,
        max_horizon: int,
        target_dates_by_horizon: dict[int, pd.Timestamp],
        season_length: int,
    ) -> dict[int, float]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError("empty finite history")
        seasonal = bool(
            model_name != "autoarima"
            and season_length > 1
            and len(values) >= max(2 * int(season_length), int(max_horizon) + int(season_length) + 2)
        )
        model_season = int(season_length) if seasonal else 1

        if model_name == "autoarima":
                                                                          
                                                                               
                                                                            
            fit_values = values[-min(len(values), 104) :]
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
        elif model_name == "autoets":
                                                                                
            model = AutoETS(season_length=model_season, model="ZZZ")
        elif model_name == "autotheta":
                                                                              
                                                                     
            model = AutoTheta(season_length=model_season, decomposition_type="additive")
        elif model_name == "autoces":
            model = AutoCES(season_length=model_season)
        else:
            raise ValueError(f"unknown StatsForecast model: {model_name}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="possible convergence problem.*")
            warnings.filterwarnings("ignore", message="Stepwise search was stopped early.*")
            if model_name == "autoces":
                                                                               
                                                                           
                                                                              
                                                                               
                pred = model.forecast(y=values, h=int(max_horizon))
            else:
                model.fit(fit_values if model_name == "autoarima" else values)
                pred = model.predict(h=int(max_horizon))
        mean = np.asarray(pred["mean"], dtype=float)
        if len(mean) == 0:
            raise ValueError("StatsForecast returned empty mean vector")
        return {int(h): float(mean[min(max(int(h) - 1, 0), len(mean) - 1)]) for h in target_dates_by_horizon}

    return predict


def _normalise_external_model_config(
    backend: str,
    model_config: dict[str, object] | None,
) -> dict[str, object]:
    ""

    config = dict(model_config or {})
    if backend != "prophet":
        if config:
            raise ValueError(
                f"model_config is only supported for prophet, got backend={backend}: "
                f"{sorted(config)}"
            )
        return {}

    allowed = {"yearly_seasonality_mode"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown Prophet model_config fields: {unknown}")
    mode = str(config.get("yearly_seasonality_mode", "auto")).strip().lower()
    if mode not in {"auto", "off"}:
        raise ValueError(
            "Prophet yearly_seasonality_mode must be one of: auto, off"
        )
    return {"yearly_seasonality_mode": mode}


def make_prophet_predictor(*, yearly_seasonality_mode: str = "auto") -> PredictionFn:
    ""







    from prophet import Prophet

    config = _normalise_external_model_config(
        "prophet", {"yearly_seasonality_mode": yearly_seasonality_mode}
    )
    yearly_seasonality_mode = str(config["yearly_seasonality_mode"])

    def predict(
        model_name: str,
        history_times: np.ndarray,
        values: np.ndarray,
        max_horizon: int,
        target_dates_by_horizon: dict[int, pd.Timestamp],
        season_length: int,
    ) -> dict[int, float]:
        train = pd.DataFrame({
            "ds": pd.to_datetime(history_times),
            "y": np.asarray(values, dtype=float),
        }).dropna()
        if train.empty:
            raise ValueError("empty Prophet training frame")
        train = train.sort_values("ds")
        yearly_auto = bool(
            season_length >= 52 and len(train) >= 2 * int(season_length)
        )
        yearly = bool(
            yearly_auto and yearly_seasonality_mode == "auto"
        )
        weekly = bool(season_length == 7 or (len(train) >= 21 and train["ds"].diff().dt.days.dropna().median() <= 1.5))
        n_changepoints = max(0, min(25, len(train) // 2 - 1))
        model = Prophet(
            growth="linear",
            n_changepoints=n_changepoints,
            changepoint_prior_scale=0.05,
            seasonality_mode="additive",
            daily_seasonality=False,
            weekly_seasonality=weekly,
            yearly_seasonality=yearly,
                                                                        
                                                                    
            uncertainty_samples=0,
        )
                                                                        
                                                                            
                                                                          
        if (
            yearly_seasonality_mode != "off"
            and season_length > 1
            and not weekly
            and not yearly
            and len(train) >= 2 * int(season_length)
        ):
            model.add_seasonality(name=f"season_{int(season_length)}", period=float(season_length), fourier_order=min(10, max(3, int(season_length) // 2)))
        model.fit(train)
        future = pd.DataFrame({"ds": pd.to_datetime(list(target_dates_by_horizon.values()))})
        forecast = model.predict(future)
        return {
            int(h): float(yhat)
            for h, yhat in zip(target_dates_by_horizon, forecast["yhat"].to_numpy(dtype=float))
        }

    return predict


def default_predictor(
    backend: str,
    model_config: dict[str, object] | None = None,
) -> PredictionFn:
    config = _normalise_external_model_config(backend, model_config)
    if backend == "statsforecast":
        return make_statsforecast_predictor()
    if backend == "prophet":
        return make_prophet_predictor(
            yearly_seasonality_mode=str(config["yearly_seasonality_mode"])
        )
    raise ValueError(f"unknown backend: {backend}")


def finite_metric_check(metrics: pd.DataFrame) -> list[str]:
    numeric_cols = [
        "mae",
        "rmse",
        "gaussian_nll",
        "coverage_50",
        "coverage_90",
        "width_50",
        "width_90",
    ]
    return [
        col
        for col in numeric_cols
        if col in metrics.columns and not np.isfinite(pd.to_numeric(metrics[col], errors="coerce")).all()
    ]


def select_manifest_rows(manifest: pd.DataFrame, dataset_keys: list[str] | None) -> pd.DataFrame:
    if not dataset_keys:
        return manifest.copy()
    wanted = set(dataset_keys)
    selected = manifest[manifest["dataset_key"].astype(str).isin(wanted)].copy()
    if selected.empty:
        selected = manifest[manifest["dataset"].astype(str).isin(wanted)].copy()
    return selected


def run_external_forecaster_from_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    model_name: str,
    backend: str,
    dataset_keys: list[str] | None = None,
    predictor: PredictionFn | None = None,
    min_train_rows: int | None = 3,
    required_packages: list[str] | None = None,
    root: Path | None = None,
    caster_root: Path | None = None,
    enable_native_sidecars: bool = False,
    native_sidecar_root: str | Path | None = None,
    fail_on_fallback: bool = False,
    model_config: dict[str, object] | None = None,
) -> Path:
    start = time.time()
    root = root or baseline_root()
    caster_root = caster_root or caster_root_from_baseline(root)
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    native_root = default_native_sidecar_root(out_dir, native_sidecar_root) if enable_native_sidecars else None
    native_availability_rows: list[NativeAvailabilityRecord] = []
    native_storage_rows: list[NativeSidecarStorageValidationRow] = []
    for stale_name in ("failed_series.csv", "blocker_report.csv", "blocker_report.md"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    blockers: list[dict[str, object]] = []
    canonical_model_config = _normalise_external_model_config(
        backend, model_config
    )

    if required_packages:
        try:
            dependency_rows = require_forecasting_dependencies(out_dir / "dependency_report.json", required_packages)
        except DependencyError as exc:
            if native_root is not None:
                write_dependency_unavailable(native_root, model_id=model_name, backend=backend, reason=str(exc))
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
        dependency_rows = []

    try:
        predictor = predictor or default_predictor(
            backend, model_config=canonical_model_config
        )
    except Exception as exc:
        if native_root is not None:
            write_dependency_unavailable(
                native_root,
                model_id=model_name,
                backend=backend,
                reason=f"{backend} predictor import failed: {type(exc).__name__}: {exc}",
            )
        blockers.append({
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"{backend} predictor import failed: {type(exc).__name__}: {exc}",
        })
        write_blocker_report(out_dir, blockers)
        raise RuntimeError(f"{backend} predictor import failed; see {out_dir / 'blocker_report.csv'}") from exc

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    selected_manifest = select_manifest_rows(manifest, dataset_keys)
    if selected_manifest.empty:
        blockers.append({
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"no manifest rows selected for dataset_keys={dataset_keys}",
        })
        write_blocker_report(out_dir, blockers)
        raise RuntimeError(f"no manifest rows selected; see {out_dir / 'blocker_report.csv'}")

    expected_rows = int(pd.to_numeric(selected_manifest["ledger_rows"], errors="raise").sum())
    rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []

    for _, manifest_row in selected_manifest.iterrows():
        dataset_key = str(manifest_row["dataset_key"])
        panel_path = resolve_manifest_path(manifest_row["panel_path"], caster_root, root)
        ledger_path = resolve_manifest_path(manifest_row["ledger_path"], caster_root, root)
        panel = pd.read_csv(panel_path)
        ledger = pd.read_csv(ledger_path, keep_default_na=False)
        if "forecast_id" not in ledger.columns:
            ledger["forecast_id"] = [
                f"{dataset_key}__ledger_row_{int(index)}"
                for index in ledger.index
            ]
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
        season_length = infer_season_length(int(manifest_row["cadence_days"]))
        effective_min_train_rows = (
            prophet_native_min_train_rows(
                int(manifest_row["cadence_days"]), season_length
            )
            if min_train_rows is None or int(min_train_rows) <= 0
            else int(min_train_rows)
        )
        context_cols = context_columns(ledger)
        forecast_rows_before = len(rows)
        checkpoint_dir = out_dir / ".checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_forecast = checkpoint_dir / f"{model_name}.{dataset_key}.forecast.csv"
        checkpoint_training = checkpoint_dir / f"{model_name}.{dataset_key}.training.csv"
        checkpoint_metadata = checkpoint_dir / f"{model_name}.{dataset_key}.json"
        checkpoint_identity = {
            "schema": "caster_external_forecast_checkpoint_v3_native_horizon",
            "model": model_name,
            "backend": backend,
            "dataset_key": dataset_key,
            "manifest_sha256": sha256_file(manifest_path),
            "panel_sha256": sha256_file(panel_path),
            "ledger_sha256": sha256_file(ledger_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "min_train_rows": effective_min_train_rows,
            "native_horizon_policy": (
                "target_date_minus_last_finite_released_target_date"
            ),
        }
        if backend == "prophet":
            checkpoint_identity["model_config"] = canonical_model_config
        completed_forecast_ids: set[str] = set()
        if native_root is None and checkpoint_metadata.is_file():
            try:
                saved_identity = json.loads(
                    checkpoint_metadata.read_text(encoding="utf-8")
                )
                identity_without_runner_matches = all(
                    saved_identity.get(key) == value
                    for key, value in checkpoint_identity.items()
                    if key != "runner_sha256"
                )
                saved_runner_sha256 = str(saved_identity.get("runner_sha256", ""))
                runner_is_compatible = saved_runner_sha256 == str(
                    checkpoint_identity["runner_sha256"]
                ) or saved_runner_sha256 in CSV_NULL_RESTORE_COMPATIBLE_RUNNER_SHA256
                if identity_without_runner_matches and runner_is_compatible:
                    partial = pd.read_csv(
                        checkpoint_forecast,
                        keep_default_na=False,
                        low_memory=False,
                    )
                    for column in (
                        "horizon",
                        "native_horizon_steps",
                        "y_true",
                        "pred_mean",
                        "pred_lower_50",
                        "pred_upper_50",
                        "pred_lower_90",
                        "pred_upper_90",
                    ):
                        if column in partial.columns:
                            partial[column] = pd.to_numeric(
                                partial[column], errors="coerce"
                            )
                    if "fallback_used" in partial.columns:
                        partial["fallback_used"] = partial["fallback_used"].map(
                            parse_bool
                        )
                    partial_training = pd.read_csv(
                        checkpoint_training,
                        keep_default_na=False,
                        low_memory=False,
                    )
                    if "fallback_used" in partial_training.columns:
                        partial_training["fallback_used"] = partial_training[
                            "fallback_used"
                        ].map(parse_bool)
                    expected_ids = set(ledger["forecast_id"].astype(str))
                    completed_forecast_ids = set(
                        partial["forecast_id"].astype(str)
                    )
                    if not completed_forecast_ids.issubset(expected_ids):
                        raise ValueError("checkpoint has forecast_ids outside ledger")
                    if partial["forecast_id"].astype(str).duplicated().any():
                        raise ValueError("checkpoint has duplicate forecast_ids")
                    rows.extend(partial.to_dict(orient="records"))
                    training_rows.extend(
                        partial_training.to_dict(orient="records")
                    )
                    print(
                        f"forecast_checkpoint_resume model={model_name} "
                        f"dataset={dataset_key} rows={len(partial)}",
                        flush=True,
                    )
            except Exception as exc:
                completed_forecast_ids = set()
                print(
                    f"forecast_checkpoint_ignored model={model_name} "
                    f"dataset={dataset_key} reason={type(exc).__name__}: {exc}",
                    flush=True,
                )

        def write_checkpoint() -> None:
            if native_root is not None:
                return
            native_rows = [
                row
                for row in rows
                if str(row.get("dataset_key", "")) == dataset_key
                and not parse_bool(row.get("fallback_used", False))
            ]
            native_training = [
                row
                for row in training_rows
                if str(row.get("dataset_key", "")) == dataset_key
                and not parse_bool(row.get("fallback_used", False))
            ]
            forecast_tmp = checkpoint_forecast.with_suffix(".csv.tmp")
            training_tmp = checkpoint_training.with_suffix(".csv.tmp")
            metadata_tmp = checkpoint_metadata.with_suffix(".json.tmp")
            pd.DataFrame(native_rows).to_csv(forecast_tmp, index=False)
            pd.DataFrame(native_training).to_csv(training_tmp, index=False)
            metadata_tmp.write_text(
                json.dumps(
                    {
                        **checkpoint_identity,
                        "completed_forecast_rows": len(native_rows),
                        "completed_group_rows": len(native_training),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            forecast_tmp.replace(checkpoint_forecast)
            training_tmp.replace(checkpoint_training)
            metadata_tmp.replace(checkpoint_metadata)

        group_cols = strategy_group_columns(ledger, [ledger_entity_col, "component", "forecast_origin"])
        grouped = ledger.groupby(group_cols, dropna=False, sort=False)
        group_total = int(grouped.ngroups)
        print(
            f"forecast_dataset_start model={model_name} dataset={dataset_key} "
            f"groups={group_total} ledger_rows={len(ledger)}",
            flush=True,
        )
        for group_number, (group_key, group) in enumerate(grouped, start=1):
            group_forecast_ids = set(group["forecast_id"].astype(str))
            if group_forecast_ids and group_forecast_ids.issubset(
                completed_forecast_ids
            ):
                continue
            if group_number == 1 or group_number % 250 == 0 or group_number == group_total:
                elapsed = time.time() - start
                print(
                    f"forecast_progress model={model_name} dataset={dataset_key} "
                    f"groups_done={group_number}/{group_total} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
            first_event = group.iloc[0]
            entity_id = str(first_event[ledger_entity_col])
            component = str(first_event["component"])
            origin_text = str(first_event["forecast_origin"])
            strategy = strategy_from_event(first_event)
            group_start = time.time()
            origin = pd.to_datetime(origin_text, errors="coerce")
            series = history_index.get((entity_id, component))
            hist = history_until_origin(series, origin) if series is not None else HistorySlice(np.asarray([], dtype="datetime64[ns]"), np.asarray([], dtype=float))
            max_nominal_horizon = int(pd.to_numeric(group["horizon"], errors="raise").max())
            target_dates_by_nominal_horizon = {
                int(pd.to_numeric(event["horizon"], errors="raise")): pd.to_datetime(event["target_time"], errors="coerce")
                for _, event in group.iterrows()
            }
            status = "model_ok"
            failure_reason = ""
            fallback_used = False
            if pd.isna(origin):
                blockers.append({
                    "dataset_key": dataset_key,
                    "ledger_row_index": int(group.index[0]),
                    "entity_id": entity_id,
                    "component": component,
                    "forecast_origin": origin_text,
                    "reason": "forecast_origin parse failed",
                })
                continue
            if series is None or len(hist.values) == 0:
                blockers.append({
                    "dataset_key": dataset_key,
                    "ledger_row_index": int(group.index[0]),
                    "entity_id": entity_id,
                    "component": component,
                    "forecast_origin": origin_text,
                    "reason": "no finite history with panel_time <= forecast_origin",
                })
                continue
            try:
                (
                    last_released_target_time,
                    requests_by_nominal_horizon,
                    max_native_horizon,
                ) = native_target_requests(
                    group,
                    hist,
                    int(manifest_row["cadence_days"]),
                )
            except Exception as exc:
                blockers.append({
                    "dataset_key": dataset_key,
                    "ledger_row_index": int(group.index[0]),
                    "entity_id": entity_id,
                    "component": component,
                    "forecast_origin": origin_text,
                    "reason": f"native horizon resolution failed: {type(exc).__name__}: {exc}",
                })
                continue
            target_dates_by_native_horizon = {
                request.native_horizon_steps: request.target_time
                for request in requests_by_nominal_horizon.values()
            }
            try:
                if len(hist.values) < effective_min_train_rows:
                    raise ValueError(
                        "insufficient training history: "
                        f"{len(hist.values)} < {effective_min_train_rows}"
                    )
                if strategy == RECURSIVE_ROLLOUT:
                    def one_step(step_times: np.ndarray, step_values: np.ndarray, step: int):
                        step_target = pd.Timestamp(step_times[-1]) + pd.Timedelta(
                            days=int(manifest_row["cadence_days"])
                        )
                        result = predictor(
                            model_name,
                            step_times,
                            step_values,
                            1,
                            {1: step_target},
                            season_length,
                        )
                        means = result.means if isinstance(result, PredictionBundle) else dict(result)
                        return float(means[1]), result

                    native_predictions, step_results = recursive_mean_path(
                        times=hist.times,
                        values=hist.values,
                        max_horizon=max_native_horizon,
                        cadence_days=int(manifest_row["cadence_days"]),
                        one_step=one_step,
                    )
                    predictions = {
                        nominal_horizon: float(
                            native_predictions[request.native_horizon_steps]
                        )
                        for nominal_horizon, request in requests_by_nominal_horizon.items()
                    }
                    prediction_result = None
                else:
                    # Prophet predicts explicit dates, so keep its established
                    # nominal-key/date call contract. StatsForecast indexes its
                    # vector by native steps and must be extended to the final
                    # actual target date.
                    predictor_max_horizon = (
                        max_nominal_horizon
                        if backend == "prophet"
                        else max_native_horizon
                    )
                    predictor_target_dates = (
                        target_dates_by_nominal_horizon
                        if backend == "prophet"
                        else target_dates_by_native_horizon
                    )
                    prediction_result = predictor(
                        model_name,
                        hist.times,
                        hist.values,
                        predictor_max_horizon,
                        predictor_target_dates,
                        season_length,
                    )
                    predictor_means = {
                        int(k): float(v)
                        for k, v in (
                            prediction_result.means.items()
                            if isinstance(prediction_result, PredictionBundle)
                            else dict(prediction_result).items()
                        )
                    }
                    if backend == "prophet":
                        predictions = {
                            nominal_horizon: predictor_means[nominal_horizon]
                            for nominal_horizon in requests_by_nominal_horizon
                        }
                    else:
                        predictions = {
                            nominal_horizon: predictor_means[
                                request.native_horizon_steps
                            ]
                            for nominal_horizon, request in requests_by_nominal_horizon.items()
                        }
                    step_results = {}
            except Exception as exc:
                status = "fallback"
                fallback_used = True
                failure_reason = f"{type(exc).__name__}: {exc}"
                predictions = fallback_predictions(
                    hist.values, sorted(target_dates_by_nominal_horizon)
                )
                prediction_result = None
                failed_rows.append({
                    "dataset_key": dataset_key,
                    "dataset": str(manifest_row["dataset"]),
                    "entity_id": entity_id,
                    "component": component,
                    "forecast_origin": origin_text,
                    "train_rows": int(len(hist.values)),
                    "max_horizon": max_native_horizon,
                    "max_nominal_horizon": max_nominal_horizon,
                    "max_native_horizon": max_native_horizon,
                    "failure_reason": failure_reason,
                    "fallback_method": "last_value",
                })

            runtime = round(time.time() - group_start, 6)
            training_rows.append({
                "dataset_key": dataset_key,
                "dataset": str(manifest_row["dataset"]),
                "entity_id": entity_id,
                "component": component,
                "forecast_origin": origin_text,
                "train_rows": int(len(hist.values)),
                "max_horizon": max_native_horizon,
                "max_nominal_horizon": max_nominal_horizon,
                "max_native_horizon": max_native_horizon,
                "min_train_rows": effective_min_train_rows,
                "status": status,
                "runtime_seconds": runtime,
                "failure_reason": failure_reason,
                "fallback_used": fallback_used,
            })

            sigma = max(residual_sigma(hist.values, float(hist.values[-1])), 1e-6)
            for ledger_idx, event in group.iterrows():
                horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
                request = requests_by_nominal_horizon[horizon]
                pred = float(predictions[horizon])
                target = pd.to_datetime(event["target_time"], errors="coerce")
                if pd.Timestamp(request.forecasted_native_target_time) != pd.Timestamp(target):
                    raise RuntimeError(
                        "native target-date assertion failed immediately before archive write: "
                        f"forecasted={format_date(request.forecasted_native_target_time)} "
                        f"ledger={format_date(target)}"
                    )
                y_true = pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0]
                if parse_bool(event.get("observed_mask", True)) and not np.isfinite(y_true):
                    blockers.append({
                        "dataset_key": dataset_key,
                        "ledger_row_index": int(ledger_idx),
                        "entity_id": entity_id,
                        "component": component,
                        "forecast_origin": origin_text,
                        "reason": "observed_mask true but observed_value missing/non-finite",
                    })
                    continue
                row = {
                    "dataset_key": dataset_key,
                    "dataset": str(event["dataset"]) if "dataset" in ledger.columns else str(manifest_row["dataset"]),
                    "method": model_name,
                    "entity_id": entity_id,
                    "forecast_origin": format_date(origin),
                    "target_time": format_date(target),
                    "component": component,
                    "horizon": horizon,
                    "last_released_target_time": format_date(last_released_target_time),
                    "native_horizon_steps": request.native_horizon_steps,
                    "forecasted_native_target_time": format_date(
                        request.forecasted_native_target_time
                    ),
                    "y_true": float(y_true) if np.isfinite(y_true) else np.nan,
                    "pred_mean": pred,
                    "pred_lower_50": float(pred - Z50 * sigma),
                    "pred_upper_50": float(pred + Z50 * sigma),
                    "pred_lower_90": float(pred - Z90 * sigma),
                    "pred_upper_90": float(pred + Z90 * sigma),
                    "split": str(event["split"]) if "split" in ledger.columns else "NA",
                    "model_status": status,
                    "fallback_used": fallback_used,
                    "failure_reason": failure_reason,
                    "fallback_method": "last_value" if fallback_used else "",
                }
                for col in context_cols:
                    row[col] = event[col]
                rows.append(row)
                if native_root is not None:
                    native_origin = f"{dataset_key}__{entity_id}__{component}__{format_date(origin)}__h{horizon}"
                    if fallback_used:
                        native_availability_rows.append(
                            status_availability(
                                model_id=model_name,
                                origin=native_origin,
                                status=STATUS_UNAVAILABLE,
                                blocker=f"model fallback used; no native distribution params available: {failure_reason}",
                                native_sidecar_required=True,
                            )
                        )
                    else:
                        native_result = (
                            step_results.get(request.native_horizon_steps)
                            if strategy == RECURSIVE_ROLLOUT
                            else prediction_result
                        )
                        prediction_key = (
                            1
                            if strategy == RECURSIVE_ROLLOUT
                            else horizon
                            if backend == "prophet"
                            else request.native_horizon_steps
                        )
                        means, availability, storage = native_rows_from_prediction(
                            native_root,
                            model_id=model_name,
                            origin=native_origin,
                            prediction_result=native_result,
                            horizon=prediction_key,
                        )
                        if prediction_key not in means:
                            availability.blocker = "; ".join(
                                x for x in [availability.blocker, f"missing forecast horizon {prediction_key} in native predictor output"] if x
                            )
                        native_availability_rows.append(availability)
                        if storage is not None:
                            native_storage_rows.append(storage)
            if group_number % 250 == 0 or group_number == group_total:
                write_checkpoint()

        dataset_summaries.append({
            "dataset_key": dataset_key,
            "dataset": str(manifest_row["dataset"]),
            "ledger_rows": declared_rows,
            "forecast_rows": int(len(rows) - forecast_rows_before),
            "group_count": group_total,
            "cadence_days": int(manifest_row["cadence_days"]),
            "season_length": season_length,
            "min_train_rows": effective_min_train_rows,
            "panel_path": str(manifest_row["panel_path"]),
            "ledger_path": str(manifest_row["ledger_path"]),
            "model_config": canonical_model_config,
        })

    pd.DataFrame(training_rows).to_csv(out_dir / "training_log.csv", index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(out_dir / "failed_series.csv", index=False)
        if fail_on_fallback:
            raise RuntimeError(
                f"{backend} produced {len(failed_rows)} fallback groups; "
                f"formal runs require native forecasts; see {out_dir / 'failed_series.csv'}"
            )

    if blockers:
        write_blocker_report(out_dir, blockers)
        raise RuntimeError(f"{backend} run blocked; see {out_dir / 'blocker_report.csv'}")

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
    if native_root is not None:
        write_availability_report(native_availability_rows, native_root / "native_likelihood_availability.csv")
        write_storage_validation(native_storage_rows, native_root / "native_sidecar_storage_validation.csv")

    fallback_count = int(sum(row["fallback_used"] for row in training_rows))
    timing = {
        "total_seconds": round(time.time() - start, 6),
        "forecast_rows": int(len(forecast)),
        "metric_rows": int(len(metrics)),
        "expected_rows": expected_rows,
        "group_rows": int(len(training_rows)),
        "fallback_group_count": fallback_count,
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8")
    run_manifest = {
        "baseline_names": [model_name],
        "model": model_name,
        "backend": backend,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_dataset_keys": dataset_keys or [],
        "expected_rows": expected_rows,
        "forecast_rows": int(len(forecast)),
        "dataset_summaries": dataset_summaries,
        "dependency_report": dependency_rows,
        "fallback_group_count": fallback_count,
        "formal_fail_on_fallback": bool(fail_on_fallback),
        "min_train_rows_configured": min_train_rows,
        "model_config": canonical_model_config,
        "model_config_by_dataset": {
            str(summary["dataset_key"]): canonical_model_config
            for summary in dataset_summaries
        },
        "min_train_rows_policy": (
            "native_feasibility_minimum_two_rows"
            if min_train_rows is None or int(min_train_rows) <= 0
            else "explicit_override"
        ),
        "failed_series_path": "failed_series.csv" if failed_rows else "",
        "no_leakage_rule": (
            "history uses only finite targets with panel_time <= forecast_origin "
            "and target release_time <= forecast_origin"
        ),
        "native_horizon_rule": (
            "native_horizon_steps=(target_time-last_released_target_time)/cadence; "
            "forecasted_native_target_time must equal ledger target_time before archive write"
        ),
        "native_horizon_audit_columns": [
            "last_released_target_time",
            "native_horizon_steps",
            "forecasted_native_target_time",
        ],
        "forecast_strategy_rule": (
            "StatsForecast direct predicts through the maximum native horizon; Prophet direct "
            "continues to predict explicit ledger target dates. recursive_rollout calls h=1 "
            "through the maximum native horizon after appending each prior predicted mean"
        ),
        "native_sidecars_enabled": bool(enable_native_sidecars),
        "native_sidecar_root": str(native_root) if native_root is not None else "",
        **forecast_strategy_manifest_fields(forecast),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir
