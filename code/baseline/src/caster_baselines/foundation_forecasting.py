from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .data_validation import baseline_root, caster_root_from_baseline, resolve_manifest_path, sha256_file
from .ledger_runner import (
    context_columns,
    build_history_index,
    choose_ledger_entity_col,
    finite_metric_check,
    forecast_strategy_manifest_fields,
    format_date,
    infer_season_length,
    parse_bool,
    residual_sigma,
    write_blocker_report,
)
from .metrics import summarize_forecasts
from .forecast_strategy import RECURSIVE_ROLLOUT, recursive_mean_path, strategy_from_event, strategy_group_columns


FOUNDATION_ALIASES = {
    "chronos": "chronos_bolt_small",
    "chronos_bolt": "chronos_bolt_small",
    "chronos_bolt_small": "chronos_bolt_small",
    "timesfm": "timesfm_2_0",
    "timesfm_2_0": "timesfm_2_0",
    "timesfm_2.0": "timesfm_2_0",
}
PACKAGE_KEYS = ("chronos-forecasting", "timesfm", "torch", "transformers")
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
NORMAL_CENTRAL_80_Z = 1.2815515655446004
NORMAL_CENTRAL_90_Z = 1.6448536269514722
CHRONOS_INTERVAL_SOURCE = "chronos_native_q25_q75_50_gaussian_proxy_90_from_q10_q90"
CHRONOS_INTERVAL_CONSTRUCTION_RULE = (
    "Chronos central 50% interval uses native q25/q75. The native q10/q90 central 80% width "
    "is converted to normal-equivalent sigma=(q90-q10)/(2*Z80), with Z80=1.2815515655446004; "
    "the exported Gaussian-proxy central 90% interval is centered on the returned q50 point "
    "forecast +/- Z90*sigma, with Z90=1.6448536269514722."
)


@dataclass(frozen=True)
class HistorySlice:
    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class FoundationPredictions:
    mean: dict[int, float]
    lower_50: dict[int, float]
    upper_50: dict[int, float]
    lower_90: dict[int, float]
    upper_90: dict[int, float]
    interval_source: str


class FoundationBackend(Protocol):
    interval_source: str

    def predict(
        self,
        values: np.ndarray,
        max_horizon: int,
        horizons: list[int],
        cadence_days: int,
    ) -> FoundationPredictions:
        ...


def canonical_foundation_model(model: str) -> str:
    key = model.strip().lower()
    if key not in FOUNDATION_ALIASES:
        raise ValueError(f"unknown foundation model {model!r}; available={sorted(FOUNDATION_ALIASES)}")
    return FOUNDATION_ALIASES[key]


def package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not_installed"


def checkpoint_sha256(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin", "config.json", "README.md"):
            candidate = p / name
            if candidate.exists():
                return sha256_file(candidate)
    return ""


def inspect_foundation_environment(
    *,
    checkpoint_id: str = "",
    checkpoint_path: str | Path | None = "",
    device: str = "cpu",
) -> dict[str, object]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "sys_prefix": sys.prefix,
        "package_versions": {package: package_version(package) for package in PACKAGE_KEYS},
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path or ""),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "device": device,
    }


def write_foundation_dependency_report(
    out_path: str | Path,
    *,
    checkpoint_id: str = "",
    checkpoint_path: str | Path | None = "",
    device: str = "cpu",
) -> dict[str, object]:
    report = inspect_foundation_environment(
        checkpoint_id=checkpoint_id,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def required_packages(model_name: str) -> tuple[str, ...]:
    if model_name == "chronos_bolt_small":
        return ("chronos-forecasting", "torch", "transformers")
    if model_name == "timesfm_2_0":
        return ("timesfm", "torch")
    raise ValueError(f"unsupported foundation model: {model_name}")


def dependency_failures(model_name: str, report: dict[str, object]) -> list[str]:
    versions = report.get("package_versions", {})
    if not isinstance(versions, dict):
        return ["package_versions missing from dependency report"]
    failures = []
    for package in required_packages(model_name):
        if versions.get(package) in {None, "", "not_installed"}:
            failures.append(f"{package} not installed")
    return failures


def history_until_origin(series, origin: pd.Timestamp) -> HistorySlice:
    if pd.isna(origin):
        return HistorySlice(np.asarray([], dtype="datetime64[ns]"), np.asarray([], dtype=float))
    origin_np = np.datetime64(pd.Timestamp(origin), "ns")
    visible = (series.times <= origin_np) & (series.releases <= origin_np)
    times = series.times[visible]
    values = series.values[visible]
    mask = np.isfinite(values)
    return HistorySlice(times=times[mask], values=values[mask])


def latest_feature_release_until_origin(series, origin: pd.Timestamp) -> pd.Timestamp:
    """Return the latest release timestamp actually admitted to model context."""
    if pd.isna(origin):
        return pd.NaT
    origin_np = np.datetime64(pd.Timestamp(origin), "ns")
    visible = (
        (series.times <= origin_np)
        & (series.releases <= origin_np)
        & np.isfinite(series.values)
    )
    if not np.any(visible):
        return pd.NaT
    return pd.Timestamp(np.max(series.releases[visible]))


def _prediction_from_quantiles(
    quantiles: np.ndarray,
    mean: np.ndarray | None,
    horizons: list[int],
    quantile_levels: list[float],
    *,
    gaussian_proxy_90_from_central_80: bool = False,
    interval_source: str = "model_quantiles",
) -> FoundationPredictions:
    q = np.asarray(quantiles, dtype=float)
    if q.ndim == 3:
        q = q[0]
    if q.shape[0] < max(horizons):
        raise ValueError(f"quantile forecast shorter than requested horizon: {q.shape[0]} < {max(horizons)}")
    mean_arr = np.asarray(mean, dtype=float) if mean is not None else None
    if mean_arr is not None and mean_arr.ndim == 2:
        mean_arr = mean_arr[0]
    idx = {float(level): i for i, level in enumerate(quantile_levels)}
    out_mean: dict[int, float] = {}
    lower_50: dict[int, float] = {}
    upper_50: dict[int, float] = {}
    lower_90: dict[int, float] = {}
    upper_90: dict[int, float] = {}
    for h in horizons:
        pos = int(h) - 1
        out_mean[h] = float(mean_arr[pos]) if mean_arr is not None else float(q[pos, idx[0.5]])
        lower_50[h] = float(q[pos, idx[0.25]])
        upper_50[h] = float(q[pos, idx[0.75]])
        q10 = float(q[pos, idx[0.1]])
        q90 = float(q[pos, idx[0.9]])
        if gaussian_proxy_90_from_central_80:
            if not np.isfinite(q10) or not np.isfinite(q90) or q90 < q10:
                raise ValueError(f"invalid q10/q90 interval at horizon {h}: q10={q10}, q90={q90}")
            sigma = (q90 - q10) / (2.0 * NORMAL_CENTRAL_80_Z)
            lower_90[h] = float(out_mean[h] - NORMAL_CENTRAL_90_Z * sigma)
            upper_90[h] = float(out_mean[h] + NORMAL_CENTRAL_90_Z * sigma)
        else:
            lower_90[h] = q10
            upper_90[h] = q90
    return FoundationPredictions(out_mean, lower_50, upper_50, lower_90, upper_90, interval_source)


def _prediction_from_point(values: np.ndarray, point: np.ndarray, horizons: list[int]) -> FoundationPredictions:
    point = np.asarray(point, dtype=float)
    if point.ndim == 2:
        point = point[0]
    if point.shape[0] < max(horizons):
        raise ValueError(f"point forecast shorter than requested horizon: {point.shape[0]} < {max(horizons)}")
    out_mean = {h: float(point[int(h) - 1]) for h in horizons}
    sigma = max(residual_sigma(values, float(point[0])), 1e-6)
    return FoundationPredictions(
        mean=out_mean,
        lower_50={h: float(out_mean[h] - 0.67448975 * sigma) for h in horizons},
        upper_50={h: float(out_mean[h] + 0.67448975 * sigma) for h in horizons},
        lower_90={h: float(out_mean[h] - 1.64485363 * sigma) for h in horizons},
        upper_90={h: float(out_mean[h] + 1.64485363 * sigma) for h in horizons},
        interval_source="residual_sigma",
    )


class ChronosBoltBackend:
    interval_source = CHRONOS_INTERVAL_SOURCE
    interval_construction_rule = CHRONOS_INTERVAL_CONSTRUCTION_RULE

    def __init__(self, checkpoint_id: str, checkpoint_path: str | Path | None = "", device: str = "cpu"):
        import torch
        from chronos import BaseChronosPipeline

        self.torch = torch
        source = str(checkpoint_path or checkpoint_id)
        self.pipeline = BaseChronosPipeline.from_pretrained(source, device_map=device)

    def predict(
        self,
        values: np.ndarray,
        max_horizon: int,
        horizons: list[int],
        cadence_days: int,
    ) -> FoundationPredictions:
        context = self.torch.tensor(np.asarray(values, dtype=np.float32))
        with self.torch.no_grad():
            quantiles, mean = self.pipeline.predict_quantiles(
                [context],
                prediction_length=int(max_horizon),
                quantile_levels=QUANTILE_LEVELS,
            )
        return _prediction_from_quantiles(
            quantiles=quantiles.detach().cpu().numpy(),
            mean=mean.detach().cpu().numpy() if mean is not None else None,
            horizons=horizons,
            quantile_levels=QUANTILE_LEVELS,
            gaussian_proxy_90_from_central_80=True,
            interval_source=self.interval_source,
        )


class TimesFMBackend:
    interval_source = "residual_sigma"

    def __init__(
        self,
        checkpoint_id: str,
        checkpoint_path: str | Path | None = "",
        device: str = "cpu",
        *,
        horizon_len: int = 128,
        context_len: int = 512,
    ):
        import timesfm

        self.timesfm = timesfm
        self.checkpoint_id = checkpoint_id
        self.checkpoint_path = str(checkpoint_path or "")
        self.device = device
        self.context_len = int(context_len)
        self.horizon_len = int(horizon_len)
        backend_name = "gpu" if str(device).lower().startswith(("cuda", "gpu")) else str(device).lower()
        if backend_name not in {"cpu", "gpu", "tpu"}:
            backend_name = "cpu"
        self.backend_name = backend_name
        if hasattr(timesfm, "TimesFm") and hasattr(timesfm, "TimesFmHparams") and hasattr(timesfm, "TimesFmCheckpoint"):
            checkpoint = timesfm.TimesFmCheckpoint(
                path=self.checkpoint_path or None,
                huggingface_repo_id=checkpoint_id if not self.checkpoint_path else None,
            )
                                                                            
                                                                              
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=self.backend_name,
                    per_core_batch_size=1,
                    context_len=self.context_len,
                    horizon_len=self.horizon_len,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=50,
                    model_dims=1280,
                    use_positional_embedding=False,
                ),
                checkpoint=checkpoint,
            )
        else:
            raise RuntimeError("TimesFM 2.0 API not available in installed timesfm package")

    def predict(
        self,
        values: np.ndarray,
        max_horizon: int,
        horizons: list[int],
        cadence_days: int,
    ) -> FoundationPredictions:
        freq = 0 if int(cadence_days) <= 1 else 1
        result = self.model.forecast(
            [np.asarray(values, dtype=np.float32)],
            freq=[freq],
            forecast_context_len=min(len(values), self.context_len),
        )
        point = result[0] if isinstance(result, tuple) else result
        pred = _prediction_from_point(values, point, horizons)
        if isinstance(result, tuple) and len(result) > 1 and result[1] is not None:
            quantiles = np.asarray(result[1], dtype=float)
            if quantiles.ndim == 3 and quantiles.shape[-1] >= 10:
                                                                             
                q = quantiles[:, : int(max_horizon), 1:10]
                try:
                    return _prediction_from_quantiles(q, np.asarray(point), horizons, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                except Exception:
                    return pred
        return pred


def real_backend(
    model_name: str,
    checkpoint_id: str,
    checkpoint_path: str | Path | None,
    device: str,
    *,
    max_horizon: int | None = None,
) -> FoundationBackend:
    if model_name == "chronos_bolt_small":
        return ChronosBoltBackend(checkpoint_id=checkpoint_id, checkpoint_path=checkpoint_path, device=device)
    if model_name == "timesfm_2_0":
        return TimesFMBackend(
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
            device=device,
            horizon_len=int(max_horizon or 128),
        )
    raise ValueError(f"unsupported foundation model: {model_name}")


def max_ledger_horizon(manifest: pd.DataFrame, root: Path, caster_root: Path) -> int:
    max_horizon = 1
    for _, manifest_row in manifest.iterrows():
        ledger_path = resolve_manifest_path(manifest_row["ledger_path"], caster_root, root)
        ledger = pd.read_csv(ledger_path, usecols=["horizon"])
        if len(ledger):
            max_horizon = max(max_horizon, int(pd.to_numeric(ledger["horizon"], errors="raise").max()))
    return max_horizon


def _expected_ledger_rows(manifest: pd.DataFrame, root: Path, caster_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, manifest_row in manifest.iterrows():
        ledger = pd.read_csv(resolve_manifest_path(manifest_row["ledger_path"], caster_root, root), keep_default_na=False)
        entity_col = choose_ledger_entity_col(ledger, str(manifest_row["panel_entity_col"]))
        for _, event in ledger.iterrows():
            records.append({
                "dataset_key": str(manifest_row["dataset_key"]),
                "dataset": str(event["dataset"]) if "dataset" in event.index else str(manifest_row["dataset"]),
                "entity_id": str(event[entity_col]),
                "forecast_origin": str(event["forecast_origin"]),
                "target_time": str(event["target_time"]),
                "component": str(event["component"]),
                "horizon": int(pd.to_numeric(event["horizon"], errors="raise")),
                "split": str(event["split"]) if "split" in event.index else "NA",
                "mode": str(event["mode"]) if "mode" in event.index else "NA",
                "y_true": pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0],
            })
    return pd.DataFrame(records)


def run_external_forecast_csv(
    *,
    manifest: pd.DataFrame,
    manifest_path: Path,
    external_csv: Path,
    out_dir: Path,
    model_name: str,
    dependency_report: dict[str, object],
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
        "split",
        "mode",
        "pred_mean",
        "generated_at",
        "features_available_until",
    }
    missing = required - set(forecast.columns)
    if missing:
        write_blocker_report(out_dir, [{"dataset_key": "ALL", "ledger_row_index": "", "entity_id": "", "component": "", "forecast_origin": "", "reason": f"external forecast csv missing columns: {sorted(missing)}"}])
        raise RuntimeError(f"external forecast csv missing columns; see {out_dir / 'blocker_report.csv'}")
    expected = _expected_ledger_rows(manifest, root, caster_root)
    forecast["horizon"] = pd.to_numeric(forecast["horizon"], errors="raise").astype(int)
    forecast_origin = pd.to_datetime(forecast["forecast_origin"], errors="raise")
    generated_at = pd.to_datetime(forecast["generated_at"], errors="coerce")
    features_available_until = pd.to_datetime(
        forecast["features_available_until"], errors="coerce"
    )
    invalid_provenance = generated_at.isna() | features_available_until.isna()
    post_origin_provenance = (
        (generated_at > forecast_origin)
        | (features_available_until > forecast_origin)
        | (features_available_until > generated_at)
    )
    if invalid_provenance.any() or post_origin_provenance.any():
        reason = (
            "external forecast CSV has invalid or non-as-of generated_at/"
            "features_available_until provenance"
        )
        write_blocker_report(
            out_dir,
            [{"dataset_key": "ALL", "ledger_row_index": "", "entity_id": "", "component": "", "forecast_origin": "", "reason": reason}],
        )
        raise RuntimeError(f"{reason}; see {out_dir / 'blocker_report.csv'}")
    key_cols = ["dataset_key", "dataset", "entity_id", "forecast_origin", "target_time", "component", "horizon", "split", "mode"]
    merged = expected[key_cols].merge(forecast[key_cols], on=key_cols, how="left", indicator=True)
    if (merged["_merge"] != "both").any():
        missing_count = int((merged["_merge"] != "both").sum())
        write_blocker_report(out_dir, [{"dataset_key": "ALL", "ledger_row_index": "", "entity_id": "", "component": "", "forecast_origin": "", "reason": f"external forecast csv missing {missing_count} ledger rows"}])
        raise RuntimeError(f"external forecast csv incomplete; see {out_dir / 'blocker_report.csv'}")

    forecast = forecast.merge(expected, on=key_cols, how="left", suffixes=("", "_ledger"))
    if "y_true" not in forecast.columns:
        forecast["y_true"] = forecast["y_true_ledger"]
    forecast["method"] = forecast.get("method", model_name)
    for col in ("pred_lower_50", "pred_upper_50", "pred_lower_90", "pred_upper_90"):
        if col not in forecast.columns:
            forecast[col] = pd.to_numeric(forecast["pred_mean"], errors="coerce")
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
        "dependency_report": dependency_report,
        "asof_provenance_rule": (
            "external forecast CSV must provide generated_at and "
            "features_available_until; both must be no later than forecast_origin, "
            "and features_available_until must be no later than generated_at"
        ),
        **dependency_report,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([{"status": "external_forecast_csv", "path": str(external_csv)}]).to_csv(out_dir / "training_log.csv", index=False)
    return out_dir


def run_foundation_from_manifest(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    model: str,
    checkpoint_id: str = "",
    checkpoint_path: str | Path | None = "",
    device: str = "cpu",
    external_forecast_csv: str | Path | None = None,
    backend: FoundationBackend | None = None,
    require_dependencies: bool = True,
    root: Path | None = None,
    caster_root: Path | None = None,
) -> Path:
    start = time.time()
    root = root or baseline_root()
    caster_root = caster_root or caster_root_from_baseline(root)
    model_name = canonical_foundation_model(model)
    default_checkpoint_id = "amazon/chronos-bolt-small" if model_name == "chronos_bolt_small" else "google/timesfm-2.0-500m-pytorch"
    checkpoint_id = checkpoint_id or default_checkpoint_id
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("failed_series.csv", "blocker_report.csv", "blocker_report.md"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    dependency_report = write_foundation_dependency_report(
        out_dir / "dependency_report.json",
        checkpoint_id=checkpoint_id,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    blockers: list[dict[str, object]] = []
    if require_dependencies:
        for reason in dependency_failures(model_name, dependency_report):
            blockers.append({"dataset_key": "ALL", "ledger_row_index": "", "entity_id": "", "component": "", "forecast_origin": "", "reason": reason})
        if blockers:
            write_blocker_report(out_dir, blockers)
            raise RuntimeError(f"foundation dependency check failed; see {out_dir / 'blocker_report.csv'}")

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    expected_rows = int(pd.to_numeric(manifest["ledger_rows"], errors="raise").sum())
    if external_forecast_csv:
        return run_external_forecast_csv(
            manifest=manifest,
            manifest_path=manifest_path,
            external_csv=Path(external_forecast_csv),
            out_dir=out_dir,
            model_name=model_name,
            dependency_report=dependency_report,
            root=root,
            caster_root=caster_root,
        )

    try:
        backend = backend or real_backend(
            model_name,
            checkpoint_id,
            checkpoint_path,
            device,
            max_horizon=max_ledger_horizon(manifest, root, caster_root),
        )
    except Exception as exc:
        write_blocker_report(out_dir, [{"dataset_key": "ALL", "ledger_row_index": "", "entity_id": "", "component": "", "forecast_origin": "", "reason": f"model load failed: {type(exc).__name__}: {exc}"}])
        raise RuntimeError(f"foundation model load failed; see {out_dir / 'blocker_report.csv'}") from exc

    rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    interval_sources: set[str] = set()

    for _, manifest_row in manifest.iterrows():
        dataset_key = str(manifest_row["dataset_key"])
        panel_path = resolve_manifest_path(manifest_row["panel_path"], caster_root, root)
        ledger_path = resolve_manifest_path(manifest_row["ledger_path"], caster_root, root)
        panel = pd.read_csv(panel_path)
        ledger = pd.read_csv(ledger_path, keep_default_na=False)
        declared_rows = int(manifest_row["ledger_rows"])
        if len(ledger) != declared_rows:
            blockers.append({"dataset_key": dataset_key, "ledger_row_index": "", "entity_id": "", "component": "", "forecast_origin": "", "reason": f"ledger row count mismatch: declared={declared_rows} actual={len(ledger)}"})
            continue

        ledger_entity_col = choose_ledger_entity_col(ledger, str(manifest_row["panel_entity_col"]))
        history_index = build_history_index(panel, manifest_row, ledger)
        context_cols = context_columns(ledger)
        group_cols = strategy_group_columns(ledger, [ledger_entity_col, "component", "forecast_origin"])
        forecast_rows_before = len(rows)
        for group_key, group in ledger.groupby(group_cols, dropna=False, sort=False):
            first_event = group.iloc[0]
            entity_id = str(first_event[ledger_entity_col])
            component = str(first_event["component"])
            origin_text = str(first_event["forecast_origin"])
            strategy = strategy_from_event(first_event)
            group_start = time.time()
            origin = pd.to_datetime(origin_text, errors="coerce")
            series = history_index.get((entity_id, component))
            hist = history_until_origin(series, origin) if series is not None else HistorySlice(np.asarray([], dtype="datetime64[ns]"), np.asarray([], dtype=float))
            latest_feature_release = (
                latest_feature_release_until_origin(series, origin)
                if series is not None
                else pd.NaT
            )
            max_horizon = int(pd.to_numeric(group["horizon"], errors="raise").max())
            horizons = sorted({int(pd.to_numeric(v, errors="raise")) for v in group["horizon"]})
            status = "model_ok"
            failure_reason = ""
            if pd.isna(origin):
                blockers.append({"dataset_key": dataset_key, "ledger_row_index": int(group.index[0]), "entity_id": entity_id, "component": component, "forecast_origin": origin_text, "reason": "forecast_origin parse failed"})
                continue
            if series is None or len(hist.values) == 0:
                blockers.append({"dataset_key": dataset_key, "ledger_row_index": int(group.index[0]), "entity_id": entity_id, "component": component, "forecast_origin": origin_text, "reason": "no finite history with panel_time <= forecast_origin"})
                continue
            if pd.isna(latest_feature_release) or latest_feature_release > origin:
                blockers.append({"dataset_key": dataset_key, "ledger_row_index": int(group.index[0]), "entity_id": entity_id, "component": component, "forecast_origin": origin_text, "reason": "released history lacks valid as-of provenance"})
                continue
            try:
                if strategy == RECURSIVE_ROLLOUT:
                    def one_step(_times: np.ndarray, step_values: np.ndarray, _step: int):
                        step_pred = backend.predict(
                            step_values,
                            1,
                            [1],
                            int(manifest_row["cadence_days"]),
                        )
                        return float(step_pred.mean[1]), step_pred

                    recursive_means, step_predictions = recursive_mean_path(
                        times=hist.times,
                        values=hist.values,
                        max_horizon=max_horizon,
                        cadence_days=int(manifest_row["cadence_days"]),
                        one_step=one_step,
                    )
                    interval_sources.update(p.interval_source for p in step_predictions.values())
                    pred = None
                else:
                    pred = backend.predict(hist.values, max_horizon, horizons, int(manifest_row["cadence_days"]))
                    interval_sources.add(pred.interval_source)
                    recursive_means = {}
                    step_predictions = {}
            except Exception as exc:
                status = "blocked"
                failure_reason = f"{type(exc).__name__}: {exc}"
                blockers.append({"dataset_key": dataset_key, "ledger_row_index": int(group.index[0]), "entity_id": entity_id, "component": component, "forecast_origin": origin_text, "reason": f"foundation inference failed: {failure_reason}"})
                training_rows.append({
                    "dataset_key": dataset_key,
                    "dataset": str(manifest_row["dataset"]),
                    "entity_id": entity_id,
                    "component": component,
                    "forecast_origin": origin_text,
                    "train_rows": int(len(hist.values)),
                    "max_horizon": max_horizon,
                    "status": status,
                    "runtime_seconds": round(time.time() - group_start, 6),
                    "failure_reason": failure_reason,
                    "fallback_used": False,
                })
                continue

            training_rows.append({
                "dataset_key": dataset_key,
                "dataset": str(manifest_row["dataset"]),
                "entity_id": entity_id,
                "component": component,
                "forecast_origin": origin_text,
                "train_rows": int(len(hist.values)),
                "max_horizon": max_horizon,
                "status": status,
                "runtime_seconds": round(time.time() - group_start, 6),
                "failure_reason": "",
                "fallback_used": False,
            })
            for ledger_idx, event in group.iterrows():
                horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
                target = pd.to_datetime(event["target_time"], errors="coerce")
                y_true = pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0]
                if parse_bool(event.get("observed_mask", True)) and not np.isfinite(y_true):
                    blockers.append({"dataset_key": dataset_key, "ledger_row_index": int(ledger_idx), "entity_id": entity_id, "component": component, "forecast_origin": origin_text, "reason": "observed_mask true but observed_value missing/non-finite"})
                    continue
                event_pred = step_predictions[horizon] if strategy == RECURSIVE_ROLLOUT else pred
                pred_mean = recursive_means[horizon] if strategy == RECURSIVE_ROLLOUT else event_pred.mean[horizon]
                pred_key = 1 if strategy == RECURSIVE_ROLLOUT else horizon
                row = {
                    "dataset_key": dataset_key,
                    "dataset": str(event["dataset"]) if "dataset" in ledger.columns else str(manifest_row["dataset"]),
                    "method": model_name,
                    "entity_id": entity_id,
                    "forecast_origin": format_date(origin),
                    "target_time": format_date(target),
                    "component": component,
                    "horizon": horizon,
                    "y_true": float(y_true) if np.isfinite(y_true) else np.nan,
                    "pred_mean": float(pred_mean),
                    "pred_lower_50": float(event_pred.lower_50[pred_key]),
                    "pred_upper_50": float(event_pred.upper_50[pred_key]),
                    "pred_lower_90": float(event_pred.lower_90[pred_key]),
                    "pred_upper_90": float(event_pred.upper_90[pred_key]),
                    "generated_at": format_date(origin),
                    "features_available_until": format_date(latest_feature_release),
                    "split": str(event["split"]) if "split" in ledger.columns else "NA",
                    "model_status": "model_ok",
                    "failure_reason": "",
                    "fallback_method": "",
                    "interval_source": event_pred.interval_source,
                }
                for col in context_cols:
                    row[col] = event[col]
                rows.append(row)

        dataset_summaries.append({
            "dataset_key": dataset_key,
            "dataset": str(manifest_row["dataset"]),
            "ledger_rows": declared_rows,
            "forecast_rows": int(len(rows) - forecast_rows_before),
            "group_count": int(ledger.groupby(group_cols, dropna=False).ngroups),
            "cadence_days": int(manifest_row["cadence_days"]),
            "season_length": infer_season_length(int(manifest_row["cadence_days"])),
            "panel_path": str(manifest_row["panel_path"]),
            "ledger_path": str(manifest_row["ledger_path"]),
        })

    pd.DataFrame(training_rows).to_csv(out_dir / "training_log.csv", index=False)
    if blockers:
        write_blocker_report(out_dir, blockers)
        raise RuntimeError(f"foundation run blocked; see {out_dir / 'blocker_report.csv'}")

    forecast = pd.DataFrame(rows)
    if len(forecast) != expected_rows:
        write_blocker_report(out_dir, [{"dataset_key": "ALL", "ledger_row_index": "", "entity_id": "", "component": "", "forecast_origin": "", "reason": f"forecast row count mismatch: expected={expected_rows} actual={len(forecast)}"}])
        raise RuntimeError(f"foundation row-count mismatch; see {out_dir / 'blocker_report.csv'}")
    forecast.to_csv(out_dir / "forecast.csv", index=False)
    metrics = summarize_forecasts(forecast)
    finite_failures = finite_metric_check(metrics)
    if finite_failures:
        write_blocker_report(out_dir, [{"dataset_key": "ALL", "ledger_row_index": "", "entity_id": "", "component": "", "forecast_origin": "", "reason": f"metrics contain non-finite values: {finite_failures}"}])
        raise RuntimeError(f"foundation run produced non-finite metrics; see {out_dir / 'blocker_report.csv'}")
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    timing = {
        "total_seconds": round(time.time() - start, 6),
        "forecast_rows": int(len(forecast)),
        "metric_rows": int(len(metrics)),
        "expected_rows": expected_rows,
        "group_rows": int(len(training_rows)),
        "fallback_group_count": 0,
        "nonfallback_scoring_rows": int((forecast["split"].astype(str) != "train").sum()),
        "nonfallback_test_rows": int((forecast["split"].astype(str) == "test").sum()),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8")
    interval_source = ";".join(sorted(interval_sources)) if interval_sources else getattr(backend, "interval_source", "")
    interval_construction_rule = str(getattr(backend, "interval_construction_rule", ""))
    run_manifest = {
        "baseline_names": [model_name],
        "model": model_name,
        "backend": "foundation",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_rows": expected_rows,
        "forecast_rows": int(len(forecast)),
        "dataset_summaries": dataset_summaries,
        "dependency_report": dependency_report,
        "dependency_report_path": "dependency_report.json",
        "fallback_group_count": 0,
        "interval_source": interval_source,
        **({"interval_construction_rule": interval_construction_rule} if interval_construction_rule else {}),
        "foundation_hparams": {
            "context_len": getattr(backend, "context_len", ""),
            "horizon_len": getattr(backend, "horizon_len", ""),
            "input_patch_len": 32 if model_name == "timesfm_2_0" else "",
            "output_patch_len": 128 if model_name == "timesfm_2_0" else "",
            "num_layers": 50 if model_name == "timesfm_2_0" else "",
            "model_dims": 1280 if model_name == "timesfm_2_0" else "",
            "use_positional_embedding": False if model_name == "timesfm_2_0" else "",
        },
        "asof_input_rule": (
            "history requires panel_time <= forecast_origin and target release_time "
            "<= forecast_origin; each row records the latest admitted release in "
            "features_available_until"
        ),
        "forecast_strategy_rule": (
            "direct makes one native multi-horizon call; recursive_rollout repeatedly calls h=1 and "
            "feeds the prior predicted mean into the next context; interval proxies remain marginal per step"
        ),
        **forecast_strategy_manifest_fields(forecast),
        **dependency_report,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir
