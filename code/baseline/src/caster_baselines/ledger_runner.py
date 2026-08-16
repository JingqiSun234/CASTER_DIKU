from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_validation import baseline_root, caster_root_from_baseline, resolve_manifest_path, sha256_file
from .metrics import summarize_forecasts
from .models import BaseBaseline
from .forecast_strategy import RECURSIVE_ROLLOUT, append_predicted_mean, strategy_from_event
from .native_sidecar import (
    NativeAvailabilityRecord,
    default_native_sidecar_root,
    status_availability,
    write_availability_report,
    write_storage_validation,
    STATUS_DETERMINISTIC_NO_NATIVE,
)

Z50 = 0.67448975
Z90 = 1.64485363
NA_VALUES = {"", "NA", "nan", "NaN", "None", "none"}
MODEL_ALIASES = {
    "lastvalue": "last_value",
    "last_value": "last_value",
    "seasonalnaive": "seasonal_naive",
    "seasonal_naive": "seasonal_naive",
}
ENTITY_CANDIDATES = ("entity_id", "jurisdiction", "location", "node_id", "node_index")


@dataclass(frozen=True)
class SeriesHistory:
    times: np.ndarray
    releases: np.ndarray
    values: np.ndarray


def canonical_model_name(model: str) -> str:
    key = model.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"unknown naive model {model!r}; available={sorted(MODEL_ALIASES)}")
    return MODEL_ALIASES[key]


def split_semicolon(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value)
    if text in NA_VALUES:
        return []
    return [part for part in text.split(";") if part and part not in NA_VALUES]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def format_date(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def infer_season_length(cadence_days: int) -> int:
    ""
    if cadence_days <= 1:
        return 7
    if 6 <= cadence_days <= 8:
        return 52
    return max(1, int(round(365.25 / max(cadence_days, 1))))


def infer_naive_season_length(cadence_days: int) -> int:
    ""





    if cadence_days <= 1:
        return 7
    if 6 <= cadence_days <= 8:
        return 8
    return max(1, int(round(56.0 / max(cadence_days, 1))))


def seasonal_period_label(cadence_days: int) -> str:
    return "7d" if cadence_days <= 1 else (
        "8w" if 6 <= cadence_days <= 8 else f"{infer_naive_season_length(cadence_days)}x"
    )


def choose_ledger_entity_col(ledger: pd.DataFrame, panel_entity_col: str) -> str:
    if "entity_id" in ledger.columns:
        return "entity_id"
    if panel_entity_col in ledger.columns:
        return panel_entity_col
    for col in ENTITY_CANDIDATES:
        if col in ledger.columns:
            return col
    raise ValueError("ledger has no usable entity column")


def _series_from_group(
    group: pd.DataFrame,
    time_col: str,
    value_col: str,
    release_col: str | None = None,
) -> SeriesHistory:
    times = pd.to_datetime(group[time_col], errors="coerce")
    releases = (
        pd.to_datetime(group[release_col], errors="coerce")
        if release_col and release_col in group.columns
        else times
    )
    values = pd.to_numeric(group[value_col], errors="coerce")
    valid = times.notna() & releases.notna()
    times_np = times[valid].to_numpy(dtype="datetime64[ns]")
    releases_np = releases[valid].to_numpy(dtype="datetime64[ns]")
    values_np = values[valid].to_numpy(dtype=float)
    order = np.argsort(times_np)
    return SeriesHistory(times=times_np[order], releases=releases_np[order], values=values_np[order])


def build_history_index(panel: pd.DataFrame, manifest_row: pd.Series, ledger: pd.DataFrame) -> dict[tuple[str, str], SeriesHistory]:
    panel_format = str(manifest_row["panel_format"])
    entity_col = str(manifest_row["panel_entity_col"])
    time_col = str(manifest_row["panel_time_col"])
    index: dict[tuple[str, str], SeriesHistory] = {}
    panel = panel.copy()
    panel[entity_col] = panel[entity_col].astype(str)
    release_col = next(
        (
            column
            for column in ("__release_time__target", "__release_time__", "release_time")
            if column in panel.columns
        ),
        None,
    )

    if panel_format == "long":
        component_col = str(manifest_row["panel_component_col"])
        value_col = str(manifest_row["panel_value_col"])
        required = {entity_col, time_col, component_col, value_col}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"long panel missing columns: {sorted(missing)}")
        for (entity, component), group in panel.groupby([entity_col, component_col], dropna=False):
            index[(str(entity), str(component))] = _series_from_group(
                group, time_col, value_col, release_col
            )
        return index

    targets = split_semicolon(manifest_row["panel_target_cols"])
    if not targets and "component" in ledger.columns:
        targets = sorted(c for c in ledger["component"].dropna().astype(str).unique() if c in panel.columns)
    for target in targets:
        if target not in panel.columns:
            continue
        for entity, group in panel.groupby(entity_col, dropna=False):
            index[(str(entity), str(target))] = _series_from_group(
                group, time_col, target, release_col
            )
    return index


def values_until_origin(series: SeriesHistory, origin: pd.Timestamp) -> np.ndarray:
    if pd.isna(origin):
        return np.asarray([], dtype=float)
    origin_np = np.datetime64(pd.Timestamp(origin), "ns")
    visible = (series.times <= origin_np) & (series.releases <= origin_np)
    values = series.values[visible]
    return values[np.isfinite(values)]


def seasonal_naive_for_target(
    series: SeriesHistory,
    origin: pd.Timestamp,
    target: pd.Timestamp,
    season_length: int,
    cadence_days: int,
) -> tuple[float, float, bool, str]:
    ""





    visible_values = values_until_origin(series, origin)
    if len(visible_values) == 0:
        raise ValueError("empty history")
    seasonal_time = pd.Timestamp(target) - pd.Timedelta(
        days=int(season_length) * int(cadence_days)
    )
    origin_np = np.datetime64(pd.Timestamp(origin), "ns")
    seasonal_np = np.datetime64(seasonal_time, "ns")
    matched = (series.times == seasonal_np) & (series.releases <= origin_np)
    matched_values = series.values[matched]
    matched_values = matched_values[np.isfinite(matched_values)]
    if len(matched_values):
        pred = float(matched_values[-1])
        return pred, residual_sigma(visible_values, pred), False, ""
    pred = float(visible_values[-1])
    reason = (
        "structural_history_unavailable: target_minus_season_not_released "
        f"target={pd.Timestamp(target).date()} seasonal_time={seasonal_time.date()}"
    )
    return pred, residual_sigma(visible_values, pred), True, reason


def residual_sigma(values: np.ndarray, pred: float) -> float:
    default = max(float(pred) * 0.1, 1.0)
    return BaseBaseline._residual_sigma(values, default=default)


def predict_from_history(
    model: str,
    values: np.ndarray,
    horizon: int,
    season_length: int,
) -> tuple[float, float, bool]:
    ""




    if len(values) == 0:
        raise ValueError("empty history")
    if model == "last_value":
        pred = float(values[-1])
        return pred, residual_sigma(values, pred), False
    if model == "seasonal_naive":
        raise ValueError(
            "seasonal_naive requires target-aware calendar lookup via seasonal_naive_for_target"
        )
    raise ValueError(f"unsupported model: {model}")


def write_blocker_report(out_dir: Path, blockers: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "blocker_report.csv"
    md_path = out_dir / "blocker_report.md"
    pd.DataFrame(blockers).to_csv(csv_path, index=False)
    lines = [
        "# Baseline Blocker Report",
        "",
        f"- blocker_count: `{len(blockers)}`",
        f"- csv: `{csv_path.name}`",
        "",
        "| Dataset key | Ledger row | Entity | Component | Forecast origin | Reason |",
        "|---|---:|---|---|---|---|",
    ]
    for row in blockers[:50]:
        lines.append(
            "| {dataset_key} | {ledger_row_index} | {entity_id} | {component} | "
            "{forecast_origin} | {reason} |".format(**{k: str(v) for k, v in row.items()})
        )
    if len(blockers) > 50:
        lines.append(f"| ... | ... | ... | ... | ... | {len(blockers) - 50} additional blockers in CSV |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def context_columns(ledger: pd.DataFrame) -> list[str]:
    wanted = [
        "protocol_version",
        "natural_event_id",
        "mode",
        "mode_kind",
        "forecast_strategy",
        "country",
        "country_code",
        "jurisdiction",
        "node_index",
        "forecast_id",
        "revision_version",
    ]
    return [col for col in wanted if col in ledger.columns]


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
    failures = []
    for col in numeric_cols:
        if col in metrics.columns and not np.isfinite(pd.to_numeric(metrics[col], errors="coerce")).all():
            failures.append(col)
    return failures


def forecast_strategy_manifest_fields(forecast: pd.DataFrame) -> dict[str, object]:
    strategies = (
        {str(k): int(v) for k, v in forecast["forecast_strategy"].astype(str).value_counts().sort_index().items()}
        if "forecast_strategy" in forecast.columns else {}
    )
    modes = (
        {str(k): int(v) for k, v in forecast["mode"].astype(str).value_counts().sort_index().items()}
        if "mode" in forecast.columns else {}
    )
    return {
        "forecast_strategy_counts": strategies,
        "forecast_mode_counts": modes,
        "rollout_uncertainty_contract": "marginal interval/pred_var proxy per horizon; no trajectory Monte Carlo",
    }


def run_baseline_from_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    model: str,
    root: Path | None = None,
    caster_root: Path | None = None,
    enable_native_sidecars: bool = False,
    native_sidecar_root: str | Path | None = None,
    fail_on_fallback: bool = False,
) -> Path:
    start = time.time()
    root = root or baseline_root()
    caster_root = caster_root or caster_root_from_baseline(root)
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("failed_series.csv", "sleeping_prefix_rows.csv", "blocker_report.csv", "blocker_report.md"):
        stale = out_dir / stale_name
        if stale.exists():
            stale.unlink()
    model_name = canonical_model_name(model)
    native_root = default_native_sidecar_root(out_dir, native_sidecar_root) if enable_native_sidecars else None
    native_availability_rows: list[NativeAvailabilityRecord] = []

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    expected_rows = int(pd.to_numeric(manifest["ledger_rows"], errors="raise").sum())
    rows: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    fallback_counts: dict[str, int] = {}
    season_lengths: dict[str, int] = {}
    seasonal_period_labels: dict[str, str] = {}

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
        cadence_days = int(manifest_row["cadence_days"])
        season_length = infer_naive_season_length(cadence_days)
        season_lengths[dataset_key] = season_length
        seasonal_period_labels[dataset_key] = seasonal_period_label(cadence_days)
        fallback_counts[dataset_key] = 0
        context_cols = context_columns(ledger)
        dataset_forecast_rows = 0

        for ledger_idx, event in ledger.iterrows():
            entity_id = str(event[ledger_entity_col])
            component = str(event["component"])
            origin = pd.to_datetime(event["forecast_origin"], errors="coerce")
            target = pd.to_datetime(event["target_time"], errors="coerce")
            horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
            key = (entity_id, component)
            series = history_index.get(key)
            values = values_until_origin(series, origin) if series is not None else np.asarray([], dtype=float)
            reason = ""
            if pd.isna(origin):
                reason = "forecast_origin parse failed"
            elif series is None:
                reason = "no panel series for entity/component"
            elif len(values) == 0:
                reason = "no finite history with panel_time <= forecast_origin"
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

            y_true = pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0]
            observed_mask = parse_bool(event.get("observed_mask", True))
            if observed_mask and not np.isfinite(y_true):
                blockers.append({
                    "dataset_key": dataset_key,
                    "ledger_row_index": int(ledger_idx),
                    "entity_id": entity_id,
                    "component": component,
                    "forecast_origin": str(event.get("forecast_origin", "")),
                    "reason": "observed_mask true but observed_value missing/non-finite",
                })
                continue

            strategy = strategy_from_event(event)
            failure_reason = ""
            if model_name == "seasonal_naive":
                pred, sigma, fallback, failure_reason = seasonal_naive_for_target(
                    series,
                    origin,
                    target,
                    season_length,
                    int(manifest_row["cadence_days"]),
                )
            else:
                if strategy == RECURSIVE_ROLLOUT:
                    recursive_values = np.asarray(values, dtype=float)
                    recursive_times = np.asarray([], dtype="datetime64[ns]")
                    fallback = False
                    pred = float(recursive_values[-1])
                    sigma = residual_sigma(recursive_values, pred)
                    for _step in range(1, horizon + 1):
                        pred, sigma, step_fallback = predict_from_history(
                            model_name,
                            recursive_values,
                            1,
                            season_length,
                        )
                        fallback = fallback or step_fallback
                        recursive_times, recursive_values = append_predicted_mean(
                            recursive_times,
                            recursive_values,
                            pred,
                            cadence_days=int(manifest_row["cadence_days"]),
                        )
                else:
                    pred, sigma, fallback = predict_from_history(model_name, values, horizon, season_length)
            if fallback:
                fallback_counts[dataset_key] += 1
            sigma = max(float(sigma), 1e-6)
            out_row = {
                "dataset_key": dataset_key,
                "dataset": str(event["dataset"]) if "dataset" in ledger.columns else str(manifest_row["dataset"]),
                "method": model_name,
                "method_variant": (
                    f"seasonal_naive_{seasonal_period_labels[dataset_key]}"
                    if model_name == "seasonal_naive"
                    else model_name
                ),
                "season_length": season_length if model_name == "seasonal_naive" else "",
                "seasonal_period_label": (
                    seasonal_period_labels[dataset_key]
                    if model_name == "seasonal_naive"
                    else ""
                ),
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
                "split": str(event["split"]) if "split" in ledger.columns else "NA",
                "forecast_status": "structural_unavailable" if fallback else "model_ok",
                "forecast_fallback_used": bool(fallback),
                "forecast_failure_reason": failure_reason,
                "forecast_fallback_method": "last_value" if fallback else "",
                "proxy_fallback_used": False,
                "unsafe_native_proxy_executed": False,
            }
            for col in context_cols:
                out_row[col] = event[col]
            rows.append(out_row)
            if native_root is not None:
                native_availability_rows.append(
                    status_availability(
                        model_id=model_name,
                        origin=f"{dataset_key}__{entity_id}__{component}__{format_date(origin)}__h{horizon}",
                        status=STATUS_DETERMINISTIC_NO_NATIVE,
                        native_likelihood_type="none",
                        blocker=(
                            f"{model_name} simple baseline emits deterministic point forecasts; "
                            "residual-sigma intervals are baseline metric proxies, not native likelihood params"
                        ),
                    )
                )
            dataset_forecast_rows += 1

        dataset_summaries.append({
            "dataset_key": dataset_key,
            "dataset": str(manifest_row["dataset"]),
            "ledger_rows": declared_rows,
            "forecast_rows": dataset_forecast_rows,
            "cadence_days": int(manifest_row["cadence_days"]),
            "season_length": season_length,
            "panel_path": str(manifest_row["panel_path"]),
            "ledger_path": str(manifest_row["ledger_path"]),
        })

    if blockers:
        write_blocker_report(out_dir, blockers)
        raise RuntimeError(f"baseline run blocked; see {out_dir / 'blocker_report.csv'}")

    forecast = pd.DataFrame(rows)
    fallback_total = int(sum(fallback_counts.values()))
    fallback_mask = forecast["forecast_fallback_used"].astype(bool)
    sleeping_prefix_mask = (
        fallback_mask
        & forecast["split"].astype(str).str.lower().eq("train")
        & forecast["forecast_status"].astype(str).eq("structural_unavailable")
        & forecast["forecast_failure_reason"].astype(str).str.startswith(
            "structural_history_unavailable:"
        )
        & ~forecast["proxy_fallback_used"].astype(bool)
        & ~forecast["unsafe_native_proxy_executed"].astype(bool)
    )
    disallowed_fallback_mask = fallback_mask & ~sleeping_prefix_mask
    sleeping_prefix_rows = forecast.loc[sleeping_prefix_mask].copy()
    disallowed_fallback_rows = forecast.loc[disallowed_fallback_mask].copy()
    if not sleeping_prefix_rows.empty:
        sleeping_prefix_rows.to_csv(out_dir / "sleeping_prefix_rows.csv", index=False)
    if fail_on_fallback and not disallowed_fallback_rows.empty:
        disallowed_fallback_rows.to_csv(out_dir / "failed_series.csv", index=False)
        raise RuntimeError(
            f"{model_name} produced {len(disallowed_fallback_rows)} non-sleeping fallback rows; "
            "formal validation/embargo/test runs require native forecasts; see "
            f"{out_dir / 'failed_series.csv'}"
        )

    if len(forecast) != expected_rows:
        blockers.append({
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"forecast row count mismatch: expected={expected_rows} actual={len(forecast)}",
        })
        write_blocker_report(out_dir, blockers)
        raise RuntimeError(f"baseline run row-count mismatch; see {out_dir / 'blocker_report.csv'}")

    forecast.to_csv(out_dir / "forecast.csv", index=False)
    metrics = summarize_forecasts(forecast)
    finite_failures = finite_metric_check(metrics)
    if finite_failures:
        blockers.append({
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"metrics contain non-finite values: {finite_failures}",
        })
        write_blocker_report(out_dir, blockers)
        raise RuntimeError(f"baseline run produced non-finite metrics; see {out_dir / 'blocker_report.csv'}")
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    if native_root is not None:
        write_availability_report(native_availability_rows, native_root / "native_likelihood_availability.csv")
        write_storage_validation([], native_root / "native_sidecar_storage_validation.csv")

    timing = {
        "total_seconds": round(time.time() - start, 6),
        "forecast_rows": int(len(forecast)),
        "metric_rows": int(len(metrics)),
        "expected_rows": expected_rows,
        "fallback_counts": fallback_counts,
        "structural_sleeping_prefix_rows": int(sleeping_prefix_mask.sum()),
        "disallowed_fallback_rows": int(disallowed_fallback_mask.sum()),
        "formal_fail_on_fallback": bool(fail_on_fallback),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    run_manifest = {
        "baseline_names": [model_name],
        "model": model_name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_rows": expected_rows,
        "forecast_rows": int(len(forecast)),
        "dataset_summaries": dataset_summaries,
        "season_lengths": season_lengths,
        "seasonal_period_labels": seasonal_period_labels,
        "fallback_counts": fallback_counts,
        "structural_sleeping_prefix_rows": int(sleeping_prefix_mask.sum()),
        "disallowed_fallback_rows": int(disallowed_fallback_mask.sum()),
        "no_leakage_rule": (
            "history uses only panel_time <= forecast_origin and "
            "release_time <= forecast_origin"
        ),
        "forecast_strategy_rule": (
            "direct requests the native requested horizon; recursive_rollout repeatedly requests h=1 "
            "and appends only predicted means; seasonal_naive always uses the actual target "
            "timestamp minus one cadence-specific season"
        ),
        "seasonal_unavailability_policy": (
            "only provenance-marked train structural prefixes may use an alignment placeholder; "
            "validation/embargo/test and all non-structural failures fail closed; train placeholders "
            "are excluded by the sleeping-model posterior/readout mask"
        ),
        "native_sidecars_enabled": bool(enable_native_sidecars),
        "native_sidecar_root": str(native_root) if native_root is not None else "",
        **forecast_strategy_manifest_fields(forecast),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return out_dir
