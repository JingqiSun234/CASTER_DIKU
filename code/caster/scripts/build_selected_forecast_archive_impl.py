from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import inspect
import json
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caster.data import (              
    add_task_columns,
    filter_frame_for_task,
    load_benchmark_b_context_contract,
    materialize_benchmark_b_adapter_panel,
    materialize_mobility_features,
    task_from_args,
    task_metadata,
)
from caster.forecast import (              
    FORECAST_ARCHIVE_COLUMNS,
    attach_ledger_context,
    build_normal_forecast_draws,
    validate_forecast_archive,
    validate_forecast_draws,
    write_forecast_archive,
    write_forecast_draws,
)
from caster.models import (              
    apply_hyperparam_overrides,
    instantiate_adapters_from_registry,
    read_registry,
)
from caster.models.candidate_adapters import (              
    _baseline_residual_var,
    _build_baseline_neuralforecast_model,
    _build_series_index,
    _freq_from_times,
    _forecast_strategy,
    _ledger_entity,
    _nf_choose_input_size,
    _nf_prediction_col,
    _nonneg,
    _normalise_panel,
    _series_from_index,
    detect_device,
)
from caster.utils import RuntimeLogger, write_timing_log              
from model_pool_contract import SHARED_FORECAST_PATHS              


BASELINE_REUSE_PATHS = dict(SHARED_FORECAST_PATHS)
FOUNDATION_REUSE_MODELS = {"chronos_external", "timesfm_external"}

CASTER_LOCAL_CPU_MODELS = {
    "sir_tau",
    "seir_tau",
    "seirs_tau",
    "tv_seir_rt",
    "renewal_rt",
    "local_level",
    "covariate_dynamic_linear_trend",
    "particle_local_level",
    "drift",
    "covariate_drift",
    "rnn_simple",
    "gru_style",
}

CASTER_LOCAL_GPU_MODELS = {"lstm_style"}

Z90 = 1.6448536269514722
BENCHMARK_B_CONTEXT_IMPLEMENTATION = (
    Path(__file__).resolve().parents[1] / "src/caster/data/benchmark_b_context.py"
)

                                                                          
                                                                           
alternate_BUILDER_SHA256_BEFORE_LSTM_RELEASE_FIX = (
    "a824dd8efc2c5bfc6d022cc3101c190e20837db6524a48c6ae6ee8c7a7521868"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_model_id(model_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(model_id))


def _checkpoint_identity_matches(
    stored: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
                                                                             
                                                                            
                                                          
    return stored == expected


def _entity_col(frame: pd.DataFrame) -> str:
    for col in ("entity_id", "jurisdiction", "region", "unit", "unique_id"):
        if col in frame.columns:
            return col
    raise ValueError("ledger must contain an entity identifier column")


def _checkpoint_path(checkpoint_dir: Path, model_id: str) -> Path:
    return checkpoint_dir / f"forecast_archive.{_safe_model_id(model_id)}.csv"


def _manifest_path(checkpoint_dir: Path, model_id: str) -> Path:
    return checkpoint_dir / f"forecast_archive.{_safe_model_id(model_id)}.manifest.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_checkpoint_identity(
    *,
    model_id: str,
    registry_row: dict[str, Any],
    adapter: Any,
    panel_path: Path,
    ledger_path: Path,
    seed: int,
    task_payload: dict[str, Any],
    reuse_baseline_forecasts: bool,
    baseline_manifest: str,
    baseline_runs_root: Path | None,
    baseline_extension_runs_root: Path | None,
    import_model_forecasts: bool = False,
    model_forecast_source_manifest: str = "",
    model_forecast_runs_root: Path | None = None,
    context_dependency_paths: list[Path] | None = None,
) -> dict[str, Any]:
    source_path = inspect.getsourcefile(type(adapter))
    source = Path(source_path).resolve() if source_path else None
    payload = {
        "schema": "caster_phase20_model_checkpoint_identity_v4",
        "model_id": str(model_id),
        "registry_row_sha256": _canonical_sha256(registry_row),
        "adapter_class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "adapter_source": str(source) if source else "",
        "adapter_source_sha256": _sha256_file(source) if source and source.is_file() else "",
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "panel_sha256": _sha256_file(panel_path),
        "ledger_sha256": _sha256_file(ledger_path),
        "context_dependencies": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for path in (context_dependency_paths or [])
            if path.is_file()
        ],
        "seed": int(seed),
        "task": task_payload,
        "reuse_baseline_forecasts": bool(reuse_baseline_forecasts),
        "baseline_manifest": str(baseline_manifest or ""),
        "baseline_manifest_sha256": (
            _sha256_file(Path(baseline_manifest))
            if baseline_manifest and Path(baseline_manifest).is_file()
            else ""
        ),
        "import_model_forecasts": bool(import_model_forecasts),
        "model_forecast_source_manifest": str(model_forecast_source_manifest or ""),
        "model_forecast_source_manifest_sha256": (
            _sha256_file(Path(model_forecast_source_manifest))
            if model_forecast_source_manifest
            and Path(model_forecast_source_manifest).is_file()
            else ""
        ),
    }
    if reuse_baseline_forecasts and model_id in BASELINE_REUSE_PATHS:
        rel = BASELINE_REUSE_PATHS[model_id]
        sources = [
            root / rel
            for root in (baseline_runs_root, baseline_extension_runs_root)
            if root is not None
        ]
        payload["baseline_forecast_sources"] = [
            {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else "",
            }
            for path in sources
        ]
    if import_model_forecasts and model_id in BASELINE_REUSE_PATHS:
        rel = BASELINE_REUSE_PATHS[model_id]
        path = model_forecast_runs_root / rel if model_forecast_runs_root else Path("")
        payload["fresh_candidate_forecast_source"] = {
            "path": str(path),
            "sha256": _sha256_file(path) if path.is_file() else "",
        }
    payload["identity_sha256"] = _canonical_sha256(payload)
    return payload


def _pred_var_from_baseline(source: pd.DataFrame, pred_mean: pd.Series) -> pd.Series:
    if "pred_var" in source.columns:
        var = pd.to_numeric(source["pred_var"], errors="coerce")
    else:
        lower_col = "pred_lower_90" if "pred_lower_90" in source.columns else ""
        upper_col = "pred_upper_90" if "pred_upper_90" in source.columns else ""
        if lower_col and upper_col:
            lower = pd.to_numeric(source[lower_col], errors="coerce")
            upper = pd.to_numeric(source[upper_col], errors="coerce")
            sigma = (upper - lower).abs() / (2.0 * Z90)
            var = sigma * sigma
        else:
            var = pd.Series(np.nan, index=source.index)
    fallback_scale = np.maximum(1.0, 0.05 * (1.0 + pred_mean.abs()))
    fallback = fallback_scale * fallback_scale
    var = pd.to_numeric(var, errors="coerce")
    return var.where(np.isfinite(var) & (var >= 0.0), fallback).astype(float)


def _boolish_source_column(source: pd.DataFrame, column: str) -> pd.Series:
    if column not in source.columns:
        return pd.Series(False, index=source.index, dtype=bool)
    values = source[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "t", "yes", "y"}
    )


def _with_forecast_provenance(archive: pd.DataFrame) -> pd.DataFrame:
    out = archive.copy()
    if "forecast_status" not in out.columns:
        out["forecast_status"] = "model_ok"
    if "forecast_fallback_used" not in out.columns:
        out["forecast_fallback_used"] = False
    else:
        out["forecast_fallback_used"] = _boolish_source_column(
            out, "forecast_fallback_used"
        )
    for column in (
        "forecast_failure_reason",
        "forecast_fallback_method",
    ):
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)
    for column in ("proxy_fallback_used", "unsafe_native_proxy_executed"):
        out[column] = _boolish_source_column(out, column)
    return out


NATIVE_HORIZON_PROVENANCE_COLUMNS = (
    "last_released_target_time",
    "native_horizon_steps",
    "forecasted_native_target_time",
)


def _build_native_horizon_provenance(
    ledger: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    panel_norm = _normalise_panel(panel)
    series_index = _build_series_index(panel_norm)
    e_col = _entity_col(ledger)
    work = ledger[
        [e_col, "component", "forecast_origin", "target_time", "horizon", "forecast_id"]
    ].copy()
    work["forecast_id"] = work["forecast_id"].astype(str)
    if work["forecast_id"].duplicated().any():
        duplicate_ids = work.loc[
            work["forecast_id"].duplicated(), "forecast_id"
        ].head(10).tolist()
        raise ValueError(f"duplicate ledger forecast_id values: {duplicate_ids}")
    work["__entity__"] = work[e_col].astype(str)
    work["component"] = work["component"].astype(str)
    work["__origin__"] = pd.to_datetime(
        work["forecast_origin"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    work["__target__"] = pd.to_datetime(
        work["target_time"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    work["__horizon__"] = pd.to_numeric(work["horizon"], errors="coerce")
    if (
        work[["__origin__", "__target__", "__horizon__"]].isna().any().any()
        or work["__horizon__"].lt(1).any()
        or work["__horizon__"].ne(np.floor(work["__horizon__"])).any()
    ):
        raise ValueError("ledger has invalid origin, target, or nominal horizon")
    work["__horizon__"] = work["__horizon__"].astype(np.int64)

    group_keys = work[["__entity__", "component", "__origin__"]].drop_duplicates()
    last_released_by_key: dict[tuple[str, str, pd.Timestamp], pd.Timestamp] = {}
    for entity, component, origin in group_keys.itertuples(index=False, name=None):
        times, releases, _ = series_index.get(
            (str(entity), str(component)),
            (
                np.asarray([], dtype="datetime64[ns]"),
                np.asarray([], dtype="datetime64[ns]"),
                np.asarray([], dtype=float),
            ),
        )
        cutoff = np.datetime64(pd.Timestamp(origin).to_datetime64())
        visible = (times <= cutoff) & (releases <= cutoff)
        if not visible.any():
            raise ValueError(
                "cannot establish native horizon without a finite released target: "
                f"entity={entity} component={component} origin={origin}"
            )
        last_released_by_key[(str(entity), str(component), pd.Timestamp(origin))] = (
            pd.Timestamp(pd.to_datetime(times[visible].max()))
        )

    work["__last_released__"] = [
        last_released_by_key[(entity, component, pd.Timestamp(origin))]
        for entity, component, origin in work[
            ["__entity__", "component", "__origin__"]
        ].itertuples(index=False, name=None)
    ]
    nominal_delta_ns = (
        work["__target__"] - work["__origin__"]
    ).astype("timedelta64[ns]").astype(np.int64)
    cadence_ns = nominal_delta_ns // work["__horizon__"].to_numpy(dtype=np.int64)
    invalid_cadence = (
        (nominal_delta_ns <= 0)
        | (cadence_ns <= 0)
        | (nominal_delta_ns % work["__horizon__"].to_numpy(dtype=np.int64) != 0)
    )
    if invalid_cadence.any():
        bad_ids = work.loc[invalid_cadence, "forecast_id"].head(10).tolist()
        raise ValueError(
            "ledger target is not on its declared nominal cadence; "
            f"forecast_id={bad_ids}"
        )
    native_delta_ns = (
        work["__target__"] - work["__last_released__"]
    ).astype("timedelta64[ns]").astype(np.int64)
    native_steps = native_delta_ns // cadence_ns
    invalid_native = (
        (native_delta_ns <= 0)
        | (native_steps < 1)
        | (native_delta_ns % cadence_ns != 0)
    )
    if invalid_native.any():
        bad_ids = work.loc[invalid_native, "forecast_id"].head(10).tolist()
        raise ValueError(
            "ledger target is not on the release-gated native cadence; "
            f"forecast_id={bad_ids}"
        )
    forecasted_target = pd.to_datetime(
        work["__last_released__"].astype("int64") + native_steps * cadence_ns
    )
    if not forecasted_target.eq(work["__target__"]).all():
        bad_ids = work.loc[
            ~forecasted_target.eq(work["__target__"]), "forecast_id"
        ].head(10).tolist()
        raise AssertionError(
            "forecasted native target does not equal ledger target; "
            f"forecast_id={bad_ids}"
        )

    return {
        forecast_id: {
            "last_released_target_time": pd.Timestamp(last_released),
            "native_horizon_steps": int(native_step),
            "forecasted_native_target_time": pd.Timestamp(target),
        }
        for forecast_id, last_released, native_step, target in zip(
            work["forecast_id"],
            work["__last_released__"],
            native_steps,
            forecasted_target,
        )
    }


def _attach_native_horizon_provenance(
    archive: pd.DataFrame,
    ledger: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    provenance_by_id: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Attach and verify the native target span for every model archive row.

    This is intentionally applied after every forecast source (local adapters,
    shared baselines, foundation models, and the grouped LSTM path) has been
    converted to the common archive.  Consequently the final concat cannot
    silently turn these fields into model-specific missing values.
    """

    if provenance_by_id is None:
        provenance_by_id = _build_native_horizon_provenance(ledger, panel)

    out = archive.copy()
    forecast_ids = out["forecast_id"].astype(str)
    unknown = sorted(set(forecast_ids) - set(provenance_by_id))
    if unknown:
        raise ValueError(
            "archive contains forecast ids outside the native-horizon ledger: "
            f"{unknown[:10]}"
        )
    for column in NATIVE_HORIZON_PROVENANCE_COLUMNS:
        expected = forecast_ids.map(
            lambda forecast_id: provenance_by_id[forecast_id][column]
        )
        if column in out.columns:
            actual = out[column]
            populated = actual.notna() & actual.astype(str).str.strip().ne("")
            if column == "native_horizon_steps":
                actual_cmp = pd.to_numeric(actual, errors="coerce")
                expected_cmp = pd.to_numeric(expected, errors="raise")
            else:
                actual_cmp = pd.to_datetime(actual, errors="coerce", utc=True)
                expected_cmp = pd.to_datetime(expected, errors="raise", utc=True)
            mismatch = populated & actual_cmp.ne(expected_cmp)
            if mismatch.any():
                bad_ids = forecast_ids[mismatch].head(10).tolist()
                raise ValueError(
                    f"archive {column} conflicts with release-gated target history; "
                    f"forecast_id={bad_ids}"
                )
        out[column] = expected.to_numpy()
    return out


def baseline_reuse_models() -> list[str]:
    return sorted(BASELINE_REUSE_PATHS)


def caster_local_models() -> list[str]:
    return sorted(CASTER_LOCAL_CPU_MODELS | CASTER_LOCAL_GPU_MODELS)


def _baseline_forecast_to_archive(
    *,
    model_id: str,
    family: str,
    forecast_paths: list[Path] | None = None,
    forecast_path: Path | None = None,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
                                                                         
                                                                            
    paths = list(forecast_paths or [])
    if forecast_path is not None:
        paths.append(forecast_path)
    if not paths:
        raise ValueError(f"no baseline forecast source supplied for {model_id}")
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"baseline forecast for {model_id} not found: {missing_paths}")
    source = pd.concat(
        [pd.read_csv(path, keep_default_na=False, low_memory=False) for path in paths],
        ignore_index=True,
    )
    required = {"forecast_id", "pred_mean"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"baseline forecasts for {model_id} missing required columns: {missing}")
    source = source.copy()
    source["forecast_id"] = source["forecast_id"].astype(str)
    if source["forecast_id"].duplicated().any():
        dup = source.loc[source["forecast_id"].duplicated(), "forecast_id"].head(10).tolist()
        raise ValueError(f"baseline forecast sources for {model_id} have duplicate forecast_id values: {dup}")

    ledger_norm = ledger.copy()
    ledger_norm["forecast_id"] = ledger_norm["forecast_id"].astype(str)
    ledger_ids = set(ledger_norm["forecast_id"])
    source = source[source["forecast_id"].isin(ledger_ids)].copy()
    if len(source) != len(ledger_ids):
        missing_ids = sorted(ledger_ids - set(source["forecast_id"]))
        raise ValueError(
            f"baseline forecast for {model_id} missing {len(missing_ids)} ledger rows; "
            f"examples={missing_ids[:10]}"
        )

    pred_mean = pd.to_numeric(source["pred_mean"], errors="coerce")
    if not np.isfinite(pred_mean).all():
        bad = source.loc[~np.isfinite(pred_mean), "forecast_id"].head(10).tolist()
        raise ValueError(f"baseline forecast for {model_id} has nonfinite pred_mean: {bad}")
    source["_pred_mean"] = pred_mean.astype(float).clip(lower=0.0)
    source["_pred_var"] = _pred_var_from_baseline(source, source["_pred_mean"])
    fallback_used = (
        _boolish_source_column(source, "forecast_fallback_used")
        | _boolish_source_column(source, "fallback_used")
    )
    for status_col in ("forecast_status", "model_status", "status"):
        if status_col in source.columns:
            fallback_used |= source[status_col].astype(str).str.lower().str.contains(
                "fallback", na=False
            )
    for method_col in ("forecast_fallback_method", "fallback_method"):
        if method_col in source.columns:
            fallback_used |= source[method_col].astype(str).str.strip().ne("")
    status_col = next(
        (col for col in ("forecast_status", "model_status", "status") if col in source.columns),
        None,
    )
    reason_col = next(
        (
            col
            for col in ("forecast_failure_reason", "failure_reason")
            if col in source.columns
        ),
        None,
    )
    method_col = next(
        (
            col
            for col in ("forecast_fallback_method", "fallback_method")
            if col in source.columns
        ),
        None,
    )
    source["_forecast_fallback_used"] = fallback_used.astype(bool)
    source["_forecast_status"] = (
        source[status_col].astype(str)
        if status_col is not None
        else pd.Series("model_ok", index=source.index)
    )
    source["_forecast_failure_reason"] = (
        source[reason_col].astype(str)
        if reason_col is not None
        else pd.Series("", index=source.index)
    )
    source["_forecast_fallback_method"] = (
        source[method_col].astype(str)
        if method_col is not None
        else pd.Series("", index=source.index)
    )
    source["_proxy_fallback_used"] = _boolish_source_column(
        source, "proxy_fallback_used"
    )
    source["_unsafe_native_proxy_executed"] = _boolish_source_column(
        source, "unsafe_native_proxy_executed"
    )
    for column in NATIVE_HORIZON_PROVENANCE_COLUMNS:
        source[f"_source_{column}"] = (
            source[column]
            if column in source.columns
            else pd.Series(pd.NA, index=source.index)
        )
    if model_id in FOUNDATION_REUSE_MODELS:
        provenance_columns = {"generated_at", "features_available_until"}
        missing_provenance = sorted(provenance_columns - set(source.columns))
        if missing_provenance:
            raise ValueError(
                f"foundation baseline forecasts for {model_id} lack required "
                f"as-of provenance columns: {missing_provenance}"
            )
        source["_generated_at"] = source["generated_at"].astype(str)
        source["_features_available_until"] = source[
            "features_available_until"
        ].astype(str)
    else:
        source["_generated_at"] = ""
        source["_features_available_until"] = ""

    e_col = _entity_col(ledger_norm)
    merged = ledger_norm.merge(
        source[
            [
                "forecast_id",
                "_pred_mean",
                "_pred_var",
                "_forecast_status",
                "_forecast_fallback_used",
                "_forecast_failure_reason",
                "_forecast_fallback_method",
                "_proxy_fallback_used",
                "_unsafe_native_proxy_executed",
                "_generated_at",
                "_features_available_until",
                *[
                    f"_source_{column}"
                    for column in NATIVE_HORIZON_PROVENANCE_COLUMNS
                ],
            ]
        ],
        on="forecast_id",
        how="left",
    )
    if merged["_pred_mean"].isna().any():
        ids = merged.loc[merged["_pred_mean"].isna(), "forecast_id"].astype(str).head(10).tolist()
        raise ValueError(f"baseline forecast merge for {model_id} unexpectedly missing ids: {ids}")

    if model_id in FOUNDATION_REUSE_MODELS:
        origin_time = pd.to_datetime(merged["forecast_origin"], errors="raise")
        generated_time = pd.to_datetime(merged["_generated_at"], errors="coerce")
        feature_time = pd.to_datetime(
            merged["_features_available_until"], errors="coerce"
        )
        invalid = generated_time.isna() | feature_time.isna()
        non_asof = (
            (generated_time > origin_time)
            | (feature_time > origin_time)
            | (feature_time > generated_time)
        )
        if invalid.any() or non_asof.any():
            bad_ids = merged.loc[invalid | non_asof, "forecast_id"].head(10).tolist()
            raise ValueError(
                f"foundation baseline forecasts for {model_id} have invalid "
                f"as-of provenance; examples={bad_ids}"
            )
        generated_at = merged["_generated_at"]
        features_available_until = merged["_features_available_until"]
    else:
        generated_at = merged["forecast_origin"]
        features_available_until = merged["forecast_origin"]

    out = pd.DataFrame({
        "dataset": merged["dataset"] if "dataset" in merged.columns else "dataset",
        "model_id": model_id,
        "family": family,
        "particle_id": 0,
        "entity_id": merged[e_col].astype(str),
        "forecast_origin": merged["forecast_origin"],
        "target_time": merged["target_time"],
        "component": merged["component"].astype(str),
        "horizon": pd.to_numeric(merged["horizon"], errors="raise").astype(int),
        "forecast_id": merged["forecast_id"].astype(str),
        "pred_mean": merged["_pred_mean"].astype(float),
        "pred_var": merged["_pred_var"].astype(float),
        "generated_at": generated_at,
        "features_available_until": features_available_until,
        "forecast_status": merged["_forecast_status"],
        "forecast_fallback_used": merged["_forecast_fallback_used"].astype(bool),
        "forecast_failure_reason": merged["_forecast_failure_reason"],
        "forecast_fallback_method": merged["_forecast_fallback_method"],
        "proxy_fallback_used": merged["_proxy_fallback_used"].astype(bool),
        "unsafe_native_proxy_executed": merged["_unsafe_native_proxy_executed"].astype(bool),
        **{
            column: merged[f"_source_{column}"]
            for column in NATIVE_HORIZON_PROVENANCE_COLUMNS
        },
    })
    return _with_forecast_provenance(attach_ledger_context(out, ledger))


def _validate_one_model_archive(archive: pd.DataFrame, ledger: pd.DataFrame, model_id: str) -> pd.DataFrame:
    violations = validate_forecast_archive(archive, ledger)
    rows: list[dict[str, object]] = []
    if not violations.empty:
        rows.extend(violations.to_dict(orient="records"))
    ledger_ids = set(ledger["forecast_id"].astype(str))
    archive_ids = set(archive["forecast_id"].astype(str))
    if archive["model_id"].astype(str).nunique() != 1 or set(archive["model_id"].astype(str)) != {model_id}:
        rows.append({"row": None, "violation": "model_id_mismatch", "details": model_id})
    missing = sorted(ledger_ids - archive_ids)
    extra = sorted(archive_ids - ledger_ids)
    if missing:
        rows.append({"row": None, "violation": "missing_ledger_predictions", "details": ",".join(missing[:10])})
    if extra:
        rows.append({"row": None, "violation": "forecast_id_not_in_ledger", "details": ",".join(extra[:10])})
    if len(archive) != len(ledger_ids):
        rows.append({"row": None, "violation": "row_count_mismatch", "details": f"expected={len(ledger_ids)} actual={len(archive)}"})
    missing_native_columns = sorted(
        set(NATIVE_HORIZON_PROVENANCE_COLUMNS) - set(archive.columns)
    )
    if missing_native_columns:
        rows.append(
            {
                "row": None,
                "violation": "missing_native_horizon_provenance",
                "details": ",".join(missing_native_columns),
            }
        )
    else:
        origin = pd.to_datetime(archive["forecast_origin"], errors="coerce", utc=True)
        target = pd.to_datetime(archive["target_time"], errors="coerce", utc=True)
        last_released = pd.to_datetime(
            archive["last_released_target_time"], errors="coerce", utc=True
        )
        forecasted_target = pd.to_datetime(
            archive["forecasted_native_target_time"], errors="coerce", utc=True
        )
        native_numeric = pd.to_numeric(
            archive["native_horizon_steps"], errors="coerce"
        )
        native_valid = (
            native_numeric.notna()
            & np.isfinite(native_numeric)
            & native_numeric.ge(1)
            & native_numeric.eq(np.floor(native_numeric))
        )
        checks = {
            "invalid_last_released_target_time": last_released.isna()
            | origin.isna()
            | last_released.gt(origin),
            "invalid_native_horizon_steps": ~native_valid,
            "forecasted_native_target_mismatch": forecasted_target.isna()
            | target.isna()
            | forecasted_target.ne(target),
        }
        for violation, mask in checks.items():
            if mask.any():
                bad_ids = archive.loc[mask, "forecast_id"].astype(str).head(10).tolist()
                rows.append(
                    {
                        "row": None,
                        "violation": violation,
                        "details": ",".join(bad_ids),
                    }
                )
    return pd.DataFrame(rows, columns=["row", "violation", "details"])


def _validate_selected_coverage(
    archive: pd.DataFrame,
    ledger: pd.DataFrame,
    selected_model_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in selected_model_ids:
        model_rows = archive[archive["model_id"].astype(str) == str(model_id)]
        violations = _validate_one_model_archive(model_rows, ledger, str(model_id))
        if not violations.empty:
            for row in violations.to_dict(orient="records"):
                rows.append({"model_id": model_id, "violation": row["violation"], "details": row["details"]})
    return pd.DataFrame(rows, columns=["model_id", "violation", "details"])


def _read_checkpoint(
    path: Path,
    ledger: pd.DataFrame,
    model_id: str,
    expected_identity: dict[str, Any],
) -> pd.DataFrame | None:
    manifest_path = _manifest_path(path.parent, model_id)
    if not path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_identity = manifest.get("checkpoint_identity")
        if not isinstance(stored_identity, dict) or not _checkpoint_identity_matches(
            stored_identity, expected_identity
        ):
            return None
        if manifest.get("archive_sha256") != _sha256_file(path):
            return None
        archive = pd.read_csv(path)
    except Exception:
        return None
    violations = _validate_one_model_archive(archive, ledger, model_id)
    if not violations.empty:
        return None
    return _with_forecast_provenance(attach_ledger_context(archive, ledger))


def _write_checkpoint(
    *,
    checkpoint_dir: Path,
    model_id: str,
    archive: pd.DataFrame,
    source: str,
    runtime_seconds: float,
    checkpoint_identity: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, model_id)
    archive.to_csv(path, index=False)
    manifest = {
        "model_id": model_id,
        "source": source,
        "archive_path": str(path),
        "archive_sha256": _sha256_file(path),
        "rows": int(len(archive)),
        "runtime_seconds": float(runtime_seconds),
        "checkpoint_identity": checkpoint_identity,
    }
    if extra:
        manifest.update(extra)
    _write_json(_manifest_path(checkpoint_dir, model_id), manifest)
    print(f"adapter_checkpoint={model_id} path={path}", flush=True)
    return path


def _build_lstm_grouped_archive(
    *,
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    family: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ""







    device = detect_device()
    if device != "gpu":
        raise RuntimeError("lstm_style requires CUDA/GPU for full Phase 20 grouped generation")
    try:
        from neuralforecast import NeuralForecast
    except Exception as exc:
        raise RuntimeError(
            f"lstm_style NeuralForecast dependency unavailable: {type(exc).__name__}: {exc}"
        ) from exc

    panel_norm = _normalise_panel(panel)
    series_index = _build_series_index(panel_norm)
    e_col = _entity_col(ledger)
    rows: list[dict[str, Any]] = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    fallback_count = 0
    model_ok_count = 0
    native_fit_count = 0
    insufficient_history_count = 0
    fallback_by_split: dict[str, int] = {}

    def cadence_days_from_times(times: np.ndarray | None) -> int:
        if times is None or len(times) < 2:
            return 1
        diffs = pd.Series(pd.to_datetime(times)).sort_values().diff().dt.days.dropna()
        if diffs.empty:
            return 1
        return max(1, int(round(float(diffs.median()))))

    def steps_to_target(last_time: object, target_time: object, cadence_days: int) -> int:
        last = pd.Timestamp(pd.to_datetime(last_time, errors="raise"))
        target = pd.Timestamp(pd.to_datetime(target_time, errors="raise"))
        raw = float((target - last).total_seconds() / (86400.0 * float(cadence_days)))
        rounded = int(round(raw))
        if rounded < 1 or abs(raw - rounded) > 1e-6:
            raise RuntimeError(
                f"target is not on the forecast cadence: last={last} target={target} "
                f"cadence_days={cadence_days}"
            )
        return rounded

    for component, comp_ledger in ledger.groupby("component", dropna=False, sort=False):
        component = str(component)
        comp_items = [(key, tv) for key, tv in series_index.items() if key[1] == component]
        if not comp_items:
            raise RuntimeError(f"lstm_style has no panel series for component={component}")

        frames = []
        cadence_times: np.ndarray | None = None
        for (entity, _component), (times, releases, values) in comp_items:
            if cadence_times is None and len(times) >= 2:
                cadence_times = times
            frames.append(
                pd.DataFrame(
                    {
                        "unique_id": str(entity),
                        "ds": pd.to_datetime(times),
                        "__release_time__": pd.to_datetime(releases),
                        "y": pd.to_numeric(values, errors="coerce"),
                    }
                )
            )
        full_df = pd.concat(frames, ignore_index=True).dropna(
            subset=["ds", "__release_time__", "y"]
        )
        full_df = full_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        if full_df.empty:
            raise RuntimeError(f"lstm_style has empty training frame for component={component}")

        freq = _freq_from_times(cadence_times)
        cadence_days = cadence_days_from_times(cadence_times)
        predictions: dict[tuple[str, str, str, int], float] = {}
        failure_reasons: dict[tuple[str, str, str, int], str] = {}
        strategy_group_cols = [
            col for col in ("mode", "forecast_strategy") if col in comp_ledger.columns
        ]
        strategy_groups = (
            comp_ledger.groupby(strategy_group_cols, dropna=False, sort=False)
            if strategy_group_cols
            else [((), comp_ledger)]
        )
        for _, strategy_ledger in strategy_groups:
            first_event = strategy_ledger.iloc[0]
            mode = str(first_event.get("mode", ""))
            strategy = _forecast_strategy(first_event)
            for origin_text, origin_group in strategy_ledger.groupby(
                "forecast_origin", dropna=False, sort=True
            ):
                origin = pd.to_datetime(origin_text, errors="coerce")
                if pd.isna(origin):
                    raise RuntimeError(f"lstm_style invalid forecast_origin={origin_text}")
                pred_input = full_df[
                    (full_df["ds"] <= origin)
                    & (full_df["__release_time__"] <= origin)
                ][["unique_id", "ds", "y"]].copy()
                if pred_input.empty:
                    for _, event in origin_group.iterrows():
                        key = (
                            mode,
                            str(event[e_col]),
                            str(origin_text),
                            int(pd.to_numeric(event["horizon"], errors="raise")),
                        )
                        failure_reasons[key] = "no released finite history at forecast origin"
                    continue

                requested_steps: dict[tuple[str, int], int] = {}
                for _, event in origin_group.iterrows():
                    entity = str(event[e_col])
                    horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
                    entity_rows = pred_input[pred_input["unique_id"].astype(str).eq(entity)]
                    if entity_rows.empty:
                        failure_reasons[(mode, entity, str(origin_text), horizon)] = (
                            "no released finite history for entity at forecast origin"
                        )
                        continue
                    requested_steps[(entity, horizon)] = steps_to_target(
                        entity_rows["ds"].max(), event["target_time"], cadence_days
                    )

                model_h = (
                    1
                    if strategy == "recursive_rollout"
                    else max(requested_steps.values(), default=1)
                )
                requested_entities = set(origin_group[e_col].astype(str))
                requested_input = pred_input[
                    pred_input["unique_id"].astype(str).isin(requested_entities)
                ].copy()
                lengths = requested_input.groupby("unique_id")["y"].size()
                potential_lengths = lengths[lengths > int(model_h)]
                min_len = (
                    int(potential_lengths.min())
                    if not potential_lengths.empty
                    else int(lengths.max()) if not lengths.empty else 0
                )
                input_size = _nf_choose_input_size(min_len, model_h)
                eligible = set(
                    lengths[
                        lengths > max(int(model_h), int(input_size))
                    ].index.astype(str)
                )
                for _, event in origin_group.iterrows():
                    entity = str(event[e_col])
                    horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
                    if entity not in eligible:
                        insufficient_history_count += 1
                        failure_reasons[(mode, entity, str(origin_text), horizon)] = (
                            f"insufficient released history for native LSTM: "
                            f"n={int(lengths.get(entity, 0))} input_size={input_size} h={model_h}"
                        )
                train_df = requested_input[
                    requested_input["unique_id"].astype(str).isin(eligible)
                ].copy()
                if train_df.empty:
                    continue

                model = _build_baseline_neuralforecast_model(
                    "lstm",
                    h=model_h,
                    input_size=input_size,
                    max_steps=int(os.environ.get("NEURAL_MAX_STEPS", "3")),
                    seed=int(seed),
                    device=device,
                )
                nf = NeuralForecast(models=[model], freq=freq)
                start = time.time()
                nf.fit(df=train_df, val_size=0, verbose=False)
                fit_seconds += time.time() - start
                native_fit_count += 1

                if strategy == "recursive_rollout":
                    requested_dates = {
                        (
                            str(event[e_col]),
                            str(
                                pd.Timestamp(
                                    pd.to_datetime(event["target_time"], errors="raise")
                                ).date()
                            ),
                        ): int(pd.to_numeric(event["horizon"], errors="raise"))
                        for _, event in origin_group.iterrows()
                        if str(event[e_col]) in eligible
                    }
                    recursive_input = train_df.copy()
                    max_steps = max(requested_steps.values(), default=0)
                    for _step in range(1, max_steps + 1):
                        start = time.time()
                        forecast = nf.predict(df=recursive_input, verbose=False).copy()
                        predict_seconds += time.time() - start
                        col = (
                            "lstm"
                            if "lstm" in forecast.columns
                            else _nf_prediction_col(forecast, "lstm")
                        )
                        forecast["unique_id"] = forecast["unique_id"].astype(str)
                        forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
                        append_rows: list[dict[str, Any]] = []
                        for _, pred_row in forecast.iterrows():
                            value = pd.to_numeric(
                                pd.Series([pred_row[col]]), errors="coerce"
                            ).iloc[0]
                            if pd.isna(pred_row["ds"]) or not np.isfinite(value):
                                continue
                            entity = str(pred_row["unique_id"])
                            target_text = str(pd.Timestamp(pred_row["ds"]).date())
                            horizon = requested_dates.get((entity, target_text))
                            if horizon is not None:
                                predictions[
                                    (mode, entity, str(origin_text), int(horizon))
                                ] = float(value)
                            append_rows.append(
                                {
                                    "unique_id": entity,
                                    "ds": pred_row["ds"],
                                    "y": float(value),
                                }
                            )
                        if not append_rows:
                            break
                        recursive_input = pd.concat(
                            [recursive_input, pd.DataFrame(append_rows)],
                            ignore_index=True,
                        )
                        recursive_input = recursive_input.sort_values(
                            ["unique_id", "ds"]
                        ).reset_index(drop=True)
                else:
                    start = time.time()
                    forecast = nf.predict(df=train_df, verbose=False).copy()
                    predict_seconds += time.time() - start
                    col = (
                        "lstm"
                        if "lstm" in forecast.columns
                        else _nf_prediction_col(forecast, "lstm")
                    )
                    forecast["unique_id"] = forecast["unique_id"].astype(str)
                    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
                    by_key = {
                        (
                            str(row["unique_id"]),
                            str(pd.Timestamp(row["ds"]).date()),
                        ): float(row[col])
                        for _, row in forecast.iterrows()
                        if pd.notna(row["ds"])
                        and np.isfinite(pd.to_numeric(row[col], errors="coerce"))
                    }
                    for _, event in origin_group.iterrows():
                        entity = str(event[e_col])
                        target = str(
                            pd.Timestamp(
                                pd.to_datetime(event["target_time"], errors="raise")
                            ).date()
                        )
                        horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
                        if (entity, target) in by_key:
                            predictions[
                                (mode, entity, str(origin_text), horizon)
                            ] = by_key[(entity, target)]

        for _, event in comp_ledger.iterrows():
            entity = str(event[e_col])
            origin_text = str(event["forecast_origin"])
            origin = pd.to_datetime(origin_text, errors="coerce")
            horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
            key = (str(event.get("mode", "")), entity, origin_text, horizon)
            pred = predictions.get(key)
            times, values = _series_from_index(series_index, entity, component, origin)
            if pred is None:
                reason = failure_reasons.get(
                    key, "native LSTM did not emit the requested target date"
                )
                raise RuntimeError(
                    "lstm_style native forecast unavailable; fallback is disabled: "
                    f"forecast_id={event['forecast_id']} split={event.get('split', 'unknown')} "
                    f"entity={entity} origin={origin_text} horizon={horizon} reason={reason}"
                )
            fallback_used = False
            model_ok_count += 1
            reason = ""
            var = _baseline_residual_var(
                values, default=max(float(pred) * 0.1, 1.0)
            )
            rows.append(
                {
                    "dataset": str(event.get("dataset", "dataset")),
                    "model_id": "lstm_style",
                    "family": family,
                    "particle_id": 0,
                    "entity_id": entity,
                    "forecast_origin": event["forecast_origin"],
                    "target_time": event["target_time"],
                    "component": component,
                    "horizon": horizon,
                    "forecast_id": event["forecast_id"],
                    "pred_mean": _nonneg(float(pred)),
                    "pred_var": float(max(var, 0.0)),
                    "generated_at": event["forecast_origin"],
                    "features_available_until": event["forecast_origin"],
                    "forecast_status": "fallback" if fallback_used else "model_ok",
                    "forecast_fallback_used": bool(fallback_used),
                    "forecast_failure_reason": reason,
                    "forecast_fallback_method": "last_value" if fallback_used else "",
                }
            )

    meta = {
        "device": device,
        "fit_seconds": round(fit_seconds, 6),
        "predict_seconds": round(predict_seconds, 6),
        "native_fit_count": int(native_fit_count),
        "fallback_count": int(fallback_count),
        "fallback_count_by_split": fallback_by_split,
        "insufficient_history_count": int(insufficient_history_count),
        "model_ok_count": int(model_ok_count),
        "forecast_span_rule": (
            "last_released_observation_to_target_date_on_declared_cadence"
        ),
        "fit_schedule": "separate_release_gated_global_fit_per_component_strategy_origin",
    }
    archive = attach_ledger_context(pd.DataFrame(rows), ledger)
    return archive, meta


def _run_adapter_archive(adapter: Any, panel: pd.DataFrame, ledger: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if str(adapter.model_id) == "lstm_style":
        return _build_lstm_grouped_archive(panel=panel, ledger=ledger, family=str(adapter.family), seed=seed)
    state = adapter.initialize(panel, seed=seed)
    return adapter.forecast_ledger(state, ledger), {}


def _build_model_archive(
    *,
    model_id: str,
    family: str,
    adapter: Any,
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    seed: int,
    reuse_baseline_forecasts: bool,
    baseline_runs_root: Path | None,
    baseline_extension_runs_root: Path | None,
    import_model_forecasts: bool = False,
    model_forecast_runs_root: Path | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if import_model_forecasts and model_id in BASELINE_REUSE_PATHS:
        if model_forecast_runs_root is None:
            raise ValueError(
                "--model-forecast-runs-root is required when --import-model-forecasts is set"
            )
        source_path = model_forecast_runs_root / BASELINE_REUSE_PATHS[model_id]
        print(f"adapter_import_fresh={model_id} source={source_path}", flush=True)
        return (
            _baseline_forecast_to_archive(
                model_id=model_id,
                family=family,
                forecast_path=source_path,
                ledger=ledger,
            ),
            f"fresh_candidate_model_source:{source_path}",
            {"fresh_candidate_source_forecast": str(source_path)},
        )
    if reuse_baseline_forecasts and model_id in BASELINE_REUSE_PATHS:
        if baseline_runs_root is None:
            raise ValueError("--baseline-runs-root is required when --reuse-baseline-forecasts is set")
        source_paths = [baseline_runs_root / BASELINE_REUSE_PATHS[model_id]]
        if baseline_extension_runs_root is not None:
            source_paths.append(baseline_extension_runs_root / BASELINE_REUSE_PATHS[model_id])
        print(f"adapter_reuse={model_id} sources={','.join(map(str, source_paths))}", flush=True)
        return (
            _baseline_forecast_to_archive(
                model_id=model_id,
                family=family,
                forecast_paths=source_paths,
                ledger=ledger,
            ),
            "baseline:" + "|".join(map(str, source_paths)),
            {"baseline_source_forecasts": [str(path) for path in source_paths]},
        )
    archive, extra = _run_adapter_archive(adapter, panel, ledger, seed)
    return archive, "caster_local", extra


def main(argv: list[str] | None = None) -> int:
    ap = ArgumentParser(description="Build immutable forecast archive for selected CASTER candidate particles.")
    ap.add_argument("--panel", required=True, help="Panel CSV used to initialize each candidate adapter.")
    ap.add_argument("--ledger", required=True, help="Event ledger CSV.")
    ap.add_argument("--registry", required=True, help="Model registry YAML/CSV/JSON.")
    ap.add_argument("--selection", required=True, help="Top-K candidate selection CSV with model_id column.")
    ap.add_argument("--out", required=True, help="Output forecast archive CSV path.")
    ap.add_argument("--draws-out", default="", help="Optional output forecast draws CSV path.")
    ap.add_argument("--n-draws", type=int, default=0, help="If >0, create normal draws from archive means/variances.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="Bounded parallel workers for candidate archive generation.")
    ap.add_argument("--baseline-manifest", default="", help="Baseline run manifest used for traceability.")
    ap.add_argument("--baseline-runs-root", default="", help="Baseline runs root containing per-method forecast.csv artifacts.")
    ap.add_argument(
        "--baseline-extension-runs-root",
        default="",
        help="Optional runs root containing disjoint extensional forecast rows, such as embargo-only origins.",
    )
    ap.add_argument("--reuse-baseline-forecasts", action="store_true", help="Reuse baseline forecast.csv artifacts for library-backed candidates.")
    ap.add_argument(
        "--import-model-forecasts",
        action="store_true",
        help="Import freshly recomputed, content-addressed candidate-model forecasts (never shared-baseline outputs).",
    )
    ap.add_argument(
        "--model-forecast-runs-root",
        default="",
        help="Fresh candidate-model run root with the standard per-model forecast layout.",
    )
    ap.add_argument(
        "--model-forecast-source-manifest",
        default="",
        help="Content-addressed manifest proving the fresh candidate-model source run.",
    )
    ap.add_argument("--checkpoint-dir", default="", help="Per-model checkpoint directory.")
    ap.add_argument("--resume", action="store_true", help="Reuse valid per-model checkpoints.")
    ap.add_argument(
        "--model-ids",
        default="",
        help="Optional comma-separated model allowlist overriding the selection rows.",
    )
    ap.add_argument(
        "--force-model-ids",
        default="",
        help="Comma-separated checkpoint identities to rebuild even under --resume.",
    )
    ap.add_argument(
        "--model-hyperparam-overrides-json",
        default="{}",
        help=(
            "Invocation-local model hyperparameter overrides as JSON, for example "
            "'{\"rnn_simple\":{\"gain\":1.005}}'. The source registry is never modified."
        ),
    )
    ap.add_argument("--task-id", default="", help="Optional Benchmark task id, e.g. benchmark_b_covid, benchmark_b_flu, or benchmark_b_pooled.")
    ap.add_argument(
        "--benchmark-b-context-contract",
        default=str(Path(__file__).resolve().parents[3] / "configs/benchmark_b_context_v26_1.yaml"),
    )
    ap.add_argument(
        "--benchmark-a-mobility-root",
        default=str(Path(__file__).resolve().parents[3] / "data/benchmark_a/raw_all"),
        help="EpiLLM-format daily mobility graph root used only by Benchmark A covariate-capable adapters.",
    )
    ap.add_argument("--target-components", default="", help="Comma-separated target components for this task.")
    ap.add_argument("--posterior-scope", default="", choices=["", "component_stratified", "pooled_sensitivity", "pooled_shared_posterior"])
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_path.parent / "phase20_checkpoints"
    baseline_runs_root = Path(args.baseline_runs_root) if args.baseline_runs_root else None
    baseline_extension_runs_root = (
        Path(args.baseline_extension_runs_root) if args.baseline_extension_runs_root else None
    )
    model_forecast_runs_root = (
        Path(args.model_forecast_runs_root) if args.model_forecast_runs_root else None
    )
    if args.import_model_forecasts and args.reuse_baseline_forecasts:
        raise SystemExit(
            "--import-model-forecasts and --reuse-baseline-forecasts are mutually exclusive"
        )

    timer = RuntimeLogger()
    with timer.measure("load_inputs"):
        panel = pd.read_csv(args.panel)
        ledger = pd.read_csv(args.ledger)
        task = task_from_args(
            task_id=args.task_id,
            target_components=args.target_components,
            posterior_scope=args.posterior_scope,
            dataset=ledger["dataset"].dropna().astype(str).iloc[0] if "dataset" in ledger.columns and not ledger.empty else "",
        )
        ledger = filter_frame_for_task(ledger, task, frame_name="ledger")
        input_metadata: dict[str, object] = {}
        dataset_name = (
            ledger["dataset"].dropna().astype(str).iloc[0]
            if "dataset" in ledger.columns and not ledger.empty
            else ""
        )
        if dataset_name == "benchmark_a":
            mobility = materialize_mobility_features(panel, args.benchmark_a_mobility_root)
            panel = mobility.panel
            if "__release_time__" not in panel.columns:
                panel["__release_time__"] = pd.to_datetime(panel["date"], errors="raise")
            benchmark_a_input_manifest = Path(args.panel).with_name("run_manifest.json")
            input_metadata = {
                **mobility.metadata,
                "benchmark_a_input_manifest": str(benchmark_a_input_manifest),
                "benchmark_a_input_manifest_sha256": (
                    _sha256_file(benchmark_a_input_manifest)
                    if benchmark_a_input_manifest.is_file()
                    else ""
                ),
                "benchmark_a_task_protocol_changed": False,
                "benchmark_a_mobility_consumers": [
                    "covariate_drift",
                    "covariate_dynamic_linear_trend",
                ],
                "algorithm_learning_policy": "no_learning_behavior_preserved",
                "input_materialization_changes_invalidate_checkpoint": True,
                "input_invalidation_policy": "checkpoint_identity_hashes_panel_ledger_mobility_graph_set_adapter_source_and_builder",
            }
        elif dataset_name == "benchmark_b":
            benchmark_b_contract = load_benchmark_b_context_contract(
                args.benchmark_b_context_contract, panel_columns=panel.columns
            )
            panel = materialize_benchmark_b_adapter_panel(panel, benchmark_b_contract)
            input_metadata = {
                "benchmark_b_context_contract_sha256": benchmark_b_contract.sha256,
                "benchmark_b_adapter_input_schema": "caster_benchmark_b_adapter_input_v1",
                "benchmark_b_adapter_receives_dynamic_covariates": True,
                "benchmark_b_adapter_receives_timestamps": True,
                "benchmark_b_adapter_receives_stream_release_times": True,
                "benchmark_b_adapter_receives_missing_masks": True,
                "benchmark_b_stream_release_columns": sorted(c for c in panel.columns if c.startswith("__release_time__")),
                "benchmark_b_missing_mask_column_count": int(sum(c.endswith("__missing_mask") for c in panel.columns)),
                "posterior_update_reads_current_x": False,
                "algorithm_learning_policy": "no_learning_behavior_preserved",
                "input_materialization_changes_invalidate_checkpoint": True,
                "input_invalidation_policy": "checkpoint_identity_hashes_panel_ledger_context_contract_adapter_source_and_builder",
            }
        registry = read_registry(args.registry)
        try:
            model_hyperparam_overrides = json.loads(
                str(args.model_hyperparam_overrides_json or "{}")
            )
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"--model-hyperparam-overrides-json is not valid JSON: {exc}"
            ) from exc
        if not isinstance(model_hyperparam_overrides, dict):
            raise SystemExit(
                "--model-hyperparam-overrides-json must be a JSON object keyed by model_id"
            )
        selection = pd.read_csv(args.selection)
        if "model_id" not in selection.columns:
            raise SystemExit("selection must contain model_id column")
        selected_model_ids = selection["model_id"].astype(str).tolist()
        requested_model_ids = [item.strip() for item in str(args.model_ids).split(",") if item.strip()]
        model_ids = requested_model_ids or selected_model_ids
        if len(model_ids) != len(set(model_ids)):
            raise SystemExit(f"duplicate model_id values requested: {model_ids}")
        registry_ids = set(registry["model_id"].astype(str))
        missing_registry = [m for m in model_ids if m not in registry_ids]
        if missing_registry:
            raise SystemExit(f"selection contains model_id not in registry: {missing_registry}")
        override_models = set(map(str, model_hyperparam_overrides))
        outside_build = sorted(override_models - set(model_ids))
        if outside_build:
            raise SystemExit(
                "model hyperparameter overrides target models outside this build: "
                + ",".join(outside_build)
            )
        reused_override_models = sorted(override_models & set(BASELINE_REUSE_PATHS))
        if reused_override_models:
            raise SystemExit(
                "model hyperparameter overrides cannot target canonical shared-baseline "
                "forecast objects: " + ",".join(reused_override_models)
            )
        try:
            registry = apply_hyperparam_overrides(
                registry,
                model_hyperparam_overrides,
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"invalid --model-hyperparam-overrides-json: {exc}"
            ) from exc
        shared_model_ids = [m for m in model_ids if m in BASELINE_REUSE_PATHS]
        if shared_model_ids and not args.reuse_baseline_forecasts:
            raise SystemExit(
                "overlapping baseline/candidate models have one canonical prediction "
                "object and cannot be fit or imported through the candidate lane; pass "
                "--reuse-baseline-forecasts with the shared baseline root: "
                + ",".join(shared_model_ids)
            )
        if args.baseline_extension_runs_root:
            raise SystemExit(
                "extensional baseline forecast files are forbidden by the single-object "
                "model-pool contract; rebuild the canonical shared baseline forecast instead"
            )

    registry_by_model = {str(r["model_id"]): r for r in registry.to_dict(orient="records")}
    adapters = instantiate_adapters_from_registry(registry, model_ids=model_ids)
    adapter_by_model = {str(a.model_id): a for a in adapters}
    if list(adapter_by_model) != model_ids:
        raise SystemExit(f"adapter order/coverage mismatch selected={model_ids} instantiated={list(adapter_by_model)}")
    if args.import_model_forecasts:
        source_manifest_path = Path(str(args.model_forecast_source_manifest or ""))
        if not source_manifest_path.is_file():
            raise SystemExit(
                "fresh candidate-model import requires an existing "
                f"--model-forecast-source-manifest: {source_manifest_path}"
            )
        try:
            source_manifest_payload = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid fresh candidate-model source manifest: {exc}") from exc
        if (
            source_manifest_payload.get("schema")
            != "candidate_model_forecast_sources_v2"
            or source_manifest_payload.get("scope") != "all"
        ):
            raise SystemExit(
                "fresh candidate-model source manifest must be schema "
                "candidate_model_forecast_sources_v2 with scope=all"
            )
        if model_forecast_runs_root is None or not model_forecast_runs_root.is_dir():
            raise SystemExit(
                "fresh candidate-model import requires an existing "
                f"--model-forecast-runs-root: {model_forecast_runs_root}"
            )
        required_source_paths = {
            model_id: model_forecast_runs_root / BASELINE_REUSE_PATHS[model_id]
            for model_id in model_ids
            if model_id in BASELINE_REUSE_PATHS
        }
        missing_source_paths = {
            model_id: path
            for model_id, path in required_source_paths.items()
            if not path.is_file()
        }
        if missing_source_paths:
            details = ", ".join(
                f"{model_id}={path}"
                for model_id, path in sorted(missing_source_paths.items())
            )
            raise SystemExit(
                "fresh candidate-model source preflight failed before adapter execution: "
                + details
            )

    task_payload = (
        {
            "task_id": str(task.task_id),
            "target_components": list(task.components),
            "posterior_scope": str(task.posterior_scope),
        }
        if task is not None
        else {
            "task_id": str(args.task_id or "generic"),
            "target_components": [item.strip() for item in str(args.target_components).split(",") if item.strip()],
            "posterior_scope": str(args.posterior_scope or "generic"),
        }
    )
    task_payload.update(input_metadata)
    if model_hyperparam_overrides:
        task_payload["model_hyperparam_overrides"] = model_hyperparam_overrides
    context_dependency_paths = (
        [Path(args.benchmark_b_context_contract), BENCHMARK_B_CONTEXT_IMPLEMENTATION]
        if dataset_name == "benchmark_b"
        else (
            [
                Path(__file__).resolve().parents[1]
                / "src/caster/data/benchmark_a_mobility.py"
            ]
            if dataset_name == "benchmark_a"
            else []
        )
    )
    checkpoint_identity_by_model = {
        model_id: _model_checkpoint_identity(
            model_id=model_id,
            registry_row=registry_by_model[model_id],
            adapter=adapter_by_model[model_id],
            panel_path=Path(args.panel),
            ledger_path=Path(args.ledger),
            seed=int(args.seed),
            task_payload=task_payload,
            reuse_baseline_forecasts=bool(args.reuse_baseline_forecasts),
            baseline_manifest=str(args.baseline_manifest or ""),
            baseline_runs_root=baseline_runs_root,
            baseline_extension_runs_root=baseline_extension_runs_root,
            import_model_forecasts=bool(args.import_model_forecasts),
            model_forecast_source_manifest=str(
                args.model_forecast_source_manifest or ""
            ),
            model_forecast_runs_root=model_forecast_runs_root,
            context_dependency_paths=context_dependency_paths,
        )
        for model_id in model_ids
    }
    force_model_ids = {item.strip() for item in str(args.force_model_ids).split(",") if item.strip()}
    unknown_force = sorted(force_model_ids - set(model_ids))
    if unknown_force:
        raise SystemExit(f"--force-model-ids contains models outside this build: {unknown_force}")

    native_horizon_provenance_by_id = _build_native_horizon_provenance(
        ledger, panel
    )

    print(
        "phase20_archive_start "
        f"models={len(model_ids)} ledger_rows={len(ledger)} "
        f"reuse_baseline={bool(args.reuse_baseline_forecasts)} "
        f"import_fresh_model_forecasts={bool(args.import_model_forecasts)} "
        f"workers={max(1, int(args.workers))}",
        flush=True,
    )

    results: dict[str, pd.DataFrame] = {}
    per_model_sources: dict[str, dict[str, Any]] = {}

    def run_one(model_id: str) -> tuple[str, pd.DataFrame, str, float, dict[str, Any]]:
        ckpt = _checkpoint_path(checkpoint_dir, model_id)
        if args.resume and model_id not in force_model_ids:
            existing = _read_checkpoint(ckpt, ledger, model_id, checkpoint_identity_by_model[model_id])
            if existing is not None:
                print(f"adapter_checkpoint_hit={model_id} path={ckpt}", flush=True)
                return model_id, existing, "checkpoint", 0.0, {"checkpoint_path": str(ckpt)}
        print(f"adapter_start={model_id}", flush=True)
        start = time.time()
        row = registry_by_model[model_id]
        family = str(row.get("family", adapter_by_model[model_id].family))
        archive, source, extra = _build_model_archive(
            model_id=model_id,
            family=family,
            adapter=adapter_by_model[model_id],
            panel=panel,
            ledger=ledger,
            seed=int(args.seed),
            reuse_baseline_forecasts=bool(args.reuse_baseline_forecasts),
            baseline_runs_root=baseline_runs_root,
            baseline_extension_runs_root=baseline_extension_runs_root,
            import_model_forecasts=bool(args.import_model_forecasts),
            model_forecast_runs_root=model_forecast_runs_root,
        )
        archive = _attach_native_horizon_provenance(
            _with_forecast_provenance(attach_ledger_context(archive, ledger)),
            ledger,
            panel,
            provenance_by_id=native_horizon_provenance_by_id,
        )
        runtime = round(time.time() - start, 6)
        violations = _validate_one_model_archive(archive, ledger, model_id)
        if not violations.empty:
            viol_path = checkpoint_dir / f"forecast_archive.{_safe_model_id(model_id)}.violations.csv"
            viol_path.parent.mkdir(parents=True, exist_ok=True)
            violations.to_csv(viol_path, index=False)
            raise RuntimeError(f"forecast archive validation failed for {model_id}; see {viol_path}")
        _write_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_id=model_id,
            archive=archive,
            source=source,
            runtime_seconds=runtime,
            checkpoint_identity=checkpoint_identity_by_model[model_id],
            extra=extra,
        )
        print(f"adapter_done={model_id} runtime_seconds={runtime}", flush=True)
        return model_id, archive, source, runtime, extra

    with timer.measure("model_forecasts"):
        workers = max(1, int(args.workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, mid): mid for mid in model_ids}
            for future in as_completed(futures):
                model_id, archive, source, runtime, extra = future.result()
                results[model_id] = archive
                per_model_sources[model_id] = {"source": source, "runtime_seconds": runtime, **extra}

    with timer.measure("concat_and_validate"):
        archives = [results[mid] for mid in model_ids]
        archive = pd.concat(archives, ignore_index=True) if archives else pd.DataFrame(columns=FORECAST_ARCHIVE_COLUMNS)
        violations = validate_forecast_archive(archive, ledger)
        if not violations.empty:
            viol_path = out_path.with_name(out_path.stem + "_violations.csv")
            violations.to_csv(viol_path, index=False)
            raise SystemExit(f"forecast archive validation failed; see {viol_path}")
        coverage_violations = _validate_selected_coverage(archive, ledger, model_ids)
        if not coverage_violations.empty:
            viol_path = out_path.with_name(out_path.stem + "_coverage_violations.csv")
            coverage_violations.to_csv(viol_path, index=False)
            raise SystemExit(f"forecast archive coverage validation failed; see {viol_path}")
        archive = add_task_columns(archive, task)
        archive_path = write_forecast_archive(archive, out_path)
    print(f"phase20_archive_done rows={len(archive)} path={archive_path}", flush=True)

    draws_out = ""
    if args.n_draws > 0:
        draws_out = args.draws_out or str(out_path.with_name("forecast_draws.csv"))
        expected_rows = int(len(archive) * int(args.n_draws))
        print(f"phase20_draws_start n_draws={args.n_draws} expected_rows={expected_rows}", flush=True)
        with timer.measure("forecast_draws"):
            draws = build_normal_forecast_draws(archive, n_draws=int(args.n_draws), seed=int(args.seed))
            draw_violations = validate_forecast_draws(draws, archive)
            if not draw_violations.empty:
                viol_path = Path(draws_out).with_name(Path(draws_out).stem + "_violations.csv")
                draw_violations.to_csv(viol_path, index=False)
                raise SystemExit(f"forecast draws validation failed; see {viol_path}")
            write_forecast_draws(draws, draws_out)
        print(f"phase20_draws_done rows={expected_rows} path={draws_out}", flush=True)

    timing_path = out_path.with_name("build_forecast_archive_timing.json")
    write_timing_log(timer.summary(seed=int(args.seed)), timing_path)
    manifest_path = out_path.with_name("forecast_archive_manifest.json")
    manifest = {
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_file(Path(archive_path)),
        "draws_path": str(draws_out) if draws_out else "",
        "draws_sha256": _sha256_file(Path(draws_out)) if draws_out else "",
        "draw_generation_distribution": (
            "nonnegative_normal_moment_projection" if draws_out else "not_materialized"
        ),
        "draw_seed_policy": (
            "sha256_global_seed_forecast_id_model_id_particle_id_v1"
            if draws_out
            else "not_materialized"
        ),
        "ledger_path": str(Path(args.ledger)),
        "ledger_sha256": _sha256_file(Path(args.ledger)),
        "panel_path": str(Path(args.panel)),
        "panel_sha256": _sha256_file(Path(args.panel)),
        "ledger_rows": int(len(ledger)),
        "ledger_split_counts": {
            str(key): int(value)
            for key, value in ledger["split"].astype(str).value_counts().sort_index().items()
        },
        "embargo_rows": int(ledger["split"].astype(str).eq("embargo").sum()),
        "embargo_forecast_coverage_required": True,
        "embargo_metric_eligible": False,
        "archive_rows": int(len(archive)),
        "selected_model_ids": model_ids,
        "models": int(archive["model_id"].nunique() if not archive.empty else 0),
        "coverage_rule": "each selected model_id has exactly one prediction for every ledger forecast_id",
        "immutable_rule": "archive and draws are content-addressed by SHA256 in this manifest",
        "reuse_baseline_forecasts": bool(args.reuse_baseline_forecasts),
        "baseline_manifest": str(args.baseline_manifest or ""),
        "baseline_runs_root": str(args.baseline_runs_root or ""),
        "baseline_extension_runs_root": str(args.baseline_extension_runs_root or ""),
        "import_model_forecasts": bool(args.import_model_forecasts),
        "model_forecast_runs_root": str(args.model_forecast_runs_root or ""),
        "model_forecast_source_manifest": str(
            args.model_forecast_source_manifest or ""
        ),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_resume": bool(args.resume),
        "forced_model_ids": sorted(force_model_ids),
        "checkpoint_identities": checkpoint_identity_by_model,
        "base_registry_path": str(Path(args.registry)),
        "base_registry_sha256": _sha256_file(Path(args.registry)),
        "model_hyperparam_overrides": model_hyperparam_overrides,
        "effective_model_hyperparams": {
            model_id: json.loads(str(registry_by_model[model_id]["hyperparams_json"]))
            for model_id in model_ids
        },
        "effective_registry_row_sha256": {
            model_id: _canonical_sha256(registry_by_model[model_id])
            for model_id in model_ids
        },
        "workers": int(args.workers),
        "baseline_reuse_models": (
            [m for m in model_ids if m in BASELINE_REUSE_PATHS]
            if args.reuse_baseline_forecasts
            else []
        ),
        "fresh_candidate_source_models": (
            [m for m in model_ids if m in BASELINE_REUSE_PATHS]
            if args.import_model_forecasts
            else []
        ),
        "caster_local_models": [m for m in model_ids if m not in BASELINE_REUSE_PATHS],
        "gpu_local_models": [m for m in model_ids if m in CASTER_LOCAL_GPU_MODELS],
        "per_model_sources": per_model_sources,
        "no_native_likelihood": True,
        "protocol_versions": sorted(ledger["protocol_version"].dropna().astype(str).unique().tolist()) if "protocol_version" in ledger.columns else [],
        "forecast_modes": sorted(ledger["mode"].dropna().astype(str).unique().tolist()) if "mode" in ledger.columns else [],
        "forecast_strategies": sorted(ledger["forecast_strategy"].dropna().astype(str).unique().tolist()) if "forecast_strategy" in ledger.columns else [],
        "strategy_execution_contract": (
            "direct=native multi-horizon from observed history; recursive_rollout=repeated h=1 with predicted-mean feedback"
        ),
        "rollout_uncertainty_contract": "marginal pred_var/draw/bridge proxy per horizon; no trajectory Monte Carlo",
        **input_metadata,
    }
    manifest.update(task_metadata(task, ledger))
    _write_json(manifest_path, manifest)
    print(f"archive={archive_path}")
    print(f"rows={len(archive)} models={archive['model_id'].nunique() if not archive.empty else 0}")
    if draws_out:
        print(f"draws={draws_out}")
    print(f"manifest={manifest_path}")
    print(f"archive_sha256={manifest['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
