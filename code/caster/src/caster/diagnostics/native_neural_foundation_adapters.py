from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import re

import pandas as pd

from caster.diagnostics.native_sidecar import (
    RETENTION_PERSIST_MINIMAL,
    RETENTION_UNAVAILABLE,
    SCHEMA_VERSION,
    STATUS_BLOCKED,
    STATUS_DETERMINISTIC_NO_NATIVE,
    STATUS_PROXY_ONLY,
    STATUS_QUANTILE_RECONSTRUCTED_ONLY,
    STATUS_SAMPLE_KERNEL_ONLY,
    STATUS_UNAVAILABLE,
    NativeAvailabilityRecord,
    NativeSidecarStorageValidationRow,
    apply_retention_policy,
    sha256_file,
)


NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA = "neural_foundation_native_artifact.v1"

DETERMINISTIC_NEURAL_MODEL_IDS = {"rnn_simple", "gru_style"}
POINT_ONLY_NEURAL_MODEL_IDS = {"lstm_style", "nbeats_basis", "nhits_hinterp", "patchtst_patched", "tft_gated"}
PROBABILISTIC_NEURAL_MODEL_IDS = {"deepar_style"}

MODEL_SAMPLE_COLUMN_PATTERN = re.compile(r"^model_sample_\d+$")
MODEL_QUANTILE_COLUMN_PATTERN = re.compile(r"^model_quantile_(?:0(?:\.\d+)?|1(?:\.0+)?)$")
EXTERNAL_FORECAST_KEY_COLUMNS = {"entity_id", "forecast_origin", "target_time", "component"}


def build_neural_foundation_native_availability(
    registry: pd.DataFrame,
    *,
    model_filters: Iterable[str] = ("neural", "foundation"),
    out_root: str | Path | None = None,
    registry_source: str = "",
    registry_blocker: str = "",
    registry_base_dir: str | Path | None = None,
) -> tuple[list[NativeAvailabilityRecord], list[NativeSidecarStorageValidationRow]]:
    ""





    filters = {str(x).strip().lower() for x in model_filters if str(x).strip()}
    include_neural = "neural" in filters
    include_foundation = "foundation" in filters
    base_dir = Path(registry_base_dir) if registry_base_dir is not None else Path.cwd()

    availability: list[NativeAvailabilityRecord] = []
    storage_rows: list[NativeSidecarStorageValidationRow] = []
    seen: set[str] = set()
    for _, row in registry.iterrows():
        model_id = str(row.get("model_id", "")).strip()
        if not model_id or model_id in seen:
            continue
        family = str(row.get("family", "")).strip().lower()
        adapter_path = str(row.get("adapter_path", "")).strip()
        if include_neural and _is_neural_row(model_id, family, adapter_path):
            availability.append(_neural_availability_record(row, registry_source, registry_blocker))
            seen.add(model_id)
            continue
        if include_foundation and _is_foundation_row(family, adapter_path):
            record, storage = _foundation_availability_record(
                row,
                out_root=out_root,
                registry_source=registry_source,
                registry_blocker=registry_blocker,
                registry_base_dir=base_dir,
            )
            availability.append(record)
            storage_rows.extend(storage)
            seen.add(model_id)
    return availability, storage_rows


def _is_neural_row(model_id: str, family: str, adapter_path: str) -> bool:
    if family == "neural":
        return True
    if model_id in DETERMINISTIC_NEURAL_MODEL_IDS | POINT_ONLY_NEURAL_MODEL_IDS | PROBABILISTIC_NEURAL_MODEL_IDS:
        return True
    return "neuralforecast" in adapter_path.lower()


def _is_foundation_row(family: str, adapter_path: str) -> bool:
    if family in {"foundation", "foundation_ts"}:
        return True
    return "foundation_adapters" in adapter_path.lower()


def _neural_availability_record(
    row: Mapping[str, Any],
    registry_source: str,
    registry_blocker: str,
) -> NativeAvailabilityRecord:
    model_id = str(row.get("model_id", "")).strip()
    if model_id in DETERMINISTIC_NEURAL_MODEL_IDS:
        return _availability_record(
            model_id=model_id,
            status=STATUS_DETERMINISTIC_NO_NATIVE,
            likelihood_type="deterministic_point_forecast",
            sidecar_required=False,
            schema="",
            retention_policy=RETENTION_UNAVAILABLE,
            blocker=f"{model_id} has deterministic point dynamics and no explicit native observation likelihood",
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        )
    if model_id in POINT_ONLY_NEURAL_MODEL_IDS:
        return _availability_record(
            model_id=model_id,
            status=STATUS_PROXY_ONLY,
            likelihood_type="point_forecast_only",
            sidecar_required=False,
            schema="",
            retention_policy=RETENTION_UNAVAILABLE,
            blocker=(
                f"{model_id} current adapter exposes point forecasts only; pred_var/residual scale is a "
                "forecast-archive proxy, not adapter-native likelihood"
            ),
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        )
    if model_id in PROBABILISTIC_NEURAL_MODEL_IDS:
        return _availability_record(
            model_id=model_id,
            status=STATUS_BLOCKED,
            likelihood_type="distribution_params_missing",
            sidecar_required=True,
            schema=NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
            retention_policy=RETENTION_UNAVAILABLE,
            blocker=(
                "blocked_missing_distribution_params: probabilistic neural model may have a native "
                "training distribution, but the current adapter/run artifact does not persist origin-time "
                "distribution parameters"
            ),
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        )
    return _availability_record(
        model_id=model_id,
        status=STATUS_BLOCKED,
        likelihood_type="unknown_neural_native_contract",
        sidecar_required=True,
        schema=NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
        retention_policy=RETENTION_UNAVAILABLE,
        blocker="blocked_missing_distribution_params: neural adapter has no diagnostic native sidecar contract",
        registry_source=registry_source,
        registry_blocker=registry_blocker,
    )


def _foundation_availability_record(
    row: Mapping[str, Any],
    *,
    out_root: str | Path | None,
    registry_source: str,
    registry_blocker: str,
    registry_base_dir: Path,
) -> tuple[NativeAvailabilityRecord, list[NativeSidecarStorageValidationRow]]:
    model_id = str(row.get("model_id", "")).strip()
    hyperparams = _parse_hyperparams(row.get("hyperparams_json"))
    forecast_path_text = str(hyperparams.get("forecast_path", "")).strip()
    if not forecast_path_text:
        record = _availability_record(
            model_id=model_id,
            status=STATUS_UNAVAILABLE,
            likelihood_type="external_artifact_missing",
            sidecar_required=True,
            schema=NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
            retention_policy=RETENTION_UNAVAILABLE,
            blocker="external foundation forecast artifact is required; API re-query is not allowed for origin-time native sidecar recovery",
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        )
        return record, []

    forecast_path = _resolve_forecast_path(forecast_path_text, registry_base_dir)
    if not forecast_path.exists():
        record = _availability_record(
            model_id=model_id,
            status=STATUS_UNAVAILABLE,
            likelihood_type="external_artifact_missing",
            sidecar_required=True,
            schema=NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
            retention_policy=RETENTION_UNAVAILABLE,
            blocker=(
                f"external forecast artifact not found: {forecast_path}; API re-query is not allowed "
                "to reconstruct origin-time native sidecars"
            ),
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        )
        return record, []

    try:
        frame = pd.read_csv(forecast_path, nrows=32)
    except Exception as exc:
        record = _availability_record(
            model_id=model_id,
            status=STATUS_UNAVAILABLE,
            likelihood_type="external_artifact_unreadable",
            sidecar_required=True,
            schema=NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
            retention_policy=RETENTION_UNAVAILABLE,
            blocker=f"could not read external artifact {forecast_path}: {type(exc).__name__}: {exc}",
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        )
        return record, []

    missing_keys = sorted(EXTERNAL_FORECAST_KEY_COLUMNS - set(frame.columns))
    if missing_keys:
        record = _availability_record(
            model_id=model_id,
            status=STATUS_UNAVAILABLE,
            likelihood_type="external_artifact_invalid",
            sidecar_required=True,
            schema=NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
            retention_policy=RETENTION_UNAVAILABLE,
            blocker=f"external artifact missing origin-time key columns: {missing_keys}",
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        )
        return record, []

    sample_columns = _model_sample_columns(frame.columns)
    quantile_columns = _model_quantile_columns(frame.columns)
    if sample_columns:
        return _artifact_evidence_record(
            model_id=model_id,
            forecast_path=forecast_path,
            columns=sample_columns,
            status=STATUS_SAMPLE_KERNEL_ONLY,
            likelihood_type="sample_kernel_only",
            sidecar_type="external_model_samples",
            out_root=out_root,
            registry_source=registry_source,
            registry_blocker=registry_blocker,
            blocker="explicit external model sample columns are available, but samples are not an analytic adapter-native log likelihood",
        )
    if quantile_columns:
        return _artifact_evidence_record(
            model_id=model_id,
            forecast_path=forecast_path,
            columns=quantile_columns,
            status=STATUS_QUANTILE_RECONSTRUCTED_ONLY,
            likelihood_type="quantile_reconstructed_only",
            sidecar_type="external_model_quantiles",
            out_root=out_root,
            registry_source=registry_source,
            registry_blocker=registry_blocker,
            blocker="explicit external model quantile columns are available, but quantile reconstruction is not analytic native likelihood",
        )
    if "pred_mean" in frame.columns:
        return (
            _availability_record(
                model_id=model_id,
                status=STATUS_PROXY_ONLY,
                likelihood_type="point_or_archive_variance_proxy",
                sidecar_required=False,
                schema="",
                retention_policy=RETENTION_UNAVAILABLE,
                blocker=(
                    f"{model_id} external artifact contains only point/proxy forecast columns; pred_var and "
                    "synthetic draws are not native likelihood samples"
                ),
                registry_source=registry_source,
                registry_blocker=registry_blocker,
            ),
            [],
        )
    return (
        _availability_record(
            model_id=model_id,
            status=STATUS_UNAVAILABLE,
            likelihood_type="external_artifact_missing_prediction_schema",
            sidecar_required=True,
            schema=NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
            retention_policy=RETENTION_UNAVAILABLE,
            blocker="external artifact lacks pred_mean, model_sample_<i>, or model_quantile_<q> columns",
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        ),
        [],
    )


def _artifact_evidence_record(
    *,
    model_id: str,
    forecast_path: Path,
    columns: list[str],
    status: str,
    likelihood_type: str,
    sidecar_type: str,
    out_root: str | Path | None,
    registry_source: str,
    registry_blocker: str,
    blocker: str,
) -> tuple[NativeAvailabilityRecord, list[NativeSidecarStorageValidationRow]]:
    storage_rows: list[NativeSidecarStorageValidationRow] = []
    artifact_path = ""
    retention_policy = RETENTION_UNAVAILABLE
    blocker_text = blocker
    if out_root is not None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "native_sidecar_schema": NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
            "model_id": model_id,
            "origin": "external_artifact_schema",
            "source_artifact_path": str(forecast_path),
            "source_artifact_hash": sha256_file(forecast_path),
            "native_likelihood_type": likelihood_type,
            "evidence_columns": columns,
            "features_available_until_policy": "required external artifact provenance; missing, invalid, or post-origin values fail closed",
            "uses_external_api_requery": False,
            "uses_synthetic_forecast_draws": False,
            "uses_bridge_scale": False,
            "supports_analytic_native_log_likelihood": False,
        }
        storage_row = apply_retention_policy(
            RETENTION_PERSIST_MINIMAL,
            out_root=out_root,
            model_id=model_id,
            origin="external_artifact_schema",
            payload=payload,
            sidecar_type=sidecar_type,
            score_reproducibility_level="artifact_schema_evidence_only",
        )
        storage_rows.append(storage_row)
        artifact_path = storage_row.artifact_path
        retention_policy = RETENTION_PERSIST_MINIMAL
    else:
        blocker_text = f"{blocker}; out_root not provided so minimal evidence sidecar was not written"

    return (
        _availability_record(
            model_id=model_id,
            status=status,
            likelihood_type=likelihood_type,
            sidecar_required=True,
            schema=NEURAL_FOUNDATION_NATIVE_SIDECAR_SCHEMA,
            retention_policy=retention_policy,
            artifact_path=artifact_path,
            blocker=blocker_text,
            registry_source=registry_source,
            registry_blocker=registry_blocker,
        ),
        storage_rows,
    )


def _model_sample_columns(columns: Iterable[str]) -> list[str]:
    return sorted(str(col) for col in columns if MODEL_SAMPLE_COLUMN_PATTERN.match(str(col)))


def _model_quantile_columns(columns: Iterable[str]) -> list[str]:
    return sorted(str(col) for col in columns if MODEL_QUANTILE_COLUMN_PATTERN.match(str(col)))


def _availability_record(
    *,
    model_id: str,
    status: str,
    likelihood_type: str,
    sidecar_required: bool,
    schema: str,
    retention_policy: str,
    blocker: str,
    registry_source: str,
    registry_blocker: str,
    artifact_path: str = "",
) -> NativeAvailabilityRecord:
    blocker_text = "; ".join(x for x in [blocker, registry_blocker, f"registry_source={registry_source}" if registry_source else ""] if x)
    return NativeAvailabilityRecord(
        model_id=model_id,
        origin="registry",
        supports_native_log_likelihood=False,
        native_likelihood_type=likelihood_type,
        native_likelihood_status=status,
        native_sidecar_required=bool(sidecar_required),
        native_sidecar_schema=schema,
        retention_policy=retention_policy,
        artifact_path=artifact_path,
        blocker=blocker_text,
    )


def _parse_hyperparams(value: object) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _resolve_forecast_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [base_dir / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
