from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json
import platform
import re

import pandas as pd


SCHEMA_VERSION = "native_sidecar.v1"
NATIVE_SIDECAR_SCHEMA = "baseline_native_sidecar.v1"
PERSISTENT_DIR_NAME = "native_sidecars"
RETENTION_PERSIST_MINIMAL = "persist_minimal"
RETENTION_UNAVAILABLE = "unavailable_if_no_feasible_sidecar"

STATUS_TRUE_NATIVE = "true_native"
STATUS_UNAVAILABLE = "unavailable"
STATUS_PROXY_ONLY = "proxy_only"
STATUS_QUANTILE_RECONSTRUCTED_ONLY = "quantile_reconstructed_only"
STATUS_DETERMINISTIC_NO_NATIVE = "deterministic_no_native"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_BLOCKED = "blocked"
STATUS_BLOCKED_NO_PANEL_LEDGER = "blocked_no_panel_ledger"
STATUS_BLOCKED_SCALE_MISMATCH = "blocked_scale_mismatch"
STATUS_BLOCKED_MISSING_DISTRIBUTION_PARAMS = "blocked_missing_distribution_params"

NATIVE_LIKELIHOOD_STATUSES = {
    STATUS_TRUE_NATIVE,
    STATUS_UNAVAILABLE,
    STATUS_PROXY_ONLY,
    STATUS_QUANTILE_RECONSTRUCTED_ONLY,
    STATUS_DETERMINISTIC_NO_NATIVE,
    STATUS_NOT_APPLICABLE,
    STATUS_BLOCKED,
    STATUS_BLOCKED_NO_PANEL_LEDGER,
    STATUS_BLOCKED_SCALE_MISMATCH,
    STATUS_BLOCKED_MISSING_DISTRIBUTION_PARAMS,
}

AVAILABILITY_COLUMNS = [
    "model_id",
    "origin",
    "supports_native_log_likelihood",
    "native_likelihood_type",
    "native_likelihood_status",
    "native_sidecar_required",
    "native_sidecar_schema",
    "retention_policy",
    "artifact_path",
    "blocker",
]

STORAGE_VALIDATION_COLUMNS = [
    "model_id",
    "origin",
    "sidecar_type",
    "persistent_size_mb",
    "temporary_size_mb",
    "retention_policy",
    "deleted_after_scoring",
    "score_reproducibility_level",
    "artifact_path",
]

TRUE_NATIVE_REQUIRED_PAYLOAD_FIELDS = [
    "distribution_params",
    "score_scale",
    "native_likelihood_type",
    "features_available_until",
    "retention_policy",
]


class NativeSidecarSchemaError(RuntimeError):
    ""


@dataclass(frozen=True)
class PredictionBundle:
    ""

    means: dict[int, float]
    native_by_horizon: dict[int, dict[str, Any]]
    native_likelihood_status: str = STATUS_TRUE_NATIVE
    native_likelihood_type: str = ""
    blocker: str = ""


@dataclass
class NativeAvailabilityRecord:
    model_id: str
    origin: str
    supports_native_log_likelihood: bool
    native_likelihood_type: str
    native_likelihood_status: str
    native_sidecar_required: bool
    native_sidecar_schema: str
    retention_policy: str
    artifact_path: str
    blocker: str = ""


@dataclass
class NativeSidecarStorageValidationRow:
    model_id: str
    origin: str
    sidecar_type: str
    persistent_size_mb: float
    temporary_size_mb: float
    retention_policy: str
    deleted_after_scoring: bool
    score_reproducibility_level: str
    artifact_path: str
    sidecar_hash: str = ""
    blocker: str = ""


def default_native_sidecar_root(out_dir: str | Path, native_sidecar_root: str | Path | None = None) -> Path:
    ""

    return Path(native_sidecar_root) if native_sidecar_root else Path(out_dir) / "native_diagnostics"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_size_mb(path: str | Path) -> float:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / (1024.0 * 1024.0)
    if p.is_dir():
        return sum(child.stat().st_size for child in p.rglob("*") if child.is_file()) / (1024.0 * 1024.0)
    return 0.0


def validate_true_native_payload(payload: Mapping[str, Any]) -> list[str]:
    ""

    missing = [field for field in TRUE_NATIVE_REQUIRED_PAYLOAD_FIELDS if field not in payload or payload.get(field) in (None, "")]
    if not isinstance(payload.get("distribution_params"), Mapping):
        missing.append("distribution_params_mapping")
    return missing


def write_true_native_sidecar(
    root: str | Path,
    *,
    model_id: str,
    origin: str,
    payload: Mapping[str, Any],
    sidecar_type: str = "baseline_native_distribution_params",
) -> tuple[NativeAvailabilityRecord, NativeSidecarStorageValidationRow]:
    ""

    missing = validate_true_native_payload(payload)
    if missing:
        raise NativeSidecarSchemaError(f"true-native payload missing required fields: {missing}")
    root_path = Path(root)
    sidecar_dir = root_path / PERSISTENT_DIR_NAME / _safe_component(origin) / _safe_component(model_id)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    payload_path = sidecar_dir / "sidecar_payload.json"
    payload_doc = {
        "schema_version": SCHEMA_VERSION,
        "native_sidecar_schema": NATIVE_SIDECAR_SCHEMA,
        "model_id": model_id,
        "origin": origin,
        "software_version": _software_version(),
        "payload": dict(payload),
    }
    _write_json(payload_path, payload_doc)
    sidecar_hash = sha256_file(payload_path)
    manifest_path = sidecar_dir / "sidecar_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "native_sidecar_schema": NATIVE_SIDECAR_SCHEMA,
        "model_id": model_id,
        "origin": origin,
        "sidecar_type": sidecar_type,
        "sidecar_hash": sidecar_hash,
        "software_version": _software_version(),
        "retention_policy": str(payload["retention_policy"]),
        "artifact_path": str(payload_path),
        "score_reproducibility_level": "baseline_native_distribution_params",
        "created_at_utc": _utc_now(),
    }
    _write_json(manifest_path, manifest)
    availability = NativeAvailabilityRecord(
        model_id=model_id,
        origin=origin,
        supports_native_log_likelihood=True,
        native_likelihood_type=str(payload["native_likelihood_type"]),
        native_likelihood_status=STATUS_TRUE_NATIVE,
        native_sidecar_required=True,
        native_sidecar_schema=NATIVE_SIDECAR_SCHEMA,
        retention_policy=str(payload["retention_policy"]),
        artifact_path=str(manifest_path),
        blocker="",
    )
    storage = NativeSidecarStorageValidationRow(
        model_id=model_id,
        origin=origin,
        sidecar_type=sidecar_type,
        persistent_size_mb=path_size_mb(sidecar_dir),
        temporary_size_mb=0.0,
        retention_policy=str(payload["retention_policy"]),
        deleted_after_scoring=False,
        score_reproducibility_level="baseline_native_distribution_params",
        artifact_path=str(manifest_path),
        sidecar_hash=sidecar_hash,
        blocker="",
    )
    return availability, storage


def status_availability(
    *,
    model_id: str,
    origin: str,
    status: str,
    native_likelihood_type: str = "none",
    blocker: str = "",
    native_sidecar_required: bool = False,
    artifact_path: str = "",
) -> NativeAvailabilityRecord:
    ""

    if status not in NATIVE_LIKELIHOOD_STATUSES:
        raise NativeSidecarSchemaError(f"unknown native likelihood status: {status}")
    return NativeAvailabilityRecord(
        model_id=model_id,
        origin=origin,
        supports_native_log_likelihood=False,
        native_likelihood_type=native_likelihood_type,
        native_likelihood_status=status,
        native_sidecar_required=bool(native_sidecar_required),
        native_sidecar_schema=NATIVE_SIDECAR_SCHEMA if native_sidecar_required else "",
        retention_policy=RETENTION_UNAVAILABLE,
        artifact_path=artifact_path,
        blocker=blocker,
    )


def native_rows_from_prediction(
    root: str | Path,
    *,
    model_id: str,
    origin: str,
    prediction_result: object,
    horizon: int,
) -> tuple[dict[int, float], NativeAvailabilityRecord, NativeSidecarStorageValidationRow | None]:
    ""

    if isinstance(prediction_result, PredictionBundle):
        means = {int(k): float(v) for k, v in prediction_result.means.items()}
        payload = dict(prediction_result.native_by_horizon.get(int(horizon), {}))
        payload.setdefault("native_likelihood_type", prediction_result.native_likelihood_type)
        missing = validate_true_native_payload(payload)
        if prediction_result.native_likelihood_status == STATUS_TRUE_NATIVE and not missing:
            availability, storage = write_true_native_sidecar(root, model_id=model_id, origin=origin, payload=payload)
            return means, availability, storage
        status = prediction_result.native_likelihood_status if prediction_result.native_likelihood_status != STATUS_TRUE_NATIVE else STATUS_BLOCKED
        blocker = prediction_result.blocker or f"incomplete PredictionBundle native payload missing={missing}"
        return means, status_availability(model_id=model_id, origin=origin, status=status, blocker=blocker, native_sidecar_required=True), None
    means = {int(k): float(v) for k, v in dict(prediction_result).items()}
    availability = status_availability(
        model_id=model_id,
        origin=origin,
        status=STATUS_PROXY_ONLY,
        blocker="predictor returned means only; residual-sigma/interval baseline output is proxy-only, not true native likelihood",
        native_sidecar_required=False,
    )
    return means, availability, None


def write_dependency_unavailable(
    root: str | Path,
    *,
    model_id: str,
    backend: str,
    reason: str,
) -> Path:
    ""

    row = status_availability(
        model_id=model_id,
        origin=f"{backend}_dependency_check",
        status=STATUS_UNAVAILABLE,
        blocker=reason,
        native_sidecar_required=True,
    )
    return write_availability_report([row], Path(root) / "native_likelihood_availability.csv")


def write_availability_report(rows: Iterable[NativeAvailabilityRecord | Mapping[str, Any]], out_csv: str | Path) -> Path:
    records = [_row_to_dict(row) for row in rows]
    for row in records:
        for col in AVAILABILITY_COLUMNS:
            row.setdefault(col, "")
    frame = pd.DataFrame(records, columns=_ordered_columns(records, AVAILABILITY_COLUMNS))
    return _write_csv(frame, out_csv)


def write_storage_validation(rows: Iterable[NativeSidecarStorageValidationRow | Mapping[str, Any]], out_csv: str | Path) -> Path:
    records = [_row_to_dict(row) for row in rows]
    for row in records:
        for col in STORAGE_VALIDATION_COLUMNS:
            row.setdefault(col, "")
    frame = pd.DataFrame(records, columns=_ordered_columns(records, STORAGE_VALIDATION_COLUMNS))
    return _write_csv(frame, out_csv)


def _row_to_dict(row: NativeAvailabilityRecord | NativeSidecarStorageValidationRow | Mapping[str, Any]) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    return dict(row)


def _ordered_columns(records: list[dict[str, Any]], required: list[str]) -> list[str]:
    extras = sorted({col for row in records for col in row} - set(required))
    return [*required, *extras]


def _write_csv(frame: pd.DataFrame, out_csv: str | Path) -> Path:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def _safe_component(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _software_version() -> str:
    return f"caster-baseline-impl:0.1.0;python:{platform.python_version()}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
