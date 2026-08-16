from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json
import platform
import re
import shutil

import pandas as pd


SCHEMA_VERSION = "native_sidecar.v1"
PERSISTENT_DIR_NAME = "native_sidecars"

RETENTION_PERSIST_MINIMAL = "persist_minimal"
RETENTION_TEMP_DELETE = "temp_until_scored_then_delete"
RETENTION_PERSIST_CHECKPOINT_IF_SMALL = "persist_checkpoint_if_small"
RETENTION_UNAVAILABLE = "unavailable_if_no_feasible_sidecar"
RETENTION_POLICIES = {
    RETENTION_PERSIST_MINIMAL,
    RETENTION_TEMP_DELETE,
    RETENTION_PERSIST_CHECKPOINT_IF_SMALL,
    RETENTION_UNAVAILABLE,
}

STATUS_TRUE_NATIVE = "true_native"
STATUS_UNAVAILABLE = "unavailable"
STATUS_PROXY_ONLY = "proxy_only"
STATUS_SAMPLE_KERNEL_ONLY = "sample_kernel_only"
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
    STATUS_SAMPLE_KERNEL_ONLY,
    STATUS_QUANTILE_RECONSTRUCTED_ONLY,
    STATUS_DETERMINISTIC_NO_NATIVE,
    STATUS_NOT_APPLICABLE,
    STATUS_BLOCKED,
    STATUS_BLOCKED_NO_PANEL_LEDGER,
    STATUS_BLOCKED_SCALE_MISMATCH,
    STATUS_BLOCKED_MISSING_DISTRIBUTION_PARAMS,
}

DEFAULT_MAX_PERSISTENT_CHECKPOINT_MB = 50.0

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

TIMING_FIXTURE_FIELDS = [
    "full_caster_bridge_update_sec",
    "native_sidecar_save_overhead_sec",
    "native_sidecar_load_overhead_sec",
    "adapter_loglik_score_sec",
    "native_posterior_update_sec",
    "adapter_native_total_sec",
    "full_rerun_wallclock_sec",
    "timing_basis",
]


class NativeSidecarPolicyError(RuntimeError):
    ""


class NativeSidecarPathError(NativeSidecarPolicyError):
    ""


@dataclass
class NativeSidecarRecord:
    ""

    schema_version: str
    model_id: str
    origin: str
    sidecar_type: str
    sidecar_hash: str
    software_version: str
    retention_policy: str
    artifact_path: str
    manifest_path: str
    persistent_size_mb: float
    temporary_size_mb: float
    deleted_after_scoring: bool
    score_reproducibility_level: str

    def to_storage_validation_row(self) -> "NativeSidecarStorageValidationRow":
        return NativeSidecarStorageValidationRow(
            model_id=self.model_id,
            origin=self.origin,
            sidecar_type=self.sidecar_type,
            persistent_size_mb=self.persistent_size_mb,
            temporary_size_mb=self.temporary_size_mb,
            retention_policy=self.retention_policy,
            deleted_after_scoring=self.deleted_after_scoring,
            score_reproducibility_level=self.score_reproducibility_level,
            artifact_path=self.manifest_path,
            sidecar_hash=self.sidecar_hash,
            deletion_log_path="",
            blocker="",
        )


@dataclass
class NativeAvailabilityRecord:
    ""

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

    def __post_init__(self) -> None:
        _validate_status_and_support(
            self.native_likelihood_status,
            self.supports_native_log_likelihood,
            label=f"{self.origin}/{self.model_id}",
        )


@dataclass
class NativeSidecarStorageValidationRow:
    ""

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
    deletion_log_path: str = ""
    blocker: str = ""


def default_software_version() -> str:
    ""

    return f"caster-algorithm-impl:0.1.0;python:{platform.python_version()}"


def sha256_file(path: str | Path) -> str:
    ""

    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: str | Path) -> str:
    ""

    p = Path(path)
    if p.is_file() or p.is_symlink():
        return sha256_file(p)
    if not p.is_dir():
        raise NativeSidecarPathError(f"artifact path does not exist or is not a file/directory: {p}")
    digest = hashlib.sha256()
    for child in sorted(x for x in p.rglob("*") if x.is_file()):
        rel = child.relative_to(p).as_posix().encode("utf-8")
        digest.update(b"path\0")
        digest.update(rel)
        digest.update(b"\0sha256\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def path_size_mb(path: str | Path) -> float:
    ""

    p = Path(path)
    if p.is_file() or p.is_symlink():
        return p.stat().st_size / (1024.0 * 1024.0)
    if p.is_dir():
        total = sum(child.stat().st_size for child in p.rglob("*") if child.is_file())
        return total / (1024.0 * 1024.0)
    return 0.0


def write_minimal_sidecar(
    *,
    out_root: str | Path,
    model_id: str,
    origin: str,
    payload: Mapping[str, Any],
    sidecar_type: str = "minimal_json",
    retention_policy: str = RETENTION_PERSIST_MINIMAL,
    software_version: str | None = None,
    score_reproducibility_level: str = "minimal_sidecar_replayable",
) -> NativeSidecarRecord:
    ""

    _require_retention_policy(retention_policy)
    sidecar_dir = _sidecar_dir(out_root, origin, model_id)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    software = software_version or default_software_version()
    payload_path = sidecar_dir / "sidecar_payload.json"
    payload_doc = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "origin": origin,
        "sidecar_type": sidecar_type,
        "software_version": software,
        "payload": dict(payload),
    }
    _write_json(payload_path, payload_doc)
    sidecar_hash = sha256_file(payload_path)
    manifest_path = sidecar_dir / "sidecar_manifest.json"
    manifest = _manifest(
        model_id=model_id,
        origin=origin,
        sidecar_type=sidecar_type,
        sidecar_hash=sidecar_hash,
        software_version=software,
        retention_policy=retention_policy,
        artifact_path=payload_path,
        score_reproducibility_level=score_reproducibility_level,
    )
    _write_json(manifest_path, manifest)
    return NativeSidecarRecord(
        schema_version=SCHEMA_VERSION,
        model_id=model_id,
        origin=origin,
        sidecar_type=sidecar_type,
        sidecar_hash=sidecar_hash,
        software_version=software,
        retention_policy=retention_policy,
        artifact_path=str(payload_path),
        manifest_path=str(manifest_path),
        persistent_size_mb=path_size_mb(sidecar_dir),
        temporary_size_mb=0.0,
        deleted_after_scoring=False,
        score_reproducibility_level=score_reproducibility_level,
    )


def delete_temporary_sidecar(
    path: str | Path,
    *,
    temp_root: str | Path,
    deletion_log_path: str | Path,
    model_id: str,
    origin: str,
    reason: str,
    retention_policy: str = RETENTION_TEMP_DELETE,
    software_version: str | None = None,
) -> dict[str, Any]:
    ""

    _require_retention_policy(retention_policy)
    target = Path(path).resolve()
    temp_root_resolved = Path(temp_root).resolve()
    _validate_deletable_temp_path(target, temp_root_resolved)
    sidecar_hash = sha256_artifact(target)
    size_mb = path_size_mb(target)
    log = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "origin": origin,
        "sidecar_hash": sidecar_hash,
        "software_version": software_version or default_software_version(),
        "retention_policy": retention_policy,
        "deleted_after_scoring": True,
        "deleted_path": str(target),
        "deleted_size_mb": size_mb,
        "reason": reason,
        "deleted_at_utc": _utc_now(),
    }
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink()
    _write_json(deletion_log_path, log)
    return log


def apply_retention_policy(
    retention_policy: str,
    *,
    out_root: str | Path,
    model_id: str,
    origin: str,
    payload: Mapping[str, Any] | None = None,
    source_artifact_path: str | Path | None = None,
    temp_root: str | Path | None = None,
    deletion_log_path: str | Path | None = None,
    sidecar_type: str = "minimal_json",
    max_persistent_mb: float = DEFAULT_MAX_PERSISTENT_CHECKPOINT_MB,
    score_reproducibility_level: str | None = None,
    blocker: str = "",
    reason: str = "temporary sidecar deleted after diagnostic scoring",
) -> NativeSidecarStorageValidationRow:
    ""

    _require_retention_policy(retention_policy)
    if retention_policy == RETENTION_PERSIST_MINIMAL:
        if payload is None:
            raise NativeSidecarPolicyError("persist_minimal requires a payload")
        record = write_minimal_sidecar(
            out_root=out_root,
            model_id=model_id,
            origin=origin,
            payload=payload,
            sidecar_type=sidecar_type,
            retention_policy=retention_policy,
            score_reproducibility_level=score_reproducibility_level or "minimal_sidecar_replayable",
        )
        return record.to_storage_validation_row()

    if retention_policy == RETENTION_TEMP_DELETE:
        if source_artifact_path is None or temp_root is None:
            raise NativeSidecarPolicyError("temp_until_scored_then_delete requires source_artifact_path and temp_root")
        log_path = Path(deletion_log_path) if deletion_log_path is not None else _default_deletion_log_path(out_root, origin, model_id)
        log = delete_temporary_sidecar(
            source_artifact_path,
            temp_root=temp_root,
            deletion_log_path=log_path,
            model_id=model_id,
            origin=origin,
            reason=reason,
            retention_policy=retention_policy,
        )
        return NativeSidecarStorageValidationRow(
            model_id=model_id,
            origin=origin,
            sidecar_type=sidecar_type,
            persistent_size_mb=path_size_mb(log_path),
            temporary_size_mb=float(log["deleted_size_mb"]),
            retention_policy=retention_policy,
            deleted_after_scoring=True,
            score_reproducibility_level=score_reproducibility_level or "hash_provenance_only",
            artifact_path=str(log_path),
            sidecar_hash=str(log["sidecar_hash"]),
            deletion_log_path=str(log_path),
            blocker="",
        )

    if retention_policy == RETENTION_PERSIST_CHECKPOINT_IF_SMALL:
        if source_artifact_path is None:
            raise NativeSidecarPolicyError("persist_checkpoint_if_small requires source_artifact_path")
        source = Path(source_artifact_path)
        size_mb = path_size_mb(source)
        if size_mb > float(max_persistent_mb):
            raise NativeSidecarPolicyError(
                f"checkpoint sidecar size {size_mb:.6f} MB exceeds max_persistent_mb={float(max_persistent_mb):.6f}"
            )
        record = _persist_checkpoint(
            source,
            out_root=out_root,
            model_id=model_id,
            origin=origin,
            sidecar_type=sidecar_type,
            retention_policy=retention_policy,
            score_reproducibility_level=score_reproducibility_level or "checkpoint_replayable",
        )
        return record.to_storage_validation_row()

    return NativeSidecarStorageValidationRow(
        model_id=model_id,
        origin=origin,
        sidecar_type=sidecar_type,
        persistent_size_mb=0.0,
        temporary_size_mb=0.0,
        retention_policy=retention_policy,
        deleted_after_scoring=False,
        score_reproducibility_level=score_reproducibility_level or "unavailable",
        artifact_path="",
        sidecar_hash="",
        deletion_log_path="",
        blocker=blocker or "no feasible diagnostic native sidecar available",
    )


def write_availability_report(rows: Iterable[NativeAvailabilityRecord | Mapping[str, Any]], out_csv: str | Path) -> Path:
    ""

    records = [_row_to_dict(row) for row in rows]
    for row in records:
        for column in AVAILABILITY_COLUMNS:
            row.setdefault(column, "")
        _validate_status_and_support(
            row["native_likelihood_status"],
            _truthy(row["supports_native_log_likelihood"]),
            label=f"{row.get('origin', '')}/{row.get('model_id', '')}",
        )
    frame = pd.DataFrame(records, columns=_ordered_columns(records, AVAILABILITY_COLUMNS))
    return _write_csv(frame, out_csv)


def write_storage_validation(rows: Iterable[NativeSidecarStorageValidationRow | Mapping[str, Any]], out_csv: str | Path) -> Path:
    ""

    records = [_row_to_dict(row) for row in rows]
    for row in records:
        for column in STORAGE_VALIDATION_COLUMNS:
            row.setdefault(column, "")
    frame = pd.DataFrame(records, columns=_ordered_columns(records, STORAGE_VALIDATION_COLUMNS))
    return _write_csv(frame, out_csv)


def write_timing_fixture(
    out_json: str | Path,
    *,
    timing_basis: str = "phase3_fixture_no_training",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    ""

    payload: dict[str, Any] = {
        "full_caster_bridge_update_sec": 0.0,
        "native_sidecar_save_overhead_sec": 0.0,
        "native_sidecar_load_overhead_sec": 0.0,
        "adapter_loglik_score_sec": 0.0,
        "native_posterior_update_sec": 0.0,
        "adapter_native_total_sec": 0.0,
        "full_rerun_wallclock_sec": 0.0,
        "timing_basis": timing_basis,
        "schema_version": SCHEMA_VERSION,
        "software_version": default_software_version(),
        "generated_at_utc": _utc_now(),
    }
    if extra:
        payload.update(dict(extra))
    _write_json(out_json, payload)
    return Path(out_json)


def _persist_checkpoint(
    source: Path,
    *,
    out_root: str | Path,
    model_id: str,
    origin: str,
    sidecar_type: str,
    retention_policy: str,
    score_reproducibility_level: str,
) -> NativeSidecarRecord:
    if not source.exists():
        raise NativeSidecarPathError(f"checkpoint sidecar path does not exist: {source}")
    sidecar_hash = sha256_artifact(source)
    sidecar_dir = _sidecar_dir(out_root, origin, model_id)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix if source.is_file() else ""
    artifact_name = f"checkpoint_{sidecar_hash[:12]}{suffix}" if source.is_file() else f"checkpoint_{sidecar_hash[:12]}"
    artifact_path = sidecar_dir / artifact_name
    if not artifact_path.exists():
        if source.is_dir() and not source.is_symlink():
            shutil.copytree(source, artifact_path)
        else:
            shutil.copy2(source, artifact_path)
    manifest_path = sidecar_dir / "sidecar_manifest.json"
    software = default_software_version()
    _write_json(
        manifest_path,
        _manifest(
            model_id=model_id,
            origin=origin,
            sidecar_type=sidecar_type,
            sidecar_hash=sidecar_hash,
            software_version=software,
            retention_policy=retention_policy,
            artifact_path=artifact_path,
            score_reproducibility_level=score_reproducibility_level,
        ),
    )
    return NativeSidecarRecord(
        schema_version=SCHEMA_VERSION,
        model_id=model_id,
        origin=origin,
        sidecar_type=sidecar_type,
        sidecar_hash=sidecar_hash,
        software_version=software,
        retention_policy=retention_policy,
        artifact_path=str(artifact_path),
        manifest_path=str(manifest_path),
        persistent_size_mb=path_size_mb(sidecar_dir),
        temporary_size_mb=0.0,
        deleted_after_scoring=False,
        score_reproducibility_level=score_reproducibility_level,
    )


def _manifest(
    *,
    model_id: str,
    origin: str,
    sidecar_type: str,
    sidecar_hash: str,
    software_version: str,
    retention_policy: str,
    artifact_path: str | Path,
    score_reproducibility_level: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "origin": origin,
        "sidecar_type": sidecar_type,
        "sidecar_hash": sidecar_hash,
        "software_version": software_version,
        "retention_policy": retention_policy,
        "artifact_path": str(artifact_path),
        "score_reproducibility_level": score_reproducibility_level,
        "created_at_utc": _utc_now(),
    }


def _validate_deletable_temp_path(target: Path, temp_root: Path) -> None:
    if not target.exists():
        raise NativeSidecarPathError(f"temporary sidecar path does not exist: {target}")
    if target == temp_root:
        raise NativeSidecarPathError("refusing to delete the temp root itself")
    if not _is_relative_to(target, temp_root):
        raise NativeSidecarPathError(f"refusing to delete non-temp sidecar path outside temp_root: {target}")
    if PERSISTENT_DIR_NAME in target.parts:
        raise NativeSidecarPathError(f"refusing to delete persistent {PERSISTENT_DIR_NAME} path: {target}")


def _require_retention_policy(policy: str) -> None:
    if policy not in RETENTION_POLICIES:
        raise NativeSidecarPolicyError(f"unknown native sidecar retention policy: {policy}")


def _validate_status_and_support(status: object, supports_native_log_likelihood: bool, *, label: str) -> None:
    status_text = str(status)
    if status_text not in NATIVE_LIKELIHOOD_STATUSES:
        raise NativeSidecarPolicyError(f"{label} has unknown native_likelihood_status={status_text!r}")
    if status_text == STATUS_TRUE_NATIVE and not supports_native_log_likelihood:
        raise NativeSidecarPolicyError(f"{label} true_native status requires supports_native_log_likelihood=True")
    if status_text != STATUS_TRUE_NATIVE and supports_native_log_likelihood:
        raise NativeSidecarPolicyError(
            f"{label} status {status_text!r} must not be marked as successful adapter-native likelihood support"
        )


def _sidecar_dir(out_root: str | Path, origin: str, model_id: str) -> Path:
    return Path(out_root) / PERSISTENT_DIR_NAME / _safe_component(origin) / _safe_component(model_id)


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _default_deletion_log_path(out_root: str | Path, origin: str, model_id: str) -> Path:
    return (
        Path(out_root)
        / "deletion_logs"
        / _safe_component(origin)
        / _safe_component(model_id)
        / f"deleted_{_utc_now().replace(':', '').replace('-', '')}.json"
    )


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def _write_csv(frame: pd.DataFrame, out_csv: str | Path) -> Path:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _row_to_dict(row: NativeAvailabilityRecord | NativeSidecarStorageValidationRow | Mapping[str, Any]) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    return dict(row)


def _ordered_columns(records: list[dict[str, Any]], required: list[str]) -> list[str]:
    extras = sorted({column for row in records for column in row} - set(required))
    return [*required, *extras]


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
