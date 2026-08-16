from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

from .spec import TaskSpec, filter_rows_to_task_spec


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def selection_fold_manifest_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(frame.columns)
    normalized = frame[columns].fillna("").astype(str).sort_values(columns).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_fold_declaration(path: str | Path, task_id: str) -> dict[str, object]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if int(payload.get("version", 0)) != 20:
        raise ValueError("selection-fold config must declare version: 20")
    if task_id not in payload.get("tasks", {}):
        raise ValueError(f"selection-fold config missing task {task_id!r}")
    declaration = dict(payload["tasks"][task_id])
    declaration["selection_fold_config_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return declaration


def materialize_selection_folds(
    ledger: pd.DataFrame,
    spec: TaskSpec,
    declaration: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "forecast_id", "natural_event_id", "dataset", "entity_id", "component", "forecast_strategy",
        "horizon", "forecast_origin", "target_time", "release_time", "revision_version", "split",
        "observed_value", "observed_mask",
    }
    if missing := sorted(required - set(ledger.columns)):
        raise ValueError(f"ledger missing selection-fold columns {missing}")
    rows = filter_rows_to_task_spec(ledger, spec, require_complete=True)
    for column in ("forecast_origin", "target_time", "release_time"):
        rows[column] = pd.to_datetime(rows[column], errors="raise")
    if str(declaration["t_sel"]) != spec.t_sel or str(declaration["first_test_origin"]) != spec.t_test:
        raise ValueError("selection-fold declaration cutoff disagrees with TaskSpec")
    first_origin = pd.Timestamp(str(declaration["first_origin"]))
    last_origin = pd.Timestamp(str(declaration["last_origin"]))
    selected = rows[
        rows["split"].astype(str).eq(str(declaration["source_split"]))
        & rows["forecast_origin"].between(first_origin, last_origin, inclusive="both")
        & rows["release_time"].le(pd.Timestamp(spec.t_sel))
    ].copy()
    if selected.empty:
        raise ValueError(f"no eligible pretest selection rows for {spec.task_id}")
    if set(selected["split"].astype(str)) != {"val"}:
        raise ValueError("formal selection folds must contain validation rows only")
    if (selected["forecast_origin"] >= pd.Timestamp(spec.t_test)).any() or (selected["release_time"] >= pd.Timestamp(spec.t_test)).any():
        raise ValueError("selection fold contains test-or-later origin or label release")
    origins = sorted(selected["forecast_origin"].unique())
    if len(origins) != int(declaration["expected_folds"]):
        raise ValueError(f"declared fold count mismatch for {spec.task_id}: {len(origins)} != {declaration['expected_folds']}")
    fold_by_origin = {pd.Timestamp(origin): f"fold_{idx + 1:03d}" for idx, origin in enumerate(origins)}
    selected["task_id"] = spec.task_id
    selected["fold_id"] = selected["forecast_origin"].map(fold_by_origin)
    for fold_id, group in selected.groupby("fold_id", sort=True):
        filter_rows_to_task_spec(group, spec, require_complete=True)
    selected["label_event_id"] = [
        _stable_id("label", row.dataset, row.entity_id, row.component, pd.Timestamp(row.target_time).date(), row.revision_version)
        for row in selected.itertuples(index=False)
    ]
    selected["t_sel"] = spec.t_sel
    selected["first_test_origin"] = spec.t_test
    selected["task_spec_sha256"] = spec.task_spec_sha256
    selected["selection_fold_config_sha256"] = str(declaration["selection_fold_config_sha256"])
    selected = selected.sort_values(["forecast_origin", "forecast_id"]).reset_index(drop=True)
    selected["fold_manifest_sha256"] = ""
    digest = selection_fold_manifest_sha256(selected.drop(columns=["fold_manifest_sha256"]))
    selected["fold_manifest_sha256"] = digest
    validation = pd.DataFrame([
        {
            "task_id": spec.task_id,
            "fold_id": fold_id,
            "forecast_origin": group["forecast_origin"].min().strftime("%Y-%m-%d"),
            "max_label_release_time": group["release_time"].max().strftime("%Y-%m-%d"),
            "first_test_origin": spec.t_test,
            "n_forecast_ids": int(group["forecast_id"].nunique()),
            "test_rows": 0,
            "release_before_test": True,
            "validation_status": "PASS",
            "fold_manifest_sha256": digest,
        }
        for fold_id, group in selected.groupby("fold_id", sort=True)
    ])
    return selected, validation
