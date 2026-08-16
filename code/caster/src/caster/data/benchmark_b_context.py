""












from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


CONTRACT_VERSION = "caster_benchmark_b_context_v26_1"
CONTEXT_SCHEMA = "caster_benchmark_b_canonical_context_v1"
ADAPTER_INPUT_SCHEMA = "caster_benchmark_b_adapter_input_v1"
POSTERIOR_BATCH_SCHEMA = "caster_benchmark_b_posterior_update_batch_v1"
RELEASE_TIME_PREFIX = "__release_time__"
MISSING_MASK_SUFFIX = "__missing_mask"


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _summary(values: Sequence[object]) -> dict[str, object]:
    arr = np.asarray([number for value in values if (number := _finite(value)) is not None])
    if not len(arr):
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _records_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    records: list[list[object]] = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        record: list[object] = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                record.append(_date(value))
            elif value is None or pd.isna(value):
                record.append(None)
            elif isinstance(value, (bool, np.bool_)):
                record.append(bool(value))
            elif isinstance(value, (float, np.floating)):
                record.append(float(value).hex())
            elif isinstance(value, (int, np.integer)):
                record.append(int(value))
            else:
                record.append(str(value))
        records.append(record)
    return sha256_json(records)


@dataclass(frozen=True)
class BenchmarkBContextContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str
    stream_columns: Mapping[str, tuple[str, ...]]

    @property
    def task(self) -> Mapping[str, Any]:
        return self.payload["task"]


@dataclass(frozen=True)
class BenchmarkBAdapterInput:
    schema: str
    task_id: str
    forecast_origin: str
    values: pd.DataFrame
    missing_mask: pd.DataFrame
    release_times: pd.DataFrame
    visible_panel_sha256: str
    contract_sha256: str


@dataclass(frozen=True)
class PosteriorUpdateBatch:
    schema: str
    task_id: str
    previous_update_time: str | None
    current_update_time: str
    released_events: pd.DataFrame
    archived_forecasts: pd.DataFrame
    released_event_sha256: str


def _resolve_stream_columns(panel_columns: Sequence[str], payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    columns = tuple(map(str, panel_columns))
    claimed: set[str] = set()
    resolved: dict[str, tuple[str, ...]] = {}
    for group, spec in payload["streams"].items():
        selected = set(map(str, spec.get("columns", [])))
        selected.update(
            column
            for prefix in map(str, spec.get("column_prefixes", []))
            for column in columns
            if column.startswith(prefix)
        )
        selected.update(
            column
            for pattern in map(str, spec.get("column_patterns", []))
            for column in columns
            if pattern in column
        )
        missing = selected - set(columns)
        _require(not missing, f"Benchmark B panel lacks configured {group} columns: {sorted(missing)}")
        overlap = claimed & selected
        _require(not overlap, f"Benchmark B stream columns overlap: {sorted(overlap)}")
        claimed.update(selected)
        resolved[str(group)] = tuple(sorted(selected))
    return resolved


def load_context_contract(path: str | Path, *, panel_columns: Sequence[str]) -> BenchmarkBContextContract:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "Benchmark B context contract must be a mapping")
    _require(payload.get("schema_version") == 1, "Benchmark B context schema_version drift")
    _require(payload.get("contract_version") == CONTRACT_VERSION, "Benchmark B context version drift")
    task = payload.get("task", {})
    _require(task.get("task_id") == "benchmark_b_pooled", "Benchmark B must be one pooled task")
    _require(tuple(task.get("target_components", [])) == ("covid_adm_per100k", "flu_adm_per100k"), "Benchmark B component drift")
    _require(task.get("direct_horizons") == [1, 2], "Benchmark B direct horizon drift")
    _require(task.get("rollout_horizons") == [1, 2, 3, 4], "Benchmark B rollout horizon drift")
    _require(task.get("release_lag_steps") == 1, "Benchmark B release lag must be one")
    _require(task.get("posterior_scope") == "pooled_shared_posterior" and task.get("posterior_count") == 1, "Benchmark B must use one shared posterior")
    _require(payload.get("causal_order", {}).get("posterior_current_x_forbidden") is True, "posterior update must forbid current x")
    canonical_payload = json.loads(canonical_json(payload))
    return BenchmarkBContextContract(
        payload=canonical_payload,
        source_path=source,
        sha256=sha256_json(canonical_payload),
        stream_columns=_resolve_stream_columns(panel_columns, canonical_payload),
    )


def annotate_stream_release_times(panel: pd.DataFrame, contract: BenchmarkBContextContract) -> pd.DataFrame:
    task = contract.task
    entity_col, time_col = str(task["entity_column"]), str(task["time_column"])
    missing = {entity_col, time_col} - set(panel.columns)
    _require(not missing, f"Benchmark B panel lacks keys: {sorted(missing)}")
    out = panel.copy()
    out[entity_col] = out[entity_col].astype(str)
    out[time_col] = pd.to_datetime(out[time_col], errors="raise")
    _require(not out.duplicated([entity_col, time_col]).any(), "Benchmark B panel has duplicate entity/time rows")
    for group, spec in contract.payload["streams"].items():
        out[f"{RELEASE_TIME_PREFIX}{group}"] = out[time_col] + pd.Timedelta(days=int(spec["release_lag_days"]))
    return out.sort_values([entity_col, time_col], kind="mergesort").reset_index(drop=True)


def materialize_adapter_panel(
    panel: pd.DataFrame, contract: BenchmarkBContextContract
) -> pd.DataFrame:
    ""







    out = annotate_stream_release_times(panel, contract)
    out["__release_time__"] = out[f"{RELEASE_TIME_PREFIX}target"]
    mask_columns: dict[str, pd.Series] = {}
    for columns in contract.stream_columns.values():
        for column in columns:
            mask_columns[f"{column}{MISSING_MASK_SUFFIX}"] = out[column].isna()
    out = out.drop(columns=list(mask_columns), errors="ignore")
    out = pd.concat([out, pd.DataFrame(mask_columns, index=out.index)], axis=1)
    out.attrs.update(
        {
            "benchmark_b_context_contract_sha256": contract.sha256,
            "benchmark_b_adapter_input_schema": ADAPTER_INPUT_SCHEMA,
            "benchmark_b_feature_cutoff": "feature_release_time_lte_forecast_origin",
        }
    )
    return out


def build_adapter_input(
    panel: pd.DataFrame,
    *,
    forecast_origin: object,
    contract: BenchmarkBContextContract,
) -> BenchmarkBAdapterInput:
    ""





    origin = pd.Timestamp(forecast_origin)
    task = contract.task
    entity_col, time_col = str(task["entity_column"]), str(task["time_column"])
    annotated = annotate_stream_release_times(panel, contract)
    annotated = annotated[annotated[time_col].le(origin)].copy()
    values = annotated[[entity_col, time_col]].copy()
    missing_mask = annotated[[entity_col, time_col]].copy()
    release_times = annotated[[entity_col, time_col]].copy()
    for group, columns in contract.stream_columns.items():
        release_col = f"{RELEASE_TIME_PREFIX}{group}"
        visible = annotated[release_col].le(origin)
        release_times[group] = annotated[release_col]
        for column in columns:
            numeric_or_text = annotated[column].where(visible)
            values[column] = numeric_or_text
            missing_mask[column] = ~visible | annotated[column].isna()
    hash_frame = pd.concat(
        [
            values.add_prefix("value:"),
            missing_mask.drop(columns=[entity_col, time_col]).add_prefix("missing:"),
            release_times.drop(columns=[entity_col, time_col]).add_prefix("release:"),
        ],
        axis=1,
    )
    hash_frame = hash_frame.sort_values([f"value:{entity_col}", f"value:{time_col}"], kind="mergesort")
    digest = _records_hash(hash_frame, list(hash_frame.columns))
    return BenchmarkBAdapterInput(
        schema=ADAPTER_INPUT_SCHEMA,
        task_id=str(task["task_id"]),
        forecast_origin=_date(origin),
        values=values,
        missing_mask=missing_mask,
        release_times=release_times,
        visible_panel_sha256=digest,
        contract_sha256=contract.sha256,
    )


def _released_label_events(ledger: pd.DataFrame, *, cutoff: pd.Timestamp, after: pd.Timestamp | None = None) -> pd.DataFrame:
    required = {
        "task_id", "entity_id", "component", "target_time", "release_time",
        "observed_value", "observed_mask", "revision_version",
    }
    _require(required <= set(ledger.columns), f"Benchmark B ledger lacks release fields: {sorted(required - set(ledger.columns))}")
    rows = ledger.copy()
    rows["target_time"] = pd.to_datetime(rows["target_time"], errors="raise")
    rows["release_time"] = pd.to_datetime(rows["release_time"], errors="raise")
    mask = rows["release_time"].le(cutoff)
    if after is not None:
        mask &= rows["release_time"].gt(after)
    rows = rows[mask].copy()
    identity = ["task_id", "entity_id", "component", "target_time", "revision_version"]
    consistency = rows.groupby(identity, dropna=False).agg(
        release_count=("release_time", "nunique"),
        value_count=("observed_value", "nunique"),
        mask_count=("observed_mask", "nunique"),
    )
    _require(
        consistency.empty or bool((consistency[["release_count", "value_count", "mask_count"]] <= 1).all().all()),
        "duplicate forecast rows disagree on released label event",
    )
    return rows.sort_values(identity + ["release_time"], kind="mergesort").drop_duplicates(identity, keep="last")


def released_event_hash(events: pd.DataFrame) -> str:
    columns = [
        "task_id", "entity_id", "component", "target_time", "release_time",
        "observed_value", "observed_mask", "revision_version",
    ]
    if events.empty:
        return sha256_json([])
    ordered = events.sort_values(columns[:5], kind="mergesort")
    return _records_hash(ordered, columns)


def _component_summary(adapter_input: BenchmarkBAdapterInput, contract: BenchmarkBContextContract) -> dict[str, object]:
    task = contract.task
    entity_col, time_col = str(task["entity_column"]), str(task["time_column"])
    result: dict[str, object] = {}
    for component in task["target_components"]:
        visible = adapter_input.values[[entity_col, time_col, component]].dropna(subset=[component])
        per_entity_last = (
            visible.sort_values(time_col).groupby(entity_col, sort=True)[component].last()
            if not visible.empty else pd.Series(dtype=float)
        )
        result[str(component)] = {
            "history_scope": "all_released_no_later_than_cutoff",
            "entity_count": int(visible[entity_col].nunique()),
            "visible_value_count": int(len(visible)),
            "history_start": None if visible.empty else _date(visible[time_col].min()),
            "history_end": None if visible.empty else _date(visible[time_col].max()),
            "value_summary": _summary(visible[component].tolist()),
            "per_entity_last_value_summary": _summary(per_entity_last.tolist()),
        }
    return result


def _stream_summary(adapter_input: BenchmarkBAdapterInput, contract: BenchmarkBContextContract) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for group, columns in contract.stream_columns.items():
        values = adapter_input.values[list(columns)]
        releases = pd.to_datetime(adapter_input.release_times[group], errors="raise")
        summaries[group] = {
            "feature_count": int(len(columns)),
            "visible_row_count": int(releases.le(pd.Timestamp(adapter_input.forecast_origin)).sum()),
            "visible_nonmissing_cell_count": int(values.notna().sum().sum()),
            "visible_missing_cell_count": int(adapter_input.missing_mask[list(columns)].sum().sum()),
            "latest_visible_release_time": None if releases.empty else _date(releases[releases.le(pd.Timestamp(adapter_input.forecast_origin))].max()),
            "release_lag_days": int(contract.payload["streams"][group]["release_lag_days"]),
            "release_time_status": str(contract.payload["streams"][group]["release_time_status"]),
            "formal_status": str(contract.payload["streams"][group]["formal_status"]),
        }
    return summaries


def _cross_stream_summary(adapter_input: BenchmarkBAdapterInput, contract: BenchmarkBContextContract) -> dict[str, object]:
    task = contract.task
    entity_col, time_col = str(task["entity_column"]), str(task["time_column"])
    left, right = map(str, task["target_components"])
    correlations: list[float] = []
    for _, group in adapter_input.values[[entity_col, time_col, left, right]].groupby(entity_col, sort=True):
        aligned = group[[left, right]].dropna()
        if len(aligned) >= 3 and bool((aligned.std(ddof=0) > 0).all()):
            correlations.append(float(aligned.corr().iloc[0, 1]))
    return {
        "target_component_entity_correlation": _summary(correlations),
        "stream_availability": _stream_summary(adapter_input, contract),
    }


def _selection_descriptor(manifest: Mapping[str, Any] | None, manifest_sha256: str) -> dict[str, object]:
    if manifest is None:
        return {
            "status": "not_bound",
            "frozen_selection_packet_sha256": manifest_sha256,
            "candidate_registry_sha256": "",
            "candidate_rows_sha256": "",
            "validation_fields": [],
        }
    source_hashes = manifest.get("source_sha256", {})
    return {
        "status": str(manifest.get("status", "")),
        "frozen_selection_packet_sha256": str(manifest_sha256),
        "candidate_registry_sha256": str(source_hashes.get("source_registry", "")),
        "candidate_rows_sha256": str(manifest.get("candidate_rows_sha256", "")),
        "candidate_count": int(manifest.get("candidate_count", 0)),
        "validation_fields": list(manifest.get("candidate_fields", [])),
        "selection_context_sha256": str(source_hashes.get("selection_context", "")),
        "validation_summary_sha256": str(source_hashes.get("validation_summary", "")),
        "t_sel": str(manifest.get("t_sel", "")),
    }


def build_canonical_context(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    forecast_origin: object,
    contract: BenchmarkBContextContract,
    frozen_selection_manifest: Mapping[str, Any] | None = None,
    frozen_selection_manifest_sha256: str = "",
) -> tuple[dict[str, object], str]:
    ""

    origin = pd.Timestamp(forecast_origin)
    task = contract.task
    task_id = str(task["task_id"])
    _require(set(ledger["task_id"].astype(str)) == {task_id}, "ledger is not the pooled Benchmark B task")
    adapter_input = build_adapter_input(panel, forecast_origin=origin, contract=contract)
    released = _released_label_events(ledger, cutoff=origin)
    query = ledger[pd.to_datetime(ledger["forecast_origin"], errors="raise").eq(origin)]
    _require(not query.empty, f"Benchmark B ledger has no forecast rows at {_date(origin)}")
    _require(pd.to_datetime(query["features_available_until"], errors="raise").le(origin).all(), "query exposes post-origin features")
    _require(pd.to_datetime(released["release_time"], errors="raise").le(origin).all(), "context exposes unreleased D_t")
    selection = _selection_descriptor(frozen_selection_manifest, frozen_selection_manifest_sha256)
    payload: dict[str, object] = {
        "schema": CONTEXT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "context_contract_sha256": contract.sha256,
        "task_id": task_id,
        "component_scope": list(task["target_components"]),
        "posterior_scope": str(task["posterior_scope"]),
        "posterior_count": int(task["posterior_count"]),
        "task_query": str(task["query"]),
        "forecast_origin": _date(origin),
        "causal_cutoff_time": _date(origin),
        "origin_query_forecast_count": int(len(query)),
        "origin_query_forecast_ids_sha256": sha256_json(sorted(query["forecast_id"].astype(str))),
        "history_summary": _component_summary(adapter_input, contract),
        "cross_stream_summary": _cross_stream_summary(adapter_input, contract),
        "visible_panel_sha256": adapter_input.visible_panel_sha256,
        "released_event_count": int(len(released)),
        "released_event_sha256": released_event_hash(released),
        "frozen_selection": selection,
        "causal_rules": {
            "feature_cutoff": "feature_release_time_lte_forecast_origin",
            "released_event_cutoff": "release_time_lte_forecast_origin",
            "router_raw_sequence_visible": False,
            "future_panel_visible": False,
            "future_released_event_visible": False,
            "posterior_current_x_forbidden": True,
        },
    }
    context_sha256 = sha256_json(payload)
    return payload, context_sha256


def build_router_context(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    contract: BenchmarkBContextContract,
    frozen_selection_manifest: Mapping[str, Any] | None = None,
    frozen_selection_manifest_sha256: str = "",
) -> tuple[dict[str, object], str]:
    ""

    return build_canonical_context(
        panel,
        ledger,
        forecast_origin=contract.task["selection_freeze_time"],
        contract=contract,
        frozen_selection_manifest=frozen_selection_manifest,
        frozen_selection_manifest_sha256=frozen_selection_manifest_sha256,
    )


def context_binding(
    payload: Mapping[str, object],
    context_sha256: str,
) -> dict[str, object]:
    return {
        "task_id": payload["task_id"],
        "forecast_origin": payload["forecast_origin"],
        "canonical_context_sha256": str(context_sha256),
        "visible_panel_sha256": payload["visible_panel_sha256"],
        "released_event_sha256": payload["released_event_sha256"],
        "frozen_selection_packet_sha256": payload["frozen_selection"]["frozen_selection_packet_sha256"],
        "forecast_origin_binding_sha256": sha256_json(
            {
                "task_id": payload["task_id"],
                "forecast_origin": payload["forecast_origin"],
                "canonical_context_sha256": context_sha256,
            }
        ),
    }


def build_posterior_update_batch(
    forecast_archive: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    current_update_time: object,
    previous_update_time: object | None = None,
    task_id: str = "benchmark_b_pooled",
) -> PosteriorUpdateBatch:
    ""

    current = pd.Timestamp(current_update_time)
    previous = None if previous_update_time is None else pd.Timestamp(previous_update_time)
    _require("forecast_id" in forecast_archive.columns, "frozen archive lacks forecast_id")
    archive_identity = ["forecast_id"]
    if "model_id" in forecast_archive.columns:
        archive_identity.append("model_id")
    if "particle_id" in forecast_archive.columns:
        archive_identity.append("particle_id")
    _require(
        not forecast_archive.duplicated(archive_identity).any(),
        f"frozen archive identity is not unique: {archive_identity}",
    )
    ledger_rows = ledger.copy()
    ledger_rows["release_time"] = pd.to_datetime(ledger_rows["release_time"], errors="raise")
    release_mask = ledger_rows["release_time"].le(current)
    if previous is not None:
        release_mask &= ledger_rows["release_time"].gt(previous)
    released_forecast_rows = ledger_rows[release_mask].copy()
    archive = forecast_archive.copy()
    archive["forecast_id"] = archive["forecast_id"].astype(str)
    released_forecast_rows["forecast_id"] = released_forecast_rows["forecast_id"].astype(str)
    joined = released_forecast_rows.merge(
        archive,
        on="forecast_id",
        how="left",
        validate="one_to_many",
        suffixes=("_ledger", "_archive"),
        indicator=True,
    )
    _require(joined["_merge"].eq("both").all(), "frozen archive lacks a released forecast row")
    joined = joined.drop(columns="_merge")
    events = _released_label_events(ledger, cutoff=current, after=previous)
    return PosteriorUpdateBatch(
        schema=POSTERIOR_BATCH_SCHEMA,
        task_id=str(task_id),
        previous_update_time=None if previous is None else _date(previous),
        current_update_time=_date(current),
        released_events=events,
        archived_forecasts=joined,
        released_event_sha256=released_event_hash(events),
    )


def input_fingerprint(
    *,
    contract_sha256: str,
    panel_sha256: str,
    ledger_sha256: str,
    frozen_selection_packet_sha256: str,
    candidate_registry_sha256: str,
) -> str:
    return sha256_json(
        {
            "context_contract_sha256": contract_sha256,
            "panel_sha256": panel_sha256,
            "ledger_sha256": ledger_sha256,
            "frozen_selection_packet_sha256": frozen_selection_packet_sha256,
            "candidate_registry_sha256": candidate_registry_sha256,
        }
    )


def require_matching_input_fingerprint(manifest: Mapping[str, object], expected: str) -> None:
    _require(
        str(manifest.get("benchmark_b_input_fingerprint", "")) == str(expected),
        "Benchmark B input changed; old forecast/posterior/Agent artifact reuse is forbidden",
    )
