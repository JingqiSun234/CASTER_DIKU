from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .spec import TaskSpec
from caster.data.benchmark_b_context import (
    build_canonical_context as build_benchmark_b_canonical_context,
    load_context_contract as load_benchmark_b_context_contract,
)
from caster.data.benchmark_a_mobility import (
    MOBILITY_FEATURE_COLUMNS,
    MOBILITY_RELEASE_COLUMN,
    MOBILITY_SCHEMA,
    materialize_mobility_features,
)


_ROOT = Path(__file__).resolve().parents[5]
_BENCHMARK_A_MOBILITY_ROOT = _ROOT / "data/benchmark_a/raw_all"
_BENCHMARK_B_CONTEXT_CONTRACT = _ROOT / "configs/benchmark_b_context_v26_1.yaml"


def _finite(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(arr.size),
        "mean": _finite(np.mean(arr)),
        "median": _finite(np.median(arr)),
        "q25": _finite(np.quantile(arr, 0.25)),
        "q75": _finite(np.quantile(arr, 0.75)),
    }


def _resolve_cutoff(spec: TaskSpec, cutoff_time: object | None) -> pd.Timestamp:
    cutoff = pd.Timestamp(spec.t_sel if cutoff_time is None else cutoff_time)
    if pd.isna(cutoff):
        raise ValueError(f"invalid selection context cutoff for {spec.task_id}: {cutoff_time!r}")
    return cutoff


def _benchmark_a_context_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ""










    join_columns = {"country", "country_code", "entity_id", "date"}
    if not join_columns.issubset(panel.columns):
                                                                             
                                                                      
        return panel.copy(), {
            "status": "not_applicable_non_graph_node_fixture",
            "schema": MOBILITY_SCHEMA,
            "feature_columns": list(MOBILITY_FEATURE_COLUMNS),
        }

    metadata_keys = {
        "benchmark_a_mobility_schema",
        "benchmark_a_mobility_graph_set_sha256",
        "benchmark_a_mobility_release_policy",
        "benchmark_a_mobility_representation",
        "benchmark_a_mobility_feature_columns",
        "benchmark_a_mobility_future_graph_access",
    }
    has_features = set(MOBILITY_FEATURE_COLUMNS).issubset(panel.columns) and MOBILITY_RELEASE_COLUMN in panel.columns
    has_metadata = metadata_keys.issubset(panel.attrs)
    if has_features and has_metadata:
        enriched = panel.copy()
        metadata = dict(panel.attrs)
    else:
        derived_columns = set(MOBILITY_FEATURE_COLUMNS) | {MOBILITY_RELEASE_COLUMN}
        derived_columns |= {f"{column}__missing_mask" for column in MOBILITY_FEATURE_COLUMNS}
        base = panel.drop(columns=sorted(derived_columns & set(panel.columns))).copy()
        materialized = materialize_mobility_features(base, _BENCHMARK_A_MOBILITY_ROOT)
        enriched = materialized.panel
        metadata = dict(materialized.metadata)

    if str(metadata.get("benchmark_a_mobility_schema", "")) != MOBILITY_SCHEMA:
        raise ValueError("Benchmark A canonical context received an invalid mobility schema")
    if bool(metadata.get("benchmark_a_mobility_future_graph_access", True)):
        raise ValueError("Benchmark A canonical context must not permit future graph access")
    return enriched, metadata


def _benchmark_a_mobility_context(
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    if str(metadata.get("status", "")) == "not_applicable_non_graph_node_fixture":
        return metadata

    visible = panel.copy()
    visible["date"] = pd.to_datetime(visible["date"], errors="raise")
    visible[MOBILITY_RELEASE_COLUMN] = pd.to_datetime(
        visible[MOBILITY_RELEASE_COLUMN], errors="raise"
    )
    visible = visible[
        visible["date"].le(cutoff)
        & visible[MOBILITY_RELEASE_COLUMN].le(cutoff)
    ].copy()
    if visible.empty:
        raise ValueError("Benchmark A canonical context has no released mobility rows")
    if visible[MOBILITY_RELEASE_COLUMN].max() > cutoff:
        raise ValueError("Benchmark A canonical context leaked a future mobility graph")

    feature_statistics: dict[str, Any] = {}
    for column in MOBILITY_FEATURE_COLUMNS:
        values = pd.to_numeric(visible[column], errors="coerce")
        feature_statistics[column] = _summary(values.dropna().astype(float).tolist())
    return {
        "status": "available",
        "schema": str(metadata["benchmark_a_mobility_schema"]),
        "graph_set_sha256": str(metadata["benchmark_a_mobility_graph_set_sha256"]),
        "release_policy": str(metadata["benchmark_a_mobility_release_policy"]),
        "representation": str(metadata["benchmark_a_mobility_representation"]),
        "feature_columns": list(MOBILITY_FEATURE_COLUMNS),
        "history_start": str(visible["date"].min().date()),
        "history_end": str(visible["date"].max().date()),
        "latest_release_time": str(visible[MOBILITY_RELEASE_COLUMN].max().date()),
        "entity_count": int(visible["entity_id"].astype(str).nunique()),
        "row_count": int(len(visible)),
        "feature_statistics": feature_statistics,
        "future_graph_access": False,
    }


def _to_long_panel(
    panel: pd.DataFrame,
    spec: TaskSpec,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if spec.dataset == "benchmark_a":
        required = {"entity_id", "date", "component", "observed_value"}
        if missing := sorted(required - set(panel.columns)):
            raise ValueError(f"Benchmark A panel missing columns {missing}")
        source = panel[pd.to_datetime(panel["date"], errors="raise") <= cutoff].copy()
        frame = source[["entity_id", "date", "component", "observed_value"]].rename(
            columns={"date": "time", "observed_value": "value"}
        )
        excluded = required | {"dataset", "country", "country_code", "raw_region_id", "node_index"}
        excluded |= set(MOBILITY_FEATURE_COLUMNS) | {MOBILITY_RELEASE_COLUMN}
        excluded |= {f"{column}__missing_mask" for column in MOBILITY_FEATURE_COLUMNS}
    else:
        required = {"jurisdiction", "week_end", *spec.target_components}
        if missing := sorted(required - set(panel.columns)):
            raise ValueError(f"Benchmark B panel missing columns {missing}")
        source = panel[pd.to_datetime(panel["week_end"], errors="raise") <= cutoff].copy()
        frame = source.melt(
            id_vars=["jurisdiction", "week_end"],
            value_vars=list(spec.target_components),
            var_name="component",
            value_name="value",
        ).rename(columns={"jurisdiction": "entity_id", "week_end": "time"})
        excluded = required | {"jurisdiction_abbr"}
    covariates = {
        str(column): float(source[column].notna().mean())
        for column in sorted(set(source.columns) - excluded)
    }
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame["entity_id"] = frame["entity_id"].astype(str)
    frame["component"] = frame["component"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["component"].isin(spec.target_components) & frame["time"].le(cutoff)]
    return frame.sort_values(["component", "entity_id", "time"]), covariates


def _component_context(group: pd.DataFrame, seasonality_lag: int, history_scope: str) -> dict[str, Any]:
    per_entity: dict[str, list[float]] = {
        "observed_count": [],
        "missing_fraction": [],
        "mean_level": [],
        "last_value": [],
        "q25": [],
        "median": [],
        "q75": [],
        "linear_trend_per_step": [],
        "difference_volatility": [],
        "seasonal_correlation": [],
    }
    for _, entity in group.groupby("entity_id", sort=True):
        entity = entity.sort_values("time")
        values = entity["value"].astype(float)
        observed = values.dropna().to_numpy(dtype=float)
        per_entity["observed_count"].append(float(observed.size))
        per_entity["missing_fraction"].append(float(values.isna().mean()))
        if observed.size:
            per_entity["mean_level"].append(float(np.mean(observed)))
            per_entity["last_value"].append(float(observed[-1]))
            per_entity["q25"].append(float(np.quantile(observed, 0.25)))
            per_entity["median"].append(float(np.median(observed)))
            per_entity["q75"].append(float(np.quantile(observed, 0.75)))
        if observed.size >= 2:
            per_entity["linear_trend_per_step"].append(float(np.polyfit(np.arange(observed.size), observed, 1)[0]))
            per_entity["difference_volatility"].append(float(np.std(np.diff(observed), ddof=0)))
        if observed.size > seasonality_lag:
            left, right = observed[:-seasonality_lag], observed[seasonality_lag:]
            if np.std(left) > 0 and np.std(right) > 0:
                per_entity["seasonal_correlation"].append(float(np.corrcoef(left, right)[0, 1]))
    return {
        "history_scope": history_scope,
        "history_start": str(group["time"].min().date()),
        "history_end": str(group["time"].max().date()),
        "entity_count": int(group["entity_id"].nunique()),
        "row_count": int(len(group)),
        "observed_count": int(group["value"].notna().sum()),
        "missing_fraction_global": float(group["value"].isna().mean()),
        "entity_statistics": {name: _summary(values) for name, values in per_entity.items()},
        "seasonality_lag": int(seasonality_lag),
    }


def _release_context(ledger: pd.DataFrame, spec: TaskSpec, cutoff: pd.Timestamp) -> dict[str, Any]:
    required = {"entity_id", "component", "target_time", "release_time", "revision_version"}
    if missing := sorted(required - set(ledger.columns)):
        raise ValueError(f"ledger missing context release columns {missing}")
    rows = ledger[ledger["component"].astype(str).isin(spec.target_components)].copy()
    rows["target_time"] = pd.to_datetime(rows["target_time"], errors="raise")
    rows["release_time"] = pd.to_datetime(rows["release_time"], errors="raise")
    rows = rows[rows["release_time"].le(cutoff)]
    rows = rows.drop_duplicates(["entity_id", "component", "target_time", "revision_version"])
    lag = (rows["release_time"] - rows["target_time"]).dt.total_seconds() / 86400.0
    return {
        "released_label_events": int(len(rows)),
        "release_lag_days": _summary(lag.astype(float).tolist()),
        "revision_versions": sorted(rows["revision_version"].dropna().astype(str).unique().tolist()),
    }


def _cross_stream_context(frame: pd.DataFrame, history_scope: str) -> dict[str, Any]:
    correlations: list[float] = []
    for _, entity in frame.groupby("entity_id", sort=True):
        wide = entity.pivot_table(index="time", columns="component", values="value", aggfunc="first").sort_index()
        aligned = wide.dropna()
        if aligned.shape[1] == 2 and len(aligned) >= 3 and all(float(aligned[col].std(ddof=0)) > 0 for col in aligned):
            correlations.append(float(aligned.corr().iloc[0, 1]))
    return {
        "history_scope": history_scope,
        "entity_correlations": _summary(correlations),
    }


def selection_context_canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


def selection_context_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(selection_context_canonical_json(payload).encode("utf-8")).hexdigest()


def build_selection_context(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    spec: TaskSpec,
    *,
    cutoff_time: object | None = None,
    context_role: str = "formal_selection",
) -> tuple[dict[str, Any], str, pd.DataFrame]:
    cutoff = _resolve_cutoff(spec, cutoff_time)
    if spec.dataset == "benchmark_b":
        contract = load_benchmark_b_context_contract(
            _BENCHMARK_B_CONTEXT_CONTRACT, panel_columns=panel.columns
        )
        payload, context_sha = build_benchmark_b_canonical_context(
            panel,
            ledger,
            forecast_origin=cutoff,
            contract=contract,
        )
                                                                            
                                                                             
                                                                       
                                                                  
        payload = {
            **payload,
            "cutoff_time": str(cutoff.date()),
            "context_role": str(context_role),
            "history_scope": "all_released_no_later_than_cutoff",
            "selection_view_task_id": spec.task_id,
            "selection_view_query": spec.natural_language_query,
            "selection_view_target_components": list(spec.target_components),
            "selection_view_posterior_scope": spec.posterior_scope,
        }
        context_sha = selection_context_sha256(payload)
        text = (
            "CASTER canonical Benchmark B task context: structured causal summary\n"
            f"task_id: {spec.task_id}\n"
            f"query: {spec.natural_language_query}\n"
            f"cutoff_time: {cutoff.date()}\n"
            f"canonical_context_sha256: {context_sha}\n"
            f"context_json: {selection_context_canonical_json(payload).strip()}\n"
        )
        validation = pd.DataFrame(
            [
                {"task_id": spec.task_id, "check": "canonical_context_contract", "status": "PASS", "value": contract.sha256},
                {"task_id": spec.task_id, "check": "router_raw_sequence_visible", "status": "PASS", "value": "false"},
                {"task_id": spec.task_id, "check": "feature_release_lte_cutoff", "status": "PASS", "value": str(cutoff.date())},
                {"task_id": spec.task_id, "check": "released_event_lte_cutoff", "status": "PASS", "value": str(cutoff.date())},
                {"task_id": spec.task_id, "check": "test_metrics_excluded", "status": "PASS", "value": "true"},
            ]
        )
        return payload, text, validation
    benchmark_a_mobility: dict[str, Any] | None = None
    if spec.dataset == "benchmark_a":
        panel, mobility_metadata = _benchmark_a_context_panel(panel)
        benchmark_a_mobility = _benchmark_a_mobility_context(panel, mobility_metadata, cutoff)

    formal_cutoff = pd.Timestamp(spec.t_sel)
    is_formal_cutoff = cutoff == formal_cutoff
    history_scope = "all_available_before_t_sel" if is_formal_cutoff else "all_available_through_cutoff"
    frame, covariates = _to_long_panel(panel, spec, cutoff)
    if frame.empty:
        raise ValueError(f"no pre-cutoff panel rows for {spec.task_id}")
    lag = int(spec.initial_context_policy["seasonality_lag"])
    payload: dict[str, Any] = {
        "schema": "caster_selection_context_full_history_v2",
        "task_id": spec.task_id,
        "task_spec_sha256": spec.task_spec_sha256,
        "t_sel": spec.t_sel,
        "cutoff_time": str(cutoff.date()),
        "context_role": str(context_role),
        "canonical_task_query": spec.natural_language_query,
        "target_components": list(spec.target_components),
        "cadence": spec.cadence,
        "direct_horizons": list(spec.direct_horizons),
        "recursive_horizons": list(spec.recursive_horizons),
        "forecast_strategies": list(spec.forecast_strategies),
        "entity_scope": spec.entity_scope,
        "history_scope": history_scope,
        "component_summaries": {
            component: _component_context(frame[frame["component"].eq(component)], lag, history_scope)
            for component in spec.target_components
        },
        "covariate_availability": covariates,
        "release_revision_summary": _release_context(ledger, spec, cutoff),
    }
    if benchmark_a_mobility is not None:
        payload["causal_mobility_context"] = benchmark_a_mobility
    if spec.dataset == "benchmark_b":
        payload["precutoff_cross_stream_association"] = _cross_stream_context(frame, history_scope)
    canonical = selection_context_canonical_json(payload)
    text = (
        "CASTER canonical task context: full causal history\n"
        f"task_id: {spec.task_id}\n"
        f"query: {spec.natural_language_query}\n"
        f"t_sel: {spec.t_sel}\n"
        f"cutoff_time: {cutoff.date()}\n"
        f"context_role: {context_role}\n"
        f"context_json: {canonical.strip()}\n"
    )
    validation_rows = [
        {"task_id": spec.task_id, "check": "history_scope", "status": "PASS", "value": history_scope},
        {"task_id": spec.task_id, "check": "panel_time_lte_cutoff", "status": "PASS", "value": str(frame["time"].max().date())},
        {"task_id": spec.task_id, "check": "cutoff_time", "status": "PASS", "value": str(cutoff.date())},
        {"task_id": spec.task_id, "check": "test_metrics_excluded", "status": "PASS", "value": "true"},
        {"task_id": spec.task_id, "check": "post_cutoff_rows_excluded", "status": "PASS", "value": "true"},
    ]
    if benchmark_a_mobility is not None:
        validation_rows.extend(
            [
                {
                    "task_id": spec.task_id,
                    "check": "benchmark_a_mobility_context",
                    "status": "PASS",
                    "value": str(benchmark_a_mobility.get("status", "")),
                },
                {
                    "task_id": spec.task_id,
                    "check": "mobility_release_lte_cutoff",
                    "status": "PASS",
                    "value": str(benchmark_a_mobility.get("latest_release_time", "not_applicable")),
                },
                {
                    "task_id": spec.task_id,
                    "check": "mobility_future_graph_excluded",
                    "status": "PASS",
                    "value": "true",
                },
            ]
        )
    validation = pd.DataFrame(validation_rows)
    return payload, text, validation
