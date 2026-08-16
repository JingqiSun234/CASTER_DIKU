""






from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from caster.data import (
    annotate_stream_release_times,
    load_benchmark_b_context_contract,
)

from .spec import TaskSpec


SEQUENCE_SKETCH_SCHEMA = "caster_causal_multiscale_released_sequence_sketch_v1"
POINT_COLUMNS = (
    "period_start",
    "period_end",
    "timepoint_count",
    "cell_count",
    "observed_count",
    "entity_count",
    "observed_entity_count",
    "mean",
    "median",
    "q25",
    "q75",
    "min",
    "max",
    "zero_fraction",
    "missing_fraction",
)

_ROOT = Path(__file__).resolve().parents[5]
_BENCHMARK_B_CONTEXT_CONTRACT = _ROOT / "configs/benchmark_b_context_v26_1.yaml"

_GLOBAL_TIER_DESIGN: Mapping[str, Mapping[str, int]] = {
    "benchmark_a": {
        "recent_count": 28,
        "medium_span": 84,
        "medium_block": 7,
        "long_block": 28,
    },
    "benchmark_b": {
        "recent_count": 16,
        "medium_span": 52,
        "medium_block": 4,
        "long_block": 13,
    },
}
_BENCHMARK_A_GROUP_TIER_DESIGN: Mapping[str, int] = {
    "recent_count": 7,
    "medium_span": 84,
    "medium_block": 7,
    "long_block": 28,
}


def sequence_sketch_canonical_json(payload: Mapping[str, Any]) -> str:
    ""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def sequence_sketch_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        sequence_sketch_canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _round_number(value: object, digits: int) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else float(rounded)


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _quantile(values: np.ndarray, probability: float, digits: int) -> float | None:
    if not len(values):
        return None
    return _round_number(np.quantile(values, probability), digits)


def _records_sha256(frame: pd.DataFrame) -> str:
    columns = ["component", "entity_id", "time", "release_time", "value"]
    records: list[list[object]] = []
    ordered = frame.sort_values(columns[:-1], kind="mergesort")
    for component, entity, time, release, value in ordered[columns].itertuples(
        index=False, name=None
    ):
        number = _finite_float(value)
        records.append(
            [
                str(component),
                str(entity),
                _date(time),
                _date(release),
                None if number is None else float(number).hex(),
            ]
        )
    canonical = json.dumps(
        records, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _released_ledger_binding(
    ledger: pd.DataFrame,
    *,
    components: Sequence[str],
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    required = {"entity_id", "component", "target_time", "release_time"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"sequence-sketch ledger missing columns {missing}")
    rows = ledger[ledger["component"].astype(str).isin(components)].copy()
    rows["target_time"] = pd.to_datetime(rows["target_time"], errors="raise")
    rows["release_time"] = pd.to_datetime(rows["release_time"], errors="raise")
    if bool(rows["release_time"].lt(rows["target_time"]).any()):
        raise ValueError("sequence-sketch ledger has release_time before target_time")
    future_count = int(rows["release_time"].gt(cutoff).sum())
    rows = rows[rows["release_time"].le(cutoff)].copy()
    optional = [
        column
        for column in ("revision_version", "observed_mask", "observed_value")
        if column in rows.columns
    ]
    identity = ["entity_id", "component", "target_time", "release_time", *optional]
    rows = rows.sort_values(identity, kind="mergesort").drop_duplicates(identity)
    records: list[list[object]] = []
    for row in rows[identity].itertuples(index=False, name=None):
        encoded: list[object] = []
        for column, value in zip(identity, row):
            if column in {"target_time", "release_time"}:
                encoded.append(_date(value))
            elif column == "observed_value":
                number = _finite_float(value)
                encoded.append(None if number is None else float(number).hex())
            elif value is None or pd.isna(value):
                encoded.append(None)
            elif isinstance(value, (bool, np.bool_)):
                encoded.append(bool(value))
            else:
                encoded.append(str(value))
        records.append(encoded)
    canonical = json.dumps(
        records, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return {
        "released_event_count": int(len(rows)),
        "future_release_event_count_excluded": future_count,
        "max_released_event_time": (
            None if rows.empty else _date(rows["release_time"].max())
        ),
        "released_event_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


def _resolve_cutoff(
    spec: TaskSpec, cutoff_time: object | None
) -> tuple[pd.Timestamp, str]:
    formal = pd.Timestamp(spec.t_sel)
    cutoff = formal if cutoff_time is None else pd.Timestamp(cutoff_time)
    if pd.isna(cutoff):
        raise ValueError(f"invalid sequence-sketch cutoff {cutoff_time!r}")
    relation = (
        "before_formal_t_sel"
        if cutoff < formal
        else (
            "after_formal_t_sel_origin_context"
            if cutoff > formal
            else "formal_t_sel"
        )
    )
    return cutoff, relation


def _prepare_benchmark_a(
    panel: pd.DataFrame,
    spec: TaskSpec,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"entity_id", "date", "component", "observed_value"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Benchmark A sequence-sketch panel missing columns {missing}")
    rows = panel[
        panel["component"].astype(str).isin(spec.target_components)
    ].copy()
    rows["time"] = pd.to_datetime(rows["date"], errors="raise")
    release_candidates = (
        "__release_time__target",
        "__release_time__",
        "target_release_time",
    )
    release_column = next(
        (column for column in release_candidates if column in rows.columns), None
    )
    if release_column is None:
        rows["release_time"] = rows["time"]
        release_policy = "benchmark_a_same_day_target_release"
    else:
        rows["release_time"] = pd.to_datetime(
            rows[release_column], errors="raise"
        )
        release_policy = f"explicit_panel_column:{release_column}"
    if bool(rows["release_time"].lt(rows["time"]).any()):
        raise ValueError("Benchmark A target release precedes target date")
    rows["entity_id"] = rows["entity_id"].astype(str)
    rows["component"] = rows["component"].astype(str)
    rows["value"] = pd.to_numeric(rows["observed_value"], errors="coerce")
    if "country" in rows.columns:
        rows["group_id"] = rows["country"].astype(str)
    duplicate = rows.duplicated(["component", "entity_id", "time"])
    if bool(duplicate.any()):
        raise ValueError("Benchmark A sequence-sketch panel has duplicate target cells")
    total_count = int(len(rows))
    visible = rows[
        rows["time"].le(cutoff) & rows["release_time"].le(cutoff)
    ].copy()
    columns = ["component", "entity_id", "time", "release_time", "value"]
    if "group_id" in visible.columns:
        columns.append("group_id")
    return (
        visible[columns].sort_values(
            ["component", "entity_id", "time"], kind="mergesort"
        ),
        {
            "release_policy": release_policy,
            "contract_sha256": None,
            "candidate_cell_count": total_count,
            "future_or_unreleased_cell_count_excluded": total_count
            - int(len(visible)),
            "sequence_components": list(spec.target_components),
        },
    )


def _prepare_benchmark_b(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = load_benchmark_b_context_contract(
        _BENCHMARK_B_CONTEXT_CONTRACT,
        panel_columns=panel.columns,
    )
    task = contract.task
    entity_column = str(task["entity_column"])
    time_column = str(task["time_column"])
    components = tuple(map(str, task["target_components"]))
    expected_release = pd.to_datetime(panel[time_column], errors="raise") + pd.Timedelta(
        days=int(contract.payload["streams"]["target"]["release_lag_days"])
    )
    release_column = "__release_time__target"
    if release_column in panel.columns:
        supplied_release = pd.to_datetime(panel[release_column], errors="raise")
        if not bool(supplied_release.eq(expected_release).all()):
            raise ValueError(
                "Benchmark B target release column disagrees with the formal "
                "context contract"
            )
    annotated = annotate_stream_release_times(panel, contract)
    records: list[pd.DataFrame] = []
    for component in components:
        part = annotated[
            [entity_column, time_column, release_column, component]
        ].rename(
            columns={
                entity_column: "entity_id",
                time_column: "time",
                release_column: "release_time",
                component: "value",
            }
        )
        part["component"] = component
        records.append(part)
    rows = pd.concat(records, ignore_index=True)
    rows["entity_id"] = rows["entity_id"].astype(str)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows["time"] = pd.to_datetime(rows["time"], errors="raise")
    rows["release_time"] = pd.to_datetime(rows["release_time"], errors="raise")
    if bool(rows["release_time"].lt(rows["time"]).any()):
        raise ValueError("Benchmark B target release precedes target week")
    duplicate = rows.duplicated(["component", "entity_id", "time"])
    if bool(duplicate.any()):
        raise ValueError("Benchmark B sequence-sketch panel has duplicate target cells")
    total_count = int(len(rows))
    visible = rows[
        rows["time"].le(cutoff) & rows["release_time"].le(cutoff)
    ].copy()
    return (
        visible.sort_values(
            ["component", "entity_id", "time"], kind="mergesort"
        ),
        {
            "release_policy": (
                "benchmark_b_context_contract_target_release_lag_"
                f"{int(contract.payload['streams']['target']['release_lag_days'])}d"
            ),
            "contract_sha256": contract.sha256,
            "candidate_cell_count": total_count,
            "future_or_unreleased_cell_count_excluded": total_count
            - int(len(visible)),
            "sequence_components": list(components),
        },
    )


def _blocks_from_right(
    values: Sequence[pd.Timestamp], block_size: int
) -> list[list[pd.Timestamp]]:
    blocks: list[list[pd.Timestamp]] = []
    end = len(values)
    while end:
        start = max(0, end - block_size)
        blocks.append(list(values[start:end]))
        end = start
    return list(reversed(blocks))


def _tier_time_blocks(
    times: Sequence[pd.Timestamp],
    design: Mapping[str, int],
) -> dict[str, list[list[pd.Timestamp]]]:
    ordered = list(sorted(pd.Timestamp(value) for value in times))
    recent_count = int(design["recent_count"])
    recent = ordered[-recent_count:]
    preceding = ordered[: -len(recent)] if recent else ordered
    medium_span = int(design["medium_span"])
    medium = preceding[-medium_span:]
    long = preceding[: -len(medium)] if medium else preceding
    return {
        "long": _blocks_from_right(long, int(design["long_block"])),
        "medium": _blocks_from_right(medium, int(design["medium_block"])),
        "recent": [[value] for value in recent],
    }


def _point(
    rows: pd.DataFrame,
    times: Iterable[pd.Timestamp],
    digits: int,
) -> list[object]:
    selected_times = list(times)
    part = rows[rows["time"].isin(selected_times)]
    numeric = pd.to_numeric(part["value"], errors="coerce")
    observed = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
    values = observed.to_numpy(dtype=float)
    observed_rows = part.loc[observed.index]
    return [
        _date(part["time"].min()),
        _date(part["time"].max()),
        int(part["time"].nunique()),
        int(len(part)),
        int(len(values)),
        int(part["entity_id"].nunique()),
        int(observed_rows["entity_id"].nunique()),
        _round_number(np.mean(values), digits) if len(values) else None,
        _round_number(np.median(values), digits) if len(values) else None,
        _quantile(values, 0.25, digits),
        _quantile(values, 0.75, digits),
        _round_number(np.min(values), digits) if len(values) else None,
        _round_number(np.max(values), digits) if len(values) else None,
        (
            _round_number(np.mean(values == 0), digits)
            if len(values)
            else None
        ),
        _round_number(1.0 - len(values) / max(len(part), 1), digits),
    ]


def _trajectory(
    rows: pd.DataFrame,
    *,
    design: Mapping[str, int],
    digits: int,
) -> dict[str, Any]:
    time_blocks = _tier_time_blocks(
        sorted(rows["time"].drop_duplicates()), design
    )
    return {
        "tier_design": {key: int(value) for key, value in design.items()},
        "tiers": {
            tier: {
                "point_count": int(len(blocks)),
                "points": [_point(rows, block, digits) for block in blocks],
            }
            for tier, blocks in time_blocks.items()
        },
    }


def _component_summary(rows: pd.DataFrame, digits: int) -> dict[str, Any]:
    numeric = pd.to_numeric(rows["value"], errors="coerce")
    observed = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
    values = observed.to_numpy(dtype=float)
    per_time_mean = (
        rows.assign(value=numeric)
        .groupby("time", sort=True)["value"]
        .mean()
        .dropna()
    )
    return {
        "history_start": _date(rows["time"].min()),
        "history_end": _date(rows["time"].max()),
        "latest_release_time": _date(rows["release_time"].max()),
        "timepoint_count": int(rows["time"].nunique()),
        "entity_count": int(rows["entity_id"].nunique()),
        "cell_count": int(len(rows)),
        "observed_count": int(len(values)),
        "missing_fraction": _round_number(
            1.0 - len(values) / max(len(rows), 1), digits
        ),
        "global_mean": (
            _round_number(np.mean(values), digits) if len(values) else None
        ),
        "global_median": (
            _round_number(np.median(values), digits) if len(values) else None
        ),
        "first_cross_entity_mean": (
            _round_number(per_time_mean.iloc[0], digits)
            if len(per_time_mean)
            else None
        ),
        "last_cross_entity_mean": (
            _round_number(per_time_mean.iloc[-1], digits)
            if len(per_time_mean)
            else None
        ),
    }


def _cross_component_summary(
    visible: pd.DataFrame,
    components: Sequence[str],
    digits: int,
) -> dict[str, Any] | None:
    if len(components) != 2:
        return None
    temporal = (
        visible.groupby(["time", "component"], sort=True)["value"]
        .mean()
        .unstack("component")
        .reindex(columns=list(components))
        .dropna()
    )
    correlation: float | None = None
    if len(temporal) >= 3 and bool((temporal.std(ddof=0) > 0).all()):
        correlation = _round_number(temporal.corr().iloc[0, 1], digits)
    return {
        "components": list(components),
        "aligned_timepoint_count": int(len(temporal)),
        "pearson_correlation_of_cross_entity_means": correlation,
    }


def _render_text(
    payload: Mapping[str, Any],
    digest: str,
) -> str:
    return (
        "CASTER causal multiscale released-sequence sketch\n"
        "Use only as an origin-causal description of observed target dynamics; "
        "it contains no validation or test metric.\n"
        f"task_id: {payload['task_id']}\n"
        f"query: {payload['task_query']}\n"
        f"cutoff_time: {payload['cutoff_time']}\n"
        f"sequence_sketch_sha256: {digest}\n"
        "point_columns: "
        + ", ".join(map(str, payload["trajectory_representation"]["point_columns"]))
        + "\n"
        f"sequence_sketch_json: {sequence_sketch_canonical_json(payload).strip()}\n"
    )


def _estimated_tokens(text: str) -> int:
                                                                          
                                                                      
    return int(math.ceil(len(text) / 3.5))


def _thin_points(points: list[list[object]]) -> list[list[object]]:
    if len(points) <= 2:
        return points
    selected = [points[0], *points[2:-1:2], points[-1]]
    return selected if len(selected) < len(points) else points


def _fit_token_budget(
    payload: dict[str, Any],
    max_estimated_tokens: int,
) -> tuple[dict[str, Any], str, str, int]:
    if max_estimated_tokens < 512:
        raise ValueError("max_estimated_tokens must be at least 512")
    while True:
        digest = sequence_sketch_sha256(payload)
        text = _render_text(payload, digest)
        estimate = _estimated_tokens(text)
        if estimate <= max_estimated_tokens:
            return payload, text, digest, estimate
        if payload.get("benchmark_a_group_trajectories"):
            payload["benchmark_a_group_trajectories"] = {}
            payload["benchmark_a_group_trajectory_status"] = (
                "omitted_to_meet_token_budget"
            )
            continue
        changed = False
        for tier in ("long", "medium", "recent"):
            for component in sorted(payload["components"]):
                tier_payload = payload["components"][component][
                    "global_trajectory"
                ]["tiers"][tier]
                old = tier_payload["points"]
                new = _thin_points(old)
                if len(new) < len(old):
                    tier_payload["points"] = new
                    tier_payload["point_count"] = len(new)
                    changed = True
            if changed:
                break
        if not changed:
            raise ValueError(
                "sequence sketch cannot satisfy max_estimated_tokens without "
                "dropping its minimum multiscale representation"
            )


def build_causal_sequence_sketch(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    spec: TaskSpec,
    *,
    cutoff_time: object | None = None,
    max_estimated_tokens: int = 8000,
    rounding_digits: int = 4,
) -> tuple[dict[str, Any], str, str, pd.DataFrame]:
    ""







    if not 0 <= int(rounding_digits) <= 10:
        raise ValueError("rounding_digits must be between 0 and 10")
    if spec.dataset not in _GLOBAL_TIER_DESIGN:
        raise ValueError(
            f"sequence sketch supports benchmark_a/benchmark_b, got {spec.dataset!r}"
        )
    cutoff, cutoff_relation = _resolve_cutoff(spec, cutoff_time)
    if spec.dataset == "benchmark_a":
        visible, preparation = _prepare_benchmark_a(panel, spec, cutoff)
    else:
        visible, preparation = _prepare_benchmark_b(panel, cutoff)
    if visible.empty:
        raise ValueError(f"no released target sequence is visible for {spec.task_id}")
    if bool(visible["time"].gt(cutoff).any()):
        raise AssertionError("sequence sketch retained a post-cutoff target time")
    if bool(visible["release_time"].gt(cutoff).any()):
        raise AssertionError("sequence sketch retained an unreleased target cell")

    sequence_components = tuple(map(str, preparation["sequence_components"]))
    component_payload: dict[str, Any] = {}
    design = _GLOBAL_TIER_DESIGN[spec.dataset]
    for component in sequence_components:
        rows = visible[visible["component"].eq(component)]
        if rows.empty:
            raise ValueError(
                f"no released sequence rows for required component {component!r}"
            )
        component_payload[component] = {
            "summary": _component_summary(rows, int(rounding_digits)),
            "global_trajectory": _trajectory(
                rows, design=design, digits=int(rounding_digits)
            ),
        }

    group_trajectories: dict[str, Any] = {}
    group_status = "not_applicable"
    if spec.dataset == "benchmark_a" and "group_id" in visible.columns:
        groups = sorted(visible["group_id"].dropna().astype(str).unique())
        if len(groups) <= 8:
            group_status = "included_country_trajectories"
            for group in groups:
                group_rows = visible[visible["group_id"].eq(group)]
                group_trajectories[group] = {
                    component: _trajectory(
                        group_rows[group_rows["component"].eq(component)],
                        design=_BENCHMARK_A_GROUP_TIER_DESIGN,
                        digits=int(rounding_digits),
                    )
                    for component in sequence_components
                }
        else:
            group_status = "omitted_more_than_8_groups"

    ledger_binding = _released_ledger_binding(
        ledger, components=sequence_components, cutoff=cutoff
    )
    payload: dict[str, Any] = {
        "schema": SEQUENCE_SKETCH_SCHEMA,
        "task_id": spec.task_id,
        "dataset": spec.dataset,
        "task_query": spec.natural_language_query,
        "selection_view_target_components": list(spec.target_components),
        "sequence_components": list(sequence_components),
        "cadence": spec.cadence,
        "entity_scope": spec.entity_scope,
        "cutoff_time": _date(cutoff),
        "formal_t_sel": _date(spec.t_sel),
        "cutoff_relation_to_formal_t_sel": cutoff_relation,
        "cutoff_defaulted_to_formal_t_sel": cutoff_time is None,
        "history_scope": "target_cells_with_target_and_release_time_lte_cutoff",
        "release_policy": preparation["release_policy"],
        "rounding_digits": int(rounding_digits),
        "trajectory_representation": {
            "value_scale": "archived_raw_target_scale",
            "time_axis_preserved": True,
            "cross_entity_statistic_order": "deterministic_entity_id_sort",
            "tiers": ["long", "medium", "recent"],
            "point_columns": list(POINT_COLUMNS),
        },
        "components": component_payload,
        "cross_component_temporal_association": _cross_component_summary(
            visible, sequence_components, int(rounding_digits)
        ),
        "benchmark_a_group_trajectory_status": group_status,
        "benchmark_a_group_trajectories": group_trajectories,
        "source_binding": {
            "visible_target_record_sha256": _records_sha256(visible),
            "benchmark_b_context_contract_sha256": preparation[
                "contract_sha256"
            ],
            "released_event_count": ledger_binding["released_event_count"],
            "max_released_event_time": ledger_binding[
                "max_released_event_time"
            ],
            "released_event_sha256": ledger_binding[
                "released_event_sha256"
            ],
        },
        "causal_guards": {
            "explicit_origin_cutoff_permitted": True,
            "target_time_lte_cutoff": True,
            "release_time_lte_cutoff": True,
            "split_membership_consulted_for_sequence_values": False,
            "validation_metrics_included": False,
            "test_metrics_included": False,
            "unreleased_labels_included": False,
        },
    }
    payload, text, digest, token_estimate = _fit_token_budget(
        payload, int(max_estimated_tokens)
    )

    validation = pd.DataFrame(
        [
            {
                "task_id": spec.task_id,
                "check": "schema",
                "status": "PASS",
                "value": SEQUENCE_SKETCH_SCHEMA,
            },
            {
                "task_id": spec.task_id,
                "check": "cutoff_relation_to_formal_t_sel",
                "status": "PASS",
                "value": cutoff_relation,
            },
            {
                "task_id": spec.task_id,
                "check": "target_time_lte_cutoff",
                "status": "PASS",
                "value": _date(visible["time"].max()),
            },
            {
                "task_id": spec.task_id,
                "check": "release_time_lte_cutoff",
                "status": "PASS",
                "value": _date(visible["release_time"].max()),
            },
            {
                "task_id": spec.task_id,
                "check": "visible_target_cell_count",
                "status": "PASS",
                "value": str(len(visible)),
            },
            {
                "task_id": spec.task_id,
                "check": "future_or_unreleased_cells_excluded",
                "status": "PASS",
                "value": str(
                    preparation["future_or_unreleased_cell_count_excluded"]
                ),
            },
            {
                "task_id": spec.task_id,
                "check": "future_release_ledger_events_excluded",
                "status": "PASS",
                "value": str(ledger_binding["future_release_event_count_excluded"]),
            },
            {
                "task_id": spec.task_id,
                "check": "deterministic_sort_and_rounding",
                "status": "PASS",
                "value": f"mergesort/{int(rounding_digits)}digits",
            },
            {
                "task_id": spec.task_id,
                "check": "estimated_text_token_budget",
                "status": "PASS",
                "value": f"{token_estimate}<={int(max_estimated_tokens)}",
            },
            {
                "task_id": spec.task_id,
                "check": "validation_and_test_metrics_excluded",
                "status": "PASS",
                "value": "true",
            },
            {
                "task_id": spec.task_id,
                "check": "sequence_sketch_sha256",
                "status": "PASS",
                "value": digest,
            },
        ]
    )
    return payload, text, digest, validation
