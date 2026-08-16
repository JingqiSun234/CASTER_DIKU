from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .context import build_selection_context, selection_context_sha256
from .sequence_sketch import (
    SEQUENCE_SKETCH_SCHEMA,
    build_causal_sequence_sketch,
)
from .spec import TaskSpec


QWEN25_MULTISCALE_CONTEXT_PROFILE = "qwen25_multiscale_released_sequence_v1"
QWEN25_MULTISCALE_CONTEXT_SCHEMA = (
    "caster_qwen25_multiscale_released_sequence_context_v1"
)


_AUDIT_ONLY_KEYS = {
    "schema",
    "profile",
    "task_id",
    "task_spec_sha256",
    "t_sel",
    "formal_t_sel",
    "cutoff_relation_to_formal_t_sel",
    "cutoff_defaulted_to_formal_t_sel",
    "context_role",
    "source_binding",
    "causal_guards",
    "causal_rules",
    "information_alignment",
    "frozen_selection",
    "contract_version",
    "posterior_scope",
    "posterior_count",
    "selection_view_task_id",
    "selection_view_posterior_scope",
    "formal_status",
    "release_time_status",
    "release_policy",
    "representation",
    "trajectory_representation",
    "future_graph_access",
    "status",
}

_SCIENTIFIC_VALUE_ALIASES = {
    "all_available_before_t_sel": "observations available by the forecast origin",
    "all_available_through_cutoff": "observations available by the forecast origin",
    "all_released_no_later_than_cutoff": "observations released by the forecast origin",
    "target_cells_with_target_and_release_time_lte_cutoff": (
        "target observations released by the forecast origin"
    ),
}


def _without_audit_metadata(value: Any) -> Any:
    """Return the scientific information view without provenance/split fields."""
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if (
                key_text in _AUDIT_ONLY_KEYS
                or key_text.endswith("_sha256")
                or key_text.startswith("__")
            ):
                continue
            clean[key_text] = _without_audit_metadata(item)
        return clean
    if isinstance(value, list):
        return [_without_audit_metadata(item) for item in value]
    if isinstance(value, str):
        return _SCIENTIFIC_VALUE_ALIASES.get(value, value)
    return value


def scientific_base_context_view(base: dict[str, Any]) -> dict[str, Any]:
    """Project one full audit context onto forecasting facts only."""
    target_components = list(
        map(
            str,
            base.get(
                "selection_view_target_components",
                base.get("target_components", base.get("component_scope", [])),
            ),
        )
    )
    visible_components = list(
        map(
            str,
            base.get(
                "component_scope",
                list(base.get("history_summary", {}).keys()) or target_components,
            ),
        )
    )
    auxiliary_components = [
        component
        for component in visible_components
        if component not in set(target_components)
    ]
    query = str(
        base.get("selection_view_query")
        or base.get("canonical_task_query")
        or base.get("task_query")
        or ""
    )

    if "history_summary" in base:
        structured = {
            "forecast_origin": base.get(
                "forecast_origin", base.get("cutoff_time", "")
            ),
            "history_scope": base.get(
                "history_scope", "all released no later than the origin"
            ),
            "history_summary": base.get("history_summary", {}),
            "cross_stream_summary": base.get("cross_stream_summary", {}),
            "released_event_count": base.get("released_event_count", 0),
        }
    else:
        structured_keys = (
            "cutoff_time",
            "cadence",
            "direct_horizons",
            "recursive_horizons",
            "forecast_strategies",
            "entity_scope",
            "history_scope",
            "component_summaries",
            "covariate_availability",
            "release_revision_summary",
        )
        structured = {
            key: base[key]
            for key in structured_keys
            if key in base
        }
        mobility = base.get("causal_mobility_context")
        if isinstance(mobility, dict):
            structured["origin_available_mobility_summary"] = {
                key: mobility[key]
                for key in (
                    "entity_count",
                    "feature_columns",
                    "feature_statistics",
                    "history_start",
                    "history_end",
                    "latest_release_time",
                    "row_count",
                )
                if key in mobility
            }

    return _without_audit_metadata(
        {
            "selection_target": {
                "query": query,
                "target_components": target_components,
                "released_auxiliary_components": auxiliary_components,
                "auxiliary_component_role": (
                    "released cross-stream context only"
                    if auxiliary_components
                    else "none"
                ),
            },
            "structured_context": structured,
        }
    )


def scientific_selection_text(payload: dict[str, Any]) -> str:
    """Canonical scientific text for non-Qwen and Qwen task embeddings."""
    return "scientific_forecasting_context=" + json.dumps(
        scientific_base_context_view(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def scientific_context_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the hash-bound augmented payload onto facts suitable for LLM input.

    The complete payload remains the source of provenance hashes and validation.
    This view removes experiment split markers, internal posterior terminology,
    implementation identifiers, and digests while retaining the same released
    histories, trajectories, horizons, covariates, and cross-stream statistics.
    """
    base = payload.get("base_context", {})
    sequence = payload.get("causal_multiscale_sequence_sketch", {})
    if not isinstance(base, dict) or not isinstance(sequence, dict):
        raise ValueError("Qwen context lacks structured or sequence evidence")

    base_for_view = dict(base)
    if "selection_view_target_components" not in base_for_view:
        base_for_view["selection_view_target_components"] = sequence.get(
            "selection_view_target_components", []
        )
    if "component_scope" not in base_for_view:
        base_for_view["component_scope"] = sequence.get("sequence_components", [])
    if not (
        base_for_view.get("selection_view_query")
        or base_for_view.get("canonical_task_query")
        or base_for_view.get("task_query")
    ):
        base_for_view["task_query"] = sequence.get("task_query", "")
    base_view = scientific_base_context_view(base_for_view)

    sequence_keys = (
        "cutoff_time",
        "cadence",
        "entity_scope",
        "history_scope",
        "rounding_digits",
        "components",
        "cross_component_temporal_association",
    )
    sequence_view = {
        key: sequence[key]
        for key in sequence_keys
        if key in sequence
    }
    if "benchmark_a_group_trajectory_status" in sequence:
        sequence_view["group_trajectory_status"] = sequence[
            "benchmark_a_group_trajectory_status"
        ]
    if "benchmark_a_group_trajectories" in sequence:
        sequence_view["group_trajectories"] = sequence[
            "benchmark_a_group_trajectories"
        ]
    return _without_audit_metadata(
        {
            **base_view,
            "release_time_valid_multiscale_sequence": sequence_view,
        }
    )


def _component_max(sequence_payload: dict[str, Any], field: str) -> str:
    values = [
        str(component.get("summary", {}).get(field, ""))
        for component in sequence_payload.get("components", {}).values()
        if isinstance(component, dict)
    ]
    return max((value for value in values if value), default="")


def build_qwen25_multiscale_context(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    spec: TaskSpec,
    *,
    cutoff_time: object | None = None,
    context_role: str = "formal_selection",
    max_estimated_sequence_tokens: int = 8000,
    rounding_digits: int = 4,
) -> tuple[dict[str, Any], str, str, pd.DataFrame]:
    ""






    base_payload, base_text, base_validation = build_selection_context(
        panel,
        ledger,
        spec,
        cutoff_time=cutoff_time,
        context_role=context_role,
    )
    sequence_payload, sequence_text, sequence_sha, sequence_validation = (
        build_causal_sequence_sketch(
            panel,
            ledger,
            spec,
            cutoff_time=cutoff_time,
            max_estimated_tokens=max_estimated_sequence_tokens,
            rounding_digits=rounding_digits,
        )
    )
    for label, frame in (
        ("structured context", base_validation),
        ("sequence sketch", sequence_validation),
    ):
        if not frame.empty and not frame["status"].astype(str).eq("PASS").all():
            raise ValueError(
                f"{label} validation failed task={spec.task_id} "
                f"cutoff={cutoff_time or spec.t_sel}"
            )

    base_sha = selection_context_sha256(base_payload)
    cutoff = str(sequence_payload["cutoff_time"])
    payload: dict[str, Any] = {
        "schema": QWEN25_MULTISCALE_CONTEXT_SCHEMA,
        "profile": QWEN25_MULTISCALE_CONTEXT_PROFILE,
        "task_id": spec.task_id,
        "task_spec_sha256": spec.task_spec_sha256,
        "t_sel": spec.t_sel,
        "cutoff_time": cutoff,
        "context_role": str(context_role),
        "history_scope": (
            "structured_causal_context_plus_released_target_multiscale_sketch"
        ),
        "history_max": _component_max(sequence_payload, "history_end"),
        "latest_target_release_time": _component_max(
            sequence_payload, "latest_release_time"
        ),
        "base_context_schema": str(base_payload.get("schema", "")),
        "base_context_sha256": base_sha,
        "sequence_sketch_schema": SEQUENCE_SKETCH_SCHEMA,
        "sequence_sketch_sha256": sequence_sha,
        "base_context": base_payload,
        "causal_multiscale_sequence_sketch": sequence_payload,
        "information_alignment": {
            "shared_by_caster_retrieval_and_agents": True,
            "same_builder_at_equal_cutoff": True,
            "validation_metrics_visible": False,
            "test_metrics_visible": False,
            "future_or_unreleased_target_values_visible": False,
        },
    }
    combined_sha = selection_context_sha256(payload)
    scientific = scientific_context_view(payload)
    text = (
        "Forecasting task context: structured evidence plus a "
        "release-time-valid multiscale target-sequence sketch\n"
        "Only scientific, origin-available information is shown.\n"
        "--- structured_context ---\n"
        f"selection_target={json.dumps(scientific['selection_target'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)}\n"
        f"structured_context={json.dumps(scientific['structured_context'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)}\n"
        "--- release_time_valid_multiscale_sequence ---\n"
        f"release_time_valid_multiscale_sequence={json.dumps(scientific['release_time_valid_multiscale_sequence'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)}\n"
    )

    validations: list[pd.DataFrame] = []
    for stage, frame in (
        ("structured_context", base_validation),
        ("causal_sequence_sketch", sequence_validation),
    ):
        part = frame.copy()
        part.insert(1, "validation_stage", stage)
        validations.append(part)
    validations.append(
        pd.DataFrame(
            [
                {
                    "task_id": spec.task_id,
                    "validation_stage": "combined_context",
                    "check": "shared_context_profile",
                    "status": "PASS",
                    "value": QWEN25_MULTISCALE_CONTEXT_PROFILE,
                },
                {
                    "task_id": spec.task_id,
                    "validation_stage": "combined_context",
                    "check": "combined_context_sha256",
                    "status": "PASS",
                    "value": combined_sha,
                },
                {
                    "task_id": spec.task_id,
                    "validation_stage": "combined_context",
                    "check": "validation_test_future_information_excluded",
                    "status": "PASS",
                    "value": "true",
                },
            ]
        )
    )
    return payload, text, combined_sha, pd.concat(validations, ignore_index=True)


__all__ = [
    "QWEN25_MULTISCALE_CONTEXT_PROFILE",
    "QWEN25_MULTISCALE_CONTEXT_SCHEMA",
    "build_qwen25_multiscale_context",
    "scientific_base_context_view",
    "scientific_context_view",
    "scientific_selection_text",
]
