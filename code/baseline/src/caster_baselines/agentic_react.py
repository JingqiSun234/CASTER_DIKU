from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agentic_archive import (
    ForecastArchiveIndex,
    ARCHIVE_FORECAST_SOURCE,
    ARCHIVE_TIMING_MODE,
    alternate_TIMING_MODE,
    archive_timing_payload,
    alternate_timing_payload,
    load_forecast_archive,
    make_forecast_row_from_archive,
    make_forecast_rows_from_archive,
    resolve_timing_mode,
    validate_archive_coverage,
)
from .agentic_llm import LLMError, QwenLocalEngine
from .agentic_skills import (
    derive_agent_selection_scope,
    eligible_registry,
    fair_registry_prompt_rows,
    group_history_values,
    is_stateful,
    load_selection_replay_log,
    load_dataset_contexts,
    make_forecast_row,
    pre_fit_predictions,
    qwen_config,
    read_candidate_registry,
    select_candidate_no_validation,
    selection_context_with_metadata,
    selection_log_row,
    total_trace_runtime,
    write_registry_snapshot,
    write_standard_artifacts,
)
from .data_validation import baseline_root, caster_root_from_baseline, sha256_file
from .selection_fairness import load_common_selection_input, merge_common_selection_input
from .benchmark_b_context import (
    annotate_forecast_context,
    context_manifest_fields as benchmark_b_context_manifest_fields,
    is_benchmark_b_context,
    prompt_payload as benchmark_b_prompt_payload,
    llm_prompt_fields as benchmark_b_llm_prompt_fields,
    prompt_text as benchmark_b_prompt_text,
)


REACT_REFERENCE = {
    "result": "yao2023react_iclr_2210.03629",
    "source": "https://github.com/ysymyth/ReAct",
    "workflow": "Thought -> as-of tool Action -> Observation -> tentative selection -> keep/repair Critic -> execute",
}
REACT_SELECTION_POLICIES = ("llm_only", "no_validation_score")
REACT_ACTIONS = ("inspect_context", "inspect_candidates", "select_model")
REACT_summary_POLICY = "asof_trajectory_compatibility_keep_or_repair_no_validation"
REACT_REPAIR_GATE = "hard_risk_allows_three_rechecked_repairs_then_fallback_to_initial"
MAX_REACT_REPAIR_ATTEMPTS = 3

FIXED_SCALAR_RECURRENCES = {"rnn_simple", "gru_style"}


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _record_react_event(
    *,
    trace_rows: list[dict[str, object]],
    dataset_key: str,
    forecast_origin: str,
    step: int,
    thought: str,
    action: str,
    observation: dict[str, object],
) -> None:
    trace_rows.append({
        "stage": "react_step",
        "dataset_key": dataset_key,
        "forecast_origin": forecast_origin,
        "step": int(step),
        "thought": thought,
        "action": action,
        "observation": json.dumps(observation, sort_keys=True),
        "model_path": "deterministic_react_tool",
        "fallback_used": False,
        "fallback_reason": "",
        "runtime_seconds": 0.0,
        "prompt": "",
        "response_text": json.dumps({"thought": thought, "action": action, "observation": observation}, sort_keys=True),
    })


def _call_react_thought(
    engine: Any,
    *,
    dataset_key: str,
    forecast_origin: str,
    payload: dict[str, object],
    trace_rows: list[dict[str, object]],
) -> dict[str, object]:
    call = engine.generate_json(
        stage="react_thought",
        system_prompt=(
            "You are a ReAct-style forecasting agent. Return compact JSON with keys thought, "
            "next_action, and selection_intent. The next_action must be exactly one of "
            "inspect_context, inspect_candidates, select_model. These tools expose only information "
            "available by the forecast origin; validation scores and future targets are unavailable."
        ),
        user_prompt=json.dumps(payload, sort_keys=True),
    )
    trace_rows.append({
        "stage": call.stage,
        "dataset_key": dataset_key,
        "forecast_origin": forecast_origin,
        "step": 0,
        "thought": str(call.response_json.get("thought", "")),
        "action": str(call.response_json.get("next_action", "")),
        "observation": "",
        "model_path": call.model_path,
        "fallback_used": call.fallback_used,
        "fallback_reason": call.fallback_reason,
        "runtime_seconds": call.runtime_seconds,
        "prompt": call.prompt,
        "response_text": call.response_text,
    })
    return call.response_json


def _normalise_react_action(value: object) -> str:
    action = str(value or "").strip().lower()
    aliases = {
        "score_candidates": "inspect_candidates",
        "select_candidate": "inspect_candidates",
        "execute_forecast": "select_model",
    }
    action = aliases.get(action, action)
    return action if action in REACT_ACTIONS else "inspect_candidates"


def _finite_median(values: list[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float(default)


def _asof_data_profile(ctx: Any, release: pd.DataFrame, origin_text: str) -> dict[str, object]:
    ""
    origin = pd.to_datetime(origin_text, errors="coerce")
    lengths: list[int] = []
    changes: list[float] = []
    relative_changes: list[float] = []
    volatility_ratios: list[float] = []
    if not pd.isna(origin):
        for (entity, component), _ in release.groupby([ctx.ledger_entity_col, "component"], dropna=False):
            values = np.asarray(group_history_values(ctx, str(entity), str(component), origin), dtype=float)
            values = values[np.isfinite(values)]
            lengths.append(int(len(values)))
            if len(values) >= 2:
                lookback = min(4, len(values) - 1)
                change = float(values[-1] - values[-1 - lookback])
                scale = max(abs(float(values[-1 - lookback])), float(np.median(np.abs(values))), 1.0)
                changes.append(change)
                relative_changes.append(change / scale)
                diffs = np.diff(values[-min(len(values), 12) :])
                if len(diffs):
                    volatility_ratios.append(float(np.median(np.abs(diffs))) / scale)

    horizons = pd.to_numeric(release.get("horizon", pd.Series(dtype=float)), errors="coerce").dropna()
    modes = release.get("mode", pd.Series(dtype=str)).astype(str).str.lower()
    strategies = release.get("forecast_strategy", pd.Series(dtype=str)).astype(str).str.lower()
    median_relative_change = _finite_median(relative_changes)
    median_abs_relative_change = _finite_median([abs(value) for value in relative_changes])
    if median_relative_change >= 0.08:
        trend_signal = "rising"
    elif median_relative_change <= -0.08:
        trend_signal = "falling"
    elif median_abs_relative_change <= 0.03:
        trend_signal = "flat"
    else:
        trend_signal = "mixed_or_mild"
    min_history = min(lengths) if lengths else 0
    median_history = _finite_median([float(value) for value in lengths])
    max_horizon = int(horizons.max()) if not horizons.empty else 0
    season_length = int(getattr(ctx, "season_length", 0) or 0)
    cadence_days = int(ctx.manifest_row.get("cadence_days", 1))
    decomposition_season_length = 7 if cadence_days <= 1 else (
        52 if 6 <= cadence_days <= 8 else max(1, round(365.25 / cadence_days))
    )
    components = sorted(str(value) for value in release.get("component", pd.Series(dtype=str)).dropna().unique())
    recursive = bool(
        strategies.eq("recursive_rollout").any()
        or modes.str.contains("rollout", regex=False).any()
    )
    return {
        "forecast_origin": origin_text,
        "history_rows_min": int(min_history),
        "history_rows_median": round(float(median_history), 3),
        "history_rows_max": int(max(lengths) if lengths else 0),
        "season_length": season_length,
        "seasonal_naive_period": (
            "7d" if cadence_days <= 1 else "8w" if 6 <= cadence_days <= 8 else str(season_length)
        ),
        "decomposition_season_length": decomposition_season_length,
        "full_season_history_available": bool(season_length > 0 and min_history >= season_length + max_horizon),
        "horizon_max": max_horizon,
        "recursive_rollout_present": recursive,
        "trend_signal": trend_signal,
        "recent_change_median": round(_finite_median(changes), 6),
        "recent_relative_change_median": round(median_relative_change, 6),
        "recent_volatility_ratio_median": round(_finite_median(volatility_ratios), 6),
        "components": components,
        "profile_uses_only_observation_and_release_time_lte_forecast_origin": True,
        "validation_score_used": False,
        "future_target_used": False,
    }


def _candidate_compatibility(profile: dict[str, object], selected: pd.Series) -> dict[str, object]:
    model_id = str(selected.get("model_id", ""))
    family = str(selected.get("family", "")).lower()
    median_history = float(profile.get("history_rows_median", 0.0))
    min_history = int(profile.get("history_rows_min", 0))
    season_length = int(profile.get("season_length", 0))
    decomposition_season_length = int(
        profile.get("decomposition_season_length", season_length)
    )
    max_horizon = int(profile.get("horizon_max", 0))
    trend = str(profile.get("trend_signal", ""))
    recursive = bool(profile.get("recursive_rollout_present", False))
    full_season = bool(profile.get("full_season_history_available", False))

    risks: list[str] = []
    strengths: list[str] = []
    if model_id == "seasonal_naive":
        if not full_season:
            risks.append("insufficient_asof_history_for_full_season_lag")
        else:
            strengths.append("full_season_lag_available")
    if model_id in {"prophet", "statsforecast_autoets", "statsforecast_autotheta"}:
        if (
            decomposition_season_length > 0
            and median_history < 2 * decomposition_season_length
        ):
            risks.append("limited_asof_cycles_for_seasonal_decomposition")
    if (
        family in {"neural", "foundation_ts"}
        and model_id not in FIXED_SCALAR_RECURRENCES
    ):
        if median_history < max(16, 4 * max(max_horizon, 1)):
            risks.append("short_asof_history_for_high_capacity_sequence_model")
        else:
            strengths.append("adequate_asof_sequence_length")
    if family in {"compartmental", "renewal"}:
        if min_history < max(8, 2 * max(max_horizon, 1)):
            risks.append("short_asof_history_for_dynamic_rate_estimation")
        else:
            strengths.append("epidemic_dynamics_structure_matches_target_domain")
    if model_id == "last_value":
        if trend in {"rising", "falling"}:
            risks.append("recency_forecast_ignores_material_directional_change")
        else:
            strengths.append("robust_recency_under_flat_or_mixed_change")
    if model_id in {"drift"}:
        if trend == "flat":
            risks.append("trend_extrapolation_not_supported_by_recent_asof_change")
        elif trend in {"rising", "falling"}:
            strengths.append("directional_model_matches_recent_asof_change")
    if (
        model_id in {"sir_tau", "seir_tau", "seirs_tau"}
        and recursive
        and max_horizon >= 4
        and trend in {"rising", "falling"}
    ):
        risks.append("fixed_compartmental_transition_may_be_rigid_for_directional_long_rollout")
    if model_id in {"tv_seir_rt", "renewal_rt"} and trend in {"rising", "falling"}:
        strengths.append("time_varying_dynamics_match_directional_change")
    if recursive:
        strengths.append("archive_has_explicit_recursive_rollout_for_requested_horizons")

    return {
        "model_id": model_id,
        "scientific_description": str(selected.get("description", "")),
        "risk_flags": sorted(set(risks)),
        "strength_flags": sorted(set(strengths)),
        "material_risk": bool(risks),
        "assessment_basis": "asof_history_horizon_strategy_and_candidate_metadata_only",
        "validation_score_used": False,
        "future_target_used": False,
    }


def _dispatch_react_action(
    action: str,
    *,
    profile: dict[str, object],
    candidates: pd.DataFrame,
) -> dict[str, object]:
    common = {
        "action_executed": action,
        "selection_score_source": "no_validation",
        "data_profile": profile,
    }
    if action == "inspect_context":
        return {"tool": "asof_context_inspector", **common}
    if action == "select_model":
        return {
            "tool": "selection_constraint_inspector",
            **common,
            "candidate_count": int(len(candidates)),
            "constraint": "select_exactly_one_executable_archive_candidate",
        }
    return {
        "tool": "candidate_compatibility_inspector",
        **common,
        "candidate_count": int(len(candidates)),
        "inspection_instruction": "match model assumptions to as-of history, horizon, strategy, and forecast target",
    }


def _trajectory_id(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _summary_candidate(
    *,
    engine: Any,
    dataset_key: str,
    forecast_origin: str,
    candidates: pd.DataFrame,
    initial_selected: pd.Series,
    profile: dict[str, object],
    trajectory: dict[str, object],
    trace_rows: list[dict[str, object]],
    selection_replay: dict[tuple[str, str], dict[str, Any]] | None,
) -> pd.Series:
    """Run the anonymous ReAct critic and validate up to three repairs."""
    initial_id = str(initial_selected["model_id"])
    rows, choice_to_model = fair_registry_prompt_rows(
        candidates,
        dataset_key=dataset_key,
        forecast_origin=forecast_origin,
    )
    valid_choice_ids = [str(row["choice_id"]) for row in rows]
    model_to_choice = {
        model_id: choice_id for choice_id, model_id in choice_to_model.items()
    }
    initial_choice_id = model_to_choice[initial_id]
    compatibility = trajectory.get("tentative_compatibility_observation", {})
    if not isinstance(compatibility, dict):
        compatibility = _candidate_compatibility(profile, initial_selected)
    material_risk = bool(compatibility.get("material_risk", False))
    target_risk_flags = {
        str(flag) for flag in compatibility.get("risk_flags", []) if str(flag)
    }
    resolving_choice_ids: list[str] = []
    if material_risk:
        for choice_id in valid_choice_ids:
            model_id = choice_to_model[choice_id]
            if model_id == initial_id:
                continue
            candidate = candidates[
                candidates["model_id"].astype(str).eq(model_id)
            ].iloc[0]
            observation = _candidate_compatibility(profile, candidate)
            risk_flags = [
                str(flag)
                for flag in observation.get("risk_flags", [])
                if str(flag)
            ]
            if not bool(observation.get("material_risk", False)) and not risk_flags:
                resolving_choice_ids.append(choice_id)

    order_strategy = "stable_hash_by_dataset_origin_model_id"
    position_neutrality_instruction = (
        "Candidate order is a stable hash order for display only, not a ranking. "
        "Every candidate position has equal weight. Do not prefer candidates because "
        "they appear earlier or later in the list."
    )

    def final_contract_fields(
        selected: pd.Series,
        *,
        attempts: list[dict[str, object]],
        fallback_after_three: bool,
        repair_skipped_reason: str,
    ) -> dict[str, object]:
        final_observation = _candidate_compatibility(profile, selected)
        final_risks = {
            str(flag) for flag in final_observation.get("risk_flags", []) if str(flag)
        }
        remaining = sorted(target_risk_flags.intersection(final_risks))
        return {
            "repair_compatibility_rechecked": True,
            "repair_target_risk_flags": sorted(target_risk_flags),
            "repair_remaining_target_risk_flags": remaining,
            "repair_target_risk_resolved": bool(
                material_risk
                and not remaining
                and not bool(final_observation.get("material_risk", False))
                and not final_risks
            ),
            "repair_resolution_available": bool(resolving_choice_ids),
            "repair_resolving_model_ids": [
                choice_to_model[choice_id] for choice_id in resolving_choice_ids
            ],
            "repair_skipped_reason": repair_skipped_reason,
            "repair_attempt_limit": MAX_REACT_REPAIR_ATTEMPTS,
            "repair_attempt_count": int(len(attempts)),
            "repair_attempts": attempts,
            "repair_attempt_consumption_policy": (
                "each_llm_proposal_consumes_one_attempt_including_invalid_same_or_repeated_choices"
            ),
            "fallback_to_initial_after_three_failed_repairs": bool(
                fallback_after_three
            ),
            "final_compatibility_observation": final_observation,
        }

    def normalize_llm_payload(
        raw: dict[str, Any],
        *,
        selected: pd.Series,
        selected_choice_id: str,
        decision: str,
        attempts: list[dict[str, object]],
        fallback_after_three: bool,
        repair_skipped_reason: str = "",
        attempt_history_available: bool = True,
        attempt_history_source: str = "semantic_critic_attempts",
    ) -> dict[str, object]:
        llm_payload: dict[str, object] = dict(raw)
        proposed_choice_id = str(
            llm_payload.get("choice_id")
            or llm_payload.get("selected_choice_id")
            or llm_payload.get("selected_model_id")
            or llm_payload.get("selected")
            or ""
        ).strip()
        for response_key in (
            "choice_id",
            "selected_choice_id",
            "selected_model_id",
            "model_id",
            "selected",
        ):
            llm_payload.pop(response_key, None)
        rank = valid_choice_ids.index(selected_choice_id) + 1
        llm_payload.update(
            {
                "decision": decision,
                "selected_choice_id": selected_choice_id,
                "llm_proposed_choice_id": proposed_choice_id,
                "llm_proposed_model_id": choice_to_model.get(
                    proposed_choice_id, ""
                ),
                "selection_method": "qwen_json_causal_react_critic",
                "candidate_order_strategy": order_strategy,
                "candidate_order": [
                    choice_to_model[choice_id] for choice_id in valid_choice_ids
                ],
                "candidate_choice_order": valid_choice_ids,
                "candidate_choice_map": choice_to_model,
                "position_neutrality_instruction": position_neutrality_instruction,
                "fairness_instruction": position_neutrality_instruction,
                "position_neutrality_instruction_present": False,
                "position_neutrality_control_trace_only": True,
                "selected_candidate_rank": int(rank),
                "n_candidates": int(len(valid_choice_ids)),
                "selected_rank_fraction": float(rank / len(valid_choice_ids)),
                "trajectory_consumed": True,
                "summary_schema_valid": True,
                "selection_score_source": "no_validation",
                "validation_score_used": False,
                "validation_context_visible": False,
                "validation_context_kind": "",
                "validation_context_scoreboard_rows": 0,
                "validation_guard_enabled": False,
                "validation_score_policy": "none",
                "validation_ledger_policy": "none",
                "llm_selected_model_id": str(selected["model_id"]),
                "repair_attempt_history_available": bool(
                    attempt_history_available
                ),
                "repair_attempt_history_source": attempt_history_source,
                "validation_rows": 0,
                "validation_best_model_id": "",
                "validation_override": False,
                "validation_override_reason": "",
                **final_contract_fields(
                    selected,
                    attempts=attempts,
                    fallback_after_three=fallback_after_three,
                    repair_skipped_reason=repair_skipped_reason,
                ),
            }
        )
        return llm_payload

    def response_choice_id(response_json: dict[str, Any]) -> str:
        choice_id = str(
            response_json.get("choice_id")
            or response_json.get("selected_choice_id")
            or response_json.get("selected_model_id")
            or response_json.get("selected")
            or ""
        ).strip()
        return choice_id

    def append_call_trace(
        call: Any,
        *,
        repair_attempt: int,
        attempt_record: dict[str, object],
    ) -> None:
        trace_rows.append(
            {
                "stage": call.stage,
                "dataset_key": dataset_key,
                "forecast_origin": forecast_origin,
                "model_path": call.model_path,
                "fallback_used": call.fallback_used,
                "fallback_reason": call.fallback_reason,
                "runtime_seconds": call.runtime_seconds,
                "prompt": call.prompt,
                "response_text": call.response_text,
                "candidate_order_strategy": order_strategy,
                "candidate_choice_map": choice_to_model,
                "position_neutrality_instruction": position_neutrality_instruction,
                "fairness_instruction": position_neutrality_instruction,
                "repair_attempt": int(repair_attempt),
                "repair_attempt_record": attempt_record,
            }
        )

    if material_risk and not resolving_choice_ids:
        skip_record = {
            "attempt": 0,
            "choice_id": initial_choice_id,
            "candidate_model_id": initial_id,
            "decision": "keep_with_risk",
            "valid_choice": True,
            "same_as_initial": True,
            "repeated_choice": False,
            "compatibility_rechecked": True,
            "risk_before": sorted(target_risk_flags),
            "risk_after": sorted(target_risk_flags),
            "target_risk_remaining": sorted(target_risk_flags),
            "validation_status": "repair_skipped_no_risk_free_candidate",
            "repair_accepted": False,
        }
        trace_rows.append(
            {
                "stage": "react_repair_skipped",
                "dataset_key": dataset_key,
                "forecast_origin": forecast_origin,
                "model_path": "deterministic_compatibility_gate",
                "fallback_used": False,
                "fallback_reason": "",
                "runtime_seconds": 0.0,
                "prompt": "",
                "response_text": json.dumps(skip_record, sort_keys=True),
                "candidate_order_strategy": order_strategy,
                "candidate_choice_map": choice_to_model,
                "position_neutrality_instruction": position_neutrality_instruction,
                "fairness_instruction": position_neutrality_instruction,
                "repair_attempt": 0,
                "repair_attempt_record": skip_record,
            }
        )
        selected = initial_selected.copy()
        llm_payload = normalize_llm_payload(
            {},
            selected=selected,
            selected_choice_id=initial_choice_id,
            decision="keep_with_risk",
            attempts=[],
            fallback_after_three=False,
            repair_skipped_reason="no_risk_free_candidate_available",
        )
        llm_payload["reason"] = "no risk-free candidate is available"
        llm_payload["updated_thought"] = (
            "Compatibility precheck found no candidate without a hard risk; retain "
            "the original tentative choice."
        )
        selected.attrs["llm_payload"] = llm_payload
        return selected

    if selection_replay:
        replay_row = dict(
            selection_replay[(str(dataset_key), str(forecast_origin))]
        )
        raw_history = replay_row.get("repair_attempts", "")
        replay_attempts: list[dict[str, object]] = []
        history_available = False
        if isinstance(raw_history, list):
            replay_attempts = [dict(row) for row in raw_history]
            history_available = True
        elif str(raw_history).strip():
            parsed_history = json.loads(str(raw_history))
            if not isinstance(parsed_history, list):
                raise ValueError("replayed repair_attempts must be a JSON list")
            replay_attempts = [dict(row) for row in parsed_history]
            history_available = True
        replay_attempt_count = int(
            float(replay_row.get("repair_attempt_count", len(replay_attempts)) or 0)
        )
        if history_available and replay_attempt_count != len(replay_attempts):
            raise ValueError(
                "replayed repair_attempt_count does not match repair_attempts"
            )
        replay_exhausted = str(
            replay_row.get(
                "fallback_to_initial_after_three_failed_repairs",
                "false",
            )
        ).strip().lower() in {"1", "true", "t", "yes", "y"}
        selected = select_candidate_no_validation(
            engine=None,
            stage="react_repair_select",
            dataset_key=dataset_key,
            forecast_origin=forecast_origin,
            candidates=candidates,
            task="Replay the final ReAct keep/repair decision.",
            trace_rows=trace_rows,
            context_text=json.dumps(trajectory, sort_keys=True),
            selection_replay=selection_replay,
        )
        replay_selected_id = str(selected["model_id"])
        replay_selected_choice_id = model_to_choice[replay_selected_id]
        replay_observation = _candidate_compatibility(profile, selected)
        replay_remaining = sorted(
            target_risk_flags.intersection(
                str(flag) for flag in replay_observation.get("risk_flags", [])
            )
        )
        repair_valid = bool(
            material_risk
            and replay_selected_id != initial_id
            and not replay_remaining
            and not bool(replay_observation.get("material_risk", False))
            and not list(replay_observation.get("risk_flags", []))
        )
        if replay_exhausted and (
            replay_attempt_count != MAX_REACT_REPAIR_ATTEMPTS
            or replay_selected_id != initial_id
        ):
            raise ValueError(
                "exhausted ReAct replay must record three attempts and finish at initial"
            )
        replay_repair_rejected = bool(
            replay_selected_id != initial_id and not repair_valid
        )
        if replay_repair_rejected:
            selected = initial_selected.copy()
            replay_selected_id = initial_id
            replay_selected_choice_id = initial_choice_id
        decision = (
            "repair"
            if replay_selected_id != initial_id
            else "keep_with_risk"
            if material_risk
            else "keep"
        )
        payload = normalize_llm_payload(
            dict(selected.attrs.get("llm_payload", {})),
            selected=selected,
            selected_choice_id=replay_selected_choice_id,
            decision=decision,
            attempts=replay_attempts,
            fallback_after_three=replay_exhausted,
            repair_skipped_reason="",
            attempt_history_available=history_available,
            attempt_history_source=(
                "replay_recorded_attempts"
                if history_available
                else "replay_final_choice_only"
            ),
        )
        payload["llm_proposed_choice_id"] = replay_selected_choice_id
        payload["llm_proposed_model_id"] = replay_selected_id
        payload["replay_repair_rejected_by_live_compatibility"] = (
            replay_repair_rejected
        )
        selected.attrs["llm_payload"] = payload
        return selected

    prompt_trajectory = json.loads(json.dumps(trajectory))
    prompt_trajectory.pop("trajectory_id", None)
    prompt_tentative = dict(prompt_trajectory.get("tentative_selection", {}))
    prompt_tentative.pop("model_id", None)
    prompt_tentative["choice_id"] = initial_choice_id
    prompt_trajectory["tentative_selection"] = prompt_tentative
    prompt_compatibility = dict(
        prompt_trajectory.get("tentative_compatibility_observation", {})
    )
    prompt_compatibility.pop("model_id", None)
    prompt_compatibility["choice_id"] = initial_choice_id
    prompt_trajectory["tentative_compatibility_observation"] = prompt_compatibility

    if not material_risk:
        payload: dict[str, object] = {
            "task": "Critique the tentative model after reading the complete ReAct trajectory.",
            "forecast_origin": forecast_origin,
            "trajectory": prompt_trajectory,
            "tentative_choice_id": initial_choice_id,
            "valid_choice_ids": valid_choice_ids,
            "candidates": rows,
            "required_response_schema": {
                "decision": "keep",
                "choice_id": "exactly one valid_choice_ids entry",
                "updated_thought": "short trajectory-aware assessment",
                "reason": "short reason",
            },
            "decision_contract": (
                "No material compatibility risk was observed. Return keep with the "
                "tentative choice_id; do not initiate repair."
            ),
            "evidence_contract": (
                "Use only the supplied as-of trajectory and scientific candidate descriptions. "
                "Do not infer validation scores or future targets."
            ),
            "strict_instruction": "Return JSON only and satisfy the decision contract exactly.",
        }
        call = engine.generate_json(
            stage="react_repair_select",
            system_prompt=(
                "You are the critic in a ReAct forecasting loop. Read Thought, Action, Observation, "
                "and tentative selection. Return compact JSON with decision, choice_id, "
                "updated_thought, and reason."
            ),
            user_prompt=json.dumps(payload, sort_keys=True),
            valid_model_ids=valid_choice_ids,
            retries=4,
        )
        decision = str(call.response_json.get("decision", "")).strip().lower()
        selected_choice_id = response_choice_id(call.response_json)
        contract_valid = bool(
            decision == "keep" and selected_choice_id == initial_choice_id
        )
        keep_record = {
            "attempt": 0,
            "choice_id": selected_choice_id,
            "candidate_model_id": choice_to_model.get(selected_choice_id, ""),
            "decision": decision,
            "compatibility_rechecked": False,
            "risk_before": [],
            "risk_after": [],
            "target_risk_remaining": [],
            "validation_status": "keep_valid" if contract_valid else "keep_contract_invalid",
            "repair_accepted": False,
        }
        append_call_trace(call, repair_attempt=0, attempt_record=keep_record)
        if not contract_valid:
            raise ValueError(
                "ReAct critic violated the no-risk keep contract for "
                f"dataset={dataset_key} origin={forecast_origin}"
            )
        selected = initial_selected.copy()
        llm_payload = normalize_llm_payload(
            call.response_json,
            selected=selected,
            selected_choice_id=initial_choice_id,
            decision="keep",
            attempts=[],
            fallback_after_three=False,
            repair_skipped_reason="initial_candidate_has_no_material_risk",
        )
        selected.attrs["llm_payload"] = llm_payload
        return selected

    attempts: list[dict[str, object]] = []
    prompt_attempts: list[dict[str, object]] = []
    tried_choice_ids: set[str] = set()
    last_response_json: dict[str, Any] = {}
    for repair_attempt in range(1, MAX_REACT_REPAIR_ATTEMPTS + 1):
        excluded_choice_ids = [
            choice_id
            for choice_id in valid_choice_ids
            if choice_id == initial_choice_id or choice_id in tried_choice_ids
        ]
        payload = {
            "task": "Repair the tentative choice to eliminate its material compatibility risk.",
            "forecast_origin": forecast_origin,
            "trajectory": prompt_trajectory,
            "tentative_choice_id": initial_choice_id,
            "initial_risk_flags": sorted(target_risk_flags),
            "repair_attempt": repair_attempt,
            "repair_attempt_limit": MAX_REACT_REPAIR_ATTEMPTS,
            "valid_choice_ids": valid_choice_ids,
            "excluded_choice_ids": excluded_choice_ids,
            "prior_repair_attempts": prompt_attempts,
            "candidates": rows,
            "required_response_schema": {
                "decision": "repair",
                "choice_id": "one non-excluded valid_choice_ids entry",
                "updated_thought": "short risk-aware assessment",
                "reason": "short reason",
            },
            "decision_contract": (
                "Return repair with a different, not previously attempted choice_id. "
                "The controller will independently recheck compatibility. Invalid, "
                "same, or repeated choices consume this repair attempt."
            ),
            "evidence_contract": (
                "Use only the supplied as-of trajectory and scientific candidate descriptions. "
                "Do not infer validation scores or future targets."
            ),
            "strict_instruction": "Return JSON only and satisfy the decision contract exactly.",
        }
        repair_system_prompt = (
            "You are the repair critic in a ReAct forecasting loop. Propose one "
            "anonymous choice_id intended to eliminate the supplied hard risk. "
            "Return compact JSON with decision, choice_id, updated_thought, and reason."
        )
        attempt_started = time.perf_counter()
        try:
            call = engine.generate_json(
                stage="react_repair_select",
                system_prompt=repair_system_prompt,
                user_prompt=json.dumps(payload, sort_keys=True),
                valid_model_ids=valid_choice_ids,
                retries=4,
            )
        except LLMError as exc:
            attempt_runtime_seconds = round(
                time.perf_counter() - attempt_started,
                6,
            )
            attempt_record = {
                "attempt": repair_attempt,
                "choice_id": "<llm_error>",
                "candidate_model_id": "",
                "decision": "",
                "valid_choice": False,
                "same_as_initial": False,
                "repeated_choice": False,
                "compatibility_rechecked": False,
                "risk_before": sorted(target_risk_flags),
                "risk_after": sorted(target_risk_flags),
                "target_risk_remaining": sorted(target_risk_flags),
                "validation_status": "llm_error",
                "repair_accepted": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            attempts.append(attempt_record)
            prompt_attempts.append(
                {
                    "attempt": repair_attempt,
                    "choice_id": "<llm_error>",
                    "decision": "",
                    "risk_after": sorted(target_risk_flags),
                    "target_risk_remaining": sorted(target_risk_flags),
                    "validation_status": "llm_error",
                }
            )
            trace_rows.append(
                {
                    "stage": "react_repair_select",
                    "dataset_key": dataset_key,
                    "forecast_origin": forecast_origin,
                    "model_path": str(getattr(engine, "active_model_path", "")),
                    "fallback_used": bool(getattr(engine, "fallback_used", False)),
                    "fallback_reason": str(getattr(engine, "fallback_reason", "")),
                    "runtime_seconds": attempt_runtime_seconds,
                    "prompt": f"{repair_system_prompt}\n\n{json.dumps(payload, sort_keys=True)}",
                    "response_text": "",
                    "candidate_order_strategy": order_strategy,
                    "candidate_choice_map": choice_to_model,
                    "position_neutrality_instruction": position_neutrality_instruction,
                    "fairness_instruction": position_neutrality_instruction,
                    "repair_attempt": repair_attempt,
                    "repair_attempt_record": attempt_record,
                }
            )
            continue
        last_response_json = dict(call.response_json)
        decision = str(call.response_json.get("decision", "")).strip().lower()
        proposed_choice_id = response_choice_id(call.response_json)
        proposed_model_id = choice_to_model.get(proposed_choice_id, "")
        valid_choice = proposed_choice_id in valid_choice_ids
        same_as_initial = proposed_choice_id == initial_choice_id
        repeated_choice = proposed_choice_id in tried_choice_ids
        compatibility_rechecked = False
        risk_after = sorted(target_risk_flags)
        target_risk_remaining = sorted(target_risk_flags)
        repaired_observation: dict[str, object] | None = None
        if valid_choice:
            proposed = candidates[
                candidates["model_id"].astype(str).eq(proposed_model_id)
            ].iloc[0]
            repaired_observation = _candidate_compatibility(profile, proposed)
            compatibility_rechecked = True
            risk_after = sorted(
                str(flag)
                for flag in repaired_observation.get("risk_flags", [])
                if str(flag)
            )
            target_risk_remaining = sorted(
                target_risk_flags.intersection(risk_after)
            )
        risk_resolved = bool(
            decision == "repair"
            and valid_choice
            and not same_as_initial
            and not repeated_choice
            and repaired_observation is not None
            and not bool(repaired_observation.get("material_risk", False))
            and not risk_after
            and not target_risk_remaining
        )
        if not valid_choice:
            validation_status = "invalid_choice"
        elif same_as_initial:
            validation_status = "same_as_initial"
        elif repeated_choice:
            validation_status = "repeated_choice"
        elif decision != "repair":
            validation_status = "invalid_decision"
        elif risk_resolved:
            validation_status = "risk_resolved"
        else:
            validation_status = "risk_remains"
        attempt_record = {
            "attempt": repair_attempt,
            "choice_id": (
                proposed_choice_id if valid_choice else "<invalid_choice>"
            ),
            "candidate_model_id": proposed_model_id,
            "decision": decision,
            "valid_choice": valid_choice,
            "same_as_initial": same_as_initial,
            "repeated_choice": repeated_choice,
            "compatibility_rechecked": compatibility_rechecked,
            "risk_before": sorted(target_risk_flags),
            "risk_after": risk_after,
            "target_risk_remaining": target_risk_remaining,
            "validation_status": validation_status,
            "repair_accepted": risk_resolved,
        }
        attempts.append(attempt_record)
        append_call_trace(
            call,
            repair_attempt=repair_attempt,
            attempt_record=attempt_record,
        )
        prompt_attempts.append(
            {
                "attempt": repair_attempt,
                "choice_id": attempt_record["choice_id"],
                "decision": decision,
                "risk_after": risk_after,
                "target_risk_remaining": target_risk_remaining,
                "validation_status": validation_status,
            }
        )
        if valid_choice:
            tried_choice_ids.add(proposed_choice_id)
        if risk_resolved:
            assert repaired_observation is not None
            selected = candidates[
                candidates["model_id"].astype(str).eq(proposed_model_id)
            ].iloc[0].copy()
            llm_payload = normalize_llm_payload(
                call.response_json,
                selected=selected,
                selected_choice_id=proposed_choice_id,
                decision="repair",
                attempts=attempts,
                fallback_after_three=False,
            )
            selected.attrs["llm_payload"] = llm_payload
            return selected

    selected = initial_selected.copy()
    llm_payload = normalize_llm_payload(
        last_response_json,
        selected=selected,
        selected_choice_id=initial_choice_id,
        decision="keep_with_risk",
        attempts=attempts,
        fallback_after_three=True,
    )
    llm_payload["reason"] = "three repair attempts failed compatibility validation"
    llm_payload["updated_thought"] = (
        "All three repair proposals failed; revert to the original tentative choice."
    )
    selected.attrs["llm_payload"] = llm_payload
    return selected


def run_react_agent_from_manifest(
    *,
    manifest_path: str | Path,
    registry_path: str | Path,
    out_dir: str | Path,
    engine: Any | None = None,
    dataset_keys: list[str] | tuple[str, ...] | None = None,
    selection_policy: str = "llm_only",
    selection_replay_log: str | Path | None = None,
    forecast_archive: str | Path | None = None,
    archive_mode: str = "off",
    selection_input_candidates: str | Path | None = None,
    selection_input_manifest: str | Path | None = None,
    require_selection_input_parity: bool = False,
    charge_selection: bool = False,
    timing_mode: str | None = None,
    exclude_llm_load_from_timing: bool = False,
    root: Path | None = None,
    caster_root: Path | None = None,
) -> Path:
    start = time.time()
    if selection_policy not in REACT_SELECTION_POLICIES:
        raise ValueError(f"selection_policy must be one of {REACT_SELECTION_POLICIES}; got {selection_policy!r}")
    resolved_timing_mode = resolve_timing_mode(archive_mode=archive_mode, timing_mode=timing_mode)
    archive_backed = resolved_timing_mode == ARCHIVE_TIMING_MODE
    archive_index: ForecastArchiveIndex | None = load_forecast_archive(forecast_archive) if archive_backed else None
    root = root or baseline_root()
    caster_root = caster_root or caster_root_from_baseline(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection_replay = load_selection_replay_log(selection_replay_log)
    if engine is None and not selection_replay:
        engine = QwenLocalEngine()
    llm_load_seconds_excluded = 0.0
    if exclude_llm_load_from_timing and engine is not None and hasattr(engine, "warm_load"):
        llm_load_seconds_excluded = float(engine.warm_load())

    registry = read_candidate_registry(registry_path)
    frozen_selection = None
    if selection_input_candidates is not None or selection_input_manifest is not None:
        if selection_input_candidates is None or selection_input_manifest is None:
            raise ValueError("common selection candidates and manifest must be supplied together")
        frozen_selection = load_common_selection_input(selection_input_candidates, selection_input_manifest)
        registry = merge_common_selection_input(registry, frozen_selection)
    if require_selection_input_parity and frozen_selection is None:
        raise ValueError("formal noLearning ReAct requires the common selection packet")
    selection_provenance = frozen_selection.provenance() if frozen_selection else {}
    eligible = eligible_registry(registry)
    if eligible.empty:
        raise RuntimeError("no ReAct-eligible candidates in registry")
    registry_meta = write_registry_snapshot(registry, out, registry_path)
    contexts, _manifest_expected_rows = load_dataset_contexts(manifest_path, root=root, caster_root=caster_root)
    all_dataset_keys = sorted({str(ctx.dataset_key) for ctx in contexts})
    requested_dataset_keys = [str(key) for key in dataset_keys] if dataset_keys is not None else []
    if "all" in requested_dataset_keys:
        requested_dataset_keys = []
    if requested_dataset_keys:
        wanted = set(requested_dataset_keys)
        contexts = [ctx for ctx in contexts if ctx.dataset_key in wanted]
        if not contexts:
            raise RuntimeError(f"no manifest rows selected for agent_react dataset_keys={requested_dataset_keys}")
    selected_dataset_keys = sorted({str(ctx.dataset_key) for ctx in contexts})
    if any(is_benchmark_b_context(ctx) for ctx in contexts):
        if not archive_backed:
            raise ValueError("benchmark_b_pooled ReAct executor must use the shared frozen archive")
    excluded_dataset_keys = sorted(set(all_dataset_keys) - set(selected_dataset_keys))
    dataset_scope = "all_available" if not excluded_dataset_keys else "custom"
    agent_selection_scope = derive_agent_selection_scope(selected_dataset_keys, excluded_dataset_keys)
    expected_rows = int(sum(len(ctx.ledger) for ctx in contexts))

    trace_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    react_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    forecast_frames: list[pd.DataFrame] = []
    proposal_seconds = 0.0
    train_seconds = 0.0
    forecast_seconds = 0.0

    for ctx in contexts:
        for origin_text, release in ctx.ledger.groupby("forecast_origin", dropna=False, sort=True):
            origin_text = str(origin_text)
            canonical_context = (
                benchmark_b_prompt_payload(ctx, forecast_origin=origin_text, frozen_selection=frozen_selection)
                if is_benchmark_b_context(ctx)
                else {}
            )
            step = 1
            group_start = time.time()
            if canonical_context:
                base_context = benchmark_b_prompt_text(
                    canonical_context["canonical_context"],
                    str(canonical_context["canonical_context_sha256"]),
                )
                context_metadata = {
                    "selection_context_schema": canonical_context["canonical_context"]["schema"],
                    "selection_context_sha256": canonical_context["canonical_context_sha256"],
                    "selection_context_cutoff": origin_text,
                    "selection_context_builder": "caster.data.benchmark_b_context.build_canonical_context",
                }
            else:
                base_context, context_metadata = selection_context_with_metadata(
                    ctx, release=release, origin_text=origin_text
                )
            data_profile = _asof_data_profile(ctx, release, origin_text)
            thought_start = time.time()
            thought_payload = {
                "forecast_origin": origin_text,
                "release_rows": int(len(release)),
                "dataset_context": base_context,
                "task": (
                    "Choose one inspection action using released history but no validation score or unreleased outcome. "
                    "The action observation will be consumed by the selector and critic."
                ),
                "required_response": {
                    "thought": "short reason",
                    "next_action": "inspect_context | inspect_candidates | select_model",
                    "selection_intent": "short model-assumption criterion",
                },
                **(
                    benchmark_b_llm_prompt_fields(canonical_context)
                    if canonical_context
                    else {}
                ),
            }
            if engine is None:
                thought_response: dict[str, object] = {
                    "thought": "Inspect candidate compatibility before replaying the recorded final choice.",
                    "next_action": "inspect_candidates",
                    "selection_intent": "match model assumptions to the as-of history and horizons",
                }
                trace_rows.append({
                    "stage": "react_thought",
                    "dataset_key": ctx.dataset_key,
                    "forecast_origin": origin_text,
                    "step": 0,
                    "thought": str(thought_response["thought"]),
                    "action": str(thought_response["next_action"]),
                    "observation": "",
                    "model_path": "deterministic_replay_controller",
                    "fallback_used": False,
                    "fallback_reason": "",
                    "runtime_seconds": 0.0,
                    "prompt": "",
                    "response_text": json.dumps(thought_response, sort_keys=True),
                })
            else:
                thought_response = _call_react_thought(
                    engine,
                    dataset_key=ctx.dataset_key,
                    forecast_origin=origin_text,
                    payload=thought_payload,
                    trace_rows=trace_rows,
                )
            proposal_seconds += round(time.time() - thought_start, 6)

            react_action = _normalise_react_action(thought_response.get("next_action", ""))
            action_observation = _dispatch_react_action(
                react_action,
                profile=data_profile,
                candidates=eligible,
            )
            if canonical_context:
                action_observation = {
                    "tool_result": action_observation,
                    **benchmark_b_llm_prompt_fields(canonical_context),
                }
            _record_react_event(
                trace_rows=trace_rows,
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                step=step,
                thought=str(thought_response.get("thought", "")),
                action=react_action,
                observation=action_observation,
            )
            step += 1

            preselection_trajectory = {
                "initial_thought": str(thought_response.get("thought", "")),
                "next_action": react_action,
                "selection_intent": str(thought_response.get("selection_intent", "")),
                "action_observation": action_observation,
            }
            select_start = time.time()
            initial_selected = select_candidate_no_validation(
                engine=engine,
                stage="react_select",
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                candidates=eligible,
                task=(
                    "Select one tentative forecasting method. The supplied as-of trajectory is evidence: "
                    "use its Thought, dispatched Action, and release-time-valid Observation."
                ),
                trace_rows=trace_rows,
                context_text=(
                    base_context
                    + "; asof_react_preselection_trajectory="
                    + json.dumps(preselection_trajectory, sort_keys=True, separators=(",", ":"))
                ),
                context_metadata=context_metadata,
                selection_replay=selection_replay or None,
            )
            initial_selected.attrs.setdefault("llm_payload", {})["agent_selection_scope"] = agent_selection_scope
            initial_selected.attrs.setdefault("llm_payload", {}).update(canonical_context)
            initial_selected.attrs.setdefault("llm_payload", {}).update(selection_provenance)
            proposal_seconds += round(time.time() - select_start, 6)
            initial_selected_id = str(initial_selected["model_id"])
            compatibility_observation = _candidate_compatibility(data_profile, initial_selected)
            _record_react_event(
                trace_rows=trace_rows,
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                step=step,
                thought="Inspect the tentative model against the as-of data profile before execution.",
                action=f"inspect_tentative_model[{initial_selected_id}]",
                observation=compatibility_observation,
            )
            step += 1

            trajectory = {
                **preselection_trajectory,
                "tentative_selection": {
                    "model_id": initial_selected_id,
                    "scientific_description": str(initial_selected.get("description", "")),
                    "selection_reason": str(initial_selected.attrs.get("llm_payload", {}).get("reason", "")),
                },
                "tentative_compatibility_observation": compatibility_observation,
            }
            trajectory_id = _trajectory_id(trajectory)
            trajectory["trajectory_id"] = trajectory_id
            summary_start = time.time()
            selected = _summary_candidate(
                engine=engine,
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                candidates=eligible,
                initial_selected=initial_selected,
                profile=data_profile,
                trajectory=trajectory,
                trace_rows=trace_rows,
                selection_replay=selection_replay or None,
            )
            selected_payload = selected.attrs.setdefault("llm_payload", {})
            selected_payload["agent_selection_scope"] = agent_selection_scope
            selected_payload.update(context_metadata)
            selected_payload.update(canonical_context)
            selected_payload.update(selection_provenance)
            proposal_seconds += round(time.time() - summary_start, 6)
            selected_id = str(selected["model_id"])
            repair_applied = selected_id != initial_selected_id
            summary_decision = str(selected.attrs.get("llm_payload", {}).get("decision", ""))
            final_compatibility_observation = selected.attrs.get("llm_payload", {}).get(
                "final_compatibility_observation",
                _candidate_compatibility(data_profile, selected),
            )
            remaining_target_risks = selected.attrs.get("llm_payload", {}).get(
                "repair_remaining_target_risk_flags",
                [],
            )
            repair_attempt_count = int(
                selected_payload.get("repair_attempt_count", 0)
            )
            repair_attempt_limit = int(
                selected_payload.get(
                    "repair_attempt_limit",
                    MAX_REACT_REPAIR_ATTEMPTS,
                )
            )
            repair_attempts = selected_payload.get("repair_attempts", [])
            fallback_after_three = bool(
                selected_payload.get(
                    "fallback_to_initial_after_three_failed_repairs",
                    False,
                )
            )
            repair_skipped_reason = str(
                selected_payload.get("repair_skipped_reason", "")
            )
            _record_react_event(
                trace_rows=trace_rows,
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                step=step,
                thought=str(
                    selected.attrs.get("llm_payload", {}).get(
                        "updated_thought",
                        (
                            "The critic repaired the tentative choice after consuming the trajectory."
                            if repair_applied
                            else "The critic kept the tentative choice after consuming the trajectory."
                        ),
                    )
                ),
                action=(f"repair_model[{selected_id}]" if repair_applied else f"keep_model[{selected_id}]"),
                observation={
                    "initial_selected_model_id": initial_selected_id,
                    "final_selected_model_id": selected_id,
                    "decision": summary_decision,
                    "repair_applied": bool(repair_applied),
                    "repair_compatibility_rechecked": True,
                    "repair_remaining_target_risk_flags": remaining_target_risks,
                    "repair_attempt_count": repair_attempt_count,
                    "repair_attempt_limit": repair_attempt_limit,
                    "repair_attempts": repair_attempts,
                    "fallback_to_initial_after_three_failed_repairs": fallback_after_three,
                    "repair_skipped_reason": repair_skipped_reason,
                    "final_compatibility_observation": final_compatibility_observation,
                    "trajectory_id": trajectory_id,
                    "trajectory_consumed": True,
                    "selection_score_source": "no_validation",
                    "validation_context_visible": False,
                    **canonical_context,
                },
            )
            step += 1

            selection_row = selection_log_row(
                stage="react_repair_select",
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                selected=selected,
                llm_payload=selected.attrs.get("llm_payload", {}),
            )
            selection_row.update(
                {
                    "initial_selected_model_id": initial_selected_id,
                    "final_selected_model_id": selected_id,
                    "react_action": react_action,
                    "react_trajectory_id": trajectory_id,
                    "summary_decision": summary_decision,
                    "compatibility_material_risk": bool(compatibility_observation["material_risk"]),
                    "compatibility_risk_flags": json.dumps(compatibility_observation["risk_flags"], sort_keys=True),
                    "repair_compatibility_rechecked": True,
                    "repair_remaining_target_risk_flags": json.dumps(
                        remaining_target_risks,
                        sort_keys=True,
                    ),
                    "final_compatibility_risk_flags": json.dumps(
                        final_compatibility_observation.get("risk_flags", []),
                        sort_keys=True,
                    ),
                    "repair_attempt_count": repair_attempt_count,
                    "repair_attempt_limit": repair_attempt_limit,
                    "repair_attempts": json.dumps(repair_attempts, sort_keys=True),
                    "fallback_to_initial_after_three_failed_repairs": fallback_after_three,
                    "repair_skipped_reason": repair_skipped_reason,
                    "repair_applied": bool(repair_applied),
                    "summary_policy": REACT_summary_POLICY,
                }
            )
            selection_rows.append(selection_row)

            fit_start = time.time()
            predictions_cache = None
            if archive_backed:
                assert archive_index is not None
                validate_archive_coverage(
                    ctx=ctx,
                    ledger_subset=release,
                    selected_model_id=str(selected["model_id"]),
                    archive_index=archive_index,
                    out_dir=out,
                )
            elif is_stateful(selected):
                predictions_cache = pre_fit_predictions(selected, ctx, release)
            train_seconds += 0.0 if archive_backed else round(time.time() - fit_start, 6)

            fcst_start = time.time()
            release_forecast_rows = []
            if archive_backed:
                assert archive_index is not None
                release_frame = make_forecast_rows_from_archive(
                    ctx=ctx,
                    ledger_subset=release,
                    selected=selected,
                    method="agent_react",
                    archive_index=archive_index,
                )
                fcst_runtime = round(time.time() - fcst_start, 6)
                release_frame["agent_selection_scope"] = agent_selection_scope
                release_frame["initial_selected_model_id"] = initial_selected_id
                release_frame["react_action"] = react_action
                release_frame["react_trajectory_id"] = trajectory_id
                release_frame["react_summary_decision"] = summary_decision
                release_frame["react_compatibility_material_risk"] = bool(compatibility_observation["material_risk"])
                release_frame["react_repair_applied"] = bool(repair_applied)
                release_frame["react_repair_attempt_count"] = repair_attempt_count
                release_frame["react_repair_attempt_limit"] = repair_attempt_limit
                release_frame[
                    "react_fallback_to_initial_after_three_failed_repairs"
                ] = fallback_after_three
                for key, value in selection_provenance.items():
                    release_frame[key] = value
                release_frame = annotate_forecast_context(
                    release_frame, ctx=ctx, frozen_selection=frozen_selection
                )
                forecast_frames.append(release_frame)
                release_forecast_count = int(len(release_frame))
            else:
                for ledger_idx, event in release.iterrows():
                    row, _row_train = make_forecast_row(
                        ctx=ctx,
                        event=event,
                        ledger_idx=int(ledger_idx),
                        selected=selected,
                        method="agent_react",
                        predictions_cache=predictions_cache,
                    )
                    row["agent_selection_scope"] = agent_selection_scope
                    row["initial_selected_model_id"] = initial_selected_id
                    row["react_action"] = react_action
                    row["react_trajectory_id"] = trajectory_id
                    row["react_summary_decision"] = summary_decision
                    row["react_compatibility_material_risk"] = bool(compatibility_observation["material_risk"])
                    row["react_repair_applied"] = bool(repair_applied)
                    row["react_repair_attempt_count"] = repair_attempt_count
                    row["react_repair_attempt_limit"] = repair_attempt_limit
                    row[
                        "react_fallback_to_initial_after_three_failed_repairs"
                    ] = fallback_after_three
                    release_forecast_rows.append(row)
                release_forecast_count = int(len(release_forecast_rows))
                fcst_runtime = round(time.time() - fcst_start, 6)
            forecast_seconds += fcst_runtime
            if not archive_backed:
                forecast_rows.extend(release_forecast_rows)
            _record_react_event(
                trace_rows=trace_rows,
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                step=step,
                thought="The selected skill has been executed on the current release group; stop without posterior maintenance.",
                action="execute_forecast",
                observation={
                    "forecast_rows": release_forecast_count,
                    "selected_model_id": str(selected["model_id"]),
                    "forecast_source": ARCHIVE_FORECAST_SOURCE if archive_backed else "baseline_local_refit_forecast_recipe",
                    "elapsed_group_seconds": round(time.time() - group_start, 6),
                    **canonical_context,
                },
            )
            react_rows.append({
                "dataset_key": ctx.dataset_key,
                "forecast_origin": origin_text,
                "selected_model_id": str(selected["model_id"]),
                "initial_selected_model_id": initial_selected_id,
                "react_action": react_action,
                "react_trajectory_id": trajectory_id,
                "summary_decision": summary_decision,
                "compatibility_material_risk": bool(compatibility_observation["material_risk"]),
                "compatibility_risk_flags": json.dumps(compatibility_observation["risk_flags"], sort_keys=True),
                "initial_thought_consumed": True,
                "action_observation_consumed": True,
                "repair_applied": bool(repair_applied),
                "repair_attempt_count": repair_attempt_count,
                "repair_attempt_limit": repair_attempt_limit,
                "repair_attempts": json.dumps(repair_attempts, sort_keys=True),
                "fallback_to_initial_after_three_failed_repairs": fallback_after_three,
                "repair_skipped_reason": repair_skipped_reason,
                "repair_attempt_history_available": bool(
                    selected_payload.get("repair_attempt_history_available", True)
                ),
                "repair_attempt_history_source": str(
                    selected_payload.get(
                        "repair_attempt_history_source",
                        "semantic_critic_attempts",
                    )
                ),
                "forecast_rows": release_forecast_count,
                "validation_rows": 0,
                "selection_policy": selection_policy,
                "selection_method": str(selected.attrs.get("llm_payload", {}).get("selection_method", "")),
                "selection_score_source": "no_validation",
                "status": "ok",
            })

    forecast = pd.concat(forecast_frames, ignore_index=True) if archive_backed else pd.DataFrame(forecast_rows)
    if len(forecast) != expected_rows:
        raise RuntimeError(f"forecast row mismatch expected={expected_rows} actual={len(forecast)}")

    pd.DataFrame(selection_rows).to_csv(out / "candidate_selection_log.csv", index=False)
    pd.DataFrame(react_rows).to_csv(out / "react_log.csv", index=False)
    _jsonl(out / "react_trace.jsonl", trace_rows)
    config = qwen_config(engine)
    config.update({
        "llm_runtime_seconds": total_trace_runtime(trace_rows),
        "primary_model_path": getattr(engine, "primary_model_path", ""),
        "fallback_model_path": getattr(engine, "fallback_model_path", ""),
    })
    (out / "qwen_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    action_counts = {
        action: int(sum(str(row.get("react_action", "")) == action for row in react_rows))
        for action in REACT_ACTIONS
    }
    material_risk_count = int(sum(bool(row.get("compatibility_material_risk", False)) for row in react_rows))
    repair_attempt_total = int(
        sum(int(row.get("repair_attempt_count", 0)) for row in react_rows)
    )
    repair_exhaustion_count = int(
        sum(
            bool(row.get("fallback_to_initial_after_three_failed_repairs", False))
            for row in react_rows
        )
    )
    repair_skipped_count = int(
        sum(bool(str(row.get("repair_skipped_reason", ""))) for row in react_rows)
    )
    replay_attempt_history_unavailable_count = int(
        sum(
            not bool(row.get("repair_attempt_history_available", True))
            for row in react_rows
        )
    )

    timing_extra = {
        **config,
        "restart_type": "archive_backed_true_selection_readout"
        if archive_backed and not selection_replay
        else ("archive_backed_selection_replay_readout" if archive_backed else "react_selection_refit_forecast"),
        "proposal_seconds": round(proposal_seconds, 6),
        "llm_runtime_seconds": total_trace_runtime(trace_rows),
        "llm_load_seconds_excluded_from_update": round(float(llm_load_seconds_excluded), 6),
        "llm_generation_seconds_charged_to_update": round(total_trace_runtime(trace_rows), 6),
        "selection_engine": "selection_replay" if selection_replay else "qwen",
        "react_group_rows": int(len(react_rows)),
        "react_repair_count": int(sum(bool(row.get("repair_applied", False)) for row in react_rows)),
        "react_repair_attempt_limit_per_group": MAX_REACT_REPAIR_ATTEMPTS,
        "react_repair_attempt_total": repair_attempt_total,
        "react_repair_exhaustion_count": repair_exhaustion_count,
        "react_repair_skipped_count": repair_skipped_count,
        "react_replay_attempt_history_unavailable_count": replay_attempt_history_unavailable_count,
        "react_material_risk_count": material_risk_count,
        "react_action_counts": action_counts,
        "react_summary_policy": REACT_summary_POLICY,
        "react_causal_trajectory": True,
        "react_initial_thought_consumed": True,
        "react_action_dispatched": True,
        "react_observation_consumed": True,
        "react_summary_schema": "decision=keep|repair|keep_with_risk;actual_selected_model_id=valid_candidate;repair_attempts=0..3;trajectory_consumed=true",
        "react_repair_gate": REACT_REPAIR_GATE,
        "selection_policy": selection_policy,
        "selection_score_source": "no_validation",
        "validation_score_used": False,
        "selection_replay_used": bool(selection_replay),
        "agent_selection_scope": agent_selection_scope,
    }
    if archive_backed:
        timing = archive_timing_payload(
            start_time=start,
            archive_lookup_seconds=forecast_seconds,
            agent_control_seconds=0.0,
            selection_seconds=proposal_seconds,
            charge_selection=charge_selection,
            forecast_rows=int(len(forecast)),
            expected_rows=int(expected_rows),
            extra=timing_extra,
        )
    else:
        timing = alternate_timing_payload({
            "restart_type": "react_selection_refit_forecast",
            "artifact_reuse": False,
            "proposal_seconds": round(proposal_seconds, 6),
            "train_seconds": round(train_seconds, 6),
            "forecast_seconds": round(forecast_seconds, 6),
            "total_seconds": round(time.time() - start, 6),
            "llm_runtime_seconds": total_trace_runtime(trace_rows),
            "forecast_rows": int(len(forecast)),
            "expected_rows": int(expected_rows),
            "react_group_rows": int(len(react_rows)),
            "react_repair_count": int(
                sum(bool(row.get("repair_applied", False)) for row in react_rows)
            ),
            "react_repair_attempt_limit_per_group": MAX_REACT_REPAIR_ATTEMPTS,
            "react_repair_attempt_total": repair_attempt_total,
            "react_repair_exhaustion_count": repair_exhaustion_count,
            "react_repair_skipped_count": repair_skipped_count,
            "react_replay_attempt_history_unavailable_count": replay_attempt_history_unavailable_count,
            "react_material_risk_count": material_risk_count,
            "react_action_counts": action_counts,
            "react_summary_policy": REACT_summary_POLICY,
            "react_causal_trajectory": True,
            "react_initial_thought_consumed": True,
            "react_action_dispatched": True,
            "react_observation_consumed": True,
            "react_summary_schema": "decision=keep|repair|keep_with_risk;actual_selected_model_id=valid_candidate;repair_attempts=0..3;trajectory_consumed=true",
            "react_repair_gate": REACT_REPAIR_GATE,
            "selection_policy": selection_policy,
            "selection_score_source": "no_validation",
            "validation_score_used": False,
            "selection_replay_used": bool(selection_replay),
        })
    run_manifest = {
        "baseline_names": ["agent_react"],
        "model": "agent_react",
        "backend": "react_style_qwen_local",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "selected_dataset_keys": selected_dataset_keys,
        "excluded_dataset_keys": excluded_dataset_keys,
        "dataset_scope": dataset_scope,
        "agent_selection_scope": agent_selection_scope,
        **registry_meta,
        **REACT_REFERENCE,
        **config,
        "restart_type": "archive_backed_true_selection_readout"
        if archive_backed and not selection_replay
        else ("archive_backed_selection_replay_readout" if archive_backed else "react_selection_refit_forecast"),
        "timing_mode": resolved_timing_mode,
        "timing_semantics": "agent_selection_control_plus_immutable_forecast_archive_readout" if archive_backed else alternate_TIMING_MODE,
        "forecast_source": ARCHIVE_FORECAST_SOURCE if archive_backed else "baseline_local_refit_forecast_recipe",
        "forecast_archive": str(forecast_archive or ""),
        "archive_mode": str(archive_mode),
        "formal_timing_valid": bool(archive_backed),
        "model_compute_excluded": bool(archive_backed),
        "selection_charged_to_update": bool(charge_selection),
        "artifact_reuse": False,
        "selection_policy": selection_policy,
        "selection_guard": "disabled; no validation scores or validation prompt context are used",
        "selection_score_source": "no_validation",
        "validation_score_used": False,
        "validation_context_visible": False,
        "validation_guard_enabled": False,
        "selection_replay_used": bool(selection_replay),
        "selection_replay_log": str(selection_replay_log or ""),
        "selection_replay_log_sha256": sha256_file(Path(selection_replay_log)) if selection_replay_log else "",
        "selection_guard_scoreboard": "",
        "selection_guard_weighted_scoreboard": "",
        "selection_weighted_validation": "",
        "expected_rows": int(expected_rows),
        "react_group_rows": int(len(react_rows)),
        "react_repair_count": int(sum(bool(row.get("repair_applied", False)) for row in react_rows)),
        "react_repair_attempt_limit_per_group": MAX_REACT_REPAIR_ATTEMPTS,
        "react_repair_attempt_total": repair_attempt_total,
        "react_repair_exhaustion_count": repair_exhaustion_count,
        "react_repair_skipped_count": repair_skipped_count,
        "react_replay_attempt_history_unavailable_count": replay_attempt_history_unavailable_count,
        "react_material_risk_count": material_risk_count,
        "react_action_counts": action_counts,
        "react_summary_policy": REACT_summary_POLICY,
        "react_causal_trajectory": True,
        "react_initial_thought_consumed": True,
        "react_action_dispatched": True,
        "react_observation_consumed": True,
        "react_summary_schema": "decision=keep|repair|keep_with_risk;actual_selected_model_id=valid_candidate;repair_attempts=0..3;trajectory_consumed=true",
        "react_repair_gate": REACT_REPAIR_GATE,
        "react_observation_basis": "asof_history_horizon_strategy_and_candidate_metadata_only",
        "no_leakage_rule": (
            "ReAct profiles use only target values whose observation and release times do not exceed "
            "the forecast origin plus scientific candidate descriptions; no validation scores, "
            "current unreleased outcomes, or future targets are used for selection"
        ),
        **selection_provenance,
    }
    for context in contexts:
        run_manifest.update(benchmark_b_context_manifest_fields(context, frozen_selection))
    return write_standard_artifacts(out_dir=out, forecast=forecast, timing=timing, run_manifest=run_manifest)
