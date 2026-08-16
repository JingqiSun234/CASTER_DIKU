from __future__ import annotations

import json

import pandas as pd
import pytest

from caster_baselines import agentic_react
from caster_baselines.agentic_llm import LLMError, ReplayJSONEngine
from caster_baselines.agentic_top_one import (
    _call_stage as _call_top_one_stage,
    _task_planning_context,
    _task_planning_trace_metadata,
)
from caster_baselines.agentic_skills import (
    fair_registry_prompt_rows,
    select_candidate_with_llm,
    select_candidate_with_replay,
)


DATASET_KEY = "unit_task"
FORECAST_ORIGIN = "2025-01-04"


def _candidates(model_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "family": "unit_family",
                "candidate_type": "unit_type",
                "description": f"Scientific mechanism card {index}.",
                "priority": 100 - index,
                "recipe": f"private_recipe_{index}",
            }
            for index, model_id in enumerate(model_ids, start=1)
        ]
    )


def _choice_for(candidates: pd.DataFrame, model_id: str) -> str:
    _rows, choice_to_model = fair_registry_prompt_rows(
        candidates,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
    )
    return next(
        choice_id
        for choice_id, mapped_model_id in choice_to_model.items()
        if mapped_model_id == model_id
    )


def _assert_prompt_surface_is_anonymous(prompt: str, model_ids: list[str]) -> None:
    assert DATASET_KEY not in prompt
    for model_id in model_ids:
        assert model_id not in prompt
    for forbidden_control in (
        '"priority"',
        '"recipe"',
        '"candidate_order_strategy"',
        '"position_neutrality_instruction"',
        '"fairness_instruction"',
        '"dataset_key"',
        "benchmark_",
    ):
        assert forbidden_control not in prompt


def test_selector_exposes_only_deterministic_anonymous_scientific_cards() -> None:
    candidates = _candidates(["internal_alpha", "internal_beta", "internal_gamma"])
    rows_a, choice_map_a = fair_registry_prompt_rows(
        candidates,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
    )
    rows_b, choice_map_b = fair_registry_prompt_rows(
        candidates,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
    )
    assert rows_a == rows_b
    assert choice_map_a == choice_map_b
    assert all(set(row) == {"choice_id", "scientific_description"} for row in rows_a)

    selected_choice_id = rows_a[1]["choice_id"]
    engine = ReplayJSONEngine(
        {"choice_id": selected_choice_id, "reason": "mechanism fits"}
    )
    trace_rows: list[dict[str, object]] = []
    selected = select_candidate_with_llm(
        engine=engine,
        stage="unit_select",
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        task="Choose from the scientific cards.",
        trace_rows=trace_rows,
        context_text="Daily released observations with a two-step horizon.",
    )

    assert selected["model_id"] == choice_map_a[str(selected_choice_id)]
    _assert_prompt_surface_is_anonymous(
        engine.calls[0].prompt,
        candidates["model_id"].astype(str).tolist(),
    )
    prompt_payload = json.loads(engine.calls[0].prompt.split("\n\n", maxsplit=1)[1])
    assert set(prompt_payload["required_response_schema"]) == {"choice_id", "reason"}
    assert trace_rows[0]["candidate_choice_map"] == choice_map_a
    llm_payload = selected.attrs["llm_payload"]
    assert llm_payload["candidate_choice_map"] == choice_map_a
    assert llm_payload["selected_choice_id"] == selected_choice_id
    assert "candidate_order_strategy" in llm_payload
    assert "position_neutrality_instruction" in llm_payload
    assert "fairness_instruction" in llm_payload
    assert llm_payload["position_neutrality_instruction_present"] is False
    assert llm_payload["position_neutrality_control_trace_only"] is True


def test_top_one_planner_keeps_control_claims_out_of_llm_prompt() -> None:
    static_context = {
        "selection_context_profile": "qwen25_multiscale_released_sequence_v1",
        "scientific_context": {"released_history_rows": 12},
        "canonical_context_sha256": "audit-only-hash",
    }
    planning_payload = {
        "task": "Plan a deterministic top-1 baseline.",
        "required_response": {"plan": "short plan"},
        **_task_planning_context(static_context),
    }
    engine = ReplayJSONEngine({"plan": "inspect released history"})
    trace_rows: list[dict[str, object]] = []
    _call_top_one_stage(
        engine,
        stage="task_planning",
        dataset_key=DATASET_KEY,
        prompt_payload=planning_payload,
        trace_rows=trace_rows,
    )
    trace_rows[-1].update(_task_planning_trace_metadata(static_context))

    prompt = engine.calls[0].prompt
    assert "scientific_context" in prompt
    assert "full_augmented_context_used_by_model_selection" not in prompt
    assert "planner_output_consumed_by_selector" not in prompt
    assert trace_rows[-1]["full_augmented_context_used_by_model_selection"] is True
    assert trace_rows[-1]["full_augmented_context_control_trace_only"] is True
    assert trace_rows[-1]["planner_output_consumed_by_selector"] is False


def _compatibility(risks_by_model: dict[str, list[str]]):
    def fake_compatibility(_profile, selected):
        model_id = str(selected["model_id"])
        risks = list(risks_by_model.get(model_id, []))
        return {
            "model_id": model_id,
            "scientific_description": str(selected["description"]),
            "risk_flags": risks,
            "strength_flags": [],
            "material_risk": bool(risks),
        }

    return fake_compatibility


def _profile(**overrides: object) -> dict[str, object]:
    profile: dict[str, object] = {
        "history_rows_median": 100.0,
        "history_rows_min": 100,
        "season_length": 7,
        "decomposition_season_length": 7,
        "horizon_max": 1,
        "trend_signal": "mixed_or_mild",
        "recursive_rollout_present": False,
        "full_season_history_available": True,
    }
    profile.update(overrides)
    return profile


@pytest.mark.parametrize(
    ("model_id", "family", "profile", "expected_risk"),
    [
        (
            "seasonal_naive",
            "statistical",
            _profile(full_season_history_available=False),
            "insufficient_asof_history_for_full_season_lag",
        ),
        (
            "prophet",
            "statistical",
            _profile(history_rows_median=10.0),
            "limited_asof_cycles_for_seasonal_decomposition",
        ),
        (
            "lstm_style",
            "neural",
            _profile(history_rows_median=8.0, horizon_max=4),
            "short_asof_history_for_high_capacity_sequence_model",
        ),
        (
            "chronos_external",
            "foundation_ts",
            _profile(history_rows_median=8.0, horizon_max=4),
            "short_asof_history_for_high_capacity_sequence_model",
        ),
        (
            "renewal_rt",
            "renewal",
            _profile(history_rows_min=4, horizon_max=4),
            "short_asof_history_for_dynamic_rate_estimation",
        ),
        (
            "last_value",
            "statistical",
            _profile(trend_signal="rising"),
            "recency_forecast_ignores_material_directional_change",
        ),
        (
            "drift",
            "state_space",
            _profile(trend_signal="flat"),
            "trend_extrapolation_not_supported_by_recent_asof_change",
        ),
        (
            "sir_tau",
            "compartmental",
            _profile(
                horizon_max=4,
                trend_signal="falling",
                recursive_rollout_present=True,
            ),
            "fixed_compartmental_transition_may_be_rigid_for_directional_long_rollout",
        ),
    ],
)
def test_react_restores_frozen_material_risk_branches(
    model_id: str,
    family: str,
    profile: dict[str, object],
    expected_risk: str,
) -> None:
    selected = pd.Series(
        {"model_id": model_id, "family": family, "description": "unit card"}
    )
    result = agentic_react._candidate_compatibility(profile, selected)
    assert expected_risk in result["risk_flags"]
    assert result["material_risk"] is True


@pytest.mark.parametrize("model_id", ["rnn_simple", "gru_style"])
def test_fixed_scalar_recurrences_keep_neural_family_without_capacity_risk(
    model_id: str,
) -> None:
    selected = pd.Series(
        {"model_id": model_id, "family": "neural", "description": "unit card"}
    )
    result = agentic_react._candidate_compatibility(
        _profile(history_rows_median=1.0, horizon_max=4),
        selected,
    )
    assert "short_asof_history_for_high_capacity_sequence_model" not in result[
        "risk_flags"
    ]
    assert result["material_risk"] is False


def _trajectory(initial: pd.Series, compatibility) -> dict[str, object]:
    return {
        "tentative_selection": {"model_id": str(initial["model_id"])},
        "tentative_compatibility_observation": compatibility({}, initial),
        "trajectory_id": "audit-only-id",
    }


@pytest.mark.parametrize("successful_attempt", [1, 2, 3])
def test_react_accepts_first_risk_free_repair_on_attempt(
    monkeypatch,
    successful_attempt: int,
) -> None:
    model_ids = [
        "initial_internal",
        "risky_internal_1",
        "risky_internal_2",
        "safe_internal",
    ]
    candidates = _candidates(model_ids)
    initial = candidates[candidates["model_id"].eq("initial_internal")].iloc[0].copy()
    compatibility = _compatibility(
        {
            "initial_internal": ["unit_hard_risk"],
            "risky_internal_1": ["unit_hard_risk"],
            "risky_internal_2": ["unit_hard_risk"],
            "safe_internal": [],
        }
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    attempted_models = ["risky_internal_1", "risky_internal_2", "safe_internal"][
        :successful_attempt
    ]
    attempted_models[-1] = "safe_internal"
    engine = ReplayJSONEngine(
        [
            {
                "decision": "repair",
                "choice_id": _choice_for(candidates, model_id),
                "updated_thought": "recheck compatibility",
                "reason": "scientific repair proposal",
            }
            for model_id in attempted_models
        ]
    )
    trace_rows: list[dict[str, object]] = []
    selected = agentic_react._summary_candidate(
        engine=engine,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=trace_rows,
        selection_replay=None,
    )

    payload = selected.attrs["llm_payload"]
    assert selected["model_id"] == "safe_internal"
    assert len(engine.calls) == successful_attempt
    assert payload["repair_attempt_count"] == successful_attempt
    assert payload["repair_target_risk_resolved"] is True
    assert payload["fallback_to_initial_after_three_failed_repairs"] is False
    assert payload["llm_selected_model_id"] == "safe_internal"
    assert payload["repair_attempts"][-1]["validation_status"] == "risk_resolved"
    for index, call in enumerate(engine.calls, start=1):
        _assert_prompt_surface_is_anonymous(call.prompt, model_ids)
        assert "audit-only-id" not in call.prompt
        prompt_payload = json.loads(call.prompt.split("\n\n", maxsplit=1)[1])
        assert prompt_payload["repair_attempt"] == index
        assert "choice_id" in prompt_payload["required_response_schema"]


def test_react_invalid_and_same_choices_consume_attempts(monkeypatch) -> None:
    model_ids = ["initial_internal", "risky_internal", "safe_internal"]
    candidates = _candidates(model_ids)
    initial = candidates[candidates["model_id"].eq("initial_internal")].iloc[0].copy()
    compatibility = _compatibility(
        {
            "initial_internal": ["unit_hard_risk"],
            "risky_internal": ["unit_hard_risk"],
            "safe_internal": [],
        }
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    initial_choice = _choice_for(candidates, "initial_internal")
    safe_choice = _choice_for(candidates, "safe_internal")
    engine = ReplayJSONEngine(
        [
            {
                "decision": "repair",
                "choice_id": "C999",
                "reason": "invalid opaque label",
            },
            {
                "decision": "repair",
                "choice_id": initial_choice,
                "reason": "same opaque label",
            },
            {
                "decision": "repair",
                "choice_id": safe_choice,
                "reason": "risk resolved",
            },
        ]
    )
    trace_rows: list[dict[str, object]] = []
    selected = agentic_react._summary_candidate(
        engine=engine,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=trace_rows,
        selection_replay=None,
    )

    assert selected["model_id"] == "safe_internal"
    attempts = selected.attrs["llm_payload"]["repair_attempts"]
    assert [row["validation_status"] for row in attempts] == [
        "invalid_choice",
        "same_as_initial",
        "risk_resolved",
    ]
    assert attempts[0]["compatibility_rechecked"] is False
    assert attempts[1]["compatibility_rechecked"] is True


def test_react_three_risky_repairs_fall_back_to_initial(monkeypatch) -> None:
    model_ids = [
        "initial_internal",
        "risky_internal_1",
        "risky_internal_2",
        "risky_internal_3",
        "safe_internal",
    ]
    candidates = _candidates(model_ids)
    initial = candidates[candidates["model_id"].eq("initial_internal")].iloc[0].copy()
    compatibility = _compatibility(
        {
            "initial_internal": ["unit_hard_risk"],
            "risky_internal_1": ["unit_hard_risk"],
            "risky_internal_2": ["unit_hard_risk"],
            "risky_internal_3": ["unit_hard_risk"],
            "safe_internal": [],
        }
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    engine = ReplayJSONEngine(
        [
            {
                "decision": "repair",
                "choice_id": _choice_for(candidates, f"risky_internal_{index}"),
                "reason": "risk remains",
            }
            for index in (1, 2, 3)
        ]
    )
    selected = agentic_react._summary_candidate(
        engine=engine,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=[],
        selection_replay=None,
    )

    payload = selected.attrs["llm_payload"]
    assert selected["model_id"] == "initial_internal"
    assert payload["decision"] == "keep_with_risk"
    assert payload["repair_attempt_count"] == 3
    assert payload["fallback_to_initial_after_three_failed_repairs"] is True
    assert payload["llm_selected_model_id"] == "initial_internal"
    assert payload["llm_proposed_model_id"] == "risky_internal_3"
    assert payload["repair_target_risk_resolved"] is False
    assert all(
        row["validation_status"] == "risk_remains"
        for row in payload["repair_attempts"]
    )


def test_react_does_not_accept_a_new_hard_risk(monkeypatch) -> None:
    model_ids = ["initial_internal", "new_risk_internal", "safe_internal"]
    candidates = _candidates(model_ids)
    initial = candidates[candidates["model_id"].eq("initial_internal")].iloc[0].copy()
    compatibility = _compatibility(
        {
            "initial_internal": ["initial_hard_risk"],
            "new_risk_internal": ["different_hard_risk"],
            "safe_internal": [],
        }
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    engine = ReplayJSONEngine(
        [
            {
                "decision": "repair",
                "choice_id": _choice_for(candidates, "new_risk_internal"),
            },
            {
                "decision": "repair",
                "choice_id": _choice_for(candidates, "safe_internal"),
            },
        ]
    )
    selected = agentic_react._summary_candidate(
        engine=engine,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=[],
        selection_replay=None,
    )

    attempts = selected.attrs["llm_payload"]["repair_attempts"]
    assert attempts[0]["target_risk_remaining"] == []
    assert attempts[0]["risk_after"] == ["different_hard_risk"]
    assert attempts[0]["validation_status"] == "risk_remains"
    assert attempts[1]["validation_status"] == "risk_resolved"


def test_react_skips_repair_when_no_risk_free_candidate_exists(monkeypatch) -> None:
    model_ids = ["initial_internal", "risky_internal"]
    candidates = _candidates(model_ids)
    initial = candidates.iloc[0].copy()
    compatibility = _compatibility(
        {model_id: ["unit_hard_risk"] for model_id in model_ids}
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    engine = ReplayJSONEngine([])
    trace_rows: list[dict[str, object]] = []
    selected = agentic_react._summary_candidate(
        engine=engine,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=trace_rows,
        selection_replay=None,
    )

    payload = selected.attrs["llm_payload"]
    assert engine.calls == []
    assert payload["repair_resolution_available"] is False
    assert payload["repair_attempt_count"] == 0
    assert payload["repair_skipped_reason"] == "no_risk_free_candidate_available"
    assert payload["fallback_to_initial_after_three_failed_repairs"] is False
    assert trace_rows[-1]["repair_attempt_record"]["validation_status"] == (
        "repair_skipped_no_risk_free_candidate"
    )


def test_react_no_initial_hard_risk_calls_keep_critic_not_repair(monkeypatch) -> None:
    model_ids = ["initial_internal", "other_internal"]
    candidates = _candidates(model_ids)
    initial = candidates.iloc[0].copy()
    compatibility = _compatibility({})
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    initial_choice = _choice_for(candidates, str(initial["model_id"]))
    engine = ReplayJSONEngine(
        {"decision": "keep", "choice_id": initial_choice, "reason": "no hard risk"}
    )
    trace_rows: list[dict[str, object]] = []
    selected = agentic_react._summary_candidate(
        engine=engine,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=trace_rows,
        selection_replay=None,
    )

    payload = selected.attrs["llm_payload"]
    assert selected["model_id"] == initial["model_id"]
    assert len(engine.calls) == 1
    assert payload["decision"] == "keep"
    assert payload["repair_attempt_count"] == 0
    assert trace_rows[-1]["repair_attempt"] == 0
    prompt_payload = json.loads(engine.calls[0].prompt.split("\n\n", maxsplit=1)[1])
    assert "repair_attempt" not in prompt_payload


def test_react_llm_errors_consume_three_attempts_then_fall_back(monkeypatch) -> None:
    model_ids = ["initial_internal", "safe_internal"]
    candidates = _candidates(model_ids)
    initial = candidates.iloc[0].copy()
    compatibility = _compatibility(
        {"initial_internal": ["unit_hard_risk"], "safe_internal": []}
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)

    class ErrorEngine:
        active_model_path = "unit_error_engine"
        fallback_used = False
        fallback_reason = ""

        def __init__(self):
            self.calls = 0

        def generate_json(self, **_kwargs):
            self.calls += 1
            raise LLMError("unit generation failure")

    engine = ErrorEngine()
    trace_rows: list[dict[str, object]] = []
    selected = agentic_react._summary_candidate(
        engine=engine,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=trace_rows,
        selection_replay=None,
    )

    payload = selected.attrs["llm_payload"]
    assert engine.calls == 3
    assert selected["model_id"] == "initial_internal"
    assert payload["fallback_to_initial_after_three_failed_repairs"] is True
    assert [row["validation_status"] for row in payload["repair_attempts"]] == [
        "llm_error",
        "llm_error",
        "llm_error",
    ]
    assert all(row["runtime_seconds"] >= 0.0 for row in trace_rows)


def test_react_non_llm_engine_bug_fails_closed(monkeypatch) -> None:
    model_ids = ["initial_internal", "safe_internal"]
    candidates = _candidates(model_ids)
    initial = candidates.iloc[0].copy()
    compatibility = _compatibility(
        {"initial_internal": ["unit_hard_risk"], "safe_internal": []}
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)

    class BugEngine:
        def generate_json(self, **_kwargs):
            raise KeyError("implementation bug")

    with pytest.raises(KeyError, match="implementation bug"):
        agentic_react._summary_candidate(
            engine=BugEngine(),
            dataset_key=DATASET_KEY,
            forecast_origin=FORECAST_ORIGIN,
            candidates=candidates,
            initial_selected=initial,
            profile={},
            trajectory=_trajectory(initial, compatibility),
            trace_rows=[],
            selection_replay=None,
        )


def test_react_replay_uses_distinct_initial_and_final_roles(monkeypatch) -> None:
    model_ids = ["initial_internal", "safe_internal"]
    candidates = _candidates(model_ids)
    compatibility = _compatibility(
        {"initial_internal": ["unit_hard_risk"], "safe_internal": []}
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    replay_row = {
        "initial_selected_model_id": "initial_internal",
        "final_selected_model_id": "safe_internal",
        "selected_model_id": "safe_internal",
        "llm_selected_model_id": "safe_internal",
        "repair_attempt_count": 1,
        "repair_attempts": json.dumps(
            [
                {
                    "attempt": 1,
                    "candidate_model_id": "safe_internal",
                    "validation_status": "risk_resolved",
                    "repair_accepted": True,
                }
            ]
        ),
        "fallback_to_initial_after_three_failed_repairs": False,
    }
    selection_replay = {(DATASET_KEY, FORECAST_ORIGIN): replay_row}
    initial = select_candidate_with_replay(
        stage="react_select",
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        trace_rows=[],
        selection_replay=selection_replay,
    )
    final = select_candidate_with_replay(
        stage="react_repair_select",
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        trace_rows=[],
        selection_replay=selection_replay,
    )
    assert initial["model_id"] == "initial_internal"
    assert final["model_id"] == "safe_internal"

    selected = agentic_react._summary_candidate(
        engine=None,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=[],
        selection_replay=selection_replay,
    )
    payload = selected.attrs["llm_payload"]
    assert selected["model_id"] == "safe_internal"
    assert payload["decision"] == "repair"
    assert payload["repair_attempt_history_available"] is True
    assert payload["repair_attempt_history_source"] == "replay_recorded_attempts"
    assert payload["repair_attempt_count"] == 1
    assert payload["llm_selected_model_id"] == "safe_internal"


def test_react_replay_accepts_consistent_three_attempt_exhaustion(monkeypatch) -> None:
    model_ids = ["initial_internal", "safe_internal"]
    candidates = _candidates(model_ids)
    initial = candidates.iloc[0].copy()
    compatibility = _compatibility(
        {"initial_internal": ["unit_hard_risk"], "safe_internal": []}
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    recorded_attempts = [
        {"attempt": index, "validation_status": "risk_remains"}
        for index in (1, 2, 3)
    ]
    selection_replay = {
        (DATASET_KEY, FORECAST_ORIGIN): {
            "initial_selected_model_id": "initial_internal",
            "final_selected_model_id": "initial_internal",
            "repair_attempt_count": 3,
            "repair_attempts": json.dumps(recorded_attempts),
            "fallback_to_initial_after_three_failed_repairs": True,
        }
    }
    selected = agentic_react._summary_candidate(
        engine=None,
        dataset_key=DATASET_KEY,
        forecast_origin=FORECAST_ORIGIN,
        candidates=candidates,
        initial_selected=initial,
        profile={},
        trajectory=_trajectory(initial, compatibility),
        trace_rows=[],
        selection_replay=selection_replay,
    )
    payload = selected.attrs["llm_payload"]
    assert selected["model_id"] == "initial_internal"
    assert payload["decision"] == "keep_with_risk"
    assert payload["repair_attempt_count"] == 3
    assert payload["fallback_to_initial_after_three_failed_repairs"] is True


def test_react_replay_rejects_inconsistent_exhaustion_count(monkeypatch) -> None:
    model_ids = ["initial_internal", "safe_internal"]
    candidates = _candidates(model_ids)
    initial = candidates.iloc[0].copy()
    compatibility = _compatibility(
        {"initial_internal": ["unit_hard_risk"], "safe_internal": []}
    )
    monkeypatch.setattr(agentic_react, "_candidate_compatibility", compatibility)
    selection_replay = {
        (DATASET_KEY, FORECAST_ORIGIN): {
            "initial_selected_model_id": "initial_internal",
            "final_selected_model_id": "initial_internal",
            "repair_attempt_count": 2,
            "repair_attempts": json.dumps(
                [{"attempt": 1}, {"attempt": 2}]
            ),
            "fallback_to_initial_after_three_failed_repairs": True,
        }
    }
    with pytest.raises(ValueError, match="must record three attempts"):
        agentic_react._summary_candidate(
            engine=None,
            dataset_key=DATASET_KEY,
            forecast_origin=FORECAST_ORIGIN,
            candidates=candidates,
            initial_selected=initial,
            profile={},
            trajectory=_trajectory(initial, compatibility),
            trace_rows=[],
            selection_replay=selection_replay,
        )
