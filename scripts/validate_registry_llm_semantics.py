#!/usr/bin/env python3
"""Static contract checks for scientific registry text and Qwen-facing fields."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "code/caster/configs/model_registry.yaml"

DESCRIPTION_FIELDS = (
    "mechanism",
    "inputs",
    "fit_scope",
    "update",
    "uncertainty",
    "limitation",
)

EXPECTED_MODEL_IDS = {
    "last_value",
    "seasonal_naive",
    "drift",
    "covariate_drift",
    "sir_tau",
    "seir_tau",
    "seirs_tau",
    "tv_seir_rt",
    "renewal_rt",
    "local_level",
    "covariate_dynamic_linear_trend",
    "particle_local_level",
    "statsforecast_autoarima",
    "statsforecast_autoets",
    "statsforecast_autotheta",
    "statsforecast_autoces",
    "prophet",
    "rnn_simple",
    "lstm_style",
    "gru_style",
    "deepar_style",
    "nbeats_basis",
    "nhits_hinterp",
    "patchtst_patched",
    "tft_gated",
    "chronos_external",
    "timesfm_external",
}

# These terms describe experiment identity, implementation environment, relative
# status, or an unstated comparator.  They are not scientific model mechanisms.
FORBIDDEN_DESCRIPTION_PATTERNS = {
    "baseline": r"\bbaseline(?:[- ]aligned)?\b",
    "aligned": r"\baligned\b",
    "reference": r"\breference\b",
    "same": r"\bsame\b",
    "style": r"\bstyle\b",
    "dataset": r"\bdataset\b",
    "benchmark": r"\bbenchmark\b",
    "CASTER": r"\bcaster\b",
    "EpiLLM": r"\bepillm\b",
    "official": r"\bofficial\b",
    "canonical": r"\bcanonical\b",
    "primary": r"\bprimary\b",
    "environment": r"\benvironment\b",
    "dependency": r"\bdependency\b",
    "hardware": r"\b(?:cuda|gpu|cpu)\b",
    "value judgement": r"\b(?:best|preferred|recommended|superior)\b",
}

FITTED_SEQUENCE_MODELS = {
    "deepar_style",
    "nbeats_basis",
    "nhits_hinterp",
    "patchtst_patched",
    "tft_gated",
}

EXPECTED_REACT_MATERIAL_RISK_FLAGS = [
    "insufficient_asof_history_for_full_season_lag",
    "limited_asof_cycles_for_seasonal_decomposition",
    "short_asof_history_for_high_capacity_sequence_model",
    "short_asof_history_for_dynamic_rate_estimation",
    "recency_forecast_ignores_material_directional_change",
    "trend_extrapolation_not_supported_by_recent_asof_change",
    "fixed_compartmental_transition_may_be_rigid_for_directional_long_rollout",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(path: Path, function_name: str) -> str:
    source = _read(path)
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            lines = source.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"function {function_name!r} not found in {path}")


def _dict_literal_key_sets(function_source: str) -> list[set[str]]:
    """Return statically declared string keys for every dict in a function."""
    tree = ast.parse(function_source)
    key_sets: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        key_sets.append(
            {
                str(key.value)
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
        )
    return key_sets


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "f", "no", "n", "disabled"}


def _require_fragments(
    failures: list[str], model_id: str, description: str, fragments: tuple[str, ...]
) -> None:
    lowered = description.lower()
    for fragment in fragments:
        if fragment.lower() not in lowered:
            failures.append(f"{model_id}: missing factual fragment {fragment!r}")


def _validate_descriptions(registry_path: Path, failures: list[str]) -> list[dict[str, Any]]:
    payload = yaml.safe_load(_read(registry_path)) or {}
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    if not isinstance(candidates, list):
        failures.append("registry candidates must be a list")
        return []
    if len(candidates) != 27:
        failures.append(f"registry must contain 27 candidates; found {len(candidates)}")

    ids = [str(row.get("model_id", "")) for row in candidates if isinstance(row, dict)]
    if len(ids) != len(set(ids)):
        failures.append("registry model_id values are not unique")
    if set(ids) != EXPECTED_MODEL_IDS:
        failures.append(
            "registry model IDs differ from the 27-model contract: "
            f"missing={sorted(EXPECTED_MODEL_IDS - set(ids))} "
            f"extra={sorted(set(ids) - EXPECTED_MODEL_IDS)}"
        )

    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(candidates):
        if not isinstance(row, dict):
            failures.append(f"candidate row {index} is not a mapping")
            continue
        model_id = str(row.get("model_id", ""))
        by_id[model_id] = row
        if not _enabled(row.get("enabled", True)):
            failures.append(f"{model_id}: candidate is not enabled")
        if "skill_embedding_text" in row:
            failures.append(
                f"{model_id}: source YAML must have one semantic source of truth; "
                "remove skill_embedding_text"
            )

        description = str(row.get("description", "")).strip()
        parts = [part.strip() for part in description.split(";")]
        if len(parts) != len(DESCRIPTION_FIELDS):
            failures.append(
                f"{model_id}: description must contain exactly six semicolon-delimited fields"
            )
            continue
        for field, part in zip(DESCRIPTION_FIELDS, parts):
            prefix = f"{field}="
            if not part.startswith(prefix) or not part[len(prefix) :].strip():
                failures.append(f"{model_id}: expected non-empty field {prefix}")
        for label, pattern in FORBIDDEN_DESCRIPTION_PATTERNS.items():
            if re.search(pattern, description, flags=re.IGNORECASE):
                failures.append(f"{model_id}: forbidden non-scientific term class {label!r}")

    if not by_id:
        return candidates

    _require_fragments(
        failures,
        "seasonal_naive",
        str(by_id["seasonal_naive"].get("description", "")),
        ("minus 7 days", "minus 8 weeks", "exact lag"),
    )
    _require_fragments(
        failures,
        "particle_local_level",
        str(by_id["particle_local_level"].get("description", "")),
        ("replaces rather than accumulates particle weights",),
    )
    for model_id in ("rnn_simple", "gru_style"):
        _require_fragments(
            failures,
            model_id,
            str(by_id[model_id].get("description", "")),
            ("no trained neural weights", "per-series", "recursive horizons"),
        )
    _require_fragments(
        failures,
        "lstm_style",
        str(by_id["lstm_style"].get("description", "")),
        (
            "observation-time and release-time filtering",
            "component, forecast strategy, and origin",
            "three optimization steps",
            "refitted at every origin",
        ),
    )
    for model_id in sorted(FITTED_SEQUENCE_MODELS):
        _require_fragments(
            failures,
            model_id,
            str(by_id[model_id].get("description", "")),
            (
                "observation-time and release-time filtering",
                "earliest viable scoring origin",
                "three",
                "weights remain fixed",
                "weights do not adapt",
            ),
        )
    _require_fragments(
        failures,
        "chronos_external",
        str(by_id["chronos_external"].get("description", "")),
        (
            "returned q50 median as point forecast",
            "one-step q50 point forecasts are fed back recursively",
            "q25 and q75",
            "q10-to-q90",
            "normal quantiles",
            "central 90 gaussian-equivalent proxy",
            "q05 and q95 are not native",
        ),
    )
    return candidates


def _validate_candidate_semantic_paths(failures: list[str]) -> None:
    qwen_embedding = ROOT / "code/caster/src/caster/models/qwen25_embedding.py"
    retrieval = ROOT / "code/caster/src/caster/models/retrieval.py"
    ranking_builder = ROOT / "scripts/build_qwen_formal_top10.py"
    registry_loader = ROOT / "code/caster/src/caster/models/registry.py"
    agent_skills = ROOT / "code/baseline/src/caster_baselines/agentic_skills.py"

    qwen_embed_func = _function_source(qwen_embedding, "embed_registry_qwen25")
    legacy_embed_func = _function_source(retrieval, "embed_registry")
    ranking_text_func = _function_source(ranking_builder, "_candidate_embedding_text")
    prompt_rows_func = _function_source(agent_skills, "fair_registry_prompt_rows")
    selector_func = _function_source(agent_skills, "select_candidate_with_llm")
    make_registry_func = _function_source(registry_loader, "make_registry")
    baseline_registry_func = _function_source(agent_skills, "read_candidate_registry")

    for label, source in (
        ("Qwen candidate embedding", qwen_embed_func),
        ("hashed candidate embedding", legacy_embed_func),
        ("formal ranking candidate text", ranking_text_func),
    ):
        if 'row.get("description"' not in source:
            failures.append(f"{label}: description is not the direct semantic source")
        if "skill_embedding_text" in source:
            failures.append(f"{label}: stale skill_embedding_text override remains")

    required_prompt_tokens = (
        '"choice_id": choice_id',
        '"scientific_description": description',
        "choice_to_model[choice_id] = model_id",
    )
    for token in required_prompt_tokens:
        if token not in prompt_rows_func:
            failures.append(f"candidate prompt rows: missing {token}")
    for forbidden_key in (
        "model_id",
        "family",
        "candidate_type",
        "priority",
        "recipe",
        "skill_embedding_text",
    ):
        if f'"{forbidden_key}":' in prompt_rows_func:
            failures.append(f"candidate prompt rows expose internal field {forbidden_key}")

    selector_prompt_key_sets = [
        keys
        for keys in _dict_literal_key_sets(selector_func)
        if {"candidates", "valid_choice_ids"}.issubset(keys)
    ]
    if len(selector_prompt_key_sets) != 1:
        failures.append(
            "selector must declare exactly one anonymous candidate user-prompt payload"
        )
    else:
        selector_prompt_keys = selector_prompt_key_sets[0]
        for required_key in ("candidates", "valid_choice_ids"):
            if required_key not in selector_prompt_keys:
                failures.append(f"selector prompt is missing {required_key}")
        for forbidden_key in (
            "valid_model_ids",
            "dataset_key",
            "candidate_order_strategy",
            "position_neutrality_instruction",
            "fairness_instruction",
        ):
            if forbidden_key in selector_prompt_keys:
                failures.append(
                    f"selector prompt exposes trace/control field {forbidden_key}"
                )
    for trace_token in (
        '"candidate_choice_map": choice_to_model',
        '"candidate_order_strategy": order_strategy',
        '"position_neutrality_instruction": position_neutrality_instruction',
        '"fairness_instruction": position_neutrality_instruction',
        'llm_payload.setdefault("position_neutrality_instruction_present", False)',
        'llm_payload.setdefault("position_neutrality_control_trace_only", True)',
    ):
        if trace_token not in selector_func:
            failures.append(f"selector trace is missing {trace_token}")

    expected_default = 'row["skill_embedding_text"] = str(row.get("description", "")).strip()'
    if expected_default not in make_registry_func:
        failures.append("CASTER registry normalization does not mirror description into retrieval text")
    if expected_default not in baseline_registry_func:
        failures.append("baseline registry normalization does not mirror description into retrieval text")

    shared_builder = ROOT / "scripts/build_shared_formal_selection.py"
    shared_main = _function_source(shared_builder, "main")
    formal_retrieval = _function_source(retrieval, "select_top_k_candidates_formal")
    if "hashed_text_embedding(selection_text" not in formal_retrieval:
        failures.append("formal hashed retrieval does not embed its supplied selection text")
    if "scientific_text = scientific_selection_text(context)" not in shared_main:
        failures.append("formal selection does not build the scientific-only selection text")
    if "selection_text=scientific_text" not in shared_main:
        failures.append("select_top_k_candidates_formal does not receive scientific_text")
    if 'task_root / "scientific_selection_context.txt"' not in shared_main:
        failures.append("formal selection does not save scientific_selection_context.txt")


def _validate_scientific_context(failures: list[str]) -> None:
    caster_src = ROOT / "code/caster/src"
    if str(caster_src) not in sys.path:
        sys.path.insert(0, str(caster_src))
    try:
        from caster.tasks.qwen_context import scientific_context_view
    except Exception as exc:  # pragma: no cover - reported as a contract failure
        failures.append(f"cannot import scientific_context_view: {type(exc).__name__}: {exc}")
        return

    component_query = (
        "Forecast weekly COVID-19 hospital admissions across US jurisdictions "
        "with direct and recursive horizons."
    )
    synthetic = {
        "schema": "internal",
        "task_id": "benchmark_b_pooled",
        "task_spec_sha256": "deadbeef",
        "formal_t_sel": "2099-01-01",
        "base_context": {
            "selection_view_query": component_query,
            "selection_view_task_id": "benchmark_b_covid",
            "selection_view_posterior_scope": "component_conditioned_posterior",
            "task_query": "one shared posterior",
            "posterior_scope": "pooled_shared_posterior",
            "history_summary": {"covid_adm_per100k": {"history_end": "2025-01-01"}},
            "cross_stream_summary": {},
            "released_event_count": 1,
            "source_binding": {"sha256": "deadbeef"},
            "history_scope": "all_available_before_t_sel",
            "covariate_availability": {
                "population": 1.0,
                "__release_time__": 1.0,
            },
        },
        "causal_multiscale_sequence_sketch": {
            "task_query": "Joint forecast with one shared posterior.",
            "selection_view_target_components": ["covid_adm_per100k"],
            "sequence_components": ["covid_adm_per100k", "flu_adm_per100k"],
            "formal_t_sel": "2099-01-01",
            "components": {},
            "history_scope": "target_cells_with_target_and_release_time_lte_cutoff",
            "source_binding": {"sha256": "deadbeef"},
        },
    }
    view = scientific_context_view(synthetic)
    rendered = json.dumps(view, sort_keys=True).lower()
    if view.get("selection_target", {}).get("query") != component_query:
        failures.append("Benchmark B scientific context does not prefer component-specific query")
    if view.get("selection_target", {}).get("target_components") != ["covid_adm_per100k"]:
        failures.append("Benchmark B scientific context does not retain one target component")
    if view.get("selection_target", {}).get("released_auxiliary_components") != [
        "flu_adm_per100k"
    ]:
        failures.append("Benchmark B scientific context does not label the other stream as auxiliary")
    for forbidden in (
        "benchmark_b_pooled",
        "pooled_shared_posterior",
        "one shared posterior",
        "formal_t_sel",
        "sha256",
        "deadbeef",
        "__release_time__",
        "all_available_before_t_sel",
        "target_cells_with_target_and_release_time_lte_cutoff",
    ):
        if forbidden in rendered:
            failures.append(f"scientific context leaks audit/pooled marker {forbidden!r}")

    qwen_context = ROOT / "code/caster/src/caster/tasks/qwen_context.py"
    benchmark_b_context = ROOT / "code/baseline/src/caster_baselines/benchmark_b_context.py"
    qwen_builder = _function_source(qwen_context, "build_qwen25_multiscale_context")
    llm_fields = _function_source(benchmark_b_context, "llm_prompt_fields")
    if "scientific = scientific_context_view(payload)" not in qwen_builder:
        failures.append("Qwen context text is not projected through scientific_context_view")
    if "provenance hashes" in qwen_builder or "experiment split markers" in qwen_builder:
        failures.append("Qwen context text describes trace-only provenance controls")
    if 'return {"scientific_context": dict(scientific)}' not in llm_fields:
        failures.append("Benchmark B LLM field gate exposes more than scientific_context")
    for agent_name in ("agentic_top_one.py", "agentic_full_recovery.py", "agentic_react.py"):
        source = _read(ROOT / "code/baseline/src/caster_baselines" / agent_name)
        if "benchmark_b_llm_prompt_fields" not in source:
            failures.append(f"{agent_name}: Benchmark B scientific field gate is not used")


def _validate_react_contract(failures: list[str]) -> None:
    path = ROOT / "code/baseline/src/caster_baselines/agentic_react.py"
    source = _read(path)
    compatibility = _function_source(path, "_candidate_compatibility")
    summary = _function_source(path, "_summary_candidate")
    exact_actions = 'REACT_ACTIONS = ("inspect_context", "inspect_candidates", "select_model")'
    if exact_actions not in source:
        failures.append("ReAct action vocabulary/order changed")
    risk_appends = re.findall(r'risks\.append\("([^"]+)"\)', compatibility)
    if risk_appends != EXPECTED_REACT_MATERIAL_RISK_FLAGS:
        failures.append(
            "ReAct material-risk set/order differs from the frozen contract: "
            f"{risk_appends}"
        )
    if '"material_risk": bool(risks)' not in compatibility:
        failures.append("ReAct material_risk is not derived from the frozen risk set")
    for token in (
        'FIXED_SCALAR_RECURRENCES = {"rnn_simple", "gru_style"}',
        'and model_id not in FIXED_SCALAR_RECURRENCES',
    ):
        if token not in source and token not in compatibility:
            failures.append(
                "ReAct fixed-scalar neural-family exception is missing: " + token
            )
    for token in (
        "preselection_trajectory = {",
        '"action_observation": action_observation',
        '"tentative_compatibility_observation": compatibility_observation',
        "selected = _summary_candidate(",
    ):
        if token not in source:
            failures.append(f"ReAct thought/action/observation/critic flow changed: missing {token}")
    for token in (
        'prompt_tentative.pop("model_id", None)',
        'prompt_compatibility.pop("model_id", None)',
        "for repair_attempt in range(1, MAX_REACT_REPAIR_ATTEMPTS + 1)",
        "repaired_observation = _candidate_compatibility(profile, proposed)",
        'and not bool(repaired_observation.get("material_risk", False))',
        "and not risk_after",
        '"repair_remaining_target_risk_flags"',
        '"fallback_to_initial_after_three_failed_repairs"',
        'except LLMError as exc:',
        'repair_skipped_reason="no_risk_free_candidate_available"',
    ):
        if token not in summary:
            failures.append(f"ReAct repair/anonymous-choice contract missing: {token}")
    react_prompt_key_sets = [
        keys
        for keys in _dict_literal_key_sets(summary)
        if {"candidates", "valid_choice_ids", "trajectory"}.issubset(keys)
    ]
    if len(react_prompt_key_sets) != 2:
        failures.append(
            "ReAct critic must declare one no-risk keep prompt and one hard-risk repair prompt"
        )
    else:
        for prompt_keys in react_prompt_key_sets:
            if "dataset_key" in prompt_keys:
                failures.append("ReAct critic prompt exposes dataset_key")
        for forbidden_key in (
            "valid_model_ids",
            "tentative_model_id",
            "candidate_order_strategy",
            "position_neutrality_instruction",
            "fairness_instruction",
        ):
            if any(forbidden_key in keys for keys in react_prompt_key_sets):
                failures.append(
                    f"ReAct critic prompt exposes trace/control field {forbidden_key}"
                )
    run_react = _function_source(path, "run_react_agent_from_manifest")
    if "profile=data_profile" not in run_react:
        failures.append("ReAct critic is not passed the as-of profile for repair revalidation")
    if '"dataset_key": ctx.dataset_key' in _function_source(
        path, "run_react_agent_from_manifest"
    ).split("thought_payload = {", maxsplit=1)[-1].split(
        "if engine is None:", maxsplit=1
    )[0]:
        failures.append("ReAct thought prompt exposes dataset_key")


def _validate_planner_boundary(failures: list[str]) -> None:
    top_one_path = ROOT / "code/baseline/src/caster_baselines/agentic_top_one.py"
    top_one_prompt_context = _function_source(top_one_path, "_task_planning_context")
    top_one_trace_metadata = _function_source(
        top_one_path, "_task_planning_trace_metadata"
    )
    control_field = "full_augmented_context_used_by_model_selection"
    if control_field in top_one_prompt_context:
        failures.append("Top-1 planner prompt context exposes control-only metadata")
    for trace_token in (
        control_field,
        "full_augmented_context_control_trace_only",
        "planner_output_consumed_by_selector",
    ):
        if trace_token not in top_one_trace_metadata:
            failures.append(f"Top-1 planner trace metadata is missing {trace_token}")

    for filename, function_name in (
        ("agentic_top_one.py", "run_agentic_top_one"),
        ("agentic_full_recovery.py", "run_agentic_full_recovery"),
    ):
        path = ROOT / "code/baseline/src/caster_baselines" / filename
        function_source = _function_source(path, function_name)
        tree = ast.parse(function_source)
        selection_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func.id if isinstance(node.func, ast.Name) else ""
            if called == "select_candidate_no_validation":
                selection_calls.append(ast.unparse(node))
        if not selection_calls:
            failures.append(f"{filename}: no selector call found")
            continue
        if any("planner" in call.lower() for call in selection_calls):
            failures.append(f"{filename}: planner output is passed into selector")

    prompt_specs = (
        ("agentic_top_one.py", "run_agentic_top_one"),
        ("agentic_full_recovery.py", "run_agentic_full_recovery"),
        ("agentic_react.py", "run_react_agent_from_manifest"),
    )
    forbidden_prompt_provenance = {
        "dataset_key",
        "dataset",
        "result_framework",
        "run_dir",
        "selected_models",
    }
    for filename, function_name in prompt_specs:
        path = ROOT / "code/baseline/src/caster_baselines" / filename
        function_source = _function_source(path, function_name)
        llm_payload_key_sets = [
            keys
            for keys in _dict_literal_key_sets(function_source)
            if "required_response" in keys
        ]
        for prompt_keys in llm_payload_key_sets:
            leaked = sorted(forbidden_prompt_provenance.intersection(prompt_keys))
            if leaked:
                failures.append(
                    f"{filename}: LLM prompt payload exposes project provenance {leaked}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate registry descriptions and Qwen-facing semantic contracts."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    failures: list[str] = []
    candidates = _validate_descriptions(registry_path, failures)
    _validate_candidate_semantic_paths(failures)
    _validate_scientific_context(failures)
    _validate_react_contract(failures)
    _validate_planner_boundary(failures)

    if failures:
        print(f"FAIL registry={registry_path} failures={len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(
        "PASS "
        f"registry={registry_path} models={len(candidates)} fields={','.join(DESCRIPTION_FIELDS)} "
        "candidate_semantics=description_only benchmark_b_prompt=component_specific "
        "react_material_risk=frozen_compatibility_set topology=unchanged"
    )


if __name__ == "__main__":
    main()
