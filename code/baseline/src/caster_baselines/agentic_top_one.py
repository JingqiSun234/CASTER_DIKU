from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .agentic_archive import (
    ForecastArchiveIndex,
    ARCHIVE_FORECAST_SOURCE,
    ARCHIVE_TIMING_SEMANTICS,
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
from .agentic_llm import QwenLocalEngine
from .agentic_skills import (
    derive_agent_selection_scope,
    eligible_registry,
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


def _trace_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _record_dispatch(
    *,
    trace_rows: list[dict[str, object]],
    dataset_key: str,
    selected_model_id: str,
    forecast_rows: int,
    archive_backed: bool = False,
    forecast_origin: str = "",
    context_payload: dict[str, object] | None = None,
) -> None:
    response = {
        "status": "ok",
        "selected_model_id": selected_model_id,
        "forecast_rows": int(forecast_rows),
        "executor": "immutable forecast archive lookup" if archive_backed else "baseline-local refit/forecast recipe",
        **(context_payload or {}),
    }
    trace_rows.append({
        "stage": "execution_dispatch",
        "dataset_key": dataset_key,
        "forecast_origin": forecast_origin,
        "model_path": "deterministic_dispatcher",
        "fallback_used": False,
        "fallback_reason": "",
        "runtime_seconds": 0.0,
        "prompt": "",
        "response_text": json.dumps(response, sort_keys=True),
    })


def _record_deterministic_stage(
    *,
    trace_rows: list[dict[str, object]],
    stage: str,
    dataset_key: str,
    payload: dict[str, object],
) -> None:
    trace_rows.append({
        "stage": stage,
        "dataset_key": dataset_key,
        "forecast_origin": "",
        "model_path": "deterministic_replay_controller",
        "fallback_used": False,
        "fallback_reason": "",
        "runtime_seconds": 0.0,
        "prompt": "",
        "response_text": json.dumps(payload, sort_keys=True),
    })


def _call_stage(engine: Any, *, stage: str, dataset_key: str, prompt_payload: dict[str, object], trace_rows: list[dict[str, object]]) -> dict[str, object]:
    call = engine.generate_json(
        stage=stage,
        system_prompt="You are the agentic top-one controller for a forecasting baseline.",
        user_prompt=json.dumps(prompt_payload, sort_keys=True),
    )
    trace_rows.append({
        "stage": call.stage,
        "dataset_key": dataset_key,
        "forecast_origin": "",
        "model_path": call.model_path,
        "fallback_used": call.fallback_used,
        "fallback_reason": call.fallback_reason,
        "runtime_seconds": call.runtime_seconds,
        "prompt": call.prompt,
        "response_text": call.response_text,
    })
    return call.response_json


def _task_planning_context(
    static_context: dict[str, object],
) -> dict[str, object]:
    ""








    if (
        str(static_context.get("selection_context_profile", ""))
        != "qwen25_multiscale_released_sequence_v1"
    ):
        return static_context
    return benchmark_b_llm_prompt_fields(static_context)


def _task_planning_trace_metadata(
    static_context: dict[str, object],
) -> dict[str, object]:
    """Return control metadata that must remain outside the planner prompt."""
    if (
        str(static_context.get("selection_context_profile", ""))
        != "qwen25_multiscale_released_sequence_v1"
    ):
        return {}
    return {
        "full_augmented_context_used_by_model_selection": True,
        "full_augmented_context_control_trace_only": True,
        "planner_output_consumed_by_selector": False,
    }


def run_agentic_top_one(
    *,
    manifest_path: str | Path,
    registry_path: str | Path,
    out_dir: str | Path,
    engine: Any | None = None,
    dataset_keys: list[str] | tuple[str, ...] | None = None,
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
        raise ValueError("formal noLearning Top-1 requires the common selection packet")
    selection_provenance = frozen_selection.provenance() if frozen_selection else {}
    eligible = eligible_registry(registry)
    if eligible.empty:
        raise RuntimeError("no selection-eligible candidates in registry")
    registry_meta = write_registry_snapshot(registry, out, registry_path)
    contexts, expected_rows = load_dataset_contexts(manifest_path, root=root, caster_root=caster_root)
    all_dataset_keys = sorted({str(ctx.dataset_key) for ctx in contexts})
    requested_dataset_keys = [str(key) for key in dataset_keys] if dataset_keys is not None else []
    if "all" in requested_dataset_keys:
        requested_dataset_keys = []
    if requested_dataset_keys:
        wanted = set(requested_dataset_keys)
        contexts = [ctx for ctx in contexts if ctx.dataset_key in wanted]
        if not contexts:
            raise RuntimeError(f"no manifest rows selected for agentic_top_one dataset_keys={requested_dataset_keys}")
        expected_rows = int(sum(len(ctx.ledger) for ctx in contexts))
    selected_dataset_keys = sorted({str(ctx.dataset_key) for ctx in contexts})
    if any(is_benchmark_b_context(ctx) for ctx in contexts):
        if not archive_backed:
            raise ValueError("benchmark_b_pooled Top-1 executor must use the shared frozen archive")
    excluded_dataset_keys = sorted(set(all_dataset_keys) - set(selected_dataset_keys))
    dataset_scope = "all_available" if not excluded_dataset_keys else "custom"
    agent_selection_scope = derive_agent_selection_scope(selected_dataset_keys, excluded_dataset_keys)
    trace_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    forecast_frames: list[pd.DataFrame] = []
    selected_by_dataset: dict[str, pd.Series] = {}
    selection_seconds = 0.0
    archive_lookup_seconds = 0.0
    agent_control_seconds = 0.0

    for ctx in contexts:
        static_context = (
            benchmark_b_prompt_payload(ctx, forecast_origin=None, frozen_selection=frozen_selection)
            if is_benchmark_b_context(ctx)
            else {}
        )
        control_start = time.time()
        planning_payload = {
                "ledger_rows": int(len(ctx.ledger)),
                "task": "Plan a deterministic top-1 model-skill forecasting baseline.",
                "required_response": {"plan": "short plan"},
                **_task_planning_context(static_context),
            }
        if engine is None:
            _record_deterministic_stage(
                trace_rows=trace_rows,
                stage="task_planning",
                dataset_key=ctx.dataset_key,
                payload={"plan": "replay previous LLM selection without validation guard", **planning_payload},
            )
        else:
            _call_stage(
                engine,
                stage="task_planning",
                dataset_key=ctx.dataset_key,
                prompt_payload=planning_payload,
                trace_rows=trace_rows,
            )
        trace_rows[-1].update(_task_planning_trace_metadata(static_context))
        agent_control_seconds += round(time.time() - control_start, 6)
        if static_context:
            base_context = benchmark_b_prompt_text(
                static_context["canonical_context"],
                str(static_context["canonical_context_sha256"]),
            )
            context_metadata = {
                "selection_context_schema": static_context["canonical_context"]["schema"],
                "selection_context_sha256": static_context["canonical_context_sha256"],
                "selection_context_cutoff": static_context["forecast_origin"],
                "selection_context_builder": "caster.data.benchmark_b_context.build_router_context",
            }
        else:
            base_context, context_metadata = selection_context_with_metadata(ctx)
        select_start = time.time()
        selected = select_candidate_no_validation(
            engine=engine,
            stage="model_selection",
            dataset_key=ctx.dataset_key,
            forecast_origin="DATASET_LEVEL",
            candidates=eligible,
            task="Select one executable model skill for all release groups in this dataset.",
            trace_rows=trace_rows,
            context_text=base_context,
            context_metadata=context_metadata,
            selection_replay=selection_replay or None,
        )
        selected.attrs.setdefault("llm_payload", {})["agent_selection_scope"] = agent_selection_scope
        selected.attrs.setdefault("llm_payload", {}).update(static_context)
        selected.attrs.setdefault("llm_payload", {}).update(selection_provenance)
        selection_seconds += round(time.time() - select_start, 6)
        selected_by_dataset[ctx.dataset_key] = selected
        selection_rows.append(selection_log_row(
            stage="model_selection",
            dataset_key=ctx.dataset_key,
            forecast_origin="DATASET_LEVEL",
            selected=selected,
            llm_payload=selected.attrs.get("llm_payload", {}),
        ))

        predictions_cache = None
        if archive_backed:
            assert archive_index is not None
            validate_archive_coverage(
                ctx=ctx,
                ledger_subset=ctx.ledger,
                selected_model_id=str(selected["model_id"]),
                archive_index=archive_index,
                out_dir=out,
            )
        elif is_stateful(selected):
            predictions_cache = pre_fit_predictions(selected, ctx, ctx.ledger)

        if archive_backed:
            assert archive_index is not None
            lookup_start = time.time()
            frame = make_forecast_rows_from_archive(
                ctx=ctx,
                ledger_subset=ctx.ledger,
                selected=selected,
                method="agentic_top_one",
                archive_index=archive_index,
            )
            archive_lookup_seconds += round(time.time() - lookup_start, 6)
            for key, value in selection_provenance.items():
                frame[key] = value
                                                                             
                                                                           
            frame = annotate_forecast_context(frame, ctx=ctx, frozen_selection=frozen_selection)
            forecast_frames.append(frame)
            forecast_frames[-1]["agent_selection_scope"] = agent_selection_scope
        else:
            for ledger_idx, event in ctx.ledger.iterrows():
                row, _train_rows = make_forecast_row(
                    ctx=ctx,
                    event=event,
                    ledger_idx=int(ledger_idx),
                    selected=selected,
                    method="agentic_top_one",
                    predictions_cache=predictions_cache,
                )
                row["agent_selection_scope"] = agent_selection_scope
                forecast_rows.append(row)
        dispatch_frame = forecast_frames[-1] if archive_backed else pd.DataFrame(forecast_rows)
        for origin, origin_frame in dispatch_frame.groupby("forecast_origin", sort=True):
            origin_context = (
                benchmark_b_prompt_payload(ctx, forecast_origin=origin, frozen_selection=frozen_selection)
                if is_benchmark_b_context(ctx)
                else {}
            )
            _record_dispatch(
                trace_rows=trace_rows,
                dataset_key=ctx.dataset_key,
                selected_model_id=str(selected["model_id"]),
                forecast_rows=int(len(origin_frame)),
                archive_backed=archive_backed,
                forecast_origin=str(origin),
                context_payload=origin_context,
            )

    forecast = pd.concat(forecast_frames, ignore_index=True) if archive_backed else pd.DataFrame(forecast_rows)
    if len(forecast) != expected_rows:
        raise RuntimeError(f"forecast row mismatch expected={expected_rows} actual={len(forecast)}")

    summary_payload = {
            "forecast_rows": int(len(forecast)),
            "required_response": {"summary": "short run summary"},
        }
    if engine is None:
        _record_deterministic_stage(
            trace_rows=trace_rows,
            stage="response_summary",
            dataset_key="ALL",
            payload={"summary": "replay run completed", **summary_payload},
        )
    else:
        _call_stage(
            engine,
            stage="response_summary",
            dataset_key="ALL",
            prompt_payload=summary_payload,
            trace_rows=trace_rows,
        )

    pd.DataFrame(selection_rows).to_csv(out / "candidate_selection_log.csv", index=False)
    _trace_jsonl(out / "agentic_top_one_trace.jsonl", trace_rows)
    config = qwen_config(engine)
    config.update({
        "llm_runtime_seconds": total_trace_runtime(trace_rows),
        "primary_model_path": getattr(engine, "primary_model_path", ""),
        "fallback_model_path": getattr(engine, "fallback_model_path", ""),
    })
    (out / "qwen_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    timing_extra = {
        **config,
        "llm_runtime_seconds": total_trace_runtime(trace_rows),
        "llm_load_seconds_excluded_from_update": round(float(llm_load_seconds_excluded), 6),
        "llm_generation_seconds_charged_to_update": round(total_trace_runtime(trace_rows), 6),
        "selection_engine": "selection_replay" if selection_replay else "qwen",
        "restart_type": "archive_backed_true_selection_readout" if archive_backed and not selection_replay else ("archive_backed_selection_replay_readout" if archive_backed else "top1_refit_forecast"),
        "selection_score_source": "no_validation",
        "validation_score_used": False,
        "selection_replay_used": bool(selection_replay),
        "agent_selection_scope": agent_selection_scope,
    }
    if archive_backed:
        timing = archive_timing_payload(
            start_time=start,
            archive_lookup_seconds=archive_lookup_seconds,
            agent_control_seconds=agent_control_seconds,
            selection_seconds=selection_seconds,
            charge_selection=charge_selection,
            forecast_rows=int(len(forecast)),
            expected_rows=int(expected_rows),
            extra=timing_extra,
        )
    else:
        timing = alternate_timing_payload({
            "total_seconds": round(time.time() - start, 6),
            "llm_runtime_seconds": total_trace_runtime(trace_rows),
            "forecast_rows": int(len(forecast)),
            "expected_rows": int(expected_rows),
            "artifact_reuse": False,
            "selection_score_source": "no_validation",
            "validation_score_used": False,
            "selection_replay_used": bool(selection_replay),
        })
    run_manifest = {
        "baseline_names": ["agentic_top_one"],
        "model": "agentic_top_one",
        "backend": "agentic_top_one_style_qwen_local",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "selected_dataset_keys": selected_dataset_keys,
        "excluded_dataset_keys": excluded_dataset_keys,
        "dataset_scope": dataset_scope,
        "agent_selection_scope": agent_selection_scope,
        **registry_meta,
        **config,
        "timing_mode": resolved_timing_mode,
        "timing_semantics": ARCHIVE_TIMING_SEMANTICS if archive_backed else alternate_TIMING_MODE,
        "restart_type": timing_extra["restart_type"],
        "forecast_source": ARCHIVE_FORECAST_SOURCE if archive_backed else "baseline_local_refit_forecast_recipe",
        "forecast_archive": str(forecast_archive or ""),
        "archive_mode": str(archive_mode),
        "formal_timing_valid": bool(archive_backed),
        "model_compute_excluded": bool(archive_backed),
        "selection_charged_to_update": bool(charge_selection),
        "selected_models_by_dataset": {k: str(v["model_id"]) for k, v in selected_by_dataset.items()},
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
        "no_leakage_rule": "each forecast uses only panel_time <= forecast_origin",
        "artifact_reuse": False,
        **selection_provenance,
    }
    for context in contexts:
        run_manifest.update(benchmark_b_context_manifest_fields(context, frozen_selection))
    return write_standard_artifacts(out_dir=out, forecast=forecast, timing=timing, run_manifest=run_manifest)
