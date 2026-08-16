from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
from .agentic_llm import QwenLocalEngine
from .agentic_skills import (
    derive_agent_selection_scope,
    eligible_registry,
    forecast_horizon_group,
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


FULL_RECOVERY_DATASET_EXCLUDED_MODEL_IDS: dict[str, set[str]] = {}
DEFAULT_FULL_RECOVERY_DATASET_KEYS: tuple[str, ...] | None = None
FULL_RECOVERY_SELECTION_POLICIES = ("llm_only", "no_validation_score")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _record_agent_event(
    *,
    agent: str,
    dataset_key: str,
    forecast_origin: str,
    payload: dict[str, object],
    trace_rows: list[dict[str, object]],
) -> None:
    trace_rows.append({
        "agent": agent,
        "stage": agent,
        "dataset_key": dataset_key,
        "forecast_origin": forecast_origin,
        "model_path": "deterministic_agent_validation",
        "fallback_used": False,
        "fallback_reason": "",
        "runtime_seconds": 0.0,
        "prompt": "",
        "response_text": json.dumps(payload, sort_keys=True),
    })


def _agent_call(
    engine: Any,
    *,
    agent: str,
    dataset_key: str,
    forecast_origin: str,
    payload: dict[str, object],
    trace_rows: list[dict[str, object]],
) -> dict[str, object]:
    if agent == "PlannerAgent":
        system_prompt = (
            "You are PlannerAgent in an Agentic Full Recovery forecasting baseline. "
            "Return exactly one compact JSON object: {\"plan\":\"ok\"}."
        )
    else:
        system_prompt = (
            f"You are {agent} in an Agentic Full Recovery multi-agent "
            "forecasting baseline."
        )
    call = engine.generate_json(
        stage=agent,
        system_prompt=system_prompt,
        user_prompt=json.dumps(payload, sort_keys=True),
    )
    trace_rows.append({
        "agent": agent,
        "stage": call.stage,
        "dataset_key": dataset_key,
        "forecast_origin": forecast_origin,
        "model_path": call.model_path,
        "fallback_used": call.fallback_used,
        "fallback_reason": call.fallback_reason,
        "runtime_seconds": call.runtime_seconds,
        "prompt": call.prompt,
        "response_text": call.response_text,
    })
    return call.response_json


def _release_train_rows(ctx, release: pd.DataFrame, origin_text: str) -> int:
    total = 0
    origin = pd.to_datetime(origin_text, errors="coerce")
    for (entity, component), _g in release.groupby([ctx.ledger_entity_col, "component"], dropna=False):
        values = group_history_values(ctx, str(entity), str(component), origin)
        total += int(len(values))
    return total


def _full_recovery_candidates_for_dataset(candidates: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    excluded = FULL_RECOVERY_DATASET_EXCLUDED_MODEL_IDS.get(str(dataset_key), set())
    if not excluded:
        return candidates
    filtered = candidates[~candidates["model_id"].astype(str).isin(excluded)].copy()
    if filtered.empty:
        raise RuntimeError(
            "no full-recovery candidates after exclusions for "
            f"dataset_key={dataset_key}"
        )
    return filtered


def _candidate_by_id(candidates: pd.DataFrame, model_id: str, fallback: pd.Series) -> pd.Series:
    matches = candidates[candidates["model_id"].astype(str) == str(model_id)]
    if matches.empty:
        return fallback.copy()
    return matches.iloc[0].copy()


def _row_horizon_group(ctx, row: pd.Series) -> str:
    return forecast_horizon_group(ctx.dataset_key, row.get("mode", ""), row.get("horizon", 0))


def _release_with_horizon_group(ctx, release: pd.DataFrame) -> pd.DataFrame:
    out = release.copy()
    out["_horizon_group"] = [_row_horizon_group(ctx, row) for _, row in out.iterrows()]
    return out


def _validation_metric_columns(prefix: str, summary: dict[str, object]) -> dict[str, object]:
    return {
        f"{prefix}_n_eval": summary.get("n_eval", 0),
        f"{prefix}_rmse": summary.get("rmse", ""),
        f"{prefix}_mae": summary.get("mae", ""),
        f"{prefix}_rank": summary.get("rank", ""),
        f"{prefix}_guarded_rank_score": summary.get("guarded_rank_score", ""),
        f"{prefix}_origin_growth_guard_p95": summary.get("origin_growth_guard_p95", ""),
        f"{prefix}_status": summary.get("status", ""),
    }


def run_agentic_full_recovery(
    *,
    manifest_path: str | Path,
    registry_path: str | Path,
    out_dir: str | Path,
    engine: Any | None = None,
    dataset_keys: list[str] | tuple[str, ...] | None = DEFAULT_FULL_RECOVERY_DATASET_KEYS,
    selection_policy: str = "llm_only",
    method_name: str | None = None,
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
    if selection_policy not in FULL_RECOVERY_SELECTION_POLICIES:
        raise ValueError(
            "selection_policy must be one of "
            f"{FULL_RECOVERY_SELECTION_POLICIES}; got {selection_policy!r}"
        )
    if method_name is None:
        method_name = "agentic_full_recovery"
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

    registry = read_candidate_registry(registry_path)
    frozen_selection = None
    if selection_input_candidates is not None or selection_input_manifest is not None:
        if selection_input_candidates is None or selection_input_manifest is None:
            raise ValueError("common selection candidates and manifest must be supplied together")
        frozen_selection = load_common_selection_input(selection_input_candidates, selection_input_manifest)
        registry = merge_common_selection_input(registry, frozen_selection)
    if require_selection_input_parity and frozen_selection is None:
        raise ValueError(
            "Agentic Full Recovery requires the common CASTER--Agent selection packet"
        )
    selection_provenance = frozen_selection.provenance() if frozen_selection else {}
    eligible = eligible_registry(registry)
    if eligible.empty:
        raise RuntimeError("no full-recovery candidates in registry")
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
            raise RuntimeError(f"no manifest rows selected for {method_name} dataset_keys={requested_dataset_keys}")
    selected_dataset_keys = sorted({str(ctx.dataset_key) for ctx in contexts})
    if any(is_benchmark_b_context(ctx) for ctx in contexts):
        if not archive_backed:
            raise ValueError("benchmark_b_pooled Agent executor must use the shared frozen archive")
    excluded_dataset_keys = sorted(set(all_dataset_keys) - set(selected_dataset_keys))
    dataset_scope = "all_available" if not excluded_dataset_keys else "custom"
    agent_selection_scope = derive_agent_selection_scope(selected_dataset_keys, excluded_dataset_keys)
    expected_rows = int(sum(len(ctx.ledger) for ctx in contexts))
    conversation_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    restart_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    forecast_frames: list[pd.DataFrame] = []
    archive_readout_validation_rows: list[dict[str, object]] = []
    proposal_seconds = 0.0
    planner_stage_seconds = 0.0
    selector_stage_seconds = 0.0
    train_seconds = 0.0
    forecast_seconds = 0.0
    archive_coverage_validation_seconds = 0.0
    readout_enrichment_seconds = 0.0
    readout_materialization_seconds = 0.0
    readout_validation_seconds = 0.0
    readout_artifact_write_seconds = 0.0
    score_seconds = 0.0
    llm_load_seconds_excluded = 0.0

    if exclude_llm_load_from_timing and engine is not None and hasattr(engine, "warm_load"):
        llm_load_seconds_excluded = float(engine.warm_load())

    for ctx in contexts:
        for origin_text, release in ctx.ledger.groupby("forecast_origin", dropna=False, sort=True):
            origin_text = str(origin_text)
            canonical_context = (
                benchmark_b_prompt_payload(
                    ctx, forecast_origin=origin_text, frozen_selection=frozen_selection
                )
                if is_benchmark_b_context(ctx)
                else {}
            )
            group_start = time.time()
            planner_stage_start = time.perf_counter()
            planner_payload = {
                    "forecast_origin": origin_text,
                    "release_rows": int(len(release)),
                    "task": "Plan candidate selection and archived forecast readout.",
                    "required_response": {"plan": "ok"},
                    **(
                        benchmark_b_llm_prompt_fields(canonical_context)
                        if canonical_context
                        else {}
                    ),
                }
            if engine is None:
                _record_agent_event(
                    agent="PlannerAgent",
                    dataset_key=ctx.dataset_key,
                    forecast_origin=origin_text,
                    payload={"plan": "replay previous LLM selection", **planner_payload},
                    trace_rows=conversation_rows,
                )
            else:
                _agent_call(
                    engine,
                    agent="PlannerAgent",
                    dataset_key=ctx.dataset_key,
                    forecast_origin=origin_text,
                    payload=planner_payload,
                    trace_rows=conversation_rows,
                )
            planner_stage_end = time.perf_counter()
            planner_stage_runtime = planner_stage_end - planner_stage_start
            planner_trace_row = conversation_rows[-1]
            if (
                str(planner_trace_row.get("stage", "")) != "PlannerAgent"
                or str(planner_trace_row.get("dataset_key", ""))
                != str(ctx.dataset_key)
                or str(planner_trace_row.get("forecast_origin", ""))
                != origin_text
            ):
                raise RuntimeError("Planner-stage trace binding mismatch")
            planner_qwen_runtime = float(
                planner_trace_row.get("runtime_seconds", 0.0)
            )
            if planner_stage_runtime + 1e-6 < planner_qwen_runtime:
                raise RuntimeError(
                    "Planner-stage wall time is smaller than its Qwen runtime"
                )
            planner_trace_row.update(
                {
                    "stage_wall_seconds": round(planner_stage_runtime, 6),
                    "stage_non_qwen_seconds": round(
                        planner_stage_runtime - planner_qwen_runtime, 6
                    ),
                    "stage_timing_schema": "agentic_full_recovery_planner_stage_wall_v1",
                }
            )
                                                                               
                                                                             
                                                                                
            selector_stage_start = time.perf_counter()
            origin_candidates = _full_recovery_candidates_for_dataset(
                eligible, ctx.dataset_key
            )
            if canonical_context:
                context_text = benchmark_b_prompt_text(
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
                context_text, context_metadata = selection_context_with_metadata(
                    ctx, release=release, origin_text=origin_text
                )
            selected = select_candidate_no_validation(
                engine=engine,
                stage="SelectorAgent",
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                candidates=origin_candidates,
                task="Select one model skill for this release/origin group.",
                trace_rows=conversation_rows,
                context_text=context_text,
                context_metadata=context_metadata,
                selection_replay=selection_replay or None,
            )
            selected.attrs.setdefault("llm_payload", {})["agent_selection_scope"] = agent_selection_scope
            selected.attrs.setdefault("llm_payload", {}).update(canonical_context)
            selected.attrs.setdefault("llm_payload", {}).update(selection_provenance)
            selector_stage_end = time.perf_counter()
            selector_stage_runtime = selector_stage_end - selector_stage_start
            selector_trace_row = conversation_rows[-1]
            if (
                str(selector_trace_row.get("stage", "")) != "SelectorAgent"
                or str(selector_trace_row.get("dataset_key", ""))
                != str(ctx.dataset_key)
                or str(selector_trace_row.get("forecast_origin", ""))
                != origin_text
            ):
                raise RuntimeError("Selector-stage trace binding mismatch")
            selector_qwen_runtime = float(
                selector_trace_row.get("runtime_seconds", 0.0)
            )
            if selector_stage_runtime + 1e-6 < selector_qwen_runtime:
                raise RuntimeError(
                    "Selector-stage wall time is smaller than its Qwen runtime"
                )
            selector_trace_row.update(
                {
                    "stage_wall_seconds": round(selector_stage_runtime, 6),
                    "stage_non_qwen_seconds": round(
                        selector_stage_runtime - selector_qwen_runtime, 6
                    ),
                    "stage_timing_schema": "agentic_full_recovery_selector_stage_wall_v1",
                }
            )
            prop_runtime = planner_stage_runtime + selector_stage_runtime
            proposal_seconds += prop_runtime
            planner_stage_seconds += planner_stage_runtime
            selector_stage_seconds += selector_stage_runtime
            selection_rows.append(selection_log_row(
                stage="SelectorAgent",
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                selected=selected,
                llm_payload=selected.attrs.get("llm_payload", {}),
            ))
            selected_id = str(selected["model_id"])

            fit_start = time.time()
            if archive_backed:
                assert archive_index is not None
                train_rows = 0
                predictions_cache = None
                fit_runtime = 0.0
            else:
                train_rows = _release_train_rows(ctx, release, origin_text)
                predictions_cache = pre_fit_predictions(selected, ctx, release) if is_stateful(selected) else None
                fit_runtime = round(time.time() - fit_start, 6)
            train_seconds += fit_runtime

            release_forecast_rows = []
            if archive_backed:
                assert archive_index is not None
                coverage_validation_start = time.perf_counter()
                validate_archive_coverage(
                    ctx=ctx,
                    ledger_subset=release,
                    selected_model_id=selected_id,
                    archive_index=archive_index,
                    out_dir=out,
                )
                coverage_validation_runtime = (
                    time.perf_counter() - coverage_validation_start
                )
                archive_coverage_validation_seconds += coverage_validation_runtime
                fcst_start = time.perf_counter()
                release_frame = make_forecast_rows_from_archive(
                    ctx=ctx,
                    ledger_subset=release,
                    selected=selected,
                    method=method_name,
                    archive_index=archive_index,
                )
                fcst_runtime = time.perf_counter() - fcst_start
                enrichment_start = time.perf_counter()
                release_frame["agent_selection_scope"] = agent_selection_scope
                release_frame["selection_policy"] = selection_policy
                release_frame["selection_horizon_group"] = [
                    _row_horizon_group(ctx, event) for _, event in release.iterrows()
                ]
                release_frame["overall_selected_model_id"] = selected_id
                release_frame["selection_score_source"] = "no_validation"
                for key, value in selection_provenance.items():
                    release_frame[key] = value
                release_frame = annotate_forecast_context(
                    release_frame, ctx=ctx, frozen_selection=frozen_selection
                )
                forecast_frames.append(release_frame)
                release_forecast_count = int(len(release_frame))
                archive_readout_validation_rows.append(
                    {
                        "dataset_key": ctx.dataset_key,
                        "forecast_origin": origin_text,
                        "selected_model_id": selected_id,
                        "expected_rows": int(len(release)),
                        "readout_rows": release_forecast_count,
                        "missing_archive_rows": 0,
                        "archive_lookup_status": "hit",
                        "status": "PASS",
                    }
                )
                enrichment_runtime = time.perf_counter() - enrichment_start
                readout_enrichment_seconds += enrichment_runtime
            else:
                fcst_start = time.perf_counter()
                for ledger_idx, event in release.iterrows():
                    event_group = _row_horizon_group(ctx, event)
                    row, _row_train = make_forecast_row(
                        ctx=ctx,
                        event=event,
                        ledger_idx=int(ledger_idx),
                        selected=selected,
                        method=method_name,
                        predictions_cache=predictions_cache,
                    )
                    row["agent_selection_scope"] = agent_selection_scope
                    row["selection_policy"] = selection_policy
                    row["selection_horizon_group"] = event_group
                    row["overall_selected_model_id"] = selected_id
                    row["selection_score_source"] = "no_validation"
                    release_forecast_rows.append(row)
                release_forecast_count = int(len(release_forecast_rows))
                fcst_runtime = time.perf_counter() - fcst_start
                coverage_validation_runtime = 0.0
                enrichment_runtime = 0.0
            forecast_seconds += fcst_runtime
            if not archive_backed:
                forecast_rows.extend(release_forecast_rows)
            _record_agent_event(
                agent="ExecutorAgent",
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                payload={
                    "status": "ok",
                    "selected_model_id": selected_id,
                    "selection_policy": selection_policy,
                    "selection_score_source": "no_validation",
                    "forecast_rows": release_forecast_count,
                    "executor": "immutable forecast archive lookup" if archive_backed else "baseline-local instantiate/refit/forecast recipe",
                    **canonical_context,
                },
                trace_rows=conversation_rows,
            )

            critic_start = time.time()
            _record_agent_event(
                agent="CriticAgent",
                dataset_key=ctx.dataset_key,
                forecast_origin=origin_text,
                payload={
                    "dataset_key": ctx.dataset_key,
                    "forecast_origin": origin_text,
                    "selected_model_id": selected_id,
                    "selection_policy": selection_policy,
                    "selection_score_source": "no_validation",
                    "forecast_rows": release_forecast_count,
                    "train_rows": int(train_rows),
                    "status": "ok",
                    "checks": ["no artifact reuse", "history <= forecast_origin", "schema rows complete"],
                    **canonical_context,
                    "notes": (
                        "deterministic validation passed after immutable archive lookup"
                        if archive_backed
                        else "deterministic validation passed after local execution"
                    ),
                },
                trace_rows=conversation_rows,
            )
            critic_runtime = round(time.time() - critic_start, 6)
            score_seconds += critic_runtime
            restart_rows.append({
                "dataset_key": ctx.dataset_key,
                "forecast_origin": origin_text,
                "selected_model_id": selected_id,
                "selection_policy": selection_policy,
                "selection_score_source": "no_validation",
                "validation_rows": 0,
                "method": method_name,
                "execute_short_model_id": selected_id,
                "execute_long_model_id": selected_id,
                "train_rows": int(train_rows),
                "proposal_seconds": round(prop_runtime, 6),
                "planner_stage_seconds": round(planner_stage_runtime, 6),
                "planner_qwen_seconds": round(planner_qwen_runtime, 6),
                "planner_non_qwen_seconds": round(
                    planner_stage_runtime - planner_qwen_runtime, 6
                ),
                "selector_stage_seconds": round(selector_stage_runtime, 6),
                "selector_qwen_seconds": round(selector_qwen_runtime, 6),
                "selector_non_qwen_seconds": round(
                    selector_stage_runtime - selector_qwen_runtime, 6
                ),
                "train_seconds": fit_runtime,
                "forecast_seconds": round(fcst_runtime, 6),
                "archive_coverage_validation_seconds": round(
                    coverage_validation_runtime, 6
                ),
                "readout_enrichment_seconds": round(enrichment_runtime, 6),
                "total_seconds": round(time.time() - group_start, 6),
                "status": "ok",
            })
            print(
                "restart_group_done "
                f"dataset_key={ctx.dataset_key} forecast_origin={origin_text} "
                f"selected_model_id={selected_id} selection_policy={selection_policy} rows={release_forecast_count}",
                flush=True,
            )

    if archive_backed:
        materialization_start = time.perf_counter()
        forecast = pd.concat(forecast_frames, ignore_index=True)
        if len(forecast) != expected_rows:
            raise RuntimeError(
                f"forecast row mismatch expected={expected_rows} actual={len(forecast)}"
            )
        forecast_readout = forecast[
            forecast["split"].astype(str).str.lower().eq("test")
        ].copy()
        readout_materialization_seconds = (
            time.perf_counter() - materialization_start
        )
        validation_start = time.perf_counter()
        group_validation = pd.DataFrame(archive_readout_validation_rows)
        if len(group_validation) != len(restart_rows):
            raise RuntimeError(
                "archive readout validation group mismatch "
                f"expected={len(restart_rows)} actual={len(group_validation)}"
            )
        if int(group_validation["expected_rows"].sum()) != expected_rows:
            raise RuntimeError("archive readout validation expected-row total mismatch")
        if int(group_validation["readout_rows"].sum()) != expected_rows:
            raise RuntimeError("archive readout validation materialized-row total mismatch")
        if int(group_validation["missing_archive_rows"].sum()) != 0:
            raise RuntimeError("archive readout validation found missing archive rows")
        if not group_validation["status"].astype(str).eq("PASS").all():
            raise RuntimeError("archive readout validation did not pass")
        validation_columns = [
            "forecast_id",
            "dataset_key",
            "method",
            "forecast_origin",
            "target_time",
            "split",
            "selected_model_id",
            "archive_model_id",
            "forecast_source",
            "archive_lookup_status",
        ]
        missing_validation_columns = [
            column for column in validation_columns if column not in forecast_readout.columns
        ]
        if missing_validation_columns:
            raise RuntimeError(
                "archive forecast readout validation missing columns: "
                f"{missing_validation_columns}"
            )
        archive_readout_validation = forecast_readout[validation_columns].copy()
        archive_readout_validation["selected_archive_model_match"] = (
            archive_readout_validation["selected_model_id"].astype(str)
            == archive_readout_validation["archive_model_id"].astype(str)
        )
        archive_readout_validation["immutable_archive_hit"] = (
            archive_readout_validation["forecast_source"]
            .astype(str)
            .eq(ARCHIVE_FORECAST_SOURCE)
            & archive_readout_validation["archive_lookup_status"]
            .astype(str)
            .eq("hit")
        )
        archive_readout_validation["validation_status"] = (
            archive_readout_validation["selected_archive_model_match"]
            & archive_readout_validation["immutable_archive_hit"]
            & archive_readout_validation["split"].astype(str).str.lower().eq("test")
        ).map({True: "PASS", False: "FAIL"})
        if forecast_readout.empty:
            raise RuntimeError("archive forecast readout has no test rows")
        if not archive_readout_validation["validation_status"].astype(str).eq("PASS").all():
            raise RuntimeError("archive forecast readout validation did not pass")
        if archive_readout_validation["forecast_id"].astype(str).duplicated().any():
            raise RuntimeError("archive forecast readout has duplicate forecast_id values")
        readout_validation_seconds = time.perf_counter() - validation_start
        artifact_write_start = time.perf_counter()
        forecast_readout.to_csv(out / "forecast_readout.csv", index=False)
        archive_readout_validation.to_csv(
            out / "archive_forecast_readout_validation.csv", index=False
        )
        readout_artifact_write_seconds = (
            time.perf_counter() - artifact_write_start
        )
    else:
        forecast = pd.DataFrame(forecast_rows)
        if len(forecast) != expected_rows:
            raise RuntimeError(
                f"forecast row mismatch expected={expected_rows} actual={len(forecast)}"
            )

    pd.DataFrame(selection_rows).to_csv(out / "candidate_selection_log.csv", index=False)
    pd.DataFrame(restart_rows).to_csv(out / "restart_log.csv", index=False)
    _jsonl(out / "agentic_full_recovery_conversation.jsonl", conversation_rows)
    config = qwen_config(engine)
    config.update({
        "llm_runtime_seconds": total_trace_runtime(conversation_rows),
        "primary_model_path": getattr(engine, "primary_model_path", ""),
        "fallback_model_path": getattr(engine, "fallback_model_path", ""),
    })
    planner_qwen_seconds = sum(
        float(row.get("runtime_seconds", 0.0))
        for row in conversation_rows
        if str(row.get("stage", row.get("agent", ""))) == "PlannerAgent"
    )
    selector_qwen_seconds = sum(
        float(row.get("runtime_seconds", 0.0))
        for row in conversation_rows
        if str(row.get("stage", row.get("agent", ""))) == "SelectorAgent"
    )
    if planner_stage_seconds + 1e-6 < planner_qwen_seconds:
        raise RuntimeError(
            "Planner-stage wall time is smaller than its recorded Qwen runtime"
        )
    if selector_stage_seconds + 1e-6 < selector_qwen_seconds:
        raise RuntimeError(
            "Selector-stage wall time is smaller than its recorded Qwen runtime"
        )
    (out / "qwen_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    if archive_backed and not selection_replay:
        restart_type = "archive_backed_true_selection_readout"
        timing_semantics = "agent_true_selection_control_plus_immutable_forecast_archive_readout"
    elif archive_backed:
        restart_type = "archive_backed_selection_replay_readout"
        timing_semantics = "agent_replayed_selection_control_plus_immutable_forecast_archive_readout"
    else:
        restart_type = "true_selection_refit_forecast"
        timing_semantics = alternate_TIMING_MODE

    timing_extra = {
        **config,
        "model": method_name,
        "restart_type": restart_type,
        "timing_semantics": timing_semantics,
        "selection_engine": "selection_replay"
        if selection_replay
        else (
            "deterministic_no_model_compute"
            if str(getattr(engine, "active_model_path", "")) == "deterministic_no_model_compute"
            else "qwen"
        ),
        "selection_policy": selection_policy,
        "selection_score_source": "no_validation",
        "validation_score_used": False,
        "selection_replay_used": bool(selection_replay),
        "agent_selection_scope": agent_selection_scope,
        "llm_load_seconds_excluded_from_update": round(float(llm_load_seconds_excluded), 6),
        "llm_generation_seconds_charged_to_update": round(total_trace_runtime(conversation_rows), 6),
        "proposal_seconds": round(proposal_seconds, 6),
        "planner_stage_seconds": round(planner_stage_seconds, 6),
        "planner_qwen_seconds": round(planner_qwen_seconds, 6),
        "planner_non_qwen_seconds": round(
            planner_stage_seconds - planner_qwen_seconds, 6
        ),
        "planner_stage_count": int(
            sum(
                1
                for row in conversation_rows
                if str(row.get("stage", row.get("agent", "")))
                == "PlannerAgent"
            )
        ),
        "selector_stage_seconds": round(selector_stage_seconds, 6),
        "selector_qwen_seconds": round(selector_qwen_seconds, 6),
        "selector_non_qwen_seconds": round(
            selector_stage_seconds - selector_qwen_seconds, 6
        ),
        "selector_stage_count": int(
            sum(
                1
                for row in conversation_rows
                if str(row.get("stage", row.get("agent", "")))
                == "SelectorAgent"
            )
        ),
        "planner_stage_boundary": (
            "before planner_payload construction through PlannerAgent Qwen "
            "response handling and in-memory trace append"
        ),
        "planner_stage_timing_schema": "agentic_full_recovery_planner_stage_wall_v1",
        "selector_stage_timing_schema": "agentic_full_recovery_selector_stage_wall_v1",
        "selection_seconds_excluding_planner": round(
            selector_stage_seconds, 6
        ),
        "archive_readout_compute_seconds": round(forecast_seconds, 6),
        "archive_coverage_validation_seconds": round(
            archive_coverage_validation_seconds, 6
        ),
        "readout_enrichment_seconds": round(readout_enrichment_seconds, 6),
        "readout_materialization_seconds": round(
            readout_materialization_seconds, 6
        ),
        "readout_validation_seconds": round(readout_validation_seconds, 6),
        "readout_artifact_write_seconds": round(
            readout_artifact_write_seconds, 6
        ),
        "forecast_readout_seconds": round(
            forecast_seconds
            + archive_coverage_validation_seconds
            + readout_enrichment_seconds
            + readout_materialization_seconds
            + readout_validation_seconds
            + readout_artifact_write_seconds,
            6,
        ),
        "forecast_readout_scope": (
            "archive coverage validation, immutable archive lookup, forecast "
            "annotations, matched test-split materialization and validation, "
            "forecast_readout.csv and archive_forecast_readout_validation.csv writes"
        ),
        "readout_timing_schema": "agent_archive_readout_artifacts_v1",
        "forecast_readout_rows": int(
            len(forecast_readout) if archive_backed else 0
        ),
        "forecast_readout_artifact": (
            str(out / "forecast_readout.csv") if archive_backed else ""
        ),
        "archive_forecast_readout_validation": (
            str(out / "archive_forecast_readout_validation.csv")
            if archive_backed
            else ""
        ),
        "score_seconds": round(score_seconds, 6),
        "critic_trace_seconds": round(score_seconds, 6),
        "runtime_update_sec": round(
            selector_stage_seconds
            + forecast_seconds
            + archive_coverage_validation_seconds
            + readout_enrichment_seconds
            + readout_materialization_seconds
            + readout_validation_seconds
            + readout_artifact_write_seconds,
            6,
        ),
        "runtime_timing_semantics": (
            "complete PlannerAgent stage and deterministic CriticAgent trace "
            "excluded; SelectorAgent stage plus matched validationed archive "
            "readout and CSV persistence retained"
        ),
        "llm_runtime_seconds": total_trace_runtime(conversation_rows),
        "restart_group_rows": int(len(restart_rows)),
    }
    if archive_backed:
        timing = archive_timing_payload(
            start_time=start,
            archive_lookup_seconds=forecast_seconds,
            agent_control_seconds=score_seconds,
            selection_seconds=proposal_seconds,
            charge_selection=charge_selection,
            forecast_rows=int(len(forecast)),
            expected_rows=int(expected_rows),
            extra=timing_extra,
        )
    else:
        timing = alternate_timing_payload({
            **timing_extra,
            "model": method_name,
            "restart_type": "true_selection_refit_forecast",
            "selection_policy": selection_policy,
            "selection_score_source": "no_validation",
            "validation_score_used": False,
            "selection_replay_used": bool(selection_replay),
            "artifact_reuse": False,
            "proposal_seconds": round(proposal_seconds, 6),
            "train_seconds": round(train_seconds, 6),
            "forecast_seconds": round(forecast_seconds, 6),
            "score_seconds": round(score_seconds, 6),
            "total_seconds": round(time.time() - start, 6),
            "llm_runtime_seconds": total_trace_runtime(conversation_rows),
            "forecast_rows": int(len(forecast)),
            "expected_rows": int(expected_rows),
            "restart_group_rows": int(len(restart_rows)),
        })
    run_manifest = {
        "baseline_names": [method_name],
        "model": method_name,
        "backend": "agentic_full_recovery_style_qwen_local",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "selected_dataset_keys": selected_dataset_keys,
        "excluded_dataset_keys": excluded_dataset_keys,
        "dataset_scope": dataset_scope,
        "agent_selection_scope": agent_selection_scope,
        **registry_meta,
        **config,
        "restart_type": restart_type,
        "timing_mode": resolved_timing_mode,
        "timing_semantics": timing_semantics,
        "forecast_source": ARCHIVE_FORECAST_SOURCE if archive_backed else "baseline_local_refit_forecast_recipe",
        "forecast_archive": str(forecast_archive or ""),
        "archive_mode": str(archive_mode),
        "formal_timing_valid": bool(archive_backed),
        "model_compute_excluded": bool(archive_backed),
        "selection_charged_to_update": bool(charge_selection),
        "selection_policy": selection_policy,
        "selection_score_source": "no_validation",
        "validation_score_used": False,
        "validation_context_visible": False,
        "validation_guard_enabled": False,
        "selection_replay_used": bool(selection_replay),
        "selection_replay_log": str(selection_replay_log or ""),
        "selection_replay_log_sha256": sha256_file(Path(selection_replay_log)) if selection_replay_log else "",
        "artifact_reuse": False,
        "selection_guard": "disabled; no validation scores or validation prompt context are used",
        "selection_guard_scoreboard": "",
        "selection_guard_weighted_scoreboard": "",
        "selection_weighted_validation": "",
        "selection_horizon_scoreboard": "",
        "selection_horizon_validation": "",
        "expected_rows": int(expected_rows),
        "restart_group_rows": int(len(restart_rows)),
        "no_leakage_rule": "each restart group refits and forecasts using only panel_time <= forecast_origin; no validation labels are scored for selection",
        **selection_provenance,
    }
    for context in contexts:
        run_manifest.update(benchmark_b_context_manifest_fields(context, frozen_selection))
    return write_standard_artifacts(out_dir=out, forecast=forecast, timing=timing, run_manifest=run_manifest)
