from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
NEW_METHOD_SRC = ROOT / "code/caster/src"
if str(NEW_METHOD_SRC) not in sys.path:
    sys.path.insert(0, str(NEW_METHOD_SRC))

from caster.data.benchmark_b_context import (              
    canonical_json,
)
from caster.tasks import (              
    build_selection_context,
    load_task_specs,
    scientific_context_view,
    selection_context_sha256,
)
from .agentic_skills import (              
    QWEN25_MULTISCALE_CONTEXT_SCHEMA,
    _qwen25_multiscale_context_api,
    _qwen25_multiscale_context_metadata,
    selection_context_profile,
    uses_qwen25_multiscale_context,
)


CONTRACT_PATH = ROOT / "configs/benchmark_b_context_v26_1.yaml"
TASK_SPECS_PATH = ROOT / "configs/caster_task_specs_v20.yaml"
POOLED_LEDGER_PATH = (
    ROOT
    / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/event_ledger.csv"
)


                                                                              
                                                                           
                                                                             
                                                                              
                                                                            
                                                                               
                                                                               
_AGENT_CONTEXT_CACHE: dict[
    tuple[str, str, str],
    tuple[dict[str, object], str, dict[str, object]],
] = {}
_QWEN25_CONTEXT_TEXT_BY_SHA: dict[str, str] = {}


def _context_source_key(ctx: Any) -> str:
    row = getattr(ctx, "manifest_row", {})
    panel_sha = str(row.get("panel_sha256", "")) if hasattr(row, "get") else ""
    ledger_sha = str(row.get("ledger_sha256", "")) if hasattr(row, "get") else ""
    profile = (
        selection_context_profile(ctx)
        if uses_qwen25_multiscale_context(ctx)
        else ""
    )
    profile_suffix = (
        f"|context_profile={profile}"
        if profile
        else ""
    )
    if panel_sha and ledger_sha:
        return (
            f"{ctx.dataset_key}|panel={panel_sha}|ledger={ledger_sha}"
            f"{profile_suffix}"
        )
    return f"{ctx.dataset_key}|object={id(ctx)}{profile_suffix}"


@lru_cache(maxsize=1)
def _task_specs():
    return load_task_specs(TASK_SPECS_PATH)


@lru_cache(maxsize=1)
def _pooled_ledger() -> pd.DataFrame:
    return pd.read_csv(POOLED_LEDGER_PATH, keep_default_na=False, low_memory=False)


def is_benchmark_b_context(ctx: Any) -> bool:
    return str(ctx.dataset) == "benchmark_b" and set(ctx.ledger.get("task_id", pd.Series(dtype=str)).astype(str)) == {"benchmark_b_pooled"}


def build_agent_context(ctx: Any, *, forecast_origin: object | None, frozen_selection: Any | None = None) -> tuple[dict[str, object], str, dict[str, object]]:
    if not is_benchmark_b_context(ctx):
        raise ValueError("canonical Benchmark B context requested for another task")
    normalized_origin = (
        "__FORMAL_SELECTION__"
        if forecast_origin is None
        else pd.Timestamp(forecast_origin).isoformat()
    )
    frozen_sha = "" if frozen_selection is None else str(frozen_selection.manifest_sha256)
    cache_key = (_context_source_key(ctx), normalized_origin, frozen_sha)
    cached = _AGENT_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    spec = _task_specs()[str(ctx.dataset_key)]
    cutoff = spec.t_sel if forecast_origin is None else forecast_origin
    role = "formal_selection" if forecast_origin is None else "agent_origin_selection"
    if uses_qwen25_multiscale_context(ctx):
        build_augmented_context = _qwen25_multiscale_context_api()
        payload, text, digest, validation = build_augmented_context(
            ctx.panel,
            _pooled_ledger().copy(),
            spec,
            cutoff_time=cutoff,
            context_role=role,
        )
        metadata = _qwen25_multiscale_context_metadata(
            payload, text, digest, validation
        )
        sequence_components = set(
            map(
                str,
                payload.get("causal_multiscale_sequence_sketch", {}).get(
                    "sequence_components", []
                ),
            )
        )
        required_streams = {"covid_adm_per100k", "flu_adm_per100k"}
        if sequence_components != required_streams:
            raise ValueError(
                "canonical Benchmark B Qwen context must expose both released "
                f"target streams; found={sorted(sequence_components)}"
            )
        base_payload = payload.get("base_context", {})
        if not isinstance(base_payload, Mapping):
            raise ValueError("canonical Benchmark B Qwen context has no base context")
        _QWEN25_CONTEXT_TEXT_BY_SHA[digest] = text
    else:
        payload, _text, validation = build_selection_context(
            ctx.panel,
            _pooled_ledger().copy(),
            spec,
            cutoff_time=cutoff,
            context_role=role,
        )
        if not validation.empty and not validation["status"].astype(str).eq("PASS").all():
            raise ValueError(
                f"canonical Benchmark B context validation failed task={ctx.dataset_key} cutoff={cutoff}"
            )
        digest = selection_context_sha256(payload)
        base_payload = payload
        metadata = {}
    binding = {
        "task_id": str(ctx.dataset_key),
        "forecast_origin": str(payload.get("forecast_origin", payload.get("cutoff_time", cutoff))),
        "canonical_context_sha256": digest,
        "visible_panel_sha256": str(base_payload.get("visible_panel_sha256", "")),
        "released_event_sha256": str(base_payload.get("released_event_sha256", "")),
        "frozen_selection_packet_sha256": frozen_sha,
    }
    if metadata:
        binding.update(metadata)
    binding["forecast_origin_binding_sha256"] = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = (payload, digest, binding)
    _AGENT_CONTEXT_CACHE[cache_key] = result
    return result


def prompt_payload(ctx: Any, *, forecast_origin: object | None, frozen_selection: Any | None = None) -> dict[str, object]:
    payload, digest, binding = build_agent_context(ctx, forecast_origin=forecast_origin, frozen_selection=frozen_selection)
    text = prompt_text(payload, digest)
    history_ends = [
        str(summary.get("history_end", ""))
        for summary in payload.get("history_summary", {}).values()
        if isinstance(summary, Mapping) and str(summary.get("history_end", ""))
    ]
    result = {
        "canonical_context": payload,
        "scientific_context": (
            scientific_context_view(payload)
            if str(payload.get("schema", "")) == QWEN25_MULTISCALE_CONTEXT_SCHEMA
            else payload
        ),
        **binding,
        "selection_context_schema": str(payload.get("schema", "")),
        "selection_context_sha256": digest,
        "selection_context_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "selection_context_cutoff": str(
            payload.get("forecast_origin", payload.get("cutoff_time", ""))
        ),
        "selection_context_role": str(payload.get("context_role", "agent_origin_selection")),
        "selection_context_history_scope": "all_released_no_later_than_cutoff",
        "selection_context_history_max": max(history_ends) if history_ends else "",
        "selection_context_builder": "caster.tasks.build_selection_context",
        "selection_context_validation_visible": False,
    }
    if uses_qwen25_multiscale_context(ctx):
        result.update(
            {
                key: value
                for key, value in binding.items()
                if key.startswith("selection_context_")
                or key.startswith("combined_context_")
                or key.startswith("base_selection_context_")
                or key.startswith("sequence_sketch_")
            }
        )
    return result


def llm_prompt_fields(context_payload: Mapping[str, object]) -> dict[str, object]:
    """Return only the scientific context intended for an LLM call."""
    scientific = context_payload.get("scientific_context", {})
    if not isinstance(scientific, Mapping):
        raise ValueError("Benchmark B prompt payload has no scientific context")
    return {"scientific_context": dict(scientific)}


def prompt_text(payload: Mapping[str, object], digest: str) -> str:
    if str(payload.get("schema", "")) == QWEN25_MULTISCALE_CONTEXT_SCHEMA:
        if digest not in _QWEN25_CONTEXT_TEXT_BY_SHA:
            raise ValueError(
                "Qwen multiscale Benchmark B prompt text is not bound to this "
                "process context cache"
            )
        return _QWEN25_CONTEXT_TEXT_BY_SHA[digest]
    return "canonical_benchmark_b_context=" + canonical_json(payload) + f"; canonical_context_sha256={digest}"


def annotate_forecast_context(forecast: pd.DataFrame, *, ctx: Any, frozen_selection: Any | None = None) -> pd.DataFrame:
    if not is_benchmark_b_context(ctx):
        return forecast
    out = forecast.copy()
    for origin, positions in out.groupby("forecast_origin", sort=False).groups.items():
        payload, digest, binding = build_agent_context(ctx, forecast_origin=origin, frozen_selection=frozen_selection)
        out.loc[positions, "canonical_context_schema"] = str(payload.get("schema", ""))
        for column in ("canonical_context_sha256", "visible_panel_sha256", "released_event_sha256", "frozen_selection_packet_sha256", "forecast_origin_binding_sha256"):
            out.loc[positions, column] = binding[column]
        if uses_qwen25_multiscale_context(ctx):
            for column in (
                "selection_context_profile",
                "combined_context_sha256",
                "combined_context_text_sha256",
                "sequence_sketch_schema",
                "sequence_sketch_sha256",
                "sequence_sketch_validation_status",
                "sequence_sketch_validation_sha256",
            ):
                out.loc[positions, column] = binding[column]
    return out


def context_manifest_fields(ctx: Any, frozen_selection: Any | None = None) -> dict[str, object]:
    if not is_benchmark_b_context(ctx):
        return {}
    payload, digest, binding = build_agent_context(ctx, forecast_origin=None, frozen_selection=frozen_selection)
    fields = {
        "benchmark_b_context_contract": str(CONTRACT_PATH),
        "benchmark_b_context_contract_sha256": (
            payload.get("base_context", payload)["context_contract_sha256"]
        ),
        "benchmark_b_router_context_sha256": digest,
        "benchmark_b_router_visible_panel_sha256": binding["visible_panel_sha256"],
        "benchmark_b_router_released_event_sha256": binding["released_event_sha256"],
        "benchmark_b_context_shared_by_caster_and_agents": True,
        "benchmark_b_context_payload_embedded_in_agent_prompts": True,
        "benchmark_b_old_artifact_reuse_allowed": False,
        "algorithm_learning_policy": "no_learning_behavior_preserved",
    }
    if uses_qwen25_multiscale_context(ctx):
        fields.update(
            {
                key: value
                for key, value in binding.items()
                if key.startswith("selection_context_")
                or key.startswith("combined_context_")
                or key.startswith("base_selection_context_")
                or key.startswith("sequence_sketch_")
            }
        )
    return fields
