#!/usr/bin/env python3
""







from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from formal_candidate_bank import (
    FORMAL_ARCHIVE_NAME,
    FORMAL_CACHE_TAG,
    FORMAL_CANDIDATE_COUNT,
    FORMAL_POOLED_MANIFEST_NAME,
    formal_candidate_model_ids,
)


ROOT = Path(__file__).resolve().parents[1]
NEW_METHOD_SRC = ROOT / "code/caster/src"
if str(NEW_METHOD_SRC) not in sys.path:
    sys.path.insert(0, str(NEW_METHOD_SRC))

from caster.models import apply_hyperparam_overrides
from caster.tasks import filter_rows_to_task_spec, load_task_spec


DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_SHARED = ROOT / "experiments/result_runs_v3_direct_rollout/shared_baseline"
TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
METHODS = ("agentic_top_one", "react", "agentic_full_recovery")
QWEN25_MULTISCALE_PROFILE = "qwen25_multiscale_released_sequence_v1"
QWEN25_MULTISCALE_CONTEXT_SCHEMA = (
    "caster_qwen25_multiscale_released_sequence_context_v1"
)
QWEN25_MULTISCALE_SKETCH_SCHEMA = (
    "caster_causal_multiscale_released_sequence_sketch_v1"
)
SUPPORTED_VERSIONED_RUN_PROFILES = (QWEN25_MULTISCALE_PROFILE,)
FORMAL_AGENT_POOL_POLICY_PATH = ROOT / "configs/caster_candidate_pool_policy_v21.yaml"
FORMAL_AGENT_POOL_POLICY = yaml.safe_load(
    FORMAL_AGENT_POOL_POLICY_PATH.read_text(encoding="utf-8")
)
FORMAL_AGENT_POOL_PROFILE = str(FORMAL_AGENT_POOL_POLICY.get("profile", ""))


def _task_policy_excluded_model_ids(task: str) -> tuple[str, ...]:
    task_policy = FORMAL_AGENT_POOL_POLICY.get("task_policies", {}).get(task, {})
    return tuple(
        str(row.get("model_id", "")).strip()
        for row in task_policy.get("excluded_from_caster_pool", [])
    )


for _task in TASKS:
    _task_policy = FORMAL_AGENT_POOL_POLICY.get("task_policies", {}).get(_task, {})
    _excluded = _task_policy_excluded_model_ids(_task)
    if (
        FORMAL_AGENT_POOL_PROFILE
        not in {
            "formal_27_country_macro_v1",
            "formal_27_country_macro_v1",
        }
        or FORMAL_AGENT_POOL_POLICY.get("formal_agent_pool_applies") is not True
        or int(_task_policy.get("candidate_bank_size", -1)) != FORMAL_CANDIDATE_COUNT
        or int(_task_policy.get("eligible_candidate_count", -1))
        != FORMAL_CANDIDATE_COUNT - len(_excluded)
    ):
        raise RuntimeError(
            f"invalid formal agent candidate-pool policy for task={_task}: "
            f"{FORMAL_AGENT_POOL_POLICY_PATH}"
        )


FORMAL_AGENT_EXCLUDED_MODEL_IDS = _task_policy_excluded_model_ids(TASKS[0])
FORMAL_AGENT_REQUIRED_MODEL_IDS = ("covariate_dynamic_linear_trend",)
FORMAL_AGENT_MODEL_IDS = tuple(
    model_id
    for model_id in formal_candidate_model_ids()
    if model_id not in FORMAL_AGENT_EXCLUDED_MODEL_IDS
)
FORMAL_AGENT_CANDIDATE_COUNT = len(FORMAL_AGENT_MODEL_IDS)
FORMAL_AGENT_REGISTRY_SUFFIX = f"agent_registry_all{FORMAL_AGENT_CANDIDATE_COUNT}.csv"
if FORMAL_AGENT_CANDIDATE_COUNT != (
    FORMAL_CANDIDATE_COUNT - len(FORMAL_AGENT_EXCLUDED_MODEL_IDS)
):
    raise RuntimeError(
        "formal agent policy count does not match its declared exclusions: "
        f"source={FORMAL_CANDIDATE_COUNT} eligible={FORMAL_AGENT_CANDIDATE_COUNT}"
    )
if not set(FORMAL_AGENT_REQUIRED_MODEL_IDS) <= set(FORMAL_AGENT_MODEL_IDS):
    raise RuntimeError(
        "formal agent policy must retain covariate_dynamic_linear_trend"
    )
_EXCLUDED_MODEL_TOKEN = (
    re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(
            re.escape(model_id) for model_id in FORMAL_AGENT_EXCLUDED_MODEL_IDS
        )
        + r")(?![A-Za-z0-9_])"
    )
    if FORMAL_AGENT_EXCLUDED_MODEL_IDS
    else re.compile(r"(?!x)x")
)
METHOD_SPEC = {
    "agentic_top_one": {
        "runner": ROOT / "code/baseline/scripts/run_agentic_top_one.py",
        "module": ROOT / "code/baseline/src/caster_baselines/agentic_top_one.py",
        "model": "agentic_top_one",
        "max_new_tokens": 256,
    },
    "react": {
        "runner": ROOT / "code/baseline/scripts/run_react_agent.py",
        "module": ROOT / "code/baseline/src/caster_baselines/agentic_react.py",
        "model": "agent_react",
        "max_new_tokens": 384,
    },
    "agentic_full_recovery": {
        "runner": ROOT / "code/baseline/scripts/run_agentic_full_recovery.py",
        "module": ROOT / "code/baseline/src/caster_baselines/agentic_full_recovery.py",
        "model": "agentic_full_recovery",
        "max_new_tokens": 512,
    },
}
COMMAND_LOG_LOCK = threading.Lock()
SHA256_CACHE_LOCK = threading.Lock()
SHA256_CACHE: dict[tuple[str, int, int], concurrent.futures.Future[str]] = {}


def sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    with SHA256_CACHE_LOCK:
        future = SHA256_CACHE.get(key)
        owner = future is None
        if owner:
            future = concurrent.futures.Future()
            SHA256_CACHE[key] = future
    assert future is not None
    if not owner:
        return future.result()
    try:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
    except BaseException as exc:
        future.set_exception(exc)
        with SHA256_CACHE_LOCK:
            SHA256_CACHE.pop(key, None)
        raise
    future.set_result(value)
    return value


def canonical_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def parse_csv_choice(value: str, allowed: tuple[str, ...]) -> list[str]:
    parts = list(allowed) if str(value).strip() == "all" else [x.strip() for x in str(value).split(",") if x.strip()]
    unknown = sorted(set(parts) - set(allowed))
    if unknown:
        raise SystemExit(f"unknown values {unknown}; allowed={list(allowed)}")
    return [x for x in allowed if x in parts]


def resolve_data_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def validate_run_profiles(
    run_profile: str,
    selection_context_profile: str,
) -> tuple[str, str]:
    ""

    run_profile = str(run_profile).strip()
    selection_context_profile = str(selection_context_profile).strip()
    if not run_profile and not selection_context_profile:
        return "", ""
    if run_profile not in SUPPORTED_VERSIONED_RUN_PROFILES:
        raise SystemExit(
            f"unsupported --run-profile={run_profile!r}; "
            f"allowed={list(SUPPORTED_VERSIONED_RUN_PROFILES)}"
        )
    if selection_context_profile != run_profile:
        raise SystemExit(
            "a versioned agent run requires --selection-context-profile to "
            f"exactly match --run-profile ({run_profile})"
        )
    return run_profile, selection_context_profile


def agent_input_dir(cache: Path, task: str, run_profile: str = "") -> Path:
    base = cache / "agent_inputs"
    return base / run_profile / task if run_profile else base / task


def agent_output_profile(run_profile: str = "") -> str:
    return run_profile if run_profile else "archive_backed"


def agent_log_root(shared: Path, run_profile: str = "") -> Path:
    return (
        shared / f"logs/formal_agents_{run_profile}"
        if run_profile
        else shared / "logs/formal_agents_b6"
    )


def stamp_selection_context_profile(manifest: Path, profile: str) -> None:
    ""

    if not profile:
        return
    frame = pd.read_csv(manifest, keep_default_na=False)
    if len(frame) != 1:
        raise RuntimeError(
            f"formal agent manifest must contain one row before profile binding: {manifest}"
        )
    frame["selection_context_profile"] = profile
    temporary = manifest.with_suffix(manifest.suffix + ".profile.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, manifest)


def agent_manifest_input_hashes(manifest: Path) -> dict[str, str]:
    frame = pd.read_csv(manifest, keep_default_na=False)
    if len(frame) != 1:
        raise RuntimeError(f"formal agent manifest must contain one row: {manifest}")
    row = frame.iloc[0]
    result: dict[str, str] = {}
    for field in ("panel_path", "ledger_path"):
        path = resolve_data_path(row[field])
        if not path.is_file():
            raise FileNotFoundError(path)
        result[f"{field}_sha256"] = sha256(path)
    return result


def archive_models(path: Path) -> set[str]:
    models: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["model_id"], chunksize=250_000):
        models.update(chunk["model_id"].dropna().astype(str))
    return models


def validate_materialized_agent_registry(
    registry_frame: pd.DataFrame,
    *,
    task: str,
) -> None:
    ""

    excluded_model_ids = _task_policy_excluded_model_ids(task)
    expected_agent_ids = tuple(
        model_id
        for model_id in formal_candidate_model_ids()
        if model_id not in excluded_model_ids
    )
    if "model_id" not in registry_frame.columns:
        raise RuntimeError("formal agent registry is missing model_id")
    registry_ids = registry_frame["model_id"].astype(str).tolist()
    if registry_ids != list(expected_agent_ids):
        raise RuntimeError(
            "formal agent registry violates the candidate-pool contract: "
            f"task={task} expected={list(expected_agent_ids)} actual={registry_ids}"
        )
    if len(registry_ids) != len(set(registry_ids)):
        raise RuntimeError("formal agent registry contains duplicate model_id values")
    if "enabled" not in registry_frame.columns:
        raise RuntimeError("formal agent registry is missing enabled")
    if not registry_frame["enabled"].map(truthy).all():
        raise RuntimeError("formal agent registry contains a disabled eligible model")
    if set(registry_ids) & set(excluded_model_ids):
        raise RuntimeError("a policy-excluded model leaked into the formal agent registry")
    if not set(FORMAL_AGENT_REQUIRED_MODEL_IDS) <= set(registry_ids):
        raise RuntimeError("covariate_dynamic_linear_trend is missing from the formal agent registry")


def build_agent_registry(
    source_registry: pd.DataFrame,
    archive_model_ids: set[str],
    *,
    task: str,
) -> pd.DataFrame:
    ""

    if "model_id" not in source_registry.columns:
        raise RuntimeError("formal source registry is missing model_id")
    source_ids = source_registry["model_id"].astype(str).tolist()
    expected_source_ids = list(formal_candidate_model_ids())
    if source_ids != expected_source_ids:
        raise RuntimeError(
            "formal source registry does not match the complete archive contract: "
            f"expected={expected_source_ids} actual={source_ids}"
        )
    if archive_model_ids != set(expected_source_ids):
        raise RuntimeError(
            "formal archive model set does not match the complete source registry: "
            f"missing={sorted(set(expected_source_ids) - archive_model_ids)} "
            f"extra={sorted(archive_model_ids - set(expected_source_ids))}"
        )
    excluded_model_ids = _task_policy_excluded_model_ids(task)
    projected = source_registry[
        ~source_registry["model_id"].astype(str).isin(excluded_model_ids)
    ].copy()
    validate_materialized_agent_registry(projected, task=task)
    return projected


def apply_task_manifest_hyperparam_overrides(
    source_registry: pd.DataFrame,
    archive_manifest: Mapping[str, Any],
    *,
    task: str,
) -> pd.DataFrame:
    ""







    if "model_hyperparam_overrides" not in archive_manifest:
        return source_registry
    overrides = archive_manifest["model_hyperparam_overrides"]
    if overrides is None or overrides == {}:
        return source_registry
    if not isinstance(overrides, Mapping):
        raise RuntimeError(
            "task archive model_hyperparam_overrides must be a model-to-parameters "
            f"mapping: task={task}"
        )
    try:
        return apply_hyperparam_overrides(source_registry, overrides)
    except ValueError as exc:
        raise RuntimeError(
            f"invalid task archive model_hyperparam_overrides task={task}: {exc}"
        ) from exc


def frame_mentions_excluded_model(frame: pd.DataFrame) -> bool:
    ""

    for column in frame.columns:
        if frame[column].astype(str).str.contains(_EXCLUDED_MODEL_TOKEN, regex=True).any():
            return True
    return False


def assert_archive_covers_formal_ledger(
    archive: Path,
    ledger: pd.DataFrame,
    model_ids: set[str],
) -> None:
    forecast_ids = set(ledger["forecast_id"].astype(str))
    coverage: dict[str, set[str]] = {forecast_id: set() for forecast_id in forecast_ids}
    for chunk in pd.read_csv(archive, usecols=["forecast_id", "model_id"], chunksize=250_000):
        hit = chunk[chunk["forecast_id"].astype(str).isin(forecast_ids)]
        for forecast_id, group in hit.groupby("forecast_id", sort=False):
            coverage[str(forecast_id)].update(group["model_id"].astype(str))
    bad = {
        forecast_id: sorted(model_ids - present)
        for forecast_id, present in coverage.items()
        if not model_ids <= present
    }
    if bad:
        sample = list(bad.items())[:5]
        raise RuntimeError(
            f"formal archive does not cover the projected task ledger: "
            f"bad_forecast_ids={len(bad)} sample={sample}"
        )


def prepare_task_inputs(
    *,
    python: Path,
    shared: Path,
    candidate_cache_root: Path | None = None,
    task: str,
    dry_run: bool,
    run_profile: str = "",
    selection_context_profile: str = "",
) -> tuple[Path, Path, Path]:
    cache = (
        candidate_cache_root.resolve()
        if candidate_cache_root is not None
        else (shared / "caster_candidates" / FORMAL_CACHE_TAG).resolve()
    )
    archive = cache / task / FORMAL_ARCHIVE_NAME
    registry_source = cache / "model_registry.formal.csv"
    baseline_manifest = shared / "baseline/data/full_manifest.csv"
    source_ledger = (
        ROOT / "data/benchmark_a/curated_full_v3_direct_rollout7/event_ledger.csv"
        if task == "benchmark_a"
        else ROOT / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/event_ledger.csv"
    )
    panel = (
        ROOT / "data/benchmark_a/curated_full_v3_direct_rollout7/daily_panel.csv"
        if task == "benchmark_a"
        else ROOT / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/weekly_panel.csv"
    )
    for required in [archive, registry_source, baseline_manifest, source_ledger, panel]:
        if not required.exists():
            raise FileNotFoundError(required)
    input_dir = agent_input_dir(cache, task, run_profile)
    manifest = input_dir / f"{task}_manifest.csv"
    registry = input_dir / f"{task}_{FORMAL_AGENT_REGISTRY_SUFFIX}"
    ledger = input_dir / "event_ledger.csv"
    if dry_run:
        return manifest, registry, archive
    input_dir.mkdir(parents=True, exist_ok=True)
    source_ledger_frame = pd.read_csv(source_ledger, keep_default_na=False, low_memory=False)
    archive_ledger_frame = source_ledger_frame.copy()
    component = {
        "benchmark_b_covid": "covid_adm_per100k",
        "benchmark_b_flu": "flu_adm_per100k",
    }.get(task)
    if component is not None:
        archive_ledger_frame = archive_ledger_frame[
            archive_ledger_frame["component"].astype(str).eq(component)
        ].copy()
    spec = load_task_spec(ROOT / "configs/caster_task_specs_v20.yaml", task)
    ledger_frame = filter_rows_to_task_spec(
        archive_ledger_frame,
        spec,
        require_complete=True,
    )
    if ledger_frame.empty or not ledger_frame["split"].astype(str).eq("embargo").any():
        raise RuntimeError(f"formal agent ledger lacks embargo origins task={task}")
    archive_manifest_path = archive.with_name(
        FORMAL_POOLED_MANIFEST_NAME
        if task == "benchmark_b_pooled"
        else "forecast_archive_manifest.json"
    )
    if not archive_manifest_path.is_file():
        raise FileNotFoundError(archive_manifest_path)
    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    expected_archive_embargo = int(archive_ledger_frame["split"].astype(str).eq("embargo").sum())
    if (
        str(archive_manifest.get("ledger_sha256", "")) != sha256(source_ledger)
        or int(archive_manifest.get("ledger_rows", -1)) != len(archive_ledger_frame)
        or int(archive_manifest.get("embargo_rows", -1)) != expected_archive_embargo
        or int(archive_manifest.get("archive_rows", -1))
        != len(archive_ledger_frame) * FORMAL_CANDIDATE_COUNT
        or int(archive_manifest.get("models", -1)) != FORMAL_CANDIDATE_COUNT
    ):
        raise RuntimeError(f"formal agent archive is stale or lacks embargo coverage task={task}")
    ledger_frame.to_csv(ledger, index=False)
    prepare = ROOT / "code/caster/scripts/prepare_archive_backed_agent_inputs.py"
    subprocess.run(
        [
            str(python),
            str(prepare),
            "--task",
            task,
            "--baseline-manifest",
            str(baseline_manifest),
            "--task-ledger",
            str(ledger),
            "--panel",
            str(panel),
            "--out-manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=True,
    )
    stamp_selection_context_profile(manifest, selection_context_profile)
    registry_frame = pd.read_csv(registry_source, keep_default_na=False)
    registry_frame = apply_task_manifest_hyperparam_overrides(
        registry_frame,
        archive_manifest,
        task=task,
    )
    models = archive_models(archive)
    agent_registry = build_agent_registry(registry_frame, models, task=task)
                                                                            
                                                                          
    assert_archive_covers_formal_ledger(
        archive,
        ledger_frame,
        set(agent_registry["model_id"].astype(str)),
    )
    registry_tmp = registry.with_suffix(registry.suffix + ".tmp")
    agent_registry.to_csv(registry_tmp, index=False)
    os.replace(registry_tmp, registry)
    return manifest, registry, archive


def expected_identity(
    task: str,
    method: str,
    manifest: Path,
    registry: Path,
    archive: Path,
    *,
    candidate_cache_root: Path | None = None,
    run_profile: str = "",
    selection_context_profile: str = "",
) -> dict[str, Any]:
    spec = METHOD_SPEC[method]
    cache_root = (
        candidate_cache_root.resolve()
        if candidate_cache_root is not None
        else archive.resolve().parent.parent
    )
    cache_manifest = cache_root / "manifest.json"
    payload: dict[str, Any] = {
        "schema": "formal_agent_identity_v3_formal_direct_pool",
        "task": task,
        "method": method,
        "model": spec["model"],
        "manifest_sha256": sha256(manifest),
        "registry_sha256": sha256(registry),
        "forecast_archive_sha256": sha256(archive),
        "context_builder_sha256": sha256(
            ROOT / "code/baseline/src/caster_baselines/agentic_skills.py"
        ),
        "canonical_context_sha256": sha256(
            ROOT / "code/caster/src/caster/tasks/context.py"
        ),
        "task_specs_sha256": sha256(ROOT / "configs/caster_task_specs_v20.yaml"),
        "method_module_sha256": sha256(Path(spec["module"])),
        "runner_sha256": sha256(Path(spec["runner"])),
        "llm_engine_sha256": sha256(
            ROOT
            / "code/baseline/src/caster_baselines/agentic_llm.py"
        ),
        "orchestrator_sha256": sha256(Path(__file__)),
        "max_new_tokens": int(spec["max_new_tokens"]),
        "selection_policy": "llm_only",
        "timing_mode": "archive_backed",
        "llm_load_charged": False,
        "selection_charged": True,
        "validation_score_used": False,
        "source_archive_candidate_count": FORMAL_CANDIDATE_COUNT,
        "candidate_count": FORMAL_AGENT_CANDIDATE_COUNT,
        "eligible_candidate_count": FORMAL_AGENT_CANDIDATE_COUNT,
        "candidate_pool_policy_profile": FORMAL_AGENT_POOL_PROFILE,
        "candidate_pool_policy_sha256": sha256(FORMAL_AGENT_POOL_POLICY_PATH),
        "excluded_model_ids": list(_task_policy_excluded_model_ids(task)),
        "eligible_model_ids": list(FORMAL_AGENT_MODEL_IDS),
        "embargo_forecast_required": True,
        "embargo_metric_eligible": False,
        **agent_manifest_input_hashes(manifest),
    }
    if run_profile:
        payload.update(
            {
                "run_profile": run_profile,
                "selection_context_profile": selection_context_profile,
            }
        )
        optional_context_implementations = {
            "sequence_sketch_implementation_sha256": (
                ROOT
                / "code/caster/src/caster/tasks/sequence_sketch.py"
            ),
            "qwen25_embedding_implementation_sha256": (
                ROOT
                / "code/caster/src/caster/models/qwen25_embedding.py"
            ),
        }
        for field, implementation in optional_context_implementations.items():
            if implementation.is_file():
                payload[field] = sha256(implementation)
        qwen_context_sources = [
            ROOT
            / "code/caster/src/caster/tasks/context.py",
            ROOT
            / "code/caster/src/caster/tasks/qwen_context.py",
            ROOT
            / "code/caster/src/caster/tasks/sequence_sketch.py",
            ROOT
            / "code/baseline/src/caster_baselines/agentic_skills.py",
            ROOT
            / "code/baseline/src/caster_baselines/benchmark_b_context.py",
        ]
        qwen_context_hashes = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in qwen_context_sources
            if path.is_file()
        }
        if qwen_context_hashes:
            payload["qwen_context_implementation_hashes"] = qwen_context_hashes
            payload["qwen_context_implementation_sha256"] = canonical_sha(
                qwen_context_hashes
            )
    if task.startswith("benchmark_b"):
        payload.update(
            {
                "benchmark_b_context_implementation_sha256": sha256(
                    ROOT
                    / "code/caster/src/caster/data/benchmark_b_context.py"
                ),
                "benchmark_b_context_contract_sha256": sha256(
                    ROOT / "configs/benchmark_b_context_v26_1.yaml"
                ),
                "benchmark_b_agent_context_wrapper_sha256": sha256(
                    ROOT
                    / "code/baseline/src/caster_baselines/benchmark_b_context.py"
                ),
            }
        )
    if task == "benchmark_a":
        benchmark_a_manifest = (
            ROOT / "data/benchmark_a/curated_full_v3_direct_rollout7/run_manifest.json"
        )
        payload.update(
            {
                "benchmark_a_mobility_implementation_sha256": sha256(
                    ROOT
                    / "code/caster/src/caster/data/benchmark_a_mobility.py"
                ),
                "benchmark_a_input_manifest_sha256": sha256(benchmark_a_manifest),
                "benchmark_a_graph_set_sha256": str(
                    json.loads(benchmark_a_manifest.read_text(encoding="utf-8")).get(
                        "authority_graph_set_sha256", ""
                    )
                ),
            }
        )
                                                                              
                                                                              
                                                                         
    payload["identity_sha256"] = canonical_sha(payload)
    payload["candidate_cache_root"] = str(cache_root)
    payload["candidate_cache_manifest_sha256"] = (
        sha256(cache_manifest) if cache_manifest.is_file() else ""
    )
    return payload


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def validate_selection_context_contract(
    selection: pd.DataFrame,
    *,
    task: str,
    selection_context_profile: str = "",
) -> tuple[bool, str]:
    context_cols = {
        "selection_context_schema",
        "selection_context_sha256",
        "selection_context_cutoff",
        "selection_context_role",
        "selection_context_history_max",
        "selection_context_builder",
        "selection_context_validation_visible",
    }
    if selection.empty or not context_cols <= set(selection.columns):
        return False, "missing_context_identity"
    if selection_context_profile:
        sketch_cols = {"sequence_sketch_schema", "sequence_sketch_sha256"}
        if not sketch_cols <= set(selection.columns):
            return False, "missing_sequence_sketch_identity"
        if not selection["selection_context_schema"].astype(str).eq(
            QWEN25_MULTISCALE_CONTEXT_SCHEMA
        ).all():
            return False, "noncanonical_context"
        if not selection["sequence_sketch_schema"].astype(str).eq(
            QWEN25_MULTISCALE_SKETCH_SCHEMA
        ).all():
            return False, "noncanonical_sequence_sketch"
        if selection["sequence_sketch_sha256"].astype(str).str.len().ne(64).any():
            return False, "invalid_sequence_sketch_sha"
    else:
        expected_schema = (
            "caster_benchmark_b_canonical_context_v1"
            if task.startswith("benchmark_b")
            else "caster_selection_context_full_history_v2"
        )
        if not selection["selection_context_schema"].astype(str).eq(
            expected_schema
        ).all():
            return False, "noncanonical_context"
    if selection["selection_context_sha256"].astype(str).str.len().ne(64).any():
        return False, "invalid_context_sha"
    if selection["selection_context_validation_visible"].map(truthy).any():
        return False, "selection_context_validation_visible"
    cutoff = pd.to_datetime(selection["selection_context_cutoff"], errors="coerce")
    history_max = pd.to_datetime(
        selection["selection_context_history_max"], errors="coerce"
    )
    if (history_max.notna() & cutoff.notna() & history_max.gt(cutoff)).any():
        return False, "causal_cutoff_violation"
    return True, "ok"


def output_is_current(
    out: Path,
    identity: dict[str, Any],
    manifest_path: Path,
    registry: Path,
    archive: Path,
) -> tuple[bool, str]:
    required = [
        out / "forecast.csv",
        out / "metrics.csv",
        out / "candidate_selection_log.csv",
        out / "candidate_registry_snapshot.csv",
        out / "timing.json",
        out / "run_manifest.json",
        out / "qwen_config.json",
        out / "formal_agent_identity.json",
    ]
    missing = [p.name for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        return False, f"missing={missing}"
    try:
        stored = json.loads((out / "formal_agent_identity.json").read_text(encoding="utf-8"))
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
        registry_frame = pd.read_csv(registry, keep_default_na=False)
        registry_snapshot = pd.read_csv(
            out / "candidate_registry_snapshot.csv", keep_default_na=False
        )
        selection = pd.read_csv(out / "candidate_selection_log.csv", keep_default_na=False)
        forecast = pd.read_csv(out / "forecast.csv", keep_default_na=False, low_memory=False)
        metrics = pd.read_csv(out / "metrics.csv", keep_default_na=False)
    except Exception as exc:
        return False, f"unreadable={exc}"
    if stored != identity:
        return False, "identity_mismatch"
    if str(manifest.get("model", "")) != str(identity["model"]):
        return False, "model_mismatch"
    if str(manifest.get("agent_selection_scope", "")) != str(identity["task"]):
        return False, "scope_mismatch"
    if str(manifest.get("registry_sha256", "")) != sha256(registry):
        return False, "registry_mismatch"
    try:
        validate_materialized_agent_registry(registry_frame, task=str(identity["task"]))
        validate_materialized_agent_registry(registry_snapshot, task=str(identity["task"]))
    except RuntimeError as exc:
        return False, f"agent_registry_contract={exc}"
    if sha256(out / "candidate_registry_snapshot.csv") != sha256(registry):
        return False, "registry_snapshot_mismatch"
    try:
        manifest_archive = Path(str(manifest.get("forecast_archive", ""))).resolve()
    except Exception:
        return False, "archive_path_invalid"
    if manifest_archive != archive.resolve():
        return False, "archive_path_mismatch"
    if int(manifest.get("enabled_candidates", -1)) != FORMAL_AGENT_CANDIDATE_COUNT:
        return False, "enabled_candidate_count_mismatch"
    if int(manifest.get("restart_eligible_candidates", -1)) != FORMAL_AGENT_CANDIDATE_COUNT:
        return False, "candidate_count_mismatch"
    if truthy(manifest.get("validation_score_used", True)) or truthy(manifest.get("validation_context_visible", True)):
        return False, "validation_leakage"
    if str(timing.get("timing_mode", "")) != "archive_backed":
        return False, "timing_mode_mismatch"
    if str(timing.get("forecast_source", "")) != "immutable_forecast_archive":
        return False, "forecast_source_mismatch"
    if truthy(timing.get("selection_replay_used", False)):
        return False, "selection_replay_used"
    if "split" not in forecast.columns or "forecast_id" not in forecast.columns:
        return False, "forecast_missing_split_or_id"
    manifest_frame = pd.read_csv(manifest_path, keep_default_na=False)
    if len(manifest_frame) != 1:
        return False, "agent_manifest_row_count"
    selection_context_profile = str(
        identity.get("selection_context_profile", "")
    ).strip()
    if selection_context_profile:
        if "selection_context_profile" not in manifest_frame.columns:
            return False, "agent_manifest_missing_context_profile"
        if not manifest_frame["selection_context_profile"].astype(str).eq(
            selection_context_profile
        ).all():
            return False, "agent_manifest_context_profile_mismatch"
    ledger_path = Path(str(manifest_frame.iloc[0]["ledger_path"]))
    if not ledger_path.is_absolute():
        ledger_path = ROOT / ledger_path
    ledger = pd.read_csv(ledger_path, usecols=["forecast_id", "split"], keep_default_na=False)
    if len(forecast) != len(ledger):
        return False, "forecast_ledger_row_count_mismatch"
    if set(forecast["forecast_id"].astype(str)) != set(ledger["forecast_id"].astype(str)):
        return False, "forecast_ledger_id_mismatch"
    expected_embargo = int(ledger["split"].astype(str).eq("embargo").sum())
    actual_embargo = int(forecast["split"].astype(str).eq("embargo").sum())
    if expected_embargo <= 0 or actual_embargo != expected_embargo:
        return False, "embargo_forecast_coverage_mismatch"
    if "split" in metrics.columns and metrics["split"].astype(str).eq("embargo").any():
        return False, "embargo_in_metrics"
    if int(manifest.get("embargo_forecast_rows", -1)) != expected_embargo:
        return False, "embargo_manifest_count_mismatch"
    if int(manifest.get("embargo_metric_rows", -1)) != 0:
        return False, "embargo_metric_manifest_nonzero"
    valid_context, context_reason = validate_selection_context_contract(
        selection,
        task=str(identity["task"]),
        selection_context_profile=selection_context_profile,
    )
    if not valid_context:
        return False, context_reason
    if frame_mentions_excluded_model(selection):
        return False, "excluded_model_leakage"
    if "n_candidates" not in selection.columns:
        return False, "missing_selection_candidate_count"
    selection_counts = pd.to_numeric(selection["n_candidates"], errors="coerce")
    if selection_counts.isna().any() or not selection_counts.eq(
        FORMAL_AGENT_CANDIDATE_COUNT
    ).all():
        return False, "selection_candidate_count_mismatch"
    return True, "ok"


def command_for(
    *, python: Path, method: str, task: str, manifest: Path, registry: Path, archive: Path, out: Path
) -> list[str]:
    spec = METHOD_SPEC[method]
    cmd = [
        str(python),
        str(spec["runner"]),
        "--manifest",
        str(manifest.resolve()),
        "--registry",
        str(registry.resolve()),
        "--out",
        str(out.resolve()),
        "--runtime-budget-seconds",
        "7200",
        "--max-new-tokens",
        str(spec["max_new_tokens"]),
        "--forecast-archive",
        str(archive.resolve()),
        "--archive-mode",
        "required",
        "--charge-selection",
        "--timing-mode",
        "archive_backed",
        "--exclude-llm-load-from-timing",
        "--dataset-key",
        task,
    ]
    if method in {"react", "agentic_full_recovery"}:
        cmd.extend(["--selection-policy", "llm_only"])
    return cmd


def run_method(
    *,
    python: Path,
    shared: Path,
    candidate_cache_root: Path,
    task: str,
    method: str,
    gpu: str,
    manifest: Path,
    registry: Path,
    archive: Path,
    log_root: Path,
    command_log: Path,
    dry_run: bool,
    run_profile: str = "",
    selection_context_profile: str = "",
) -> dict[str, Any]:
    output_profile = agent_output_profile(run_profile)
    out = shared / "baseline/runs/agentic" / output_profile / task / method
    cmd = command_for(
        python=python,
        method=method,
        task=task,
        manifest=manifest,
        registry=registry,
        archive=archive,
        out=out,
    )
    if dry_run:
        return {
            "task": task,
            "method": method,
            "gpu": gpu,
            "status": "dry_run",
            "command": cmd,
            "candidate_cache_root": str(candidate_cache_root.resolve()),
            "forecast_archive_sha256": sha256(archive),
        }
    identity = expected_identity(
        task,
        method,
        manifest,
        registry,
        archive,
        candidate_cache_root=candidate_cache_root,
        run_profile=run_profile,
        selection_context_profile=selection_context_profile,
    )
    current, reason = output_is_current(out, identity, manifest, registry, archive)
    if current:
        return {
            "task": task,
            "method": method,
            "gpu": gpu,
            "status": "reused",
            "reason": reason,
            "identity_sha256": identity["identity_sha256"],
            "candidate_cache_root": identity["candidate_cache_root"],
            "forecast_archive_sha256": identity["forecast_archive_sha256"],
        }
    if out.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = (
            shared
            / "baseline/runs/agentic"
            / output_profile
            / "_invalidated"
            / stamp
            / task
            / method
        )
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(archived))
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_root / task / f"{method}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "method": method,
        "gpu": gpu,
        "identity_sha256": identity["identity_sha256"],
        "command": cmd,
    }
    with COMMAND_LOG_LOCK:
        with command_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(command_record, sort_keys=True) + "\n")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"agent failed task={task} method={method} exit={completed.returncode}; log={log_path}")
    (out / "formal_agent_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    valid, validation_reason = output_is_current(out, identity, manifest, registry, archive)
    if not valid:
        raise RuntimeError(
            f"post-run validation failed task={task} method={method}: {validation_reason}; log={log_path}"
        )
    return {
        "task": task,
        "method": method,
        "gpu": gpu,
        "status": "executed",
        "reason": reason,
        "identity_sha256": identity["identity_sha256"],
        "candidate_cache_root": identity["candidate_cache_root"],
        "forecast_archive_sha256": identity["forecast_archive_sha256"],
    }


def update_shared_agent_manifest(*, python: Path, shared: Path, log_root: Path) -> Path:
    ""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = log_root / f"manifest_refresh_{stamp}"
    stage.mkdir(parents=True, exist_ok=False)
    aggregate = ROOT / "code/baseline/scripts/aggregate_baseline_results.py"
    manifest = stage / "baseline_run_manifest.csv"
    cmd = [
        str(python),
        str(aggregate),
        "--run-root",
        str(shared / "baseline/runs"),
        "--out-metrics",
        str(stage / "baseline_metrics.diagnostic_only.csv"),
        "--out-metric-slices",
        str(stage / "baseline_metric_slices.diagnostic_only.csv"),
        "--out-manifest",
        str(manifest),
        "--report",
        str(stage / "baseline_summary.diagnostic_only.md"),
    ]
    log_path = stage / "aggregate.log"
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not manifest.exists():
        raise RuntimeError(f"shared agent manifest aggregation failed exit={completed.returncode}; log={log_path}")
    frame = pd.read_csv(manifest, keep_default_na=False)
    required = {(method, task) for method in ("agentic_top_one", "agent_react", "agentic_full_recovery") for task in TASKS}
    seen = {
        (str(row.get("method", "")), str(row.get("dataset_key", "")))
        for _, row in frame.iterrows()
        if str(row.get("restart_type", "")).startswith("archive_backed_")
    }
    missing = sorted(required - seen)
    if missing:
        raise RuntimeError(f"refreshed shared manifest is missing formal agent rows: {missing}")
    target = shared / "baseline/results/baseline_run_manifest.csv"
    backup = shared / "baseline/results/manifest_archives" / f"baseline_run_manifest.{stamp}.csv"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.copy2(target, backup)
    os.replace(manifest, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", default=str(DEFAULT_SHARED))
    parser.add_argument(
        "--candidate-cache-root",
        default=None,
        help=(
            "Explicit complete-registry candidate-cache root. Defaults to "
            "<shared>/caster_candidates/" + FORMAL_CACHE_TAG + "."
        ),
    )
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--run-profile",
        default="",
        help=(
            "Opt into an isolated versioned agent-input/output namespace. "
            "Empty preserves the alternate archive_backed layout."
        ),
    )
    parser.add_argument(
        "--selection-context-profile",
        default="",
        help=(
            "Context protocol written into each versioned task manifest. "
            "Must exactly match a nonempty --run-profile."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--update-shared-manifest",
        dest="update_shared_manifest",
        action="store_true",
        help="Rebuild the shared run manifest after all agent outputs validate.",
    )
    parser.add_argument(
        "--no-update-shared-manifest",
        dest="update_shared_manifest",
        action="store_false",
        help="Do not rebuild the shared run manifest after all agent outputs validate.",
    )
    parser.set_defaults(update_shared_manifest=True)
    args = parser.parse_args(argv)

    run_profile, selection_context_profile = validate_run_profiles(
        args.run_profile,
        args.selection_context_profile,
    )
    shared = Path(args.shared).resolve()
    candidate_cache_root = (
        Path(args.candidate_cache_root).resolve()
        if args.candidate_cache_root is not None
        else (shared / "caster_candidates" / FORMAL_CACHE_TAG).resolve()
    )
    python = Path(args.python).resolve()
    tasks = parse_csv_choice(args.tasks, TASKS)
    methods = parse_csv_choice(args.methods, METHODS)
    gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip()]
    if not gpus:
        raise SystemExit("at least one GPU is required")
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f"GPU IDs must be unique; got {gpus}")
    if run_profile and args.update_shared_manifest:
        raise SystemExit(
            "versioned agent profiles cannot publish the shared manifest; "
            "pass --no-update-shared-manifest"
        )
    if (
        not args.dry_run
        and args.update_shared_manifest
        and (tasks != list(TASKS) or methods != list(METHODS))
    ):
        raise SystemExit(
            "partial formal-agent runs cannot publish the shared manifest; "
            "select all tasks and methods or pass --no-update-shared-manifest"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                                                            
                                                                          
                                                                     
    log_root = agent_log_root(shared, run_profile)
    if not args.dry_run:
        log_root.mkdir(parents=True, exist_ok=True)
    command_log = log_root / "commands.jsonl"

    prepared: dict[str, tuple[Path, Path, Path]] = {}
    prepare_futures: dict[
        concurrent.futures.Future[tuple[Path, Path, Path]], str
    ] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for task in tasks:
            future = pool.submit(
                prepare_task_inputs,
                python=python,
                shared=shared,
                candidate_cache_root=candidate_cache_root,
                task=task,
                dry_run=bool(args.dry_run),
                run_profile=run_profile,
                selection_context_profile=selection_context_profile,
            )
            prepare_futures[future] = task
        for future in concurrent.futures.as_completed(prepare_futures):
            prepared[prepare_futures[future]] = future.result()

    gpu_pool: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)

    def scheduled_method(task: str, method: str) -> dict[str, Any]:
        gpu = gpu_pool.get()
        try:
            manifest, registry, archive = prepared[task]
            return run_method(
                python=python,
                shared=shared,
                candidate_cache_root=candidate_cache_root,
                task=task,
                method=method,
                gpu=gpu,
                manifest=manifest,
                registry=registry,
                archive=archive,
                log_root=log_root,
                command_log=command_log,
                dry_run=bool(args.dry_run),
                run_profile=run_profile,
                selection_context_profile=selection_context_profile,
            )
        finally:
            gpu_pool.put(gpu)

    futures: dict[concurrent.futures.Future[dict[str, Any]], tuple[str, str]] = {}
    records: list[dict[str, Any]] = []
    jobs = [(task, method) for method in methods for task in tasks]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(gpus), len(jobs))) as pool:
        for task, method in jobs:
            future = pool.submit(scheduled_method, task, method)
            futures[future] = (task, method)
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda row: (TASKS.index(str(row["task"])), METHODS.index(str(row["method"]))))
    refreshed_manifest = ""
    if not args.dry_run and args.update_shared_manifest:
        refreshed_manifest = str(update_shared_agent_manifest(python=python, shared=shared, log_root=log_root))
    summary = {
        "schema": (
            "formal_agents_all_tasks_run_v4_versioned_context"
            if run_profile
            else "formal_agents_all_tasks_run_v3_gpu_job_pool"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shared": str(shared),
        "candidate_cache_root": str(candidate_cache_root),
        "candidate_cache_manifest_sha256": (
            sha256(candidate_cache_root / "manifest.json")
            if (candidate_cache_root / "manifest.json").is_file()
            else ""
        ),
        "candidate_archive_sha256_by_task": {
            task: sha256(prepared[task][2]) for task in tasks
        },
        "source_archive_candidate_count": FORMAL_CANDIDATE_COUNT,
        "agent_candidate_count": FORMAL_AGENT_CANDIDATE_COUNT,
        "agent_excluded_model_ids": list(FORMAL_AGENT_EXCLUDED_MODEL_IDS),
        "agent_eligible_model_ids": list(FORMAL_AGENT_MODEL_IDS),
        "candidate_pool_policy_profile": FORMAL_AGENT_POOL_PROFILE,
        "candidate_pool_policy_sha256": sha256(FORMAL_AGENT_POOL_POLICY_PATH),
        "tasks": tasks,
        "methods": methods,
        "gpus": gpus,
        "parallel_job_limit": min(len(gpus), len(jobs)),
        "input_preparation_count": len(prepared),
        "dry_run": bool(args.dry_run),
        "records": records,
        "shared_agent_manifest": refreshed_manifest,
    }
    if run_profile:
        summary.update(
            {
                "run_profile": run_profile,
                "selection_context_profile": selection_context_profile,
                "agent_input_root": str(
                    candidate_cache_root / "agent_inputs" / run_profile
                ),
                "agent_output_root": str(
                    shared / "baseline/runs/agentic" / run_profile
                ),
                "agent_log_root": str(log_root),
            }
        )
    summary["manifest_sha256"] = canonical_sha(summary)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        out = log_root / f"run_manifest_{stamp}.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (log_root / "latest_run_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"formal agents ready: manifest={out}")
        for row in records:
            print(f"{row['task']} {row['method']} {row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
