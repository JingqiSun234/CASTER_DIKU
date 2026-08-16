#!/usr/bin/env python3
""














from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from formal_candidate_bank import (
    FORMAL_RESULT_CANDIDATE_PROFILE,
    FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT,
    FORMAL_RESULT_EXCLUDED_MODEL_IDS,
)


ROOT = Path(__file__).resolve().parents[1]
NM_CODE_ROOT = ROOT / "code/caster"
BL_CODE_ROOT = ROOT / "code/baseline"
BL_PYTHON = BL_CODE_ROOT / ".venv/bin/python"
REQUIRED_QWEN_7B_PATH = os.environ.get(
    "CASTER_QWEN_CHECKPOINT", "QWEN_CHECKPOINT_NOT_CONFIGURED"
)
PREWARMED_TIMING_REGIME = "prewarmed_online_update_readout"
CELL_MANIFEST_SCHEMA = "runtime_prewarmed_cell_v1"
FULL_RECOVERY_TIMED_ARTIFACTS = (
    "forecast_readout.csv",
    "archive_forecast_readout_validation.csv",
)
CELL_LOCK_SCHEMA = "runtime_cell_lock_v1"
                                                                           
                                                            
PRE_CELL_LOCK_SWEEP_SOURCE_SHA256 = (
    "e966cfb578fefb519ff022d0eda187c63ecedc2c17f88ea3290e6b5b28c021a8"
)
alternate_PREDICTIVE_CONTRACT = "alternate_archive_moment"
                                                                          
                                                                           
FORMAL_CANDIDATE_COUNT = FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT

DEFAULT_METHODS = ["agentic_top_one", "agent_react", "agentic_full_recovery", "caster_one_layer", "caster_hierarchical"]
REQUIRED_METHODS = set(DEFAULT_METHODS)
AGENT_METHODS = frozenset(
    {"agentic_top_one", "agent_react", "agentic_full_recovery"}
)
DEFAULT_RUNTIME_K_GRID = [
    1,
    2,
    3,
    5,
    8,
    10,
    13,
    16,
    20,
    FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT,
]
RUNTIME_SINGLE_TASKS = {"benchmark_b_covid", "benchmark_b_flu"}
RUNTIME_TASK_LABELS = {
    "benchmark_b_covid": "Benchmark B-COVID",
    "benchmark_b_flu": "Benchmark B-FLU",
}
REQUIRED_COLUMNS = [
    "method",
    "candidate_count",
    "repeat",
    "seed",
    "run_mode",
    "status",
    "total_sec",
    "algorithm_update_sec",
    "update_sec",
    "proposal_sec",
    "filter_sec",
    "forecast_sec",
    "planner_stage_sec",
    "planner_qwen_sec",
    "planner_non_qwen_sec",
    "selector_stage_sec",
    "selector_qwen_sec",
    "selector_non_qwen_sec",
    "critic_trace_sec",
    "archive_readout_compute_sec",
    "archive_coverage_validation_sec",
    "readout_enrichment_sec",
    "readout_materialization_sec",
    "readout_validation_sec",
    "readout_artifact_write_sec",
    "forecast_readout_sec",
    "runtime_update_sec",
    "planner_stage_timing_schema",
    "selector_stage_timing_schema",
    "readout_timing_schema",
    "hardware",
    "run_id",
    "started_at",
    "finished_at",
    "log_path",
    "artifact_reuse_proxy",
    "uses_legacy_timing",
    "selection_path",
    "selection_hash",
    "selected_model_ids",
    "same_selection_per_k",
    "timing_mode",
    "timing_semantics",
    "formal_timing_valid",
    "model_compute_sec",
    "forecast_source",
    "selection_charged_to_update",
    "selection_replay_used",
    "selection_engine",
    "llm_model_path",
    "llm_required_model_path",
    "llm_primary_required",
    "llm_cuda_required",
    "llm_cuda_available",
    "llm_model_device",
    "llm_fallback_used",
    "llm_fallback_allowed",
    "restart_type",
    "timing_task",
    "timing_task_label",
    "single_task_timing",
    "pooled_timing_used",
    "agent_selection_scope",
    "timing_regime",
    "process_startup_sec_included",
    "llm_load_sec_included",
    "process_total_sec",
    "persistent_agent_worker",
    "resumed_from_valid_cell",
    "resumed_from_external_agent_cache",
    "external_agent_cache_root",
    "external_cell_manifest_path",
    "external_cell_manifest_sha256",
    "external_timing_path",
    "external_timing_sha256",
]
HASH_GUARD_RELS = [
    "new_method/artifacts/benchmark_b/forecast_archive.csv",
    "new_method/artifacts/benchmark_b/bridge_config.json",
    "new_method/artifacts/benchmark_b/posterior_path.csv",
    "new_method/artifacts/benchmark_b/evidence_log.csv",
    "new_method/artifacts/benchmark_b/posterior_weights.csv",
    "new_method/artifacts/benchmark_b/hierarchical_posterior_path.csv",
    "new_method/artifacts/benchmark_b/hierarchical_evidence_log.csv",
    "new_method/artifacts/benchmark_b/hierarchical_posterior_weights.csv",
    "new_method/artifacts/benchmark_b/family_posterior.csv",
    "new_method/artifacts/benchmark_b/model_registry.csv",
    "new_method/artifacts/candidate_selection_all_enabled.csv",
    "new_method/artifacts/model_registry.formal.csv",
    "new_method/results/real_full_result_inputs/caster_metrics.csv",
    "new_method/results/real_full_result_inputs/runtime_metrics.csv",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def is_agent_method(value: object) -> bool:
    return str(value) in AGENT_METHODS


def falsy(value: object) -> bool:
    return str(value).strip().lower() in {"", "0", "false", "f", "no", "n", "nan"}


def device_is_cuda(value: object) -> bool:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return False
    if any(token in text for token in ("cpu", "disk", "meta")):
        return False
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    def _part_is_cuda(part: str) -> bool:
        if part.startswith("cuda") or part.isdigit():
            return True
        try:
            numeric = float(part)
        except ValueError:
            return False
        return numeric >= 0 and numeric.is_integer()

    return bool(parts) and all(_part_is_cuda(part) for part in parts)


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("K values must be positive integers")
    return values


def parse_csv_text(text: str) -> list[str]:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("method list cannot be empty")
    unknown = sorted(set(values) - REQUIRED_METHODS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown methods: {unknown}")
    return values


def parse_optional_csv_text(text: object) -> list[str]:
    ""

    return [value.strip() for value in str(text or "").split(",") if value.strip()]


def caster_task_scope_args(dataset_key: str) -> list[str]:
    ""

    task_id = str(dataset_key).strip()
    if task_id not in RUNTIME_SINGLE_TASKS:
        raise ValueError(
            f"runtime scaling CASTER timing requires a component task id, got {task_id!r}"
        )
    return ["--task-id", task_id]


def formal_repeat_values(args: argparse.Namespace) -> list[int]:
    offset = int(getattr(args, "repeat_offset", 0))
    return list(range(offset, offset + int(args.repeats)))


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_python_tree(root: Path) -> str:
    ""

    h = hashlib.sha256()
    if not root.exists():
        return "MISSING"
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(path).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def hash_manifest(run_root: Path) -> dict[str, str]:
    return {rel: sha256_file(run_root / rel) for rel in HASH_GUARD_RELS}


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown_cpu"


def gpu_summary() -> str:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return "no_nvidia_smi"
    try:
        proc = subprocess.run([nvidia, "-L"], text=True, capture_output=True, timeout=5)
    except Exception as exc:                                     
        return f"nvidia_smi_error:{type(exc).__name__}"
    text = ";".join(line.strip() for line in proc.stdout.splitlines() if line.strip())
    return text or "no_cuda_devices"


def hardware_id() -> str:
    return "|".join(
        [
            "host=anonymous",
            f"machine={platform.machine()}",
            f"cpu={cpu_model()}",
            f"cores={os.cpu_count() or 'unknown'}",
            f"gpu={gpu_summary()}",
        ]
    )


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _bridge_predictive_contract(path: Path) -> tuple[str, bool]:
    ""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid frozen bridge JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"frozen bridge root must be an object: {path}")
    calibration_metadata = payload.get("calibration_metadata", {})
    if not isinstance(calibration_metadata, dict):
        calibration_metadata = {}
    config_declared = "predictive_contract" in payload
    metadata_declared = "predictive_contract" in calibration_metadata
    config_contract = str(
        payload.get("predictive_contract", alternate_PREDICTIVE_CONTRACT)
    ).strip()
    metadata_contract = str(
        calibration_metadata.get(
            "predictive_contract", alternate_PREDICTIVE_CONTRACT
        )
    ).strip()
    if not config_contract or not metadata_contract:
        raise ValueError(f"empty predictive_contract in frozen bridge {path}")
    if config_contract != metadata_contract:
        raise ValueError(
            "frozen bridge predictive_contract disagrees with its calibration "
            f"metadata: path={path}, config={config_contract!r}, "
            f"metadata={metadata_contract!r}"
        )
    return config_contract, bool(config_declared or metadata_declared)


def resolve_frozen_predictive_contract(
    bridge_one_layer: Path, bridge_hierarchical: Path
) -> tuple[str, str]:
    ""

    one_contract, one_declared = _bridge_predictive_contract(bridge_one_layer)
    hierarchical_contract, hierarchical_declared = _bridge_predictive_contract(
        bridge_hierarchical
    )
    if one_declared != hierarchical_declared:
        raise ValueError(
            "one-layer and hierarchical bridge configs must both declare "
            "predictive_contract or both omit it; "
            f"one_layer={one_contract!r} (declared={one_declared}), "
            f"hierarchical={hierarchical_contract!r} "
            f"(declared={hierarchical_declared})"
        )
    if one_contract != hierarchical_contract:
        raise ValueError(
            "one-layer and hierarchical frozen bridge predictive_contract "
            f"values disagree: one_layer={one_contract!r}, "
            f"hierarchical={hierarchical_contract!r}"
        )
    if not one_declared:
        return alternate_PREDICTIVE_CONTRACT, "alternate_default_both_bridges_omitted"
    return one_contract, "explicit_frozen_bridge_configs"


def caster_predictive_contract_args(
    timing_inputs: dict[str, Path | str],
) -> list[str]:
    ""

    predictive_contract = str(timing_inputs.get("predictive_contract", "")).strip()
    if not predictive_contract:
        raise ValueError("resolved runtime scaling timing inputs omitted predictive_contract")
    return ["--predictive-contract", predictive_contract]


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_cell_lock_metadata(
    handle, payload: dict[str, object]
) -> None:
    ""

    handle.seek(0)
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def runtime_cell_lock(
    *,
    lock_path: Path,
    owner: dict[str, object],
) -> Iterator[dict[str, object]]:
    ""








    lock_path.parent.mkdir(parents=True, exist_ok=True)
    wait_started_at = utc_now()
    wait_started = time.monotonic()
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o640)
    handle = os.fdopen(lock_fd, "r+", encoding="utf-8")
    acquired = False
    metadata: dict[str, object] = {}
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        acquired = True
        metadata = {
            "schema_version": CELL_LOCK_SCHEMA,
            "status": "held",
            "owner_pid": os.getpid(),
            "owner_ppid": os.getppid(),
            "owner_host": socket.gethostname(),
            "wait_started_at": wait_started_at,
            "acquired_at": utc_now(),
            "wait_seconds": max(0.0, time.monotonic() - wait_started),
            "lock_path": str(lock_path.resolve()),
            **owner,
        }
        _write_cell_lock_metadata(handle, metadata)
        try:
            yield metadata
        except BaseException as exc:
            metadata.update(
                {
                    "status": "released_exception",
                    "released_at": utc_now(),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc)[:1000],
                }
            )
            raise
        else:
            metadata.update(
                {
                    "status": "released_ok",
                    "released_at": utc_now(),
                }
            )
        finally:
                                                                           
                                                              
            try:
                _write_cell_lock_metadata(handle, metadata)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                acquired = False
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def record_seconds(timing: dict, names: Iterable[str]) -> float:
    wanted = set(names)
    return float(sum(float(r.get("seconds", 0.0)) for r in timing.get("records", []) if str(r.get("name")) in wanted))


def runtime_k_dir(run_root: Path, k: int) -> Path:
    return run_root / "stageB/runtime_k_sweep" / f"K_{int(k)}"


def selection_paths(run_root: Path, k: int) -> dict[str, Path]:
    root = runtime_k_dir(run_root, k) / "selection/shared_topk"
    return {
        "root": root,
        "selection": root / "candidate_selection_log.csv",
        "timing": root / "candidate_selection_timing.json",
        "metadata": root / "candidate_selection_metadata.json",
        "log": run_root / "stageB/runtime_k_sweep/logs" / f"selection_k{k}.log",
    }


def resolve_timing_task_paths(run_root: Path, args: argparse.Namespace) -> dict[str, Path | str]:
    ""







    requested_task = str(args.timing_task)
    if requested_task not in RUNTIME_SINGLE_TASKS:
        raise ValueError(f"runtime scaling timing must use a single Benchmark B task, not {requested_task!r}")

    if args.task_artifact_root:
        task_root = Path(args.task_artifact_root).resolve()
        task_id = requested_task
    else:
        task_id = requested_task
        task_root = run_root / "new_method/artifacts/benchmark_b" / task_id
        if not task_root.exists():
            alternate = run_root / "new_method/artifacts/benchmark_b"
            if (alternate / "forecast_archive.csv").exists():
                task_root = alternate
                task_id = "benchmark_b"
    if not task_root.exists():
        raise FileNotFoundError(f"runtime scaling task artifact root missing: {task_root}")

    alternate_flat = task_root.name == "benchmark_b"
    if args.ledger:
        ledger = Path(args.ledger).resolve()
    else:
        task_ledger = task_root / "event_ledger.csv"
        ledger = task_ledger if task_ledger.exists() else ROOT / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/event_ledger.csv"

    if args.forecast_archive:
        archive = Path(args.forecast_archive).resolve()
    else:
        pool_size = int(
            getattr(
                args,
                "candidate_pool_size",
                FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT,
            )
        )
        pool_archive = run_root / f"stageB/runtime_k_sweep/forecast_archive_k{pool_size}.csv"
        alternate_k26 = run_root / "stageB/runtime_k_sweep/forecast_archive_k26.csv"
        if pool_archive.exists():
            archive = pool_archive
        elif alternate_k26.exists():
            archive = alternate_k26
        else:
            archive = task_root / "forecast_archive.csv"

    explicit_registry = getattr(args, "model_registry", None)
    registry_csv = (
        Path(explicit_registry).resolve()
        if explicit_registry
        else run_root / "new_method/artifacts/model_registry.formal.csv"
    )
    bridge_one = task_root / "bridge_config.one_layer.json"
    bridge_hier = task_root / "bridge_config.hierarchical.json"
    if alternate_flat:
        bridge_one = bridge_one if bridge_one.exists() else task_root / "bridge_config.json"
        bridge_hier = bridge_hier if bridge_hier.exists() else task_root / "bridge_config.json"
    explicit_full_recovery_manifest = getattr(args, "full_recovery_manifest", None)
    full_recovery_manifest = (
        Path(explicit_full_recovery_manifest).resolve()
        if explicit_full_recovery_manifest
        else resolve_full_recovery_manifest(run_root, task_root, task_id)
    )

    required = {
        "ledger": ledger,
        "archive": archive,
        "registry": registry_csv,
        "bridge_one_layer": bridge_one,
        "bridge_hierarchical": bridge_hier,
        "full_recovery_manifest": full_recovery_manifest,
    }
    missing = [f"{key}={path}" for key, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("runtime scaling timing inputs missing: " + "; ".join(missing))
    predictive_contract, predictive_contract_source = (
        resolve_frozen_predictive_contract(bridge_one, bridge_hier)
    )
    return {
        "task_id": task_id,
        "task_label": RUNTIME_TASK_LABELS.get(task_id, task_id),
        "task_root": task_root,
        "predictive_contract": predictive_contract,
        "predictive_contract_source": predictive_contract_source,
        **required,
    }


def selection_context(selection_path: Path, metadata_path: Path) -> dict[str, object]:
    metadata = read_json(metadata_path)
    if not selection_path.exists():
        raise FileNotFoundError(selection_path)
    selected = pd.read_csv(selection_path)
    if "rank" in selected.columns:
        selected = selected.sort_values("rank", kind="mergesort")
    if "model_id" not in selected.columns:
        raise ValueError(f"selection source missing model_id: {selection_path}")
    model_ids = selected["model_id"].astype(str).tolist()
    if not model_ids:
        raise ValueError(f"empty shared selection: {selection_path}")
    return {
        "selection_path": str(selection_path),
        "selection_metadata_path": str(metadata_path),
        "selection_hash": sha256_file(selection_path),
        "selected_model_ids": ",".join(model_ids),
        "selected_model_count": int(len(model_ids)),
        "same_selection_per_k": True,
        "selection_scope": str(metadata.get("selection_scope", "runtime_formal_k_sweep_shared_topk")),
        "selection_reused_across_methods": True,
        "uses_legacy_timing": False,
    }


def run_shared_topk_selection(
    run_root: Path,
    k: int,
    seed: int,
    *,
    timeout: float | None,
    formal_ranking: Path | None = None,
    expected_candidate_count: int = FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT,
    excluded_model_ids: Iterable[str] = (),
    candidate_pool_profile: str = FORMAL_RESULT_CANDIDATE_PROFILE,
    selection_input_hash: str = "",
    reuse_existing_selection: bool = False,
) -> tuple[dict[str, object] | None, Path, int, bool]:
    paths = selection_paths(run_root, k)
    expected_candidate_count = int(expected_candidate_count)
    if expected_candidate_count <= 0:
        raise ValueError("expected_candidate_count must be positive")
    if int(k) > expected_candidate_count:
        raise ValueError(f"K={k} exceeds candidate pool size {expected_candidate_count}")
    excluded = {str(model_id).strip() for model_id in excluded_model_ids if str(model_id).strip()}
    if formal_ranking is not None:
        ranking = pd.read_csv(formal_ranking)
        if "model_id" not in ranking.columns:
            raise ValueError(f"formal runtime scaling ranking missing model_id: {formal_ranking}")
        if "rank" in ranking.columns:
            ranking = ranking.sort_values("rank", kind="mergesort")
        if (
            len(ranking) != expected_candidate_count
            or ranking["model_id"].astype(str).nunique() != expected_candidate_count
        ):
            raise ValueError(
                "formal runtime scaling ranking must contain the complete candidate bank "
                f"({expected_candidate_count}): {formal_ranking}"
            )
        present_excluded = sorted(excluded & set(ranking["model_id"].astype(str)))
        if present_excluded:
            raise ValueError(
                "formal runtime scaling ranking contains excluded model ids: "
                + ",".join(present_excluded)
            )
        selected = ranking.head(int(k)).copy()
        expected_ids = selected["model_id"].astype(str).tolist()
        if reuse_existing_selection and paths["selection"].exists() and paths["metadata"].exists():
            existing_metadata = read_json(paths["metadata"])
            existing_context = selection_context(paths["selection"], paths["metadata"])
            existing_ids = str(existing_context["selected_model_ids"]).split(",")
            existing_profile = str(
                existing_metadata.get(
                    "candidate_pool_profile",
                    existing_metadata.get("retrieval_profile", ""),
                )
            )
                                                                             
                                                                            
                                                                            
                                                     
            alternate_default_profile_match = (
                str(candidate_pool_profile) == FORMAL_RESULT_CANDIDATE_PROFILE
                and existing_profile == "embedding_validation_full_history_v1"
                and "candidate_pool_profile" not in existing_metadata
            )
            profile_match = (
                existing_profile == str(candidate_pool_profile)
                or alternate_default_profile_match
            )
            existing_selection_input_hash = str(
                existing_metadata.get("selection_input_hash", "")
            )
            reusable = (
                existing_ids == expected_ids
                and str(existing_metadata.get("formal_ranking_sha256", "")) == sha256_file(formal_ranking)
                and int(existing_metadata.get("candidate_pool_size", -1)) == expected_candidate_count
                and int(existing_metadata.get("top_k", -1)) == int(k)
                and sorted(str(value) for value in existing_metadata.get("excluded_model_ids", []))
                == sorted(excluded)
                and profile_match
                and existing_selection_input_hash == str(selection_input_hash)
            )
            if reusable:
                existing_context["selection_reused_existing"] = True
                return existing_context, paths["log"], 0, False
        paths["root"].mkdir(parents=True, exist_ok=True)
        selected.to_csv(paths["selection"], index=False)
        metadata = {
            "selection_scope": "runtime_formal_task_ranking_prefix",
            "selection_reused_across_algorithms": True,
            "selection_rerun_per_algorithm": False,
            "retrieval_profile": str(candidate_pool_profile),
            "candidate_pool_profile": str(candidate_pool_profile),
            "selection_input_hash": str(selection_input_hash),
            "formal_ranking": str(formal_ranking),
            "formal_ranking_sha256": sha256_file(formal_ranking),
            "candidate_pool_size": expected_candidate_count,
            "excluded_model_ids": sorted(excluded),
            "top_k": int(k),
            "beta_val": 1.0,
            "beta_runtime": 0.0,
            "priority_weight": 0.0,
            "family_diversity_bonus": 0.0,
            "selected_model_ids": selected["model_id"].astype(str).tolist(),
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["timing"].write_text(json.dumps({
            "records": [{"name": "shared_selection_prefix_materialization", "seconds": 0.0}],
            "total_sec": 0.0,
            "selection_reused": True,
            "selection_metadata": metadata,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        paths["log"].write_text(
            f"shared_selection=skip existing source={formal_ranking} top_k={k}\n",
            encoding="utf-8",
        )
        context = selection_context(paths["selection"], paths["metadata"])
        context["selection_reused_existing"] = False
        return context, paths["log"], 0, False
    registry_csv = run_root / "new_method/artifacts/model_registry.formal.csv"
    embeddings = run_root / "new_method/artifacts/candidate_embeddings.csv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_stageB_topk_selection.py"),
        "--registry",
        str(registry_csv),
        "--embeddings",
        str(embeddings),
        "--out",
        str(paths["selection"]),
        "--timing-out",
        str(paths["timing"]),
        "--metadata-out",
        str(paths["metadata"]),
        "--benchmark",
        "runtime_k_sweep",
        "--algorithm-id",
        f"runtime_k{k}_shared_topk",
        "--selection-scope",
        "runtime_formal_k_sweep_shared_topk",
        "--shared-selection",
        "--top-k",
        str(int(k)),
        "--family-diversity-bonus",
        "0.12",
        "--seed",
        str(int(seed)),
    ]
    code, timed_out = run_logged(cmd, cwd=ROOT, log_path=paths["log"], timeout=timeout)
    if code != 0 or timed_out:
        return None, paths["log"], code, timed_out
    context = selection_context(paths["selection"], paths["metadata"])
    if int(context["selected_model_count"]) != int(k):
        raise ValueError(f"shared selection returned {context['selected_model_count']} rows for K={k}")
    return context, paths["log"], code, timed_out


def write_restart_registry(
    run_root: Path,
    model_ids: list[str],
    out: Path,
    *,
    registry_path: Path | None = None,
) -> Path:
    registry_path = (
        Path(registry_path).resolve()
        if registry_path is not None
        else run_root / "new_method/artifacts/model_registry.formal.csv"
    )
    if not registry_path.exists():
        registry_path = NM_CODE_ROOT / "configs/model_registry.yaml"
    if registry_path.suffix.lower() == ".csv":
        registry = pd.read_csv(registry_path)
        subset = registry[registry["model_id"].astype(str).isin(set(model_ids))].copy()
        subset["_order"] = subset["model_id"].astype(str).map({m: i for i, m in enumerate(model_ids)})
        subset = subset.sort_values("_order", kind="mergesort").drop(columns=["_order"])
        if len(subset) != len(model_ids):
            missing = sorted(set(model_ids) - set(subset["model_id"].astype(str)))
            raise ValueError(f"restart registry missing selected model ids: {missing}")
        out.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(out, index=False)
        return out
                                                                                    
    import yaml

    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    rows = data.get("candidates", data.get("models", [])) if isinstance(data, dict) else []
    wanted = set(model_ids)
    subset = [dict(row) for row in rows if str(row.get("model_id")) in wanted]
    order = {m: i for i, m in enumerate(model_ids)}
    subset.sort(key=lambda row: order[str(row.get("model_id"))])
    if len(subset) != len(model_ids):
        missing = sorted(wanted - {str(row.get("model_id")) for row in subset})
        raise ValueError(f"restart registry missing selected model ids: {missing}")
    out = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"candidates": subset}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _resolve_manifest_rel(path_text: str, manifest_path: Path) -> Path:
    raw = Path(str(path_text))
    if raw.is_absolute():
        return raw
    for base in [ROOT, BL_CODE_ROOT, manifest_path.parent]:
        candidate = base / raw
        if candidate.exists():
            return candidate
    return ROOT / raw


def _manifest_has_dataset(manifest_path: Path, dataset_key: str) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = pd.read_csv(manifest_path, keep_default_na=False, usecols=["dataset_key"])
    except Exception:
        return False
    return bool(manifest["dataset_key"].astype(str).eq(str(dataset_key)).any())


def resolve_full_recovery_manifest(run_root: Path, task_root: Path, task_id: str) -> Path:
    ""







    candidates = [
        task_root / "archive_backed_agent_inputs" / f"{task_id}_manifest.csv",
        task_root / f"{task_id}_manifest.csv",
        run_root / "baseline/data/full_manifest.csv",
        BL_CODE_ROOT / "data/full_manifest.csv",
    ]
    for candidate in candidates:
        if _manifest_has_dataset(candidate, task_id):
            return candidate
    existing = [str(path) for path in candidates if path.exists()]
    checked = existing if existing else [str(path) for path in candidates]
    raise FileNotFoundError(
        f"runtime scaling restart manifest has no dataset_key={task_id}; checked: "
        + ", ".join(checked)
    )


def write_agent_selection_replay_log(
    *,
    manifest_path: Path,
    dataset_key: str,
    selected_model_id: str,
    out: Path,
) -> Path:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    rows = manifest[manifest["dataset_key"].astype(str).eq(str(dataset_key))]
    if rows.empty:
        raise ValueError(f"manifest has no dataset_key={dataset_key}: {manifest_path}")
    ledger_path = _resolve_manifest_rel(str(rows.iloc[0]["ledger_path"]), manifest_path)
    ledger = pd.read_csv(ledger_path, keep_default_na=False)
    origins = sorted(str(x) for x in ledger["forecast_origin"].dropna().astype(str).unique())
    replay_rows = [{
        "stage": "model_selection",
        "dataset_key": str(dataset_key),
        "forecast_origin": "DATASET_LEVEL",
        "llm_selected_model_id": str(selected_model_id),
        "selected_model_id": str(selected_model_id),
        "llm_reason": "deterministic runtime scaling shared-topK replay selection",
    }]
    for origin in origins:
        replay_rows.append({
            "stage": "model_selection",
            "dataset_key": str(dataset_key),
            "forecast_origin": origin,
            "llm_selected_model_id": str(selected_model_id),
            "selected_model_id": str(selected_model_id),
            "llm_reason": "deterministic runtime scaling shared-topK replay selection",
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(replay_rows).to_csv(out, index=False)
    return out


def caster_update_splits(ledger_path: Path) -> str:
    ""

    ledger = pd.read_csv(ledger_path, usecols=["split"])
    declared = set(ledger["split"].dropna().astype(str).str.strip())
    required = {"train", "val"}
    missing = sorted(required - declared)
    if missing:
        raise ValueError(f"runtime scaling ledger missing required update splits {missing}: {ledger_path}")
    return "train,val,embargo" if "embargo" in declared else "train,val"


def run_logged(cmd: list[str], *, cwd: Path, log_path: Path, timeout: float | None) -> tuple[int, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
                                                                       
                                                                         
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            return int(proc.returncode), False
        except subprocess.TimeoutExpired as exc:
            log.write(f"\nTIMEOUT after {timeout} seconds\n")
            if exc.stdout:
                log.write(str(exc.stdout))
            if exc.stderr:
                log.write(str(exc.stderr))
            return 124, True


def build_persistent_agent_engine(methods: list[str]):
    ""

    agent_methods = [method for method in methods if is_agent_method(method)]
    if len(agent_methods) != 1 or len(methods) != 1:
        raise ValueError("--persistent-agent-worker requires exactly one agent method")
    method = agent_methods[0]
    max_new_tokens = {
        "agentic_top_one": 128,
        "agent_react": 192,
        "agentic_full_recovery": 256,
    }[method]
    baseline_src = str(BL_CODE_ROOT / "src")
    if baseline_src not in sys.path:
        sys.path.insert(0, baseline_src)
    from caster_baselines.agentic_llm import QwenLocalEngine

    engine = QwenLocalEngine(
        primary_model_path=REQUIRED_QWEN_7B_PATH,
        fallback_model_path="",
        allow_fallback=False,
        runtime_budget_seconds=86400.0,
        max_new_tokens=max_new_tokens,
        required_model_path=REQUIRED_QWEN_7B_PATH,
        require_cuda=True,
    )
    engine.warm_load()
    return engine


def run_persistent_agent_cell(
    *,
    method: str,
    engine: object,
    manifest_path: Path,
    registry_path: Path,
    out_dir: Path,
    dataset_key: str,
    forecast_archive: Path,
    charge_selection: bool,
    log_path: Path,
) -> tuple[int, bool]:
    ""

    baseline_src = str(BL_CODE_ROOT / "src")
    if baseline_src not in sys.path:
        sys.path.insert(0, baseline_src)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if method == "agentic_top_one":
            from caster_baselines.agentic_top_one import run_agentic_top_one

            run_agentic_top_one(
                manifest_path=manifest_path,
                registry_path=registry_path,
                out_dir=out_dir,
                engine=engine,
                dataset_keys=[dataset_key],
                forecast_archive=forecast_archive,
                archive_mode="required",
                charge_selection=charge_selection,
                timing_mode="archive_backed",
                exclude_llm_load_from_timing=True,
            )
        elif method == "agent_react":
            from caster_baselines.agentic_react import run_react_agent_from_manifest

            run_react_agent_from_manifest(
                manifest_path=manifest_path,
                registry_path=registry_path,
                out_dir=out_dir,
                engine=engine,
                dataset_keys=[dataset_key],
                selection_policy="llm_only",
                forecast_archive=forecast_archive,
                archive_mode="required",
                charge_selection=charge_selection,
                timing_mode="archive_backed",
                exclude_llm_load_from_timing=True,
            )
        elif method == "agentic_full_recovery":
            from caster_baselines.agentic_full_recovery import run_agentic_full_recovery

            run_agentic_full_recovery(
                manifest_path=manifest_path,
                registry_path=registry_path,
                out_dir=out_dir,
                engine=engine,
                dataset_keys=[dataset_key],
                selection_policy="llm_only",
                method_name="agentic_full_recovery",
                forecast_archive=forecast_archive,
                archive_mode="required",
                charge_selection=charge_selection,
                timing_mode="archive_backed",
                exclude_llm_load_from_timing=True,
            )
        else:                                                 
            raise ValueError(f"unsupported persistent agent method: {method}")
    except Exception:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        return 1, False
    timing_path = out_dir / "timing.json"
    timing = read_json(timing_path)
    timing["persistent_agent_worker"] = True
    timing["prewarmed_engine_reused"] = True
                                                                            
                                                                        
                                                                              
                                                                                 
                                                                            
    persistent_load = pd.to_numeric(
        pd.Series([timing.get("llm_model_load_seconds")]), errors="coerce"
    ).iloc[0]
    if pd.notna(persistent_load) and math.isfinite(float(persistent_load)) and float(persistent_load) > 0.0:
        timing["llm_load_seconds_excluded_from_update"] = float(persistent_load)
        timing["llm_load_exclusion_accounting"] = (
            "persistent_worker_qwen_loaded_once_before_timed_cells"
        )
    write_json_atomic(timing_path, timing)
    log_path.write_text(
        f"persistent_agent_worker=ok method={method} dataset={dataset_key} out={out_dir}\n",
        encoding="utf-8",
    )
    return 0, False


def caster_row_from_timing(
    *,
    method: str,
    k: int,
    repeat: int,
    seed: int,
    run_mode: str,
    status: str,
    timing: dict,
    hw: str,
    run_id: str,
    started_at: str,
    finished_at: str,
    log_path: Path,
    artifact_reuse_proxy: bool,
    selection: dict[str, object],
) -> dict[str, object]:
    if method == "caster_hierarchical":
        filter_sec = record_seconds(timing, ["hierarchical_filter"])
        forecast_sec = record_seconds(timing, ["hierarchical_forecast_readout"])
    else:
        filter_sec = record_seconds(timing, ["sequential_filter"])
        forecast_sec = record_seconds(timing, ["forecast_readout"])
    score_sec = record_seconds(timing, ["score_update_rows"])
    update_sec = score_sec + filter_sec + forecast_sec
    return {
        "method": method,
        "candidate_count": int(k),
        "repeat": int(repeat),
        "seed": int(seed),
        "run_mode": run_mode,
        "status": status,
        "total_sec": update_sec if status == "ok" else float("nan"),
        "algorithm_update_sec": update_sec if status == "ok" else float("nan"),
        "update_sec": update_sec if status == "ok" else float("nan"),
        "proposal_sec": 0.0,
        "filter_sec": filter_sec if status == "ok" else float("nan"),
        "forecast_sec": forecast_sec if status == "ok" else float("nan"),
        "planner_stage_sec": 0.0,
        "planner_qwen_sec": 0.0,
        "planner_non_qwen_sec": 0.0,
        "selector_stage_sec": 0.0,
        "selector_qwen_sec": 0.0,
        "selector_non_qwen_sec": 0.0,
        "critic_trace_sec": 0.0,
        "archive_readout_compute_sec": 0.0,
        "archive_coverage_validation_sec": 0.0,
        "readout_enrichment_sec": 0.0,
        "readout_materialization_sec": 0.0,
        "readout_validation_sec": 0.0,
        "readout_artifact_write_sec": 0.0,
        "forecast_readout_sec": forecast_sec if status == "ok" else float("nan"),
        "runtime_update_sec": update_sec if status == "ok" else float("nan"),
        "planner_stage_timing_schema": "not_applicable",
        "selector_stage_timing_schema": "not_applicable",
        "readout_timing_schema": "caster_native_timed_readout",
        "hardware": hw,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "log_path": str(log_path),
        "artifact_reuse_proxy": bool(artifact_reuse_proxy),
        "uses_legacy_timing": False,
        **selection,
        "timing_semantics": "incremental_update_readout_on_frozen_archive_bridge",
        "timing_mode": "archive_backed",
        "formal_timing_valid": bool(status == "ok" and not artifact_reuse_proxy),
        "model_compute_sec": 0.0,
        "forecast_source": "immutable_forecast_archive",
        "selection_charged_to_update": False,
        "selection_replay_used": False,
        "selection_engine": "",
        "llm_model_path": "",
        "llm_required_model_path": "",
        "llm_primary_required": False,
        "llm_cuda_required": False,
        "llm_cuda_available": False,
        "llm_model_device": "",
        "llm_fallback_used": False,
        "llm_fallback_allowed": False,
        "restart_type": "",
        "timing_regime": PREWARMED_TIMING_REGIME,
        "process_startup_sec_included": False,
        "llm_load_sec_included": False,
        "process_total_sec": float(timing.get("total_sec", float("nan"))) if timing else float("nan"),
    }


BAD_FORMAL_TIMING_TOKENS = ("refit", "reforecast")


def formal_archive_backed_agent_ok(
    timing: dict, *, method: str = ""
) -> tuple[bool, str]:
    if not timing:
        return False, "missing_timing_json"
    if str(timing.get("timing_mode", "")) != "archive_backed":
        return False, "timing_mode_not_archive_backed"
    if str(timing.get("forecast_source", "")) != "immutable_forecast_archive":
        return False, "forecast_source_not_immutable_archive"
    if str(timing.get("formal_timing_valid", "")).strip().lower() not in {"1", "true", "yes"}:
        return False, "formal_timing_valid_not_true"
    model_compute = pd.to_numeric(pd.Series([timing.get("model_compute_sec")]), errors="coerce").iloc[0]
    if pd.isna(model_compute) or abs(float(model_compute)) > 1e-12:
        return False, "model_compute_sec_not_zero"
    if str(timing.get("artifact_reuse_proxy", "false")).strip().lower() in {"1", "true", "yes"}:
        return False, "artifact_reuse_proxy_true"
    semantics = str(timing.get("timing_semantics", "")).lower()
    if any(tok in semantics for tok in BAD_FORMAL_TIMING_TOKENS):
        return False, "alternate_refit_reforecast_semantics"
    if str(timing.get("selection_replay_used", "false")).strip().lower() in {"1", "true", "yes"}:
        return False, "selection_replay_used"
    if str(timing.get("selection_charged_to_update", "false")).strip().lower() not in {"1", "true", "yes"}:
        return False, "selection_not_charged_to_update"
    if str(timing.get("selection_engine", "")) != "qwen":
        return False, "selection_engine_not_qwen"
    if str(timing.get("llm_model_path", "")) != REQUIRED_QWEN_7B_PATH:
        return False, "llm_model_path_not_required_qwen7b"
    if str(timing.get("llm_required_model_path", "")) != REQUIRED_QWEN_7B_PATH:
        return False, "llm_required_model_path_not_qwen7b"
    if not truthy(timing.get("llm_primary_required", False)):
        return False, "llm_primary_not_required"
    if not truthy(timing.get("llm_cuda_required", False)):
        return False, "llm_cuda_not_required"
    if not truthy(timing.get("llm_cuda_available", False)):
        return False, "llm_cuda_not_available"
    if not device_is_cuda(timing.get("llm_model_device", "")):
        return False, "llm_model_device_not_cuda"
    if truthy(timing.get("llm_fallback_used", False)):
        return False, "llm_fallback_used"
    if truthy(timing.get("llm_fallback_allowed", False)):
        return False, "llm_fallback_allowed"
    restart_type = str(timing.get("restart_type", ""))
    if restart_type != "archive_backed_true_selection_readout":
        return False, "restart_type_not_archive_backed_true_selection_readout"
    algorithm_update = pd.to_numeric(pd.Series([timing.get("algorithm_update_sec")]), errors="coerce").iloc[0]
    if pd.isna(algorithm_update) or not math.isfinite(float(algorithm_update)) or float(algorithm_update) <= 0.0:
        return False, "missing_algorithm_update_sec"
    excluded_load = pd.to_numeric(pd.Series([timing.get("llm_load_seconds_excluded_from_update")]), errors="coerce").iloc[0]
    persistent_prewarm = (
        truthy(timing.get("persistent_agent_worker", False))
        and truthy(timing.get("prewarmed_engine_reused", False))
    )
    model_load = pd.to_numeric(
        pd.Series([timing.get("llm_model_load_seconds")]), errors="coerce"
    ).iloc[0]
    has_persistent_excluded_load = (
        persistent_prewarm
        and pd.notna(model_load)
        and math.isfinite(float(model_load))
        and float(model_load) > 0.0
    )
    if (
        pd.isna(excluded_load)
        or not math.isfinite(float(excluded_load))
        or float(excluded_load) <= 0.0
    ) and not has_persistent_excluded_load:
        return False, "qwen_load_was_not_prewarmed_and_excluded"
    generation_sec = pd.to_numeric(
        pd.Series([timing.get("llm_generation_seconds_charged_to_update", timing.get("llm_runtime_seconds"))]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(generation_sec) or not math.isfinite(float(generation_sec)) or float(generation_sec) <= 0.0:
        return False, "qwen_generation_not_charged_to_update"
    if method == "agentic_full_recovery":
        expected_schemas = {
            "planner_stage_timing_schema": "agentic_full_recovery_planner_stage_wall_v1",
            "selector_stage_timing_schema": "agentic_full_recovery_selector_stage_wall_v1",
            "readout_timing_schema": "agent_archive_readout_artifacts_v1",
        }
        for key, expected in expected_schemas.items():
            if str(timing.get(key, "")) != expected:
                return False, f"{key}_invalid"

        numeric_keys = [
            "selection_seconds",
            "planner_stage_seconds",
            "planner_qwen_seconds",
            "planner_non_qwen_seconds",
            "selector_stage_seconds",
            "selector_qwen_seconds",
            "selector_non_qwen_seconds",
            "agent_control_seconds",
            "archive_lookup_seconds",
            "archive_readout_compute_seconds",
            "archive_coverage_validation_seconds",
            "readout_enrichment_seconds",
            "readout_materialization_seconds",
            "readout_validation_seconds",
            "readout_artifact_write_seconds",
            "forecast_readout_seconds",
            "runtime_update_sec",
        ]
        values: dict[str, float] = {}
        for key in numeric_keys:
            value = pd.to_numeric(
                pd.Series([timing.get(key)]), errors="coerce"
            ).iloc[0]
            if pd.isna(value) or not math.isfinite(float(value)):
                return False, f"{key}_missing_or_nonfinite"
            values[key] = float(value)
        if values["planner_stage_seconds"] + 2e-6 < values["planner_qwen_seconds"]:
            return False, "planner_qwen_exceeds_stage"
        if values["selector_stage_seconds"] + 2e-6 < values["selector_qwen_seconds"]:
            return False, "selector_qwen_exceeds_stage"
        if not math.isclose(
            values["planner_non_qwen_seconds"],
            values["planner_stage_seconds"] - values["planner_qwen_seconds"],
            abs_tol=3e-6,
        ):
            return False, "planner_stage_component_identity_failed"
        if not math.isclose(
            values["selector_non_qwen_seconds"],
            values["selector_stage_seconds"] - values["selector_qwen_seconds"],
            abs_tol=3e-6,
        ):
            return False, "selector_stage_component_identity_failed"
        if not math.isclose(
            values["selection_seconds"],
            values["planner_stage_seconds"] + values["selector_stage_seconds"],
            abs_tol=3e-6,
        ):
            return False, "selection_stage_identity_failed"
        readout_component_sum = sum(
            values[key]
            for key in [
                "archive_readout_compute_seconds",
                "archive_coverage_validation_seconds",
                "readout_enrichment_seconds",
                "readout_materialization_seconds",
                "readout_validation_seconds",
                "readout_artifact_write_seconds",
            ]
        )
        if not math.isclose(
            values["forecast_readout_seconds"],
            readout_component_sum,
            abs_tol=6e-6,
        ):
            return False, "forecast_readout_component_identity_failed"
        if not math.isclose(
            values["runtime_update_sec"],
            values["selector_stage_seconds"]
            + values["forecast_readout_seconds"],
            abs_tol=3e-6,
        ):
            return False, "runtime_update_identity_failed"
        if values["readout_artifact_write_seconds"] <= 0.0:
            return False, "readout_artifact_write_not_positive"

        try:
            planner_count = int(timing.get("planner_stage_count", -1))
            selector_count = int(timing.get("selector_stage_count", -1))
            restart_count = int(timing.get("restart_group_rows", -1))
            forecast_rows = int(timing.get("forecast_rows", -1))
            expected_rows = int(timing.get("expected_rows", -1))
        except (TypeError, ValueError):
            return False, "stage_or_row_count_invalid"
        if restart_count <= 0:
            return False, "restart_group_rows_not_positive"
        if planner_count != restart_count or selector_count != restart_count:
            return False, "planner_selector_restart_count_mismatch"
        if forecast_rows <= 0 or forecast_rows != expected_rows:
            return False, "forecast_expected_row_count_mismatch"

        readout_path = Path(str(timing.get("forecast_readout_artifact", "")))
        validation_path = Path(
            str(timing.get("archive_forecast_readout_validation", ""))
        )
        if not readout_path.is_file():
            return False, "forecast_readout_artifact_missing"
        if not validation_path.is_file():
            return False, "archive_forecast_readout_validation_missing"
        try:
            readout = pd.read_csv(readout_path, low_memory=False)
            validation = pd.read_csv(validation_path, low_memory=False)
        except Exception:
            return False, "timed_readout_artifact_unreadable"
        expected_readout_rows = int(timing.get("forecast_readout_rows", -1))
        if expected_readout_rows <= 0 or len(readout) != expected_readout_rows:
            return False, "forecast_readout_row_count_mismatch"
        if len(validation) != expected_readout_rows:
            return False, "archive_readout_validation_row_count_mismatch"
        if set(readout.get("split", pd.Series(dtype=str)).astype(str).str.lower()) != {"test"}:
            return False, "forecast_readout_not_test_only"
        if not validation.get("validation_status", pd.Series(dtype=str)).astype(str).eq("PASS").all():
            return False, "archive_readout_validation_not_all_pass"
    return True, ""


def agent_row_from_timing(
    *,
    method: str,
    k: int,
    repeat: int,
    seed: int,
    run_mode: str,
    status: str,
    timing: dict,
    hw: str,
    run_id: str,
    started_at: str,
    finished_at: str,
    log_path: Path,
    artifact_reuse_proxy: bool,
    selection: dict[str, object],
) -> dict[str, object]:
    original_update = (
        float(timing.get("algorithm_update_sec", float("nan")))
        if status == "ok"
        else float("nan")
    )
    update = (
        float(timing.get("runtime_update_sec", float("nan")))
        if status == "ok" and method == "agentic_full_recovery"
        else original_update
    )
    total_process = float(timing.get("total_process_sec", timing.get("total_seconds", float("nan")))) if timing else float("nan")
    excluded_load = pd.to_numeric(
        pd.Series([timing.get("llm_load_seconds_excluded_from_update")]), errors="coerce"
    ).iloc[0]
    if (
        (pd.isna(excluded_load) or not math.isfinite(float(excluded_load)) or float(excluded_load) <= 0.0)
        and truthy(timing.get("persistent_agent_worker", False))
        and truthy(timing.get("prewarmed_engine_reused", False))
    ):
        excluded_load = pd.to_numeric(
            pd.Series([timing.get("llm_model_load_seconds")]), errors="coerce"
        ).iloc[0]
    return {
        "method": method,
        "candidate_count": int(k),
        "repeat": int(repeat),
        "seed": int(seed),
        "run_mode": run_mode,
        "status": status,
                                                                      
                                                                              
                                                         
        "total_sec": update,
        "algorithm_update_sec": update,
        "update_sec": update,
        "proposal_sec": float(timing.get("selection_seconds", timing.get("proposal_seconds", float("nan")))) if status == "ok" else float("nan"),
        "filter_sec": float(timing.get("agent_control_seconds", 0.0)) if status == "ok" else float("nan"),
        "forecast_sec": float(timing.get("archive_lookup_seconds", float("nan"))) if status == "ok" else float("nan"),
        "planner_stage_sec": float(timing.get("planner_stage_seconds", float("nan"))) if status == "ok" else float("nan"),
        "planner_qwen_sec": float(timing.get("planner_qwen_seconds", float("nan"))) if status == "ok" else float("nan"),
        "planner_non_qwen_sec": float(timing.get("planner_non_qwen_seconds", float("nan"))) if status == "ok" else float("nan"),
        "selector_stage_sec": float(timing.get("selector_stage_seconds", float("nan"))) if status == "ok" else float("nan"),
        "selector_qwen_sec": float(timing.get("selector_qwen_seconds", float("nan"))) if status == "ok" else float("nan"),
        "selector_non_qwen_sec": float(timing.get("selector_non_qwen_seconds", float("nan"))) if status == "ok" else float("nan"),
        "critic_trace_sec": float(timing.get("critic_trace_seconds", timing.get("agent_control_seconds", float("nan")))) if status == "ok" else float("nan"),
        "archive_readout_compute_sec": float(timing.get("archive_readout_compute_seconds", timing.get("archive_lookup_seconds", float("nan")))) if status == "ok" else float("nan"),
        "archive_coverage_validation_sec": float(timing.get("archive_coverage_validation_seconds", float("nan"))) if status == "ok" else float("nan"),
        "readout_enrichment_sec": float(timing.get("readout_enrichment_seconds", float("nan"))) if status == "ok" else float("nan"),
        "readout_materialization_sec": float(timing.get("readout_materialization_seconds", float("nan"))) if status == "ok" else float("nan"),
        "readout_validation_sec": float(timing.get("readout_validation_seconds", float("nan"))) if status == "ok" else float("nan"),
        "readout_artifact_write_sec": float(timing.get("readout_artifact_write_seconds", float("nan"))) if status == "ok" else float("nan"),
        "forecast_readout_sec": float(timing.get("forecast_readout_seconds", timing.get("archive_lookup_seconds", float("nan")))) if status == "ok" else float("nan"),
        "runtime_update_sec": float(timing.get("runtime_update_sec", update)) if status == "ok" else float("nan"),
        "planner_stage_timing_schema": str(timing.get("planner_stage_timing_schema", "")),
        "selector_stage_timing_schema": str(timing.get("selector_stage_timing_schema", "")),
        "readout_timing_schema": str(timing.get("readout_timing_schema", "")),
        "hardware": hw,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "log_path": str(log_path),
        "artifact_reuse_proxy": bool(artifact_reuse_proxy),
        "uses_legacy_timing": False,
        **selection,
        "timing_semantics": timing.get("timing_semantics", ""),
        "timing_mode": timing.get("timing_mode", ""),
        "formal_timing_valid": bool(str(timing.get("formal_timing_valid", "")).strip().lower() in {"1", "true", "yes"}),
        "model_compute_sec": float(timing.get("model_compute_sec", float("nan"))) if timing else float("nan"),
        "forecast_source": timing.get("forecast_source", ""),
        "selection_charged_to_update": bool(str(timing.get("selection_charged_to_update", "false")).strip().lower() in {"1", "true", "yes"}),
        "selection_replay_used": bool(str(timing.get("selection_replay_used", "false")).strip().lower() in {"1", "true", "yes"}),
        "selection_engine": str(timing.get("selection_engine", "")),
        "llm_model_path": str(timing.get("llm_model_path", "")),
        "llm_required_model_path": str(timing.get("llm_required_model_path", "")),
        "llm_primary_required": truthy(timing.get("llm_primary_required", False)),
        "llm_cuda_required": truthy(timing.get("llm_cuda_required", False)),
        "llm_cuda_available": truthy(timing.get("llm_cuda_available", False)),
        "llm_model_device": str(timing.get("llm_model_device", "")),
        "llm_fallback_used": truthy(timing.get("llm_fallback_used", False)),
        "llm_fallback_allowed": truthy(timing.get("llm_fallback_allowed", False)),
        "restart_type": timing.get("restart_type", ""),
        "llm_load_sec_excluded": float(excluded_load) if status == "ok" else float("nan"),
        "llm_generation_sec": float(timing.get("llm_generation_seconds_charged_to_update", timing.get("llm_runtime_seconds", 0.0))) if status == "ok" else float("nan"),
        "timing_regime": PREWARMED_TIMING_REGIME,
        "process_startup_sec_included": False,
        "llm_load_sec_included": False,
        "process_total_sec": total_process if status == "ok" else float("nan"),
        "persistent_agent_worker": truthy(timing.get("persistent_agent_worker", False)),
    }


def blocked_row(
    *,
    method: str,
    k: int,
    repeat: int,
    seed: int,
    hw: str,
    run_id: str,
    started_at: str,
    finished_at: str,
    log_path: Path,
    blocker: str,
    selection: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = {
        "method": method,
        "candidate_count": int(k),
        "repeat": int(repeat),
        "seed": int(seed),
        "run_mode": "formal",
        "status": "blocked",
        "total_sec": float("nan"),
        "algorithm_update_sec": float("nan"),
        "update_sec": float("nan"),
        "proposal_sec": float("nan"),
        "filter_sec": float("nan"),
        "forecast_sec": float("nan"),
        "hardware": hw,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "log_path": str(log_path),
        "artifact_reuse_proxy": True,
        "uses_legacy_timing": False,
        "same_selection_per_k": False,
        "selection_path": "",
        "selection_hash": "",
        "selected_model_ids": "",
        "blocker": blocker,
        "timing_mode": "",
        "timing_semantics": "",
        "formal_timing_valid": False,
        "model_compute_sec": float("nan"),
        "forecast_source": "",
        "selection_charged_to_update": False,
        "selection_replay_used": False,
        "selection_engine": "",
        "llm_model_path": "",
        "llm_required_model_path": "",
        "llm_primary_required": False,
        "llm_cuda_required": False,
        "llm_cuda_available": False,
        "llm_model_device": "",
        "llm_fallback_used": False,
        "llm_fallback_allowed": False,
        "restart_type": "",
        "timing_regime": PREWARMED_TIMING_REGIME,
        "process_startup_sec_included": False,
        "llm_load_sec_included": False,
        "process_total_sec": float("nan"),
        "persistent_agent_worker": False,
    }
    if selection:
        payload.update(selection)
        payload["artifact_reuse_proxy"] = False
    return payload


def evaluate_timing_scaling(
    df: pd.DataFrame,
    *,
    required_k: Iterable[int] = DEFAULT_RUNTIME_K_GRID,
    required_methods: Iterable[str] = DEFAULT_METHODS,
    repeats: int = 3,
) -> dict[str, object]:
    missing: list[str] = []
    formal = df.copy()
    if "run_mode" in formal.columns:
        formal = formal[formal["run_mode"].astype(str).eq("formal")]
    else:
        missing.append("run_mode=formal")
        formal = formal.iloc[0:0]
    if "status" in formal.columns:
        formal = formal[formal["status"].astype(str).eq("ok")]
    else:
        missing.append("status=ok")
        formal = formal.iloc[0:0]
    if "artifact_reuse_proxy" in formal.columns:
        proxy = formal["artifact_reuse_proxy"].astype(str).str.lower().isin({"1", "true", "yes"})
        formal = formal[~proxy]
    else:
        missing.append("artifact_reuse_proxy=false")
        formal = formal.iloc[0:0]
    if "uses_legacy_timing" in formal.columns:
        legacy_timing = formal["uses_legacy_timing"].astype(str).str.lower().isin({"1", "true", "yes"})
        formal = formal[~legacy_timing]
    else:
        missing.append("uses_legacy_timing=false")
        formal = formal.iloc[0:0]
    if "same_selection_per_k" in formal.columns:
        same_selection = formal["same_selection_per_k"].astype(str).str.lower().isin({"1", "true", "yes"})
        if not bool(same_selection.all()):
            missing.append("same selection per K")
    else:
        missing.append("same_selection_per_k=true")
        formal = formal.iloc[0:0]
    if "formal_timing_valid" in formal.columns:
        valid = formal["formal_timing_valid"].astype(str).str.lower().isin({"1", "true", "yes"})
        formal = formal[valid]
    else:
        missing.append("formal_timing_valid=true")
        formal = formal.iloc[0:0]
    if "model_compute_sec" in formal.columns:
        model_compute = pd.to_numeric(formal["model_compute_sec"], errors="coerce")
        formal = formal[model_compute.fillna(float("inf")).abs() <= 1e-12]
    else:
        missing.append("model_compute_sec=0")
        formal = formal.iloc[0:0]
    if "timing_semantics" in formal.columns:
        bad_sem = formal["timing_semantics"].astype(str).str.lower().map(
            lambda text: any(tok in text for tok in BAD_FORMAL_TIMING_TOKENS)
        )
        if bool(bad_sem.any()):
            missing.append("no alternate refit/reforecast timing semantics")
            formal = formal[~bad_sem]
    else:
        missing.append("timing_semantics")
        formal = formal.iloc[0:0]
    if {"method", "timing_mode"} <= set(formal.columns):
        archive_backed = formal["timing_mode"].astype(str).eq("archive_backed")
        if not bool(archive_backed.all()):
            missing.append("timing_mode=archive_backed")
            formal = formal[archive_backed]
    else:
        missing.append("timing_mode=archive_backed")
        formal = formal.iloc[0:0]
    if "forecast_source" in formal.columns:
        immutable_archive = formal["forecast_source"].astype(str).eq("immutable_forecast_archive")
        if not bool(immutable_archive.all()):
            missing.append("forecast_source=immutable_forecast_archive")
            formal = formal[immutable_archive]
    else:
        missing.append("forecast_source=immutable_forecast_archive")
        formal = formal.iloc[0:0]
    if "timing_regime" in formal.columns:
        prewarmed = formal["timing_regime"].astype(str).eq(PREWARMED_TIMING_REGIME)
        if not bool(prewarmed.all()):
            missing.append(f"timing_regime={PREWARMED_TIMING_REGIME}")
            formal = formal[prewarmed]
    else:
        missing.append(f"timing_regime={PREWARMED_TIMING_REGIME}")
        formal = formal.iloc[0:0]
    for col, reason in [
        ("process_startup_sec_included", "process startup excluded"),
        ("llm_load_sec_included", "Qwen load excluded"),
    ]:
        if col not in formal.columns:
            missing.append(reason)
            formal = formal.iloc[0:0]
            continue
        included = formal[col].astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
        if bool(included.any()):
            missing.append(reason)
            formal = formal[~included]
    if {"total_sec", "algorithm_update_sec"} <= set(formal.columns):
        total = pd.to_numeric(formal["total_sec"], errors="coerce")
        update = pd.to_numeric(formal["algorithm_update_sec"], errors="coerce")
        same_runtime = total.notna() & update.notna() & ((total - update).abs() <= 1e-9)
        if not bool(same_runtime.all()):
            missing.append("total_sec equals prewarmed algorithm_update_sec")
            formal = formal[same_runtime]
    else:
        missing.append("total_sec equals prewarmed algorithm_update_sec")
        formal = formal.iloc[0:0]
    if {"method", "selection_charged_to_update"} <= set(formal.columns):
        agent_mask = formal["method"].astype(str).isin(AGENT_METHODS)
        agent_charged = formal.loc[agent_mask, "selection_charged_to_update"].astype(str).str.lower().isin({"1", "true", "yes"})
        if not bool(agent_charged.all()):
            missing.append("agent selection charged to update")
            drop_idx = formal.loc[agent_mask].index[~agent_charged]
            formal = formal.drop(index=drop_idx)
    else:
        missing.append("agent selection charged to update")
        formal = formal.iloc[0:0]
    if {"method", "selection_replay_used"} <= set(formal.columns):
        agent_mask = formal["method"].astype(str).isin(AGENT_METHODS)
        agent_replay = formal.loc[agent_mask, "selection_replay_used"].astype(str).str.lower().isin({"1", "true", "yes"})
        if bool(agent_replay.any()):
            missing.append("agent true selection not replay")
            drop_idx = formal.loc[agent_mask].index[agent_replay]
            formal = formal.drop(index=drop_idx)
    else:
        missing.append("agent true selection not replay")
        formal = formal.iloc[0:0]
    if {"method", "selection_engine"} <= set(formal.columns):
        agent_mask = formal["method"].astype(str).isin(AGENT_METHODS)
        agent_engine = formal.loc[agent_mask, "selection_engine"].astype(str)
        qwen_engine = agent_engine.eq("qwen")
        if not bool(qwen_engine.all()):
            missing.append("agent selection_engine=qwen")
            drop_idx = formal.loc[agent_mask].index[~qwen_engine]
            formal = formal.drop(index=drop_idx)
    else:
        missing.append("agent selection_engine=qwen")
        formal = formal.iloc[0:0]
    if "method" in formal.columns:
        agent_mask = formal["method"].astype(str).isin(AGENT_METHODS)
    else:
        agent_mask = pd.Series(False, index=formal.index)
    agent_checks = [
        ("llm_model_path", lambda s: s.astype(str).eq(REQUIRED_QWEN_7B_PATH), "agent llm_model_path=Qwen2.5-7B-Instruct"),
        ("llm_required_model_path", lambda s: s.astype(str).eq(REQUIRED_QWEN_7B_PATH), "agent llm_required_model_path=Qwen2.5-7B-Instruct"),
        ("llm_primary_required", lambda s: s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"}), "agent llm_primary_required=true"),
        ("llm_cuda_required", lambda s: s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"}), "agent llm_cuda_required=true"),
        ("llm_cuda_available", lambda s: s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"}), "agent llm_cuda_available=true"),
        ("llm_model_device", lambda s: s.map(device_is_cuda), "agent llm_model_device=cuda"),
        ("llm_fallback_used", lambda s: ~s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"}), "agent llm_fallback_used=false"),
        ("llm_fallback_allowed", lambda s: ~s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"}), "agent llm_fallback_allowed=false"),
    ]
    for col, predicate, reason in agent_checks:
        if col not in formal.columns:
            missing.append(reason)
            formal = formal.iloc[0:0]
            break
        if not bool(agent_mask.any()):
            continue
        ok = predicate(formal.loc[agent_mask, col])
        if not bool(ok.all()):
            missing.append(reason)
            drop_idx = formal.loc[agent_mask].index[~ok]
            formal = formal.drop(index=drop_idx)
    if {"method", "restart_type"} <= set(formal.columns):
        agent_mask = formal["method"].astype(str).isin(AGENT_METHODS)
        true_restart = formal.loc[agent_mask, "restart_type"].astype(str).eq("archive_backed_true_selection_readout")
        if not bool(true_restart.all()):
            missing.append("agent restart_type=archive_backed_true_selection_readout")
            drop_idx = formal.loc[agent_mask].index[~true_restart]
            formal = formal.drop(index=drop_idx)
    else:
        missing.append("agent restart_type=archive_backed_true_selection_readout")
        formal = formal.iloc[0:0]

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing.append(f"missing column {col}")
    if formal.empty:
        missing.append("formal ok timing rows")
    if "hardware" in formal.columns and formal["hardware"].nunique(dropna=True) != 1:
        missing.append("single hardware")
    required_k_set = {int(k) for k in required_k}
    k_set = set(pd.to_numeric(formal.get("candidate_count", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    if not required_k_set <= k_set:
        missing.append("required K values")
    if len(k_set) < len(required_k_set):
        missing.append(f">={len(required_k_set)} K values")
    method_set = set(formal.get("method", pd.Series(dtype=str)).astype(str).tolist())
    required_method_set = {str(m) for m in required_methods}
    if not required_method_set <= method_set:
        missing.append("required methods")
    min_repeats = 0
    if {"method", "candidate_count"} <= set(formal.columns) and not formal.empty:
        counts = formal.groupby(["method", "candidate_count"]).size()
        expected = [(m, k) for m in required_method_set for k in required_k_set]
        min_repeats = min((int(counts.get((m, k), 0)) for m, k in expected), default=0)
        if min_repeats < repeats:
            missing.append(">=3 repeats per method/K")
    else:
        missing.append("method/K grouping")
    if {"candidate_count", "selection_hash", "selected_model_ids"} <= set(formal.columns) and not formal.empty:
        for k, group in formal.groupby("candidate_count", dropna=False):
            hashes = {str(x) for x in group["selection_hash"].dropna().astype(str) if str(x).strip()}
            selected_ids = {str(x) for x in group["selected_model_ids"].dropna().astype(str) if str(x).strip()}
            if len(hashes) != 1 or len(selected_ids) != 1:
                missing.append(f"same selection hash/model ids for K={k}")
    else:
        missing.append("selection hash/model ids")
    strict_pass = not missing
    return {
        "strict_pass": strict_pass,
        "missing_reasons": sorted(set(missing)),
        "formal_rows": int(len(formal)),
        "k_values": sorted(k_set),
        "methods": sorted(method_set),
        "hardware": sorted(formal["hardware"].dropna().astype(str).unique().tolist()) if "hardware" in formal.columns else [],
        "repeats_per_cell_min": int(min_repeats),
        "median_iqr_available": bool(strict_pass),
    }


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke_rows(k_values: list[int], methods: list[str], repeats: int, hw: str, run_root: Path, out: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    started = finished = utc_now()
    log_dir = run_root / "stageB/runtime_k_sweep/logs"
    for k in k_values:
        selection = {
            "selection_path": str(selection_paths(run_root, k)["selection"]),
            "selection_hash": f"smoke_selection_hash_k{k}",
            "selected_model_ids": ",".join(f"model_{i}" for i in range(1, int(k) + 1)),
            "same_selection_per_k": True,
            "selection_scope": "runtime_formal_k_sweep_shared_topk",
            "selection_reused_across_methods": True,
            "uses_legacy_timing": False,
            "predictive_contract": "",
            "predictive_contract_source": "not_resolved_nonformal",
        }
        for method in methods:
            for repeat in range(repeats):
                log_path = log_dir / f"smoke_{method}_k{k}_rep{repeat}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("smoke schema row; not formal runtime scaling input\n", encoding="utf-8")
                rows.append(
                    {
                        "method": method,
                        "candidate_count": int(k),
                        "repeat": int(repeat),
                        "seed": int(42 + repeat),
                        "run_mode": "smoke",
                        "status": "smoke",
                        "total_sec": 0.001,
                        "algorithm_update_sec": 0.001,
                        "update_sec": 0.001,
                        "proposal_sec": 0.0,
                        "filter_sec": 0.001,
                        "forecast_sec": 0.0,
                        "hardware": hw,
                        "run_id": f"smoke__k{k}__rep{repeat}",
                        "started_at": started,
                        "finished_at": finished,
                        "log_path": str(log_path),
                        "artifact_reuse_proxy": False,
                        **selection,
                        "timing_semantics": "smoke_schema_only_not_formal",
                        "timing_mode": "archive_backed",
                        "formal_timing_valid": False,
                        "model_compute_sec": 0.0,
                        "forecast_source": "immutable_forecast_archive",
                        "selection_charged_to_update": False,
                        "selection_replay_used": False,
                        "selection_engine": "qwen" if is_agent_method(method) else "",
                        "llm_model_path": REQUIRED_QWEN_7B_PATH if is_agent_method(method) else "",
                        "llm_required_model_path": REQUIRED_QWEN_7B_PATH if is_agent_method(method) else "",
                        "llm_primary_required": is_agent_method(method),
                        "llm_cuda_required": is_agent_method(method),
                        "llm_cuda_available": is_agent_method(method),
                        "llm_model_device": "cuda:0" if is_agent_method(method) else "",
                        "llm_fallback_used": False,
                        "llm_fallback_allowed": False,
                        "restart_type": "",
                        "timing_regime": PREWARMED_TIMING_REGIME,
                        "process_startup_sec_included": False,
                        "llm_load_sec_included": False,
                        "process_total_sec": 0.001,
                    }
                )
    return rows


def cell_identity(
    *,
    method: str,
    k: int,
    repeat: int,
    seed: int,
    hardware: str,
    selection: dict[str, object],
    timing_inputs: dict[str, Path | str],
    restart_registry: Path,
    args: argparse.Namespace,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    bridge_key = "bridge_hierarchical" if method == "caster_hierarchical" else "bridge_one_layer"
    runner = {
        "caster_one_layer": NM_CODE_ROOT / "scripts/run_caster_from_archive.py",
        "caster_hierarchical": NM_CODE_ROOT / "scripts/run_hierarchical_from_archive.py",
        "agentic_top_one": BL_CODE_ROOT / "scripts/run_agentic_top_one.py",
        "agent_react": BL_CODE_ROOT / "scripts/run_react_agent.py",
        "agentic_full_recovery": BL_CODE_ROOT / "scripts/run_agentic_full_recovery.py",
    }[method]
    identity = {
        "schema_version": CELL_MANIFEST_SCHEMA,
        "timing_regime": PREWARMED_TIMING_REGIME,
        "method": method,
        "candidate_count": int(k),
        "repeat": int(repeat),
        "seed": int(seed),
        "hardware": hardware,
        "timing_task": str(timing_inputs["task_id"]),
        "selection_hash": str(selection["selection_hash"]),
        "selected_model_ids": str(selection["selected_model_ids"]),
        "ledger_sha256": sha256_file(Path(timing_inputs["ledger"])),
        "forecast_archive_sha256": sha256_file(Path(timing_inputs["archive"])),
        "registry_sha256": sha256_file(Path(timing_inputs["registry"])),
        "bridge_config_sha256": sha256_file(Path(timing_inputs[bridge_key])),
        "predictive_contract": str(timing_inputs["predictive_contract"]),
        "predictive_contract_source": str(
            timing_inputs["predictive_contract_source"]
        ),
        "full_recovery_manifest_sha256": sha256_file(Path(timing_inputs["full_recovery_manifest"])),
        "restart_registry_sha256": sha256_file(restart_registry),
        "runner_sha256": sha256_file(runner),
        "sweep_source_sha256": sha256_file(Path(__file__)),
        "source_hashes": source_hashes,
        "agent_archive_mode": str(args.agent_archive_mode),
        "agent_selection_mode": str(args.agent_selection_mode),
        "agent_selection_engine": str(args.agent_selection_engine),
        "selection_charged_to_update": bool(args.charge_selection or args.agent_selection_mode == "true"),
        "process_startup_sec_included": False,
        "llm_load_sec_included": False,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["identity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return identity


_AGENT_CELL_STABLE_IDENTITY_KEYS = (
    "schema_version",
    "timing_regime",
    "method",
    "candidate_count",
    "repeat",
    "seed",
    "hardware",
    "timing_task",
    "selection_hash",
    "selected_model_ids",
    "ledger_sha256",
    "forecast_archive_sha256",
    "registry_sha256",
    "full_recovery_manifest_sha256",
    "restart_registry_sha256",
    "runner_sha256",
    "agent_archive_mode",
    "agent_selection_mode",
    "agent_selection_engine",
    "selection_charged_to_update",
    "process_startup_sec_included",
    "llm_load_sec_included",
)


def agent_cell_identity_compatible(
    stored: dict[str, object], current: dict[str, object]
) -> bool:
    ""












    if not is_agent_method(stored.get("method", "")):
        return False
    if str(stored.get("method", "")) != str(current.get("method", "")):
        return False
    if any(stored.get(key) != current.get(key) for key in _AGENT_CELL_STABLE_IDENTITY_KEYS):
        return False
    stored_sources = stored.get("source_hashes", {})
    current_sources = current.get("source_hashes", {})
    if not isinstance(stored_sources, dict) or not isinstance(current_sources, dict):
        return False
    return stored_sources.get("baseline_python_tree_sha256") == current_sources.get(
        "baseline_python_tree_sha256"
    )


_CASTER_CELL_COMPATIBILITY_REQUIRED_KEYS = {
    "schema_version",
    "method",
    "candidate_count",
    "repeat",
    "seed",
    "timing_task",
    "selection_hash",
    "selected_model_ids",
    "ledger_sha256",
    "forecast_archive_sha256",
    "registry_sha256",
    "bridge_config_sha256",
    "predictive_contract",
    "full_recovery_manifest_sha256",
    "restart_registry_sha256",
    "runner_sha256",
    "source_hashes",
}


def caster_cell_identity_compatible(
    stored: dict[str, object], current: dict[str, object]
) -> bool:
    ""







    method = str(stored.get("method", ""))
    if method not in {"caster_one_layer", "caster_hierarchical"}:
        return False
    if method != str(current.get("method", "")):
        return False
    if not _CASTER_CELL_COMPATIBILITY_REQUIRED_KEYS <= set(stored):
        return False
    if not _CASTER_CELL_COMPATIBILITY_REQUIRED_KEYS <= set(current):
        return False
    if (
        stored.get("sweep_source_sha256")
        != PRE_CELL_LOCK_SWEEP_SOURCE_SHA256
    ):
        return False
    if not str(current.get("sweep_source_sha256", "")).strip():
        return False
    if current.get("sweep_source_sha256") == stored.get("sweep_source_sha256"):
        return False
    ignored = {"identity_sha256", "sweep_source_sha256"}
    stored_stable = {key: value for key, value in stored.items() if key not in ignored}
    current_stable = {
        key: value for key, value in current.items() if key not in ignored
    }
    return stored_stable == current_stable


def load_valid_cell(
    *,
    marker_path: Path,
    timing_path: Path,
    identity: dict[str, object],
    method: str,
) -> tuple[dict, str, str] | None:
    marker = read_json(marker_path)
    if marker.get("schema_version") != CELL_MANIFEST_SCHEMA:
        return None
    if marker.get("identity_sha256") != identity.get("identity_sha256"):
        stored_identity = marker.get("identity", {})
        compatible = bool(
            isinstance(stored_identity, dict)
            and (
                (
                    is_agent_method(method)
                    and agent_cell_identity_compatible(stored_identity, identity)
                )
                or (
                    method in {"caster_one_layer", "caster_hierarchical"}
                    and caster_cell_identity_compatible(stored_identity, identity)
                )
            )
        )
        if not compatible:
            return None
    if str(marker.get("status", "")).lower() != "ok":
        return None
    if not timing_path.exists() or marker.get("timing_sha256") != sha256_file(timing_path):
        return None
    if method == "agentic_full_recovery":
        artifact_sha256 = marker.get("timed_artifact_sha256", {})
        if not isinstance(artifact_sha256, dict):
            return None
        for name in FULL_RECOVERY_TIMED_ARTIFACTS:
            artifact_path = timing_path.parent / name
            if (
                not artifact_path.is_file()
                or str(artifact_sha256.get(name, ""))
                != sha256_file(artifact_path)
            ):
                return None
    timing = read_json(timing_path)
    if not timing:
        return None
    if is_agent_method(method):
        valid, _ = formal_archive_backed_agent_ok(timing, method=method)
        if not valid:
            return None
    else:
        names = ["hierarchical_filter", "hierarchical_forecast_readout"] if method == "caster_hierarchical" else ["sequential_filter", "forecast_readout"]
        update_sec = record_seconds(timing, ["score_update_rows", *names])
        if not math.isfinite(update_sec) or update_sec <= 0.0:
            return None
    started_at = str(marker.get("started_at", ""))
    finished_at = str(marker.get("finished_at", ""))
    if not started_at or not finished_at:
        return None
    return timing, started_at, finished_at


def load_external_agent_cell(
    *,
    cache_root: Path | None,
    k: int,
    repeat: int,
    method: str,
    identity: dict[str, object],
) -> tuple[dict, str, str, dict[str, object]] | None:
    ""









    if cache_root is None or not is_agent_method(method):
        return None
    resolved_root = Path(cache_root).resolve()
    cell_dir = resolved_root / f"K_{int(k)}" / f"rep{int(repeat)}" / method
    marker_path = cell_dir / "runtime_cell_manifest.json"
    timing_path = cell_dir / "timing.json"
    resumed = load_valid_cell(
        marker_path=marker_path,
        timing_path=timing_path,
        identity=identity,
        method=method,
    )
    if resumed is None:
        return None
    timing, started_at, finished_at = resumed
    provenance: dict[str, object] = {
        "resumed_from_external_agent_cache": True,
        "external_agent_cache_root": str(resolved_root),
        "external_cell_manifest_path": str(marker_path),
        "external_cell_manifest_sha256": sha256_file(marker_path),
        "external_timing_path": str(timing_path),
        "external_timing_sha256": sha256_file(timing_path),
    }
    return timing, started_at, finished_at, provenance


def no_external_agent_cache_provenance() -> dict[str, object]:
    return {
        "resumed_from_external_agent_cache": False,
        "external_agent_cache_root": "",
        "external_cell_manifest_path": "",
        "external_cell_manifest_sha256": "",
        "external_timing_path": "",
        "external_timing_sha256": "",
    }


def summarize_external_agent_cache(
    rows: pd.DataFrame, cache_root: Path | None
) -> dict[str, object]:
    ""

    resolved_root = str(Path(cache_root).resolve()) if cache_root is not None else ""
    resumed = rows.get(
        "resumed_from_external_agent_cache",
        pd.Series(False, index=rows.index, dtype=bool),
    )
    resumed_mask = resumed.astype(str).str.lower().isin({"1", "true", "yes"})
    source_columns = [
        "method",
        "candidate_count",
        "repeat",
        "external_cell_manifest_path",
        "external_cell_manifest_sha256",
        "external_timing_path",
        "external_timing_sha256",
    ]
    available_columns = [column for column in source_columns if column in rows.columns]
    sources = (
        rows.loc[resumed_mask, available_columns]
        .fillna("")
        .drop_duplicates()
        .to_dict(orient="records")
        if available_columns
        else []
    )
    external_caster_rows = int(
        (
            resumed_mask
            & ~rows.get("method", pd.Series("", index=rows.index))
            .astype(str)
            .isin(AGENT_METHODS)
        ).sum()
    )
    return {
        "agent_cell_cache_enabled": cache_root is not None,
        "agent_cell_cache_root": resolved_root,
        "agent_cell_cache_read_only": True,
        "external_agent_cache_resumed_rows": int(resumed_mask.sum()),
        "external_agent_cache_sources": sources,
        "external_caster_cache_resumed_rows": external_caster_rows,
    }


def save_valid_cell(
    *,
    marker_path: Path,
    timing_path: Path,
    identity: dict[str, object],
    started_at: str,
    finished_at: str,
) -> None:
    timed_artifact_sha256 = {
        name: sha256_file(timing_path.parent / name)
        for name in FULL_RECOVERY_TIMED_ARTIFACTS
        if (timing_path.parent / name).is_file()
    }
    write_json_atomic(
        marker_path,
        {
            "schema_version": CELL_MANIFEST_SCHEMA,
            "identity_sha256": identity["identity_sha256"],
            "identity": identity,
            "timing_path": str(timing_path),
            "timing_sha256": sha256_file(timing_path),
            "timed_artifact_sha256": timed_artifact_sha256,
            "started_at": started_at,
            "finished_at": finished_at,
            "validated_at_utc": utc_now(),
            "status": "ok",
        },
    )


def run_or_resume_local_cell(
    *,
    marker_path: Path,
    timing_path: Path,
    identity: dict[str, object],
    method: str,
    lock_path: Path,
    lock_owner: dict[str, object],
    execute: Callable[[], tuple[int, bool]],
    accept_timing: Callable[[int, dict], bool],
) -> dict[str, object]:
    ""

                                                                              
                                                          
    resumed = load_valid_cell(
        marker_path=marker_path,
        timing_path=timing_path,
        identity=identity,
        method=method,
    )
    if resumed is not None:
        timing, started_at, finished_at = resumed
        return {
            "timing": timing,
            "started_at": started_at,
            "finished_at": finished_at,
            "code": 0,
            "timed_out": False,
            "resumed_from_valid_cell": True,
        }

    with runtime_cell_lock(lock_path=lock_path, owner=lock_owner) as lock_metadata:
                                                                           
                                                               
        resumed = load_valid_cell(
            marker_path=marker_path,
            timing_path=timing_path,
            identity=identity,
            method=method,
        )
        if resumed is not None:
            timing, started_at, finished_at = resumed
            lock_metadata["cell_outcome"] = "resumed_after_lock_wait"
            return {
                "timing": timing,
                "started_at": started_at,
                "finished_at": finished_at,
                "code": 0,
                "timed_out": False,
                "resumed_from_valid_cell": True,
            }

        started_at = utc_now()
        code, timed_out = execute()
        finished_at = utc_now()
        timing = read_json(timing_path)
        accepted = bool(accept_timing(code, timing))
        lock_metadata["cell_outcome"] = (
            "computed_valid" if accepted else "computed_invalid"
        )
        lock_metadata["command_exit_code"] = int(code)
        lock_metadata["command_timed_out"] = bool(timed_out)
        if accepted:
            save_valid_cell(
                marker_path=marker_path,
                timing_path=timing_path,
                identity=identity,
                started_at=started_at,
                finished_at=finished_at,
            )
        return {
            "timing": timing,
            "started_at": started_at,
            "finished_at": finished_at,
            "code": int(code),
            "timed_out": bool(timed_out),
            "resumed_from_valid_cell": False,
        }


def formal_rows(args: argparse.Namespace, run_root: Path, k_values: list[int], methods: list[str], hw: str) -> list[dict[str, object]]:
    base_work = run_root / "stageB/runtime_k_sweep"
    logs = base_work / "logs"
    rows: list[dict[str, object]] = []
    timing_inputs = resolve_timing_task_paths(run_root, args)
    ledger_b = Path(timing_inputs["ledger"])
    archive_b = Path(timing_inputs["archive"])
    registry_csv = Path(timing_inputs["registry"])
    bridge_one = Path(timing_inputs["bridge_one_layer"])
    bridge_hier = Path(timing_inputs["bridge_hierarchical"])
    predictive_contract = str(timing_inputs["predictive_contract"])
    predictive_contract_source = str(timing_inputs["predictive_contract_source"])
    full_recovery_manifest = Path(timing_inputs["full_recovery_manifest"])
    candidate_pool_size = int(
        getattr(
            args,
            "candidate_pool_size",
            FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT,
        )
    )
    excluded_model_ids = parse_optional_csv_text(getattr(args, "excluded_model_ids", ""))
    update_splits = caster_update_splits(ledger_b)
    dataset_key = str(timing_inputs["task_id"])
    task_label = str(timing_inputs.get("task_label", dataset_key))
    timeout = None if args.wall_clock_budget_sec is None or args.wall_clock_budget_sec <= 0 else float(args.wall_clock_budget_sec)
    agent_selection_mode = str(getattr(args, "agent_selection_mode", "true"))
    charge_agent_selection = bool(args.charge_selection or agent_selection_mode == "true")
    source_hashes = {
        "new_method_python_tree_sha256": sha256_python_tree(NM_CODE_ROOT / "src"),
        "baseline_python_tree_sha256": sha256_python_tree(BL_CODE_ROOT / "src"),
    }
    repeat_values = formal_repeat_values(args)
    persistent_agent_worker = bool(getattr(args, "persistent_agent_worker", False))
                                                                               
                                                                           
                   
    persistent_agent_engine = None

    for k in k_values:
        k_dir = runtime_k_dir(run_root, k)
        selection, selection_log, selection_code, selection_timed_out = run_shared_topk_selection(
            run_root,
            k,
            int(args.seed_base),
            timeout=timeout,
            formal_ranking=Path(args.formal_selection_ranking).resolve() if args.formal_selection_ranking else None,
            expected_candidate_count=candidate_pool_size,
            excluded_model_ids=excluded_model_ids,
            candidate_pool_profile=str(args.candidate_pool_profile),
            selection_input_hash=str(getattr(args, "selection_input_hash", "")),
            reuse_existing_selection=bool(getattr(args, "reuse_existing_selections", False)),
        )
        if selection is None:
            started = finished = utc_now()
            for repeat in repeat_values:
                seed = int(args.seed_base + repeat)
                for method in methods:
                    rows.append(
                        blocked_row(
                            method=method,
                            k=k,
                            repeat=repeat,
                            seed=seed,
                            hw=hw,
                            run_id=f"R5__{method}__k{k}__rep{repeat}__seed{seed}",
                            started_at=started,
                            finished_at=finished,
                            log_path=selection_log,
                            blocker="shared_topk_selection_failed_or_timed_out",
                        )
                    )
            continue
        selected_ids = str(selection["selected_model_ids"]).split(",")
        restart_registry = write_restart_registry(
            run_root,
            selected_ids,
            k_dir / "selection/restart_registry_topk.csv",
            registry_path=registry_csv,
        )
        selection = {
            **selection,
            "timing_task": dataset_key,
            "timing_task_label": task_label,
            "single_task_timing": True,
            "pooled_timing_used": False,
            "agent_selection_scope": dataset_key,
            "predictive_contract": predictive_contract,
            "predictive_contract_source": predictive_contract_source,
        }
        for repeat in repeat_values:
            seed = int(args.seed_base + repeat)
            for method in methods:
                run_id = f"R5__{method}__k{k}__rep{repeat}__seed{seed}"
                identity = cell_identity(
                    method=method,
                    k=k,
                    repeat=repeat,
                    seed=seed,
                    hardware=hw,
                    selection=selection,
                    timing_inputs=timing_inputs,
                    restart_registry=restart_registry,
                    args=args,
                    source_hashes=source_hashes,
                )
                if method in {"caster_one_layer", "caster_hierarchical"}:
                    out_dir = k_dir / f"rep{repeat}/{method}"
                    script = "run_caster_from_archive.py" if method == "caster_one_layer" else "run_hierarchical_from_archive.py"
                    timing_name = "timing.json" if method == "caster_one_layer" else "hierarchical_timing.json"
                    timing_path = out_dir / timing_name
                    marker_path = out_dir / "runtime_cell_manifest.json"
                    bridge_config = bridge_one if method == "caster_one_layer" else bridge_hier
                    log_path = logs / f"{method}_k{k}_rep{repeat}.log"
                    cmd = [
                        sys.executable,
                        str(NM_CODE_ROOT / "scripts" / script),
                        "--ledger",
                        str(ledger_b),
                        "--archive",
                        str(archive_b),
                        "--registry",
                        str(registry_csv),
                        "--selection",
                        str(selection["selection_path"]),
                        "--bridge-config",
                        str(bridge_config),
                        *caster_predictive_contract_args(timing_inputs),
                        "--out",
                        str(out_dir),
                        "--update-splits",
                        update_splits,
                        "--readout-split",
                        "test",
                        "--seed",
                        str(seed),
                        *caster_task_scope_args(dataset_key),
                    ]
                    caster_timing_names = (
                        ["score_update_rows", "hierarchical_filter", "hierarchical_forecast_readout"]
                        if method == "caster_hierarchical"
                        else ["score_update_rows", "sequential_filter", "forecast_readout"]
                    )
                    result = run_or_resume_local_cell(
                        marker_path=marker_path,
                        timing_path=timing_path,
                        identity=identity,
                        method=method,
                        lock_path=out_dir / "runtime_cell.lock",
                        lock_owner={
                            "run_id": run_id,
                            "method": method,
                            "candidate_count": int(k),
                            "repeat": int(repeat),
                            "seed": int(seed),
                            "identity_sha256": str(identity["identity_sha256"]),
                            "cell_dir": str(out_dir.resolve()),
                        },
                        execute=lambda cmd=cmd, log_path=log_path: run_logged(
                            cmd,
                            cwd=ROOT,
                            log_path=log_path,
                            timeout=timeout,
                        ),
                        accept_timing=lambda code, timing, names=caster_timing_names: bool(
                            code == 0
                            and timing
                            and math.isfinite(record_seconds(timing, names))
                            and record_seconds(timing, names) > 0.0
                        ),
                    )
                    timing = dict(result["timing"])
                    started = str(result["started_at"])
                    finished = str(result["finished_at"])
                    resumed_from_valid_cell = bool(
                        result["resumed_from_valid_cell"]
                    )
                    code = int(result["code"])
                    timed_out = bool(result["timed_out"])
                    status = (
                        "ok"
                        if resumed_from_valid_cell or (code == 0 and bool(timing))
                        else ("partial" if timed_out else "blocked")
                    )
                    row = caster_row_from_timing(
                        method=method,
                        k=k,
                        repeat=repeat,
                        seed=seed,
                        run_mode="formal",
                        status=status,
                        timing=timing,
                        hw=hw,
                        run_id=run_id,
                        started_at=started,
                        finished_at=finished,
                        log_path=log_path,
                        artifact_reuse_proxy=False,
                        selection=selection,
                    )
                    row["resumed_from_valid_cell"] = resumed_from_valid_cell
                    row["cell_manifest_path"] = str(marker_path)
                    row.update(no_external_agent_cache_provenance())
                    rows.append(row)
                    if resumed_from_valid_cell:
                        print(f"runtime_cell=resume method={method} K={k} repeat={repeat}")
                else:
                    agent_scripts = {
                        "agentic_top_one": "run_agentic_top_one.py",
                        "agent_react": "run_react_agent.py",
                        "agentic_full_recovery": "run_agentic_full_recovery.py",
                    }
                    out_dir = k_dir / f"rep{repeat}/{method}"
                    timing_path = out_dir / "timing.json"
                    marker_path = out_dir / "runtime_cell_manifest.json"
                    log_path = logs / f"{method}_k{k}_rep{repeat}.log"
                    resumed = load_valid_cell(
                        marker_path=marker_path,
                        timing_path=timing_path,
                        identity=identity,
                        method=method,
                    )
                    external_provenance: dict[str, object] | None = None
                    if resumed is None:
                        external = load_external_agent_cell(
                            cache_root=getattr(args, "agent_cell_cache_root", None),
                            k=k,
                            repeat=repeat,
                            method=method,
                            identity=identity,
                        )
                        if external is not None:
                            timing, started, finished, external_provenance = external
                            resumed = timing, started, finished
                    if resumed is not None:
                        timing, started, finished = resumed
                        result = {
                            "timing": timing,
                            "started_at": started,
                            "finished_at": finished,
                            "code": 0,
                            "timed_out": False,
                            "resumed_from_valid_cell": True,
                        }
                    else:
                        def execute_agent_cell() -> tuple[int, bool]:
                            nonlocal persistent_agent_engine
                            if persistent_agent_worker:
                                if persistent_agent_engine is None:
                                    persistent_agent_engine = build_persistent_agent_engine(
                                        methods
                                    )
                                return run_persistent_agent_cell(
                                    method=method,
                                    engine=persistent_agent_engine,
                                    manifest_path=full_recovery_manifest,
                                    registry_path=restart_registry,
                                    out_dir=out_dir,
                                    dataset_key=dataset_key,
                                    forecast_archive=archive_b,
                                    charge_selection=charge_agent_selection,
                                    log_path=log_path,
                                )
                            cmd = [
                                str(BL_PYTHON if BL_PYTHON.exists() else Path(sys.executable)),
                                str(BL_CODE_ROOT / "scripts" / agent_scripts[method]),
                                "--manifest",
                                str(full_recovery_manifest),
                                "--registry",
                                str(restart_registry),
                                "--out",
                                str(out_dir),
                                "--dataset-key",
                                dataset_key,
                                "--forecast-archive",
                                str(archive_b),
                                "--archive-mode",
                                str(args.agent_archive_mode),
                            ]
                            if method == "agentic_full_recovery":
                                cmd.extend(
                                    [
                                        "--selection-engine",
                                        str(args.agent_selection_engine),
                                    ]
                                )
                            cmd.append("--exclude-llm-load-from-timing")
                            if agent_selection_mode == "replay":
                                replay_log = write_agent_selection_replay_log(
                                    manifest_path=full_recovery_manifest,
                                    dataset_key=dataset_key,
                                    selected_model_id=selected_ids[0],
                                    out=k_dir
                                    / f"rep{repeat}/selection/{method}_selection_replay.csv",
                                )
                                cmd.extend(
                                    ["--selection-replay-log", str(replay_log)]
                                )
                            if charge_agent_selection:
                                cmd.append("--charge-selection")
                            return run_logged(
                                cmd,
                                cwd=ROOT,
                                log_path=log_path,
                                timeout=timeout,
                            )

                        result = run_or_resume_local_cell(
                            marker_path=marker_path,
                            timing_path=timing_path,
                            identity=identity,
                            method=method,
                            lock_path=out_dir / "runtime_cell.lock",
                            lock_owner={
                                "run_id": run_id,
                                "method": method,
                                "candidate_count": int(k),
                                "repeat": int(repeat),
                                "seed": int(seed),
                                "identity_sha256": str(
                                    identity["identity_sha256"]
                                ),
                                "cell_dir": str(out_dir.resolve()),
                            },
                            execute=execute_agent_cell,
                            accept_timing=lambda code, timing: bool(
                                code == 0
                                and formal_archive_backed_agent_ok(
                                    timing, method=method
                                )[0]
                            ),
                        )
                    timing = dict(result["timing"])
                    started = str(result["started_at"])
                    finished = str(result["finished_at"])
                    code = int(result["code"])
                    timed_out = bool(result["timed_out"])
                    resumed_from_valid_cell = bool(
                        result["resumed_from_valid_cell"]
                    )
                    ok_agent, blocker = formal_archive_backed_agent_ok(
                        timing, method=method
                    )
                    ok_agent = bool(code == 0 and ok_agent)
                    status = "ok" if ok_agent else ("partial" if timed_out else "blocked")
                    row = agent_row_from_timing(
                        method=method,
                        k=k,
                        repeat=repeat,
                        seed=seed,
                        run_mode="formal",
                        status=status,
                        timing=timing,
                        hw=hw,
                        run_id=run_id,
                        started_at=started,
                        finished_at=finished,
                        log_path=log_path,
                        artifact_reuse_proxy=not ok_agent,
                        selection=selection,
                    )
                    row["resumed_from_valid_cell"] = resumed_from_valid_cell
                    if external_provenance is None:
                        row["cell_manifest_path"] = str(marker_path)
                        row.update(no_external_agent_cache_provenance())
                    else:
                        row["cell_manifest_path"] = str(
                            external_provenance["external_cell_manifest_path"]
                        )
                        row.update(external_provenance)
                    rows.append(row)
                    if resumed_from_valid_cell:
                        resume_source = (
                            "external_agent_cache"
                            if external_provenance is not None
                            else "current_run"
                        )
                        print(
                            f"runtime_cell=resume source={resume_source} "
                            f"method={method} K={k} repeat={repeat}"
                        )
                    if not ok_agent and rows[-1].get("blocker", "") == "":
                        rows[-1]["blocker"] = (
                            blocker or "archive_backed_agent_timing_invalid"
                        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run formal runtime scaling runtime K-sweep")
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--k-values", type=parse_csv_ints, required=True)
    ap.add_argument("--methods", type=parse_csv_text, default=",".join(DEFAULT_METHODS))
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata-out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wall-clock-budget-sec", type=float, default=None)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument(
        "--repeat-offset",
        type=int,
        default=0,
        help="Zero-based repeat index offset; supports conflict-free multi-GPU sharding.",
    )
    ap.add_argument("--timing-task", default="benchmark_b_covid", help="Task-local Benchmark B artifact directory to use when flat benchmark_b artifacts are absent.")
    ap.add_argument("--task-artifact-root", type=Path, default=None, help="Explicit task artifact root containing forecast_archive and bridge configs.")
    ap.add_argument("--ledger", type=Path, default=None, help="Explicit event ledger for timing replay.")
    ap.add_argument("--forecast-archive", type=Path, default=None)
    ap.add_argument("--model-registry", type=Path, default=None, help="Explicit registry matching the eligible runtime scaling candidate pool.")
    ap.add_argument("--full-recovery-manifest", type=Path, default=None, help="Explicit archive-backed agent manifest for the timing task.")
    ap.add_argument(
        "--candidate-pool-size",
        type=int,
        default=FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT,
    )
    ap.add_argument(
        "--candidate-pool-profile",
        default=FORMAL_RESULT_CANDIDATE_PROFILE,
    )
    ap.add_argument(
        "--selection-input-hash",
        default="",
        help=(
            "Optional immutable selection-bank identity. It is recorded in "
            "every K-prefix metadata file and must match for selection reuse."
        ),
    )
    ap.add_argument(
        "--excluded-model-ids",
        default=",".join(FORMAL_RESULT_EXCLUDED_MODEL_IDS),
        help="Comma-separated model ids that must not occur in the formal ranking.",
    )
    ap.add_argument(
        "--formal-selection-ranking",
        type=Path,
        default=None,
        help="Immutable task-specific full-bank ranking; every K uses its deterministic prefix.",
    )
    ap.add_argument(
        "--reuse-existing-selections",
        action="store_true",
        help="Reuse a materialized K-prefix only after ranking/hash/pool guards pass.",
    )
    ap.add_argument(
        "--persistent-agent-worker",
        action="store_true",
        help=(
            "Run exactly one agent method in-process so Qwen and immutable causal-context "
            "preprocessing remain prewarmed across timing cells."
        ),
    )
    ap.add_argument(
        "--agent-cell-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional read-only path to another stageB/runtime_k_sweep root. "
            "Only validated, identity-compatible agent timing cells may be "
            "resumed from it; CASTER cells are never externally reused."
        ),
    )
    ap.add_argument("--agent-archive-mode", choices=["required"], default="required")
    ap.add_argument(
        "--agent-selection-mode",
        choices=["true", "replay"],
        default="true",
        help="Formal default is true per-origin agent selection/control; replay is diagnostic only.",
    )
    ap.add_argument(
        "--agent-selection-engine",
        choices=["qwen", "deterministic_no_model_compute"],
        default="qwen",
        help="Formal default is qwen. deterministic_no_model_compute is diagnostic and excludes LLM inference.",
    )
    ap.add_argument("--charge-selection", action="store_true")
    ap.add_argument("--include-agent-react", action="store_true", help="Compatibility flag; agent_react is formal by default.")
    args = ap.parse_args(argv)

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if int(args.repeat_offset) < 0:
        raise SystemExit("--repeat-offset must be non-negative")
    if int(args.candidate_pool_size) <= 0:
        raise SystemExit("--candidate-pool-size must be positive")
    if str(args.timing_task) not in RUNTIME_SINGLE_TASKS:
        raise SystemExit(f"--timing-task must be a single Benchmark B task ({','.join(sorted(RUNTIME_SINGLE_TASKS))}); got {args.timing_task!r}")
    run_root = args.run_root.resolve()
    if args.agent_cell_cache_root is not None:
        args.agent_cell_cache_root = args.agent_cell_cache_root.resolve()
        if not args.agent_cell_cache_root.is_dir():
            raise SystemExit(
                "--agent-cell-cache-root must be an existing runtime_k_sweep "
                f"directory: {args.agent_cell_cache_root}"
            )
    out = args.out.resolve()
    if args.metadata_out:
        metadata_out = args.metadata_out.resolve()
    elif out.name == "timing_scaling.csv":
        metadata_out = out.with_name("runtime_runtime_scaling_metadata.json")
    else:
        metadata_out = out.with_name(f"{out.stem}_metadata.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    k_values = list(args.k_values)
    if len(set(k_values)) != len(k_values):
        raise SystemExit("--k-values must not contain duplicates")
    if max(k_values) > int(args.candidate_pool_size):
        raise SystemExit(
            f"maximum K={max(k_values)} exceeds --candidate-pool-size={args.candidate_pool_size}"
        )
    methods = list(args.methods)
    if args.persistent_agent_worker:
        if args.smoke or args.dry_run:
            raise SystemExit("--persistent-agent-worker is formal-only")
        if methods not in (["agentic_top_one"], ["agent_react"], ["agentic_full_recovery"]):
            raise SystemExit("--persistent-agent-worker requires exactly one agent method")
        if args.agent_selection_mode != "true" or args.agent_selection_engine != "qwen":
            raise SystemExit("--persistent-agent-worker requires true primary-only Qwen selection")
    hw = hardware_id()
    before_hashes = hash_manifest(run_root)
    started_at = utc_now()

    if args.dry_run:
        rows = run_smoke_rows(k_values, methods, int(args.repeats), hw, run_root, out)
        for row in rows:
            row["run_mode"] = "dry_run"
            row["status"] = "planned"
            row["timing_semantics"] = "dry_run_command_plan_only_not_formal"
    elif args.smoke:
        rows = run_smoke_rows(k_values, methods, int(args.repeats), hw, run_root, out)
    else:
        rows = formal_rows(args, run_root, k_values, methods, hw)

    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)

    after_hashes = hash_manifest(run_root)
    hash_guard_unchanged = before_hashes == after_hashes
    evaluation = evaluate_timing_scaling(
        df,
        required_k=k_values if not args.smoke and not args.dry_run else DEFAULT_RUNTIME_K_GRID,
        required_methods=methods if not args.smoke and not args.dry_run else DEFAULT_METHODS,
        repeats=int(args.repeats) if not args.smoke and not args.dry_run else 3,
    )
    if args.smoke or args.dry_run:
        evaluation["strict_pass"] = False
        evaluation["median_iqr_available"] = False
        evaluation["missing_reasons"] = sorted(set(evaluation["missing_reasons"]) | {"not formal run"})
    if not hash_guard_unchanged:
        evaluation["strict_pass"] = False
        evaluation["median_iqr_available"] = False
        evaluation["missing_reasons"] = sorted(set(evaluation["missing_reasons"]) | {"canonical source hash changed"})

    effective_agent_selection_charged = bool(args.charge_selection or args.agent_selection_mode == "true")
    agent_engine = str(args.agent_selection_engine) if args.agent_selection_mode == "true" else "selection_replay"
    agent_load_excluded = bool(args.agent_selection_mode == "true" and agent_engine == "qwen")
    agent_semantics = (
        "true per-origin Qwen selection/control charged to update plus immutable forecast archive lookup/readout; "
        "one-time Qwen model load and candidate model compute excluded"
        if agent_engine == "qwen"
        else "true per-origin deterministic diagnostic selection/control plus immutable forecast archive lookup/readout; model compute excluded"
    )
    metadata_timing_task = str(args.timing_task)
    metadata_timing_task_label = RUNTIME_TASK_LABELS.get(metadata_timing_task, metadata_timing_task)
    _timing_inputs_for_metadata: dict[str, Path | str] | None = None
    if not args.smoke and not args.dry_run:
        try:
            _timing_inputs_for_metadata = resolve_timing_task_paths(run_root, args)
            metadata_timing_task = str(_timing_inputs_for_metadata["task_id"])
            metadata_timing_task_label = str(_timing_inputs_for_metadata.get("task_label", metadata_timing_task))
        except Exception:
            pass
    cache_metadata = summarize_external_agent_cache(
        df, getattr(args, "agent_cell_cache_root", None)
    )
    metadata = {
        "status": "PASS" if evaluation["strict_pass"] else "PENDING_DATA",
        "strict_pass": bool(evaluation["strict_pass"]),
        "k_values": k_values,
        "methods": methods,
        "repeats_per_cell": int(args.repeats),
        "repeat_offset": int(args.repeat_offset),
        "persistent_agent_worker_requested": bool(args.persistent_agent_worker),
        "persistent_agent_worker_rows": int(
            df.get("persistent_agent_worker", pd.Series(dtype=object))
            .astype(str)
            .str.lower()
            .isin({"1", "true", "yes"})
            .sum()
        ),
        "hardware": [hw],
        "timing_task": metadata_timing_task,
        "timing_task_label": metadata_timing_task_label,
        "single_task_timing": True,
        "pooled_timing_used": False,
        "predictive_contract": (
            str(_timing_inputs_for_metadata["predictive_contract"])
            if _timing_inputs_for_metadata is not None
            else ""
        ),
        "predictive_contract_source": (
            str(_timing_inputs_for_metadata["predictive_contract_source"])
            if _timing_inputs_for_metadata is not None
            else "not_resolved_nonformal"
        ),
        "median_iqr_available": bool(evaluation["median_iqr_available"]),
        "timing_regime": PREWARMED_TIMING_REGIME,
        "process_startup_sec_included": False,
        "llm_load_sec_included": False,
        "timing_semantics": {
            "caster_one_layer": "incremental posterior update + readout on frozen forecast archive and frozen bridge",
            "caster_hierarchical": "incremental posterior update + readout on frozen forecast archive and frozen bridge",
            "agentic_top_one": agent_semantics,
            "agent_react": agent_semantics,
            "agentic_full_recovery": agent_semantics,
        },
        "caster_timing_scope": "incremental_update_readout_on_frozen_archive_bridge",
        "full_recovery_timing_scope": "archive_backed_algorithm_update_readout",
        "agent_timing_scope": "archive_backed_algorithm_update_readout",
        "model_compute_excluded_for_all_methods": True,
        "rho_gamma_grid_selection_sec_included": False,
        "bridge_calibration_sec_included": False,
        "archive_construction_sec_included": False,
        "agent_selection_mode": str(args.agent_selection_mode),
        "agent_selection_engine": agent_engine,
        "agent_qwen_generation_charged_to_update": bool(agent_engine == "qwen" and args.agent_selection_mode == "true"),
        "agent_llm_load_sec_included_in_update": False,
        "agent_llm_load_sec_excluded_from_update": agent_load_excluded,
        "llm_required_model_path": REQUIRED_QWEN_7B_PATH,
        "llm_primary_required": bool(agent_engine == "qwen" and args.agent_selection_mode == "true"),
        "llm_cuda_required": bool(agent_engine == "qwen" and args.agent_selection_mode == "true"),
        "llm_fallback_allowed": False,
        "selection_charged_to_update": effective_agent_selection_charged,
        "selection_source": str(run_root / "stageB/runtime_k_sweep/K_<K>/selection/shared_topk/candidate_selection_log.csv"),
        "selection_rule": "stageB_shared_topk_per_K",
        "candidate_pool_size": int(args.candidate_pool_size),
        "candidate_pool_profile": str(args.candidate_pool_profile),
        "selection_input_hash": str(getattr(args, "selection_input_hash", "")),
        "excluded_model_ids": parse_optional_csv_text(args.excluded_model_ids),
        "formal_selection_ranking": str(Path(args.formal_selection_ranking).resolve()) if args.formal_selection_ranking else "",
        "formal_selection_ranking_sha256": sha256_file(Path(args.formal_selection_ranking).resolve()) if args.formal_selection_ranking else "",
        "same_selection_per_k": True,
        "uses_legacy_timing": False,
        "no_performance_based_selection": True,
        "runtime_formal_filter": "run_mode=formal AND status=ok AND artifact_reuse_proxy=false AND uses_legacy_timing=false AND same_selection_per_k=true AND formal_timing_valid=true AND model_compute_sec=0 AND timing_mode=archive_backed AND forecast_source=immutable_forecast_archive AND timing_regime=prewarmed_online_update_readout AND process_startup_sec_included=false AND llm_load_sec_included=false AND total_sec=algorithm_update_sec AND agent rows have selection_replay_used=false AND selection_charged_to_update=true AND selection_engine=qwen AND llm_model_path=Qwen2.5-7B-Instruct AND llm_cuda_required=true AND llm_model_device=cuda AND llm_fallback_used=false AND restart_type=archive_backed_true_selection_readout",
        "runtime_timing_basis": "prewarmed_archive_backed_algorithm_update_readout_for_all_methods",
        "formal_k_grid_min_count": len(k_values),
        "required_k_grid": k_values,
        "canonical_hashes_before": before_hashes,
        "canonical_hashes_after": after_hashes,
        "canonical_hashes_unchanged": hash_guard_unchanged,
        "evaluation": evaluation,
        "started_at": started_at,
        "finished_at": utc_now(),
        "out": str(out),
        **cache_metadata,
    }
    if not args.smoke and not args.dry_run:
        try:
            timing_inputs = (
                _timing_inputs_for_metadata
                if _timing_inputs_for_metadata is not None
                else resolve_timing_task_paths(run_root, args)
            )
            metadata.update(
                {
                    "timing_task_id": str(timing_inputs["task_id"]),
                    "timing_task_artifact_root": str(timing_inputs["task_root"]),
                    "timing_ledger": str(timing_inputs["ledger"]),
                    "timing_forecast_archive": str(timing_inputs["archive"]),
                    "timing_model_registry": str(timing_inputs["registry"]),
                    "timing_bridge_config_one_layer": str(timing_inputs["bridge_one_layer"]),
                    "timing_bridge_config_hierarchical": str(timing_inputs["bridge_hierarchical"]),
                    "timing_bridge_config_one_layer_sha256": sha256_file(
                        Path(timing_inputs["bridge_one_layer"])
                    ),
                    "timing_bridge_config_hierarchical_sha256": sha256_file(
                        Path(timing_inputs["bridge_hierarchical"])
                    ),
                    "timing_full_recovery_manifest": str(timing_inputs["full_recovery_manifest"]),
                    "caster_update_splits": caster_update_splits(Path(timing_inputs["ledger"])),
                }
            )
        except Exception as exc:
            metadata["timing_input_resolution_error"] = f"{type(exc).__name__}: {exc}"
    write_metadata(metadata_out, metadata)
    print(json.dumps({"out": str(out), "metadata_out": str(metadata_out), **evaluation}, indent=2, sort_keys=True))
    return 0 if args.smoke or args.dry_run or evaluation["strict_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
