#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from formal_candidate_bank import (
    FORMAL_ARCHIVE_NAME,
    FORMAL_CANDIDATE_COUNT,
    FORMAL_SELECTION_DIR,
    cache_tag,
    formal_candidate_model_ids,
)
from model_pool_contract import (
    CANONICAL_SHARED_MODEL_OBJECTS,
    SHARED_BASELINE_METHODS,
    SHARED_FORECAST_PATHS,
    model_role_fields,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "code/baseline"
CASTER_ROOT = ROOT / "code/caster"
BASELINE_SCRIPT = BASELINE_ROOT / "scripts/run_real_full_baselines_all.sh"
ARCHIVE_BUILDER = CASTER_ROOT / "scripts/build_selected_forecast_archive.py"
REGISTRY_VALIDATOR = CASTER_ROOT / "scripts/validation_model_registry.py"
EMBED_DESCRIPTIONS = CASTER_ROOT / "scripts/embed_model_descriptions.py"
SELECTION_BUILDER = ROOT / "scripts/run_stageB_topk_selection.py"
REGISTRY_SOURCE = CASTER_ROOT / "configs/model_registry.yaml"
DATA_A = ROOT / "data/benchmark_a/curated_full_v3_direct_rollout7"
DATA_B = ROOT / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled"
TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
FORMAL_AGENT_ELIGIBLE_COUNT = FORMAL_CANDIDATE_COUNT
FORMAL_AGENT_EXCLUDED_MODEL_IDS: tuple[str, ...] = ()
QWEN_SELECTION_PROFILES = {
    "qwen25_structured_context_v1",
    "qwen25_multiscale_released_sequence_v1",
}
SUPPORTED_SELECTION_PROFILES = {
    "formal_27_country_macro_v1",
    *QWEN_SELECTION_PROFILES,
}
QWEN_RANKING_NAME = "candidate_ranking_all27.csv"
BASE_RANKING_NAME = "candidate_selection_all27.csv"
QWEN_TOP10_NAME = "top10_selection.csv"
DEFAULT_WORKERS = "8"
DEFAULT_CUDA_A = "3"
DEFAULT_CUDA_B = "2"
DEFAULT_CUDA_B_FLU = "1"
LOCAL_MODELS = [
    "covariate_drift",
    "sir_tau",
    "seir_tau",
    "seirs_tau",
    "tv_seir_rt",
    "renewal_rt",
    "local_level",
    "covariate_dynamic_linear_trend",
    "particle_local_level",
    "drift",
    "rnn_simple",
    "gru_style",
    "lstm_style",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_files(paths: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(
                item.resolve()
                for item in path.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix in {".py", ".sh", ".yaml", ".yml"}
            )
    return sorted(files, key=str)


def source_identity(paths: Iterable[Path], schema: str) -> dict[str, object]:
    files = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_files(paths)
    }
    payload: dict[str, object] = {"schema": schema, "files": files}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def run_checked(command: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(command))
    code = subprocess.run(command, cwd=ROOT, env=env).returncode
    if code:
        raise SystemExit(code)


def runtime_python() -> str:
    configured = os.environ.get("PHASE20_PYTHON", "").strip()
    return configured or sys.executable


def baseline_paths(shared: Path) -> dict[str, Path]:
    base = shared / "baseline"
    results = base / "results"
    checks = base / "checks"
    return {
        "run_root": base / "runs",
        "manifest": base / "data/full_manifest.csv",
        "results": results,
        "metrics": results / "baseline_metrics.csv",
        "metric_slices": results / "baseline_metric_slices.csv",
        "run_manifest": results / "baseline_run_manifest.csv",
        "summary": checks / "baseline_summary.md",
        "ledger_check": checks / "event_ledger_validation.md",
        "split_check": checks / "split_validation.md",
        "contract_checks": checks,
        "data_contract": checks / "data_contract.md",
        "foundation_check": checks / "foundation_acceptance.md",
        "logs": shared / "logs/baseline",
        "status": shared / "status.json",
    }


def baseline_source_identity() -> dict[str, object]:
    return source_identity(
        [
            BASELINE_ROOT / "scripts",
            BASELINE_ROOT / "src",
            REGISTRY_SOURCE,
            ROOT / "scripts/result_metric_contract.py",
            ROOT / "configs/caster_task_specs_v20.yaml",
        ],
        "caster_shared_baseline_source_v1",
    )


def baseline_artifacts_exist(shared: Path) -> bool:
    paths = baseline_paths(shared)
    required = [
        paths["metrics"],
        paths["run_manifest"],
        *(
            paths["run_root"] / relative
            for relative in SHARED_FORECAST_PATHS.values()
        ),
    ]
    if not all(path.is_file() for path in required):
        return False
    with paths["metrics"].open(newline="", encoding="utf-8") as handle:
        fields = set(csv.DictReader(handle).fieldnames or [])
    return {"n", "mae", "rmse", "nll", "coverage_90", "width_90"} <= fields


def baseline_complete(shared: Path) -> bool:
    status = read_json(baseline_paths(shared)["status"])
    identity = baseline_source_identity()
    return (
        status.get("status") == "succeeded"
        and status.get("source_identity_sha256") == identity["identity_sha256"]
        and baseline_artifacts_exist(shared)
    )


def move_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.previous_{stamp}")
    index = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.previous_{stamp}_{index}")
        index += 1
    shutil.move(str(path), str(destination))
    return destination


def baseline_environment(args: argparse.Namespace, shared: Path) -> dict[str, str]:
    paths = baseline_paths(shared)
    env = os.environ.copy()
    if args.gpu_ids:
        env["GPU_IDS"] = args.gpu_ids
    env["CUDA_A"] = args.cuda_a or env.get("CUDA_A", DEFAULT_CUDA_A)
    env["CUDA_B"] = args.cuda_b or env.get("CUDA_B", DEFAULT_CUDA_B)
    env["CUDA_B_FLU"] = args.cuda_b_flu or env.get(
        "CUDA_B_FLU", DEFAULT_CUDA_B_FLU
    )
    env.update(
        {
            "PYTHON": runtime_python(),
            "DATA_A": str(DATA_A),
            "DATA_B": str(DATA_B),
            "BENCHMARK_PROTOCOL": args.benchmark_protocol,
            "RUN_ROOT": str(paths["run_root"]),
            "MANIFEST": str(paths["manifest"]),
            "LOG_DIR": str(paths["logs"]),
            "RESULTS_DIR": str(paths["results"]),
            "BASELINE_METRICS": str(paths["metrics"]),
            "BASELINE_METRIC_SLICES": str(paths["metric_slices"]),
            "BASELINE_MANIFEST": str(paths["run_manifest"]),
            "summary_PACKET": str(paths["summary"]),
            "LEDGER_VALIDATION_REPORT": str(paths["ledger_check"]),
            "SPLIT_REPORT": str(paths["split_check"]),
            "CONTRACT_REPORTS_DIR": str(paths["contract_checks"]),
            "V3_CONTRACT_REPORT": str(paths["data_contract"]),
            "FOUNDATION_ACCEPTANCE_REPORT": str(paths["foundation_check"]),
            "BASELINE_BRIDGE_CONFIG_ROOT": str(
                paths["results"] / "bridge_configs"
            ),
            "NEURAL_MAX_STEPS": str(args.neural_max_steps),
            "SEED": str(args.base_seed),
            "PROPHET_B_YEARLY_SEASONALITY_MODE": (
                args.prophet_b_yearly_seasonality_mode
            ),
        }
    )
    if args.resume_baseline:
        env["REUSE_COMPLETE_BASELINE_MODELS"] = "1"
    return env


def ensure_baseline(args: argparse.Namespace, shared: Path) -> Path:
    if args.rerun_baseline:
        previous = move_existing(shared)
        if previous is not None:
            print(f"previous_shared_root={rel(previous)}")

    if baseline_complete(shared):
        print(f"baseline=skip existing {rel(shared)}")
        return shared

    shared.mkdir(parents=True, exist_ok=True)
    paths = baseline_paths(shared)
    identity_at_start = baseline_source_identity()
    state = {
        "status": "running",
        "started_at_utc": utc_now(),
        "source_identity_sha256": identity_at_start["identity_sha256"],
        "base_seed": args.base_seed,
        "prophet_b_yearly_seasonality_mode": (
            args.prophet_b_yearly_seasonality_mode
        ),
    }
    write_json(paths["status"], state)
    command = ["bash", str(BASELINE_SCRIPT)]
    if not args.run_tests:
        command.append("--skip-tests")
    try:
        run_checked(command, baseline_environment(args, shared))
    except SystemExit as error:
        state.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "exit_code": int(error.code or 1),
            }
        )
        write_json(paths["status"], state)
        raise

    identity_at_end = baseline_source_identity()
    if (
        identity_at_start["identity_sha256"] != identity_at_end["identity_sha256"]
        or not baseline_artifacts_exist(shared)
    ):
        state.update({"status": "failed", "finished_at_utc": utc_now()})
        write_json(paths["status"], state)
        raise SystemExit("baseline output validation failed")
    state.update(
        {
            "status": "succeeded",
            "finished_at_utc": utc_now(),
            "source_identity_sha256": identity_at_end["identity_sha256"],
            "source_identity": identity_at_end,
        }
    )
    write_json(paths["status"], state)
    return shared


def candidate_protocol_identity() -> dict[str, object]:
    sources = source_identity(
        [
            CASTER_ROOT / "src/caster",
            ARCHIVE_BUILDER,
            CASTER_ROOT / "scripts/build_selected_forecast_archive_impl.py",
            REGISTRY_VALIDATOR,
            EMBED_DESCRIPTIONS,
            SELECTION_BUILDER,
            ROOT / "scripts/formal_candidate_bank.py",
            ROOT / "scripts/model_pool_contract.py",
            ROOT / "configs/benchmark_b_context_v26_1.yaml",
            BASELINE_ROOT / "src/caster_baselines/agentic_skills.py",
        ],
        "caster_candidate_source_v1",
    )
    payload = {
        "schema": "caster_candidate_protocol_v1",
        "source_identity_sha256": sources["identity_sha256"],
        "source_identity": sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def cache_root(shared: Path, n_draws: int, seed: int) -> Path:
    return shared / "caster_candidates" / cache_tag(
        n_draws=n_draws,
        base_seed=seed,
    )


def task_archive(root: Path, task: str) -> Path:
    return root / task / FORMAL_ARCHIVE_NAME


def shared_candidate_cache_root(
    shared: Path,
    *,
    top_k: int = FORMAL_CANDIDATE_COUNT,
    n_draws: int = 10,
    base_seed: int = 42,
) -> Path:
    if int(top_k) != FORMAL_CANDIDATE_COUNT:
        raise ValueError(f"top_k must be {FORMAL_CANDIDATE_COUNT}")
    return cache_root(shared, n_draws, base_seed)


def shared_candidate_task_archive(root: Path, task: str) -> Path:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    return task_archive(root, task)


def normalized_selection_contract(
    manifest: dict[str, object],
) -> dict[str, object]:
    profile = str(
        manifest.get("selection_profile")
        or manifest.get("profile")
        or "formal_27_country_macro_v1"
    ).strip()
    if profile not in SUPPORTED_SELECTION_PROFILES:
        raise ValueError(f"unsupported selection profile: {profile}")
    if profile in QWEN_SELECTION_PROFILES:
        identity = manifest.get("identity", {})
        if not isinstance(identity, dict):
            raise ValueError("selection identity must be an object")
        source_count = int(identity.get("source_candidate_count", 0))
        eligible_count = int(identity.get("eligible_candidate_count", 0))
        excluded = [str(item) for item in identity.get("excluded_model_ids", [])]
        ranking = QWEN_RANKING_NAME
        status = str(
            manifest.get(
                "selection_profile_status",
                "versioned_qwen25_context_input_control",
            )
        )
    else:
        source_count = int(
            manifest.get("candidate_model_count", FORMAL_CANDIDATE_COUNT)
        )
        declared = manifest.get("eligible_candidate_count_by_task", {})
        counts = (
            {int(value) for value in declared.values()}
            if isinstance(declared, dict) and declared
            else {FORMAL_CANDIDATE_COUNT}
        )
        if len(counts) != 1:
            raise ValueError("eligible candidate counts differ across tasks")
        eligible_count = next(iter(counts))
        excluded = []
        ranking = BASE_RANKING_NAME
        status = str(manifest.get("selection_profile_status", "fixed_selection"))
    preregistered = (
        truthy(manifest.get("formal_preregistration_claimed", False))
        if "formal_preregistration_claimed" in manifest
        else False
    )
    return {
        "profile": profile,
        "status": status,
        "formal_preregistration_claimed": preregistered,
        "source_candidate_count": source_count,
        "eligible_candidate_count": eligible_count,
        "excluded_model_ids": excluded,
        "ranking_filename": ranking,
    }


def require_shared_selection_bank(
    shared: Path,
    candidate_cache: Path,
    selection_profile: str = "formal_27_country_macro_v1",
) -> Path:
    if selection_profile not in SUPPORTED_SELECTION_PROFILES:
        raise SystemExit(f"unsupported selection profile: {selection_profile}")
    pointer = shared / "caster_selections" / selection_profile / "active_selection.json"
    if not pointer.is_file():
        raise SystemExit(f"selection pointer is missing: {rel(pointer)}")
    pointer_payload = read_json(pointer)
    selection_root = Path(str(pointer_payload.get("selection_root", ""))).resolve()
    manifest_path = selection_root / "selection_manifest.json"
    if (
        not selection_root.is_dir()
        or not manifest_path.is_file()
        or pointer_payload.get("selection_manifest_sha256")
        != sha256_file(manifest_path)
    ):
        raise SystemExit(f"selection pointer is invalid: {rel(pointer)}")
    manifest = read_json(manifest_path)
    contract = normalized_selection_contract(manifest)
    if contract["profile"] != selection_profile:
        raise SystemExit("selection profile does not match its pointer")
    if (
        int(contract["source_candidate_count"]) != FORMAL_CANDIDATE_COUNT
        or int(contract["eligible_candidate_count"]) != FORMAL_CANDIDATE_COUNT
        or list(contract["excluded_model_ids"])
    ):
        raise SystemExit("selection candidate counts are invalid")
    if selection_profile in QWEN_SELECTION_PROFILES:
        identity = manifest.get("identity", {})
        if not isinstance(identity, dict):
            raise SystemExit("selection identity is invalid")
        source_root = Path(str(identity.get("source_candidate_cache", ""))).resolve()
        source_hash = str(
            identity.get("source_candidate_cache_manifest_sha256", "")
        )
        if (
            source_root != candidate_cache.resolve()
            or source_hash != sha256_file(candidate_cache / "manifest.json")
        ):
            raise SystemExit("selection does not match the candidate cache")
    ranking_name = str(contract["ranking_filename"])
    for task in TASKS:
        task_root = selection_root / task
        required = [
            task_root / ranking_name,
            task_root / "selection_fold_manifest.csv",
            task_root / "candidate_pool_eligibility.csv",
        ]
        if selection_profile in QWEN_SELECTION_PROFILES:
            required.append(task_root / QWEN_TOP10_NAME)
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise SystemExit(f"selection task files are incomplete: {task}")
    print(f"selection=verified {rel(selection_root)}")
    return selection_root


def materialize_bundle_selection_prefixes(
    selection_root: Path,
    bundle: Path,
    top_k: int | dict[str, int],
) -> dict[str, Path]:
    contract = normalized_selection_contract(
        read_json(selection_root / "selection_manifest.json")
    )
    profile = str(contract["profile"])
    ranking_name = str(contract["ranking_filename"])
    outputs: dict[str, Path] = {}
    for task in TASKS:
        count = int(top_k[task]) if isinstance(top_k, dict) else int(top_k)
        if not 1 <= count <= FORMAL_CANDIDATE_COUNT:
            raise SystemExit(f"invalid Top-K for {task}: {count}")
        source = selection_root / task / ranking_name
        fold = selection_root / task / "selection_fold_manifest.csv"
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        if (
            len(rows) != FORMAL_CANDIDATE_COUNT
            or not {"rank", "model_id", "task_id"} <= set(fields)
            or [int(row["rank"]) for row in rows]
            != list(range(1, FORMAL_CANDIDATE_COUNT + 1))
            or {row["task_id"] for row in rows} != {task}
            or len({row["model_id"] for row in rows}) != FORMAL_CANDIDATE_COUNT
        ):
            raise SystemExit(f"selection ranking is invalid: {rel(source)}")
        if profile in QWEN_SELECTION_PROFILES:
            with (selection_root / task / QWEN_TOP10_NAME).open(
                newline="", encoding="utf-8"
            ) as handle:
                frozen = list(csv.DictReader(handle))
            if frozen != rows[:10]:
                raise SystemExit(f"Top-10 prefix is invalid: {task}")
        destination = (
            bundle
            / "new_method/artifacts/shared_selections"
            / task
            / "candidate_selection_log.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[:count])
        shutil.copy2(fold, destination.parent / "selection_fold_manifest.csv")
        outputs[task] = destination
    return outputs


def materialize_bundle_shared_registry(
    candidate_cache: Path,
    bundle: Path,
) -> dict[str, Path]:
    artifact_root = bundle / "new_method/artifacts"
    sources = {
        "registry": candidate_cache / "model_registry.formal.csv",
        "registry_validation": candidate_cache / "model_registry_validation.csv",
        "embeddings": candidate_cache / "candidate_embeddings.csv",
    }
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit(
            "candidate registry files are incomplete: "
            + ", ".join(rel(path) for path in missing)
        )
    outputs = {
        "registry": artifact_root / "model_registry.formal.csv",
        "registry_validation": artifact_root / "model_registry_validation.csv",
        "embeddings": artifact_root / "candidate_embeddings.csv",
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for key, source in sources.items():
        shutil.copy2(source, outputs[key])
    return outputs


def task_inputs(task: str) -> tuple[Path, Path, list[str]]:
    if task == "benchmark_a":
        return DATA_A / "daily_panel.csv", DATA_A / "event_ledger.csv", []
    component = {
        "benchmark_b_covid": "covid_adm_per100k",
        "benchmark_b_flu": "flu_adm_per100k",
    }[task]
    return (
        DATA_B / "weekly_panel.csv",
        DATA_B / "event_ledger.csv",
        [
            "--task-id",
            task,
            "--target-components",
            component,
            "--posterior-scope",
            "component_stratified",
        ],
    )


def task_current(root: Path, task: str) -> bool:
    panel, ledger, _ = task_inputs(task)
    archive = task_archive(root, task)
    manifest_path = archive.with_name("forecast_archive_manifest.json")
    if not all(path.is_file() for path in [panel, ledger, archive, manifest_path]):
        return False
    manifest = read_json(manifest_path)
    component = {
        "benchmark_b_covid": "covid_adm_per100k",
        "benchmark_b_flu": "flu_adm_per100k",
    }.get(task)
    rows = 0
    embargo_rows = 0
    with ledger.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if component is not None and row.get("component") != component:
                continue
            rows += 1
            embargo_rows += int(row.get("split") == "embargo")
    selected = tuple(map(str, manifest.get("selected_model_ids", [])))
    input_current = manifest.get("panel_sha256") == sha256_file(panel)
    if task == "benchmark_a":
        input_manifest = DATA_A / "run_manifest.json"
        input_current = (
            input_current
            and input_manifest.is_file()
            and manifest.get("benchmark_a_input_manifest_sha256")
            == sha256_file(input_manifest)
        )
    return (
        input_current
        and manifest.get("ledger_sha256") == sha256_file(ledger)
        and int(manifest.get("ledger_rows", -1)) == rows
        and int(manifest.get("embargo_rows", -1)) == embargo_rows
        and embargo_rows > 0
        and manifest.get("embargo_forecast_coverage_required") is True
        and manifest.get("embargo_metric_eligible") is False
        and int(manifest.get("models", -1)) == FORMAL_CANDIDATE_COUNT
        and set(selected) == set(formal_candidate_model_ids())
        and len(selected) == FORMAL_CANDIDATE_COUNT
        and int(manifest.get("archive_rows", -1))
        == rows * FORMAL_CANDIDATE_COUNT
        and manifest.get("archive_sha256") == sha256_file(archive)
    )


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def model_contract_current(path: Path) -> bool:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    task_columns = [f"{task}_forecast" for task in TASKS]
    required = {
        "canonical_model_id",
        "prediction_object_id",
        "canonical_prediction_source",
        "canonical_prediction_source_sha256",
        "candidate_model_id",
        "baseline_method",
        "is_baseline",
        "is_candidate",
        "model_roles",
        "prediction_result_count",
        *task_columns,
    }
    if not required <= set(frame.columns) or len(frame) != FORMAL_CANDIDATE_COUNT:
        return False
    canonical = frame["canonical_model_id"].astype(str)
    if canonical.duplicated().any():
        return False
    if canonical.tolist() != frame["candidate_model_id"].astype(str).tolist():
        return False
    if set(canonical) != set(formal_candidate_model_ids()):
        return False
    if not frame["is_candidate"].map(truthy).all():
        return False
    if not pd.to_numeric(
        frame["prediction_result_count"], errors="coerce"
    ).eq(1).all():
        return False
    shared = frame["is_baseline"].map(truthy)
    expected_shared = canonical.isin(SHARED_BASELINE_METHODS)
    if not shared.equals(expected_shared):
        return False
    expected_methods = canonical.map(SHARED_BASELINE_METHODS).fillna("")
    if not frame["baseline_method"].fillna("").astype(str).equals(expected_methods):
        return False
    expected_roles = shared.map(
        lambda value: "baseline;candidate" if value else "candidate"
    )
    if not frame["model_roles"].astype(str).equals(expected_roles):
        return False
    expected_ids = canonical.map(
        lambda model: model_role_fields(model)["prediction_object_id"]
    )
    if not frame["prediction_object_id"].astype(str).equals(expected_ids):
        return False
    sources = frame.loc[shared, "canonical_prediction_source"].astype(str).map(Path)
    hashes = frame.loc[shared, "canonical_prediction_source_sha256"].astype(str)
    if not all(
        source.is_file() and digest == sha256_file(source)
        for source, digest in zip(sources, hashes, strict=True)
    ):
        return False
    if not frame.loc[~shared, "canonical_prediction_source"].fillna("").eq("").all():
        return False
    for column in task_columns:
        paths = frame[column].astype(str).map(Path)
        if not paths.map(Path.is_file).all() or paths.astype(str).duplicated().any():
            return False
    return True


def cache_complete(root: Path) -> bool:
    selection = (
        root / "selections" / FORMAL_SELECTION_DIR / "candidate_selection_log.csv"
    )
    manifest_path = root / "manifest.json"
    contract_path = root / "shared_model_contract.csv"
    if not all(path.is_file() for path in [selection, manifest_path, contract_path]):
        return False
    manifest = read_json(manifest_path)
    current_protocol = candidate_protocol_identity()
    if manifest.get("protocol_identity_sha256") != current_protocol["identity_sha256"]:
        return False
    coverage = manifest.get("baseline_reuse_coverage", {})
    if not isinstance(coverage, dict):
        return False
    records = coverage.get("models", {})
    if (
        coverage.get("schema") != "caster_shared_baseline_single_result_coverage_v1"
        or coverage.get("extensional_extension_used") is not False
        or coverage.get("prediction_result_count_per_overlapping_model") != 1
        or not isinstance(records, dict)
        or set(records) != set(SHARED_FORECAST_PATHS)
    ):
        return False
    for record in records.values():
        if not isinstance(record, dict):
            return False
        source = Path(str(record.get("path", "")))
        if (
            not source.is_file()
            or record.get("sha256") != sha256_file(source)
            or record.get("prediction_result_count") != 1
        ):
            return False
    if manifest.get("registry_source_sha256") != sha256_file(REGISTRY_SOURCE):
        return False
    if manifest.get("shared_model_contract_sha256") != sha256_file(contract_path):
        return False
    return model_contract_current(contract_path) and all(
        task_current(root, task) for task in TASKS
    )


def baseline_coverage(run_root: Path) -> dict[str, object]:
    expected_ids: set[str] = set()
    expected_rows = 0
    for ledger_path in [DATA_A / "event_ledger.csv", DATA_B / "event_ledger.csv"]:
        ledger = pd.read_csv(
            ledger_path,
            usecols=["forecast_id"],
            keep_default_na=False,
            low_memory=False,
        )
        ids = ledger["forecast_id"].astype(str)
        if ids.duplicated().any() or expected_ids.intersection(ids):
            raise SystemExit(f"forecast_id grid is not unique: {rel(ledger_path)}")
        expected_ids.update(ids)
        expected_rows += len(ids)
    records: dict[str, dict[str, object]] = {}
    for model, relative in SHARED_FORECAST_PATHS.items():
        path = run_root / relative
        if not path.is_file():
            raise SystemExit(f"missing shared forecast for {model}: {rel(path)}")
        forecasts = pd.read_csv(
            path,
            usecols=["forecast_id"],
            keep_default_na=False,
            low_memory=False,
        )
        ids = forecasts["forecast_id"].astype(str)
        observed = set(ids)
        if (
            ids.duplicated().any()
            or len(forecasts) != expected_rows
            or observed != expected_ids
        ):
            raise SystemExit(f"shared forecast coverage mismatch for {model}")
        records[model] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(forecasts),
            "unique_forecast_ids": len(observed),
            "prediction_result_count": 1,
        }
    return {
        "schema": "caster_shared_baseline_single_result_coverage_v1",
        "expected_rows": expected_rows,
        "expected_unique_forecast_ids": len(expected_ids),
        "models": records,
        "extensional_extension_used": False,
        "prediction_result_count_per_overlapping_model": 1,
    }


def freeze_baseline_manifest(source: Path, root: Path, refresh: bool) -> Path:
    target = root / "baseline_inputs/baseline_run_manifest.non_agent.csv"
    metadata = target.with_suffix(".manifest.json")
    if target.is_file() and metadata.is_file() and not refresh:
        recorded = read_json(metadata)
        if recorded.get("snapshot_sha256") == sha256_file(target):
            return target
    if not source.is_file():
        raise SystemExit(f"missing baseline manifest: {rel(source)}")
    frame = pd.read_csv(source, keep_default_na=False, low_memory=False)
    required = {"status", "method", "run_dir", "restart_type"}
    if not required <= set(frame.columns):
        raise SystemExit("baseline manifest columns are incomplete")
    if frame["restart_type"].astype(str).str.startswith("archive_backed_").any():
        raise SystemExit("baseline manifest already contains agent rows")
    if frame["run_dir"].astype(str).map(
        lambda value: any(
            part.startswith(".") and part not in {".", ".."}
            for part in Path(value).parts
        )
    ).any():
        raise SystemExit("baseline manifest contains hidden paths")
    prophet = frame[
        frame["status"].astype(str).eq("numeric")
        & frame["method"].astype(str).eq("prophet")
    ]
    if len(prophet) != 1:
        raise SystemExit("baseline manifest requires one Prophet result")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    write_json(
        metadata,
        {
            "schema": "caster_candidate_baseline_manifest_v1",
            "created_at_utc": utc_now(),
            "source_sha256": sha256_file(source),
            "snapshot_sha256": sha256_file(target),
            "rows": len(frame),
        },
    )
    return target


def build_task(
    task: str,
    gpu: str,
    root: Path,
    registry: Path,
    selection: Path,
    baseline_manifest: Path,
    baseline_runs: Path,
    seed: int,
    force: bool,
    protocol_stale: bool,
) -> None:
    if task_current(root, task) and not force and not protocol_stale:
        print(f"candidate_task=skip existing {task}")
        return
    panel, ledger, task_args = task_inputs(task)
    command = [
        runtime_python(),
        str(ARCHIVE_BUILDER),
        "--panel",
        str(panel),
        "--ledger",
        str(ledger),
        "--registry",
        str(registry),
        "--selection",
        str(selection),
        "--out",
        str(task_archive(root, task)),
        "--seed",
        str(seed),
        "--workers",
        DEFAULT_WORKERS,
        "--checkpoint-dir",
        str(root / task / "phase20_checkpoints"),
        "--resume",
        *task_args,
        "--reuse-baseline-forecasts",
        "--baseline-manifest",
        str(baseline_manifest),
        "--baseline-runs-root",
        str(baseline_runs),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"candidate_task=start {task} device={gpu}")
    run_checked(command, env)


def ensure_candidate_cache(args: argparse.Namespace, shared: Path) -> Path:
    if args.top_k != FORMAL_CANDIDATE_COUNT:
        raise SystemExit(f"--top-k must be {FORMAL_CANDIDATE_COUNT}")
    root = cache_root(shared, args.n_draws, args.base_seed)
    if cache_complete(root) and not args.force:
        print(f"candidate_cache=skip existing {rel(root)}")
        return root
    root.mkdir(parents=True, exist_ok=True)
    baseline = baseline_paths(shared)
    baseline_manifest = freeze_baseline_manifest(
        baseline["run_manifest"],
        root,
        args.force,
    )
    registry = root / "model_registry.formal.csv"
    registry_check = root / "model_registry_validation.csv"
    embeddings = root / "candidate_embeddings.csv"
    previous = read_json(root / "manifest.json")
    protocol = candidate_protocol_identity()
    protocol_stale = previous.get("protocol_identity_sha256") != protocol[
        "identity_sha256"
    ]
    registry_stale = previous.get("registry_source_sha256") != sha256_file(
        REGISTRY_SOURCE
    )

    if registry_stale or not registry.is_file() or not registry_check.is_file():
        run_checked(
            [
                sys.executable,
                str(REGISTRY_VALIDATOR),
                "--registry",
                str(REGISTRY_SOURCE),
                "--out",
                str(registry_check),
                "--normalized-out",
                str(registry),
            ],
            os.environ.copy(),
        )
    if registry_stale or not embeddings.is_file():
        run_checked(
            [
                sys.executable,
                str(EMBED_DESCRIPTIONS),
                "--registry",
                str(REGISTRY_SOURCE),
                "--out",
                str(embeddings),
                "--dim",
                str(args.embed_dim),
            ],
            os.environ.copy(),
        )
    selection = (
        root / "selections" / FORMAL_SELECTION_DIR / "candidate_selection_log.csv"
    )
    if registry_stale or not selection.is_file():
        run_checked(
            [
                sys.executable,
                str(SELECTION_BUILDER),
                "--registry",
                str(registry),
                "--embeddings",
                str(embeddings),
                "--out",
                str(selection),
                "--timing-out",
                str(selection.with_name("candidate_selection_timing.json")),
                "--metadata-out",
                str(selection.with_name("candidate_selection_metadata.json")),
                "--benchmark",
                f"shared_baseline_all{FORMAL_CANDIDATE_COUNT}",
                "--algorithm-id",
                f"shared_baseline_k{FORMAL_CANDIDATE_COUNT}",
                "--selection-scope",
                "shared_baseline_all_result_tasks",
                "--shared-selection",
                "--top-k",
                str(FORMAL_CANDIDATE_COUNT),
                "--family-diversity-bonus",
                "0.12",
                "--seed",
                str(args.base_seed),
            ],
            os.environ.copy(),
        )

    coverage = baseline_coverage(baseline["run_root"])
    gpus = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpu-ids must contain at least one device")
    with ThreadPoolExecutor(max_workers=len(TASKS)) as executor:
        futures = [
            executor.submit(
                build_task,
                task,
                gpus[index % len(gpus)],
                root,
                registry,
                selection,
                baseline_manifest,
                baseline["run_root"],
                args.base_seed,
                args.force,
                protocol_stale,
            )
            for index, task in enumerate(TASKS)
        ]
        for future in futures:
            future.result()

    task_records: dict[str, dict[str, object]] = {}
    for task in TASKS:
        archive = task_archive(root, task)
        task_manifest = archive.with_name("forecast_archive_manifest.json")
        metadata = read_json(task_manifest)
        task_records[task] = {
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
            "manifest": str(task_manifest),
            "ledger_sha256": metadata.get("ledger_sha256", ""),
            "ledger_rows": int(metadata.get("ledger_rows", 0)),
            "embargo_rows": int(metadata.get("embargo_rows", 0)),
            "embargo_forecast_coverage_required": bool(
                metadata.get("embargo_forecast_coverage_required", False)
            ),
        }

    registry_frame = pd.read_csv(registry, low_memory=False)
    contract_rows: list[dict[str, object]] = []
    for model in registry_frame["model_id"].astype(str).drop_duplicates():
        roles = model_role_fields(model)
        baseline_method = str(roles["baseline_method"])
        source = (
            baseline["run_root"]
            / CANONICAL_SHARED_MODEL_OBJECTS[model].forecast_path
            if baseline_method
            else None
        )
        contract_rows.append(
            {
                **roles,
                "canonical_prediction_source": str(source) if source else "",
                "canonical_prediction_source_sha256": (
                    sha256_file(source) if source else ""
                ),
                "prediction_result_count_per_task": 1,
                "task_count": len(TASKS),
                "prediction_source_kind": (
                    "canonical_shared_baseline" if baseline_method else "caster_local"
                ),
                **{
                    f"{task}_forecast": str(
                        root
                        / task
                        / "phase20_checkpoints"
                        / f"forecast_archive.{model}.csv"
                    )
                    for task in TASKS
                },
            }
        )
    contract = root / "shared_model_contract.csv"
    pd.DataFrame(contract_rows).to_csv(contract, index=False)
    write_json(
        root / "manifest.json",
        {
            "schema": "caster_shared_candidate_cache_v1",
            "benchmark_protocol": args.benchmark_protocol,
            "created_at_utc": utc_now(),
            "top_k": FORMAL_CANDIDATE_COUNT,
            "n_draws_label": args.n_draws,
            "base_seed": args.base_seed,
            "candidate_training_scope": "canonical_shared_baseline_reuse",
            "formal_bundle_candidate_training_allowed": False,
            "full_bank_draws_generated": False,
            "selection": str(selection),
            "selection_sha256": sha256_file(selection),
            "registry": str(registry),
            "registry_sha256": sha256_file(registry),
            "registry_source_sha256": sha256_file(REGISTRY_SOURCE),
            "protocol_identity_sha256": protocol["identity_sha256"],
            "protocol_identity": protocol,
            "baseline_manifest": str(baseline_manifest),
            "baseline_manifest_sha256": sha256_file(baseline_manifest),
            "baseline_runs_root": str(baseline["run_root"]),
            "baseline_reuse_models": list(SHARED_FORECAST_PATHS),
            "baseline_reuse_policy": (
                "single_shared_forecast_file_per_overlapping_model"
            ),
            "baseline_reuse_coverage": coverage,
            "baseline_extension_runs_root": "",
            "fresh_candidate_source_models": [],
            "fresh_candidate_source_manifest": "",
            "fresh_candidate_source_manifest_sha256": "",
            "caster_local_models": LOCAL_MODELS,
            "shared_model_contract": str(contract),
            "shared_model_contract_sha256": sha256_file(contract),
            "shared_model_result_policy": (
                "one_prediction_result_with_baseline_candidate_roles"
            ),
            "tasks": task_records,
        },
    )
    if not cache_complete(root):
        raise SystemExit(f"candidate cache validation failed: {rel(root)}")
    print(f"candidate_cache={rel(root)}")
    return root


def check_inputs() -> None:
    code_inputs = [
        BASELINE_SCRIPT,
        ARCHIVE_BUILDER,
        CASTER_ROOT / "scripts/build_selected_forecast_archive_impl.py",
        REGISTRY_VALIDATOR,
        EMBED_DESCRIPTIONS,
        SELECTION_BUILDER,
        ROOT / "scripts/formal_candidate_bank.py",
        ROOT / "scripts/model_pool_contract.py",
        ROOT / "scripts/result_metric_contract.py",
        REGISTRY_SOURCE,
    ]
    data_inputs = [
        DATA_A / "daily_panel.csv",
        DATA_A / "event_ledger.csv",
        DATA_A / "run_manifest.json",
        DATA_B / "weekly_panel.csv",
        DATA_B / "event_ledger.csv",
    ]
    missing_code = [path for path in code_inputs if not path.is_file()]
    if missing_code:
        raise SystemExit(
            "missing code inputs: " + ", ".join(rel(path) for path in missing_code)
        )
    missing_data = [path for path in data_inputs if not path.is_file()]
    packaged_sources = (
        (ROOT / "data/benchmark_a/raw").is_dir()
        and (ROOT / "data/benchmark_b/source").is_dir()
    )
    staged_sources = (
        (ROOT / "data/benchmark_a/raw_all").is_dir()
        and (ROOT / "data/benchmark_b/raw_all/data_raw").is_dir()
    )
    if missing_data and not (packaged_sources or staged_sources):
        raise SystemExit(
            "missing data inputs: " + ", ".join(rel(path) for path in missing_data)
        )
    if len(formal_candidate_model_ids()) != FORMAL_CANDIDATE_COUNT:
        raise SystemExit("candidate registry size mismatch")
    state = "ready" if not missing_data else "created_by_data_stage"
    print(
        f"inputs=ok code_files={len(code_inputs)} data={state} "
        f"tasks={len(TASKS)} models={FORMAL_CANDIDATE_COUNT}"
    )


def command(args: argparse.Namespace) -> int:
    if args.check_only:
        check_inputs()
        return 0
    runs_root = resolve_path(args.runs_root)
    shared = (
        resolve_path(args.shared_root)
        if args.shared_root
        else runs_root / "shared_baseline"
    )
    if args.rerun_baseline or args.resume_baseline:
        ensure_baseline(args, shared)
    elif not baseline_complete(shared):
        raise SystemExit("baseline is missing; pass --rerun-baseline")
    else:
        print(f"baseline=skip existing {rel(shared)}")
    root = ensure_candidate_cache(args, shared)
    if not args.skip_agents:
        print("agents=deferred to the agent stage")
    print(f"prepared_candidate_cache={rel(root)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the shared three-task candidate cache."
    )
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument(
        "--benchmark-protocol",
        choices=["v3_direct_rollout"],
        default="v3_direct_rollout",
    )
    parser.add_argument("--top-k", type=int, default=FORMAL_CANDIDATE_COUNT)
    parser.add_argument("--n-draws", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--prophet-b-yearly-seasonality-mode",
        choices=["auto", "off"],
        default="auto",
    )
    parser.add_argument(
        "--cache-scope",
        choices=["all-result-tasks"],
        default="all-result-tasks",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--force", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rerun-baseline", action="store_true")
    mode.add_argument("--resume-baseline", action="store_true")
    parser.add_argument("--reuse-baseline-forecasts", action="store_true", default=True)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--skip-agents", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--neural-max-steps", type=int, default=3)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--cuda-a", default="")
    parser.add_argument("--cuda-b", default="")
    parser.add_argument("--cuda-b-flu", default="")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.n_draws <= 0:
        raise SystemExit("--n-draws must be positive")
    return command(args)


if __name__ == "__main__":
    raise SystemExit(main())
