#!/usr/bin/env python3
""








from __future__ import annotations

import argparse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code/caster"
DEFAULT_SOURCE = ROOT / "experiments/result_runs_v3_direct_rollout/k10__draws10__seed42__rep00__formal"
DEFAULT_OUT = ROOT / "experiments/result_runs_v3_direct_rollout/k10__draws10__seed42__rep00__rho_only_moment_t_smallval360_newton"
DEFAULT_DUAL_OUT = ROOT / "experiments/result_runs_v3_direct_rollout/k10__draws10__seed42__rep00__rho001_moment_t_plus_draw_kernel_smallval360_newton"
DEFAULT_PYTHON = Path(sys.executable)
CURRENT_SELECTION_PROFILE = "formal_27_country_macro_v1"
DEFAULT_AGENT_RUN_PROFILE = "qwen25_multiscale_released_sequence_v1"
FULL_CANDIDATE_ARCHIVE_NAME = "forecast_archive_all27.csv"
alternate_PREDICTIVE_CONTRACT = "alternate_archive_moment"
MEAN_PRESERVING_CENSORED_PREDICTIVE_CONTRACT = (
    "coherent_mean_preserving_censored_student_t"
)
CENSORING_BOUND_SCOPES = (
    "selected_topk_train",
    "eligible27_train",
)
PREDICTIVE_CONTRACTS = (
    alternate_PREDICTIVE_CONTRACT,
    "coherent_mean_preserving_truncated_t",
    MEAN_PRESERVING_CENSORED_PREDICTIVE_CONTRACT,
    "archive_mean_bridge_quantiles",
)
CURRENT_OBJECTIVE_WEIGHTS = {
    "nll": 0.20,
    "wis": 0.20,
    "short_rmse": 0.20,
    "long_rmse": 0.20,
    "mae": 0.10,
    "coverage_penalty": 0.10,
}
EXPECTED_SELECTED_ARCHIVE_ROWS = {
    "benchmark_a": 420_990,
    "benchmark_b_covid": 261_630,
    "benchmark_b_flu": 261_630,
}
DEFAULT_TOP_K = 10
MAX_ELIGIBLE_TOP_K = 27

TASKS = {
    "benchmark_a": {
        "relative": Path("benchmark_a"),
        "components": "cases",
        "scope": "",
    },
    "benchmark_b_covid": {
        "relative": Path("benchmark_b/benchmark_b_covid"),
        "components": "covid_adm_per100k",
        "scope": "component_stratified",
    },
    "benchmark_b_flu": {
        "relative": Path("benchmark_b/benchmark_b_flu"),
        "components": "flu_adm_per100k",
        "scope": "component_stratified",
    },
}
RESULT_TASKS = (
    "benchmark_a",
    "benchmark_b_covid",
    "benchmark_b_flu",
)
_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _rho_bounds_for_task(
    args: argparse.Namespace,
    task: str,
) -> tuple[float, float]:
    if task == "benchmark_a":
        lower_override = args.benchmark_a_rho_min
        upper_override = args.benchmark_a_rho_max
    else:
        lower_override = args.benchmark_b_rho_min
        upper_override = args.benchmark_b_rho_max
    lower = float(args.rho_min if lower_override is None else lower_override)
    upper = float(args.rho_max if upper_override is None else upper_override)
    if not (
        math.isfinite(lower)
        and math.isfinite(upper)
        and 0.0 < lower < upper
    ):
        raise SystemExit(
            f"invalid rho bounds for {task}: expected finite 0 < lower < upper, "
            f"found [{lower:g}, {upper:g}]"
        )
    return lower, upper


def _objective_weights(args: argparse.Namespace) -> dict[str, float]:
    declared = {
        key: float(getattr(args, f"weight_{key}"))
        for key in CURRENT_OBJECTIVE_WEIGHTS
    }
    overall_raw = getattr(args, "weight_overall_rmse", None)
    if overall_raw is not None:
        declared["overall_rmse"] = float(overall_raw)
    if any(not math.isfinite(value) or value < 0.0 for value in declared.values()):
        raise SystemExit("rho objective weights must be finite and nonnegative")
    mode = _rmse_objective_mode(args)
    if mode == "short-long":
        if overall_raw is not None and not math.isclose(
            float(overall_raw), 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SystemExit(
                "--weight-overall-rmse is only active with --rmse-objective-mode overall"
            )
        weights = {
            key: float(declared[key]) for key in CURRENT_OBJECTIVE_WEIGHTS
        }
    else:
        overall_weight = (
            float(declared["short_rmse"] + declared["long_rmse"])
            if overall_raw is None
            else float(overall_raw)
        )
        weights = {
            "nll": float(declared["nll"]),
            "overall_rmse": overall_weight,
            "mae": float(declared["mae"]),
            "wis": float(declared["wis"]),
            "coverage_penalty": float(declared["coverage_penalty"]),
        }
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("rho objective weights must sum to one")
    return weights


def _rmse_objective_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "rmse_objective_mode", "short-long"))
    if mode not in {"short-long", "overall"}:
        raise SystemExit(f"unsupported rho RMSE objective mode: {mode!r}")
    return mode


def _distribution(args: argparse.Namespace) -> str:
    value = str(getattr(args, "distribution", "student_t"))
    if value not in {"student_t", "gaussian"}:
        raise SystemExit(f"unsupported rho-only distribution: {value!r}")
    return value


def _predictive_contract(args: argparse.Namespace) -> str:
    value = str(
        getattr(args, "predictive_contract", alternate_PREDICTIVE_CONTRACT)
    )
    if value not in PREDICTIVE_CONTRACTS:
        raise SystemExit(f"unsupported predictive contract: {value!r}")
    if value in {
        "coherent_mean_preserving_truncated_t",
        MEAN_PRESERVING_CENSORED_PREDICTIVE_CONTRACT,
    } and _distribution(args) != "student_t":
        raise SystemExit(
            f"{value} requires --distribution student_t"
        )
    return value


def _fixed_c_u(args: argparse.Namespace) -> float:
    value = float(getattr(args, "fixed_c_u", 1.25))
    if not math.isfinite(value) or value < 1.0:
        raise SystemExit("--fixed-c-u must be finite and at least 1")
    return value


def _fixed_c_u_is_active(predictive_contract: str) -> bool:
    return (
        str(predictive_contract)
        == MEAN_PRESERVING_CENSORED_PREDICTIVE_CONTRACT
    )


def _censoring_bound_scope(args: argparse.Namespace) -> str:
    value = str(
        getattr(args, "censoring_bound_scope", "selected_topk_train")
    )
    if value not in CENSORING_BOUND_SCOPES:
        raise SystemExit(f"unsupported censoring bound scope: {value!r}")
    if (
        value != "selected_topk_train"
        and not _fixed_c_u_is_active(_predictive_contract(args))
    ):
        raise SystemExit(
            "--censoring-bound-scope eligible27_train is only valid for "
            f"{MEAN_PRESERVING_CENSORED_PREDICTIVE_CONTRACT}"
        )
    return value


def _recorded_predictive_contract(payload: object) -> str:
    ""

    if not isinstance(payload, dict):
        return ""
    return str(payload.get("predictive_contract", alternate_PREDICTIVE_CONTRACT))


def _input_draws(args: argparse.Namespace) -> int:
    value = int(getattr(args, "input_draws", 10))
    if value < 1:
        raise SystemExit("--input-draws must be positive")
    return value


def _candidate_cache_draws(args: argparse.Namespace) -> int:
    raw = getattr(args, "candidate_cache_draws", None)
    value = _input_draws(args) if raw is None else int(raw)
    if value < 1:
        raise SystemExit("--candidate-cache-draws must be positive")
    return value


def _task_top_k(args: argparse.Namespace, task: str) -> int:
    ""






    if task not in TASKS:
        raise SystemExit(f"unknown task for Top-K: {task!r}")
    override_name = (
        "benchmark_a_top_k"
        if task == "benchmark_a"
        else "benchmark_b_top_k"
    )
    override = getattr(args, override_name, None)
    value = int(
        getattr(args, "top_k", DEFAULT_TOP_K)
        if override is None
        else override
    )
    if value < 1 or value > MAX_ELIGIBLE_TOP_K:
        raise SystemExit(
            f"Top-K for {task} must be in [1, {MAX_ELIGIBLE_TOP_K}]: {value}"
        )
    return value


def _top_k_by_task(
    args: argparse.Namespace,
    tasks: Sequence[str] = tuple(TASKS),
) -> dict[str, int]:
    return {task: _task_top_k(args, task) for task in tasks}


def _expected_selected_archive_rows(
    task: str,
    top_k: int,
    *,
    source_archive: Path | None = None,
    ledger: Path | None = None,
    selected_model_ids: Sequence[str] | None = None,
) -> int:
    if task not in EXPECTED_SELECTED_ARCHIVE_ROWS:
        raise RuntimeError(f"unknown task for archive row validation: {task}")
    if int(top_k) == DEFAULT_TOP_K:
        return int(EXPECTED_SELECTED_ARCHIVE_ROWS[task])
    if source_archive is None or ledger is None or selected_model_ids is None:
        raise RuntimeError(
            "non-default Top-K archive validation requires the source archive, "
            "task ledger, and selected model IDs"
        )

    import pandas as pd

    selected = tuple(str(value) for value in selected_model_ids)
    if len(selected) != int(top_k) or len(set(selected)) != int(top_k):
        raise RuntimeError(
            f"selected model IDs do not match Top-K for {task}: "
            f"top_k={top_k} selected={len(selected)} unique={len(set(selected))}"
        )
    if not source_archive.is_file() or not ledger.is_file():
        raise RuntimeError(
            f"archive row validation inputs are missing for {task}: "
            f"source_archive={source_archive} ledger={ledger}"
        )
    try:
        ledger_frame = pd.read_csv(
            ledger,
            usecols=["forecast_id"],
            keep_default_na=False,
            low_memory=False,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"cannot read task ledger for {task}: {ledger}") from exc
    ledger_ids = ledger_frame["forecast_id"].astype(str)
    if ledger_ids.empty or ledger_ids.duplicated().any():
        raise RuntimeError(
            f"task ledger forecast IDs must be nonempty and unique for {task}"
        )
    required_forecasts = set(ledger_ids)
    selected_set = set(selected)
    rows_by_model = {model_id: 0 for model_id in selected}
    forecasts_by_model = {model_id: set() for model_id in selected}
    try:
        chunks = pd.read_csv(
            source_archive,
            usecols=["forecast_id", "model_id"],
            keep_default_na=False,
            low_memory=False,
            chunksize=250_000,
        )
        for chunk in chunks:
            chunk["forecast_id"] = chunk["forecast_id"].astype(str)
            chunk["model_id"] = chunk["model_id"].astype(str)
            hit = chunk[
                chunk["model_id"].isin(selected_set)
                & chunk["forecast_id"].isin(required_forecasts)
            ]
            for model_id, group in hit.groupby("model_id", sort=False):
                key = str(model_id)
                rows_by_model[key] += int(len(group))
                forecasts_by_model[key].update(group["forecast_id"].astype(str))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"cannot read source archive for {task}: {source_archive}"
        ) from exc

    incomplete = {
        model_id: len(required_forecasts - forecasts_by_model[model_id])
        for model_id in selected
        if forecasts_by_model[model_id] != required_forecasts
    }
    if incomplete:
        raise RuntimeError(
            f"source archive does not cover the task ledger for {task}: {incomplete}"
        )
    expected = sum(rows_by_model.values())
    if expected <= 0:
        raise RuntimeError(f"source archive has no selected rows for {task}")
    return int(expected)


def _top_k_identity_suffix(args: argparse.Namespace) -> str:
    top_k_by_task = _top_k_by_task(args)
    if set(top_k_by_task.values()) == {DEFAULT_TOP_K}:
        return ""
    return (
        f"__kA{top_k_by_task['benchmark_a']}"
        f"_kB{top_k_by_task['benchmark_b_covid']}"
    )


def _fixed_nu(args: argparse.Namespace) -> float | None:
    distribution = _distribution(args)
    raw = getattr(args, "fixed_nu", None)
    if distribution == "gaussian":
        if raw is not None:
            raise SystemExit("Gaussian rho-only runs do not accept --fixed-nu")
        return None
    value = 5.0 if raw is None else float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit("Student-t --fixed-nu must be finite and positive")
    return value


def _reported_nu(args: argparse.Namespace) -> float | str:
    value = _fixed_nu(args)
    return float(value) if value is not None else "inactive"


def _task_root(run_root: Path, task: str) -> Path:
    return run_root / "new_method/artifacts" / TASKS[task]["relative"]


def _draw_branch_root(out_run: Path) -> Path:
    return out_run / "branches/draw_kernel"


def _filter_root(calibration_root: Path, family: str, hierarchical: bool) -> Path:
    if family == "moment_t":
        return calibration_root
    return calibration_root / ("hierarchical" if hierarchical else "one_layer")


def _safe_link(source: Path, destination: Path, *, directory: bool = False) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        if (
            not directory
            and source.is_file()
            and destination.resolve().is_file()
            and _sha256_file(source) == _sha256_file(destination)
        ):
                                                                              
                                                                           
                                                                           
                                                           
            return
        raise RuntimeError(f"existing symlink has a different target: {destination}")
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing pilot input: {destination}")
    destination.symlink_to(source, target_is_directory=directory)


def _sha256_file(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    if cache_key in _SHA256_CACHE:
        return _SHA256_CACHE[cache_key]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _SHA256_CACHE[cache_key] = value
    return value


def _canonical_json_sha256(payload: object) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _agent_run_profile_name(value: object) -> str:
    ""

    profile = str(value).strip()
    candidate = Path(profile)
    if (
        not profile
        or candidate.is_absolute()
        or candidate.parts != (profile,)
        or profile in {".", ".."}
        or "/" in profile
        or "\\" in profile
    ):
        raise ValueError(
            "agent run profile must be one non-empty directory name "
            "(no absolute path, separators, or path traversal)"
        )
    return profile


def _agent_run_profile_arg(value: str) -> str:
    try:
        return _agent_run_profile_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _bind_agent_run_profile(args: argparse.Namespace) -> dict[str, str]:
    ""

    try:
        profile = _agent_run_profile_name(
            getattr(args, "agent_run_profile", DEFAULT_AGENT_RUN_PROFILE)
        )
    except ValueError as exc:
        raise SystemExit(f"invalid --agent-run-profile: {exc}") from exc

    record = {
        "agent_run_profile": profile,
        "agent_run_root": "",
        "agent_overall_manifest_path": "",
        "agent_overall_manifest_sha256": "",
    }
    shared_root = getattr(args, "shared_root", None)
    if shared_root is None:
        args.agent_run_profile = profile
        args.agent_run_root = None
        args.agent_overall_manifest_path = None
        args.agent_overall_manifest_sha256 = ""
        return record

    shared = Path(shared_root).resolve()
    agentic_root = (shared / "baseline/runs/agentic").resolve()
    requested_root = agentic_root / profile
    if not requested_root.is_dir():
        raise SystemExit(
            "selected agent run profile does not exist: "
            f"{requested_root}"
        )
    resolved_root = requested_root.resolve()
    try:
        resolved_root.relative_to(agentic_root)
    except ValueError as exc:
        raise SystemExit(
            "selected agent run profile escapes the shared agentic root: "
            f"{requested_root}"
        ) from exc

    log_directory = f"formal_agents_{profile}"
    manifest_candidate = (
        shared / "logs" / log_directory / "latest_run_manifest.json"
    )
    manifest_path: Path | None = None
    manifest_sha256 = ""
    if manifest_candidate.is_file():
        manifest_path = manifest_candidate.resolve()
        try:
            manifest_path.relative_to(shared)
        except ValueError as exc:
            raise SystemExit(
                "agent overall manifest escapes the shared baseline root: "
                f"{manifest_candidate}"
            ) from exc
        manifest_sha256 = _sha256_file(manifest_path)

    args.agent_run_profile = profile
    args.agent_run_root = resolved_root
    args.agent_overall_manifest_path = manifest_path
    args.agent_overall_manifest_sha256 = manifest_sha256
    return {
        "agent_run_profile": profile,
        "agent_run_root": str(resolved_root),
        "agent_overall_manifest_path": (
            str(manifest_path) if manifest_path is not None else ""
        ),
        "agent_overall_manifest_sha256": manifest_sha256,
    }


def _agent_input_ledger(
    args: argparse.Namespace,
    candidate_cache: Path,
    task: str,
) -> Path:
    profile = _agent_run_profile_name(
        getattr(args, "agent_run_profile", DEFAULT_AGENT_RUN_PROFILE)
    )
    versioned = candidate_cache / "agent_inputs" / profile / task / "event_ledger.csv"
    if versioned.is_file():
        return versioned
    raise SystemExit(
        "agent input ledger is missing for "
        f"task={task} profile={profile}: {versioned}"
    )


def _selection_fold_manifest_sha256(path: Path) -> str:
    ""

    import pandas as pd

    folds = pd.read_csv(path)
    hash_column = "fold_manifest_sha256"
    if folds.empty or hash_column not in folds.columns:
        raise ValueError("selection fold manifest is empty or lacks its hash column")
    declared = folds[hash_column].dropna().astype(str).unique()
    if len(declared) != 1:
        raise ValueError("selection fold manifest must declare exactly one hash")
    content = folds.drop(columns=[hash_column])
    columns = sorted(content.columns)
    if not columns:
        raise ValueError("selection fold manifest has no identity columns")
    normalized = (
        content[columns]
        .fillna("")
        .astype(str)
        .sort_values(columns)
        .to_csv(index=False, lineterminator="\n")
    )
    computed = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if str(declared[0]) != computed:
        raise ValueError("selection fold manifest content/hash mismatch")
    return computed


def _input_manifest_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if payload.get("schema") != "caster_rho_only_shared_selection_input_manifest_v3":
        return False
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    required_eligibility_artifacts = {
        (
            Path("new_method/artifacts")
            / TASKS[task]["relative"]
            / "candidate_pool_eligibility.csv"
        ).as_posix()
        for task in TASKS
    }
    if not required_eligibility_artifacts.issubset(artifacts):
        return False
    for entry in artifacts.values():
        if not isinstance(entry, dict):
            return False
        artifact = Path(str(entry.get("path", "")))
        if (
            not artifact.is_file()
            or str(entry.get("sha256", "")) != _sha256_file(artifact)
        ):
            return False
    return True


def _task_materialization_valid(
    *,
    task: str,
    selection: Path,
    ledger: Path,
    archive: Path,
    draws: Path,
    archive_manifest: Path,
    source_archive: Path,
    n_draws: int,
    top_k: int,
) -> bool:
    if not all(path.is_file() for path in (selection, ledger, archive, draws, archive_manifest)):
        return False
    try:
        payload = json.loads(archive_manifest.read_text(encoding="utf-8"))
        import pandas as pd

        selected_ids = pd.read_csv(selection, keep_default_na=False)[
            "model_id"
        ].astype(str).tolist()
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        len(selected_ids) == int(top_k)
        and len(set(selected_ids)) == int(top_k)
        and payload.get("selected_model_ids") == selected_ids
        and int(payload.get("models", -1)) == int(top_k)
        and int(payload.get("archive_rows", -1))
        == _expected_selected_archive_rows(
            task,
            int(top_k),
            source_archive=source_archive,
            ledger=ledger,
            selected_model_ids=selected_ids,
        )
        and int(payload.get("n_draws", -1)) == int(n_draws)
        and str(payload.get("selection_sha256", "")) == _sha256_file(selection)
        and str(payload.get("ledger_sha256", "")) == _sha256_file(ledger)
        and str(payload.get("archive_sha256", "")) == _sha256_file(archive)
        and str(payload.get("draws_sha256", "")) == _sha256_file(draws)
    )


def _prepare_shared_selection_input(
    args: argparse.Namespace,
    reference_run: Path,
    out_run: Path,
    selected_tasks: list[str],
) -> Path:
    ""

    if args.shared_root is None:
        return reference_run
    import _candidate_pipeline as manager
    import pandas as pd

    shared = args.shared_root.resolve()
    agent_profile_record = _bind_agent_run_profile(args)
    generated_draws = _input_draws(args)
    cache_draws_label = _candidate_cache_draws(args)
    top_k_by_task = _top_k_by_task(args)
    top_k_suffix = _top_k_identity_suffix(args)
    candidate_cache = manager.shared_candidate_cache_root(
        shared,
        top_k=manager.FORMAL_CANDIDATE_COUNT,
        n_draws=cache_draws_label,
        base_seed=int(args.seed),
    )
    selection_root = manager.require_shared_selection_bank(
        shared,
        candidate_cache,
        str(args.selection_profile),
    )
    requested_hash = str(args.selection_hash).strip()
    if requested_hash != "active" and selection_root.name != requested_hash:
        raise SystemExit(
            "active shared selection hash mismatch: "
            f"expected={requested_hash} actual={selection_root.name}"
        )
    shared_input_root = getattr(args, "shared_input_root", None)
    if shared_input_root is None:
        alternate_input_run = out_run / "inputs" / (
            f"{args.selection_profile}__{selection_root.name[:12]}__"
            f"draws{generated_draws}"
        )
        input_run = out_run / "inputs" / (
            f"{args.selection_profile}__{selection_root.name[:12]}__"
            f"draws{generated_draws}{top_k_suffix}"
        )
    else:
        alternate_input_run = Path(shared_input_root).resolve() / (
            f"seed{int(args.seed)}__cache_draws{cache_draws_label}__"
            f"generated_draws{generated_draws}__{args.selection_profile}__"
            f"{selection_root.name[:12]}"
        )
        input_run = Path(shared_input_root).resolve() / (
            f"seed{int(args.seed)}__cache_draws{cache_draws_label}__"
            f"generated_draws{generated_draws}__{args.selection_profile}__"
            f"{selection_root.name[:12]}{top_k_suffix}"
        )
    manifest_path = input_run / "rho_only_input_manifest.json"
    reusable_alternate_input = (
        top_k_suffix != ""
        and alternate_input_run != input_run
        and _input_manifest_valid(
            alternate_input_run / "rho_only_input_manifest.json"
        )
    )
    if _input_manifest_valid(manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("selection_input_hash") != selection_root.name
            or manifest.get("selection_profile") != str(args.selection_profile)
            or int(manifest.get("generated_draws_per_forecast", -1))
            != generated_draws
            or int(manifest.get("candidate_cache_n_draws_label", -1))
            != cache_draws_label
            or int(manifest.get("seed", -1)) != int(args.seed)
            or {
                str(task): int(value)
                for task, value in manifest.get(
                    "top_k_by_task",
                    {task: DEFAULT_TOP_K for task in TASKS},
                ).items()
            }
            != top_k_by_task
            or Path(str(manifest.get("shared_baseline_root", ""))).resolve()
            != shared
            or str(
                manifest.get(
                    "agent_run_profile",
                    DEFAULT_AGENT_RUN_PROFILE,
                )
            )
            != agent_profile_record["agent_run_profile"]
        ):
            raise SystemExit("existing rho-only input manifest has a different identity")
        print(f"[rho-only] shared input=skip verified {input_run}", flush=True)
        args.shared_selection_root = selection_root
        args.shared_candidate_cache_root = candidate_cache
        args.input_manifest_path = manifest_path
        return input_run
    if args.dry_run:
        print(
            f"[rho-only] shared input=would materialize {input_run} "
            f"selection={selection_root.name}",
            flush=True,
        )
        args.shared_selection_root = selection_root
        args.shared_candidate_cache_root = candidate_cache
        args.input_manifest_path = manifest_path
        return input_run

    input_run.mkdir(parents=True, exist_ok=True)
    selections = manager.materialize_bundle_selection_prefixes(
        selection_root,
        input_run,
        top_k_by_task,
    )
    registry_outputs = manager.materialize_bundle_shared_registry(
        candidate_cache,
        input_run,
    )
    selection_manifest = json.loads(
        (selection_root / "selection_manifest.json").read_text(encoding="utf-8")
    )
    try:
        selection_contract = manager.normalized_selection_contract(
            selection_manifest
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"invalid shared selection contract: {exc}"
        ) from exc
    candidate_manifest = json.loads(
        (candidate_cache / "manifest.json").read_text(encoding="utf-8")
    )
    expected_source_count = int(manager.FORMAL_CANDIDATE_COUNT)
    expected_eligible_count = int(manager.FORMAL_AGENT_ELIGIBLE_COUNT)
    expected_exclusions = list(manager.FORMAL_AGENT_EXCLUDED_MODEL_IDS)
    if (
        str(selection_contract.get("profile", ""))
        != str(args.selection_profile)
        or int(selection_contract.get("source_candidate_count", -1))
        != expected_source_count
        or int(selection_contract.get("eligible_candidate_count", -1))
        != expected_eligible_count
        or list(selection_contract.get("excluded_model_ids", []))
        != expected_exclusions
    ):
        raise SystemExit(
            "rho-only shared selection does not match the configured "
            "candidate-bank policy"
        )
    if (
        int(candidate_manifest.get("top_k", -1)) != expected_source_count
        or int(candidate_manifest.get("n_draws_label", -1)) != cache_draws_label
        or int(candidate_manifest.get("base_seed", -1)) != int(args.seed)
        or candidate_manifest.get("full_bank_draws_generated") is not False
    ):
        raise SystemExit(
            "shared input does not match the configured candidate archive"
        )

    task_records: dict[str, object] = {}
    artifacts: dict[str, dict[str, object]] = {}
    for task in selected_tasks:
        task_top_k = top_k_by_task[task]
        task_root = _task_root(input_run, task)
        task_root.mkdir(parents=True, exist_ok=True)
        ledger = _agent_input_ledger(args, candidate_cache, task)
        selection = selections[task]
        fold = selection.parent / "selection_fold_manifest.csv"
        eligibility = selection_root / task / "candidate_pool_eligibility.csv"
        _safe_link(ledger, task_root / "event_ledger.csv")
        _safe_link(selection, task_root / "candidate_selection_log.csv")
        _safe_link(fold, task_root / "selection_fold_manifest.csv")
        _safe_link(eligibility, task_root / "candidate_pool_eligibility.csv")
        _safe_link(
            registry_outputs["registry"],
            task_root / "model_registry.csv",
        )
        archive = task_root / "forecast_archive.csv"
        draws = task_root / "forecast_draws.csv"
        archive_manifest = task_root / "forecast_archive_manifest.json"
        if reusable_alternate_input and task_top_k == DEFAULT_TOP_K:
            alternate_task_root = _task_root(alternate_input_run, task)
            for name in (
                "forecast_archive.csv",
                "forecast_draws.csv",
                "forecast_archive_manifest.json",
            ):
                destination = task_root / name
                if not destination.exists() and not destination.is_symlink():
                    _safe_link(alternate_task_root / name, destination)
        source_archive = manager.shared_candidate_task_archive(candidate_cache, task)
        command = [
            str(args.python),
            str(ROOT / "scripts/assemble_shared_forecast_archive.py"),
            "--ledger",
            str(ledger),
            "--source-archive",
            str(source_archive),
            "--selection",
            str(selection),
            "--out",
            str(archive),
            "--draws-out",
            str(draws),
            "--n-draws",
            str(generated_draws),
            "--seed",
            str(int(args.seed)),
            "--manifest",
            str(archive_manifest),
            "--derived-view",
            *_task_args(task),
        ]
        if _task_materialization_valid(
            task=task,
            selection=selection,
            ledger=ledger,
            archive=archive,
            draws=draws,
            archive_manifest=archive_manifest,
            source_archive=source_archive,
            n_draws=generated_draws,
            top_k=task_top_k,
        ):
            print(
                f"[rho-only] {task} shared archive/draws=skip verified",
                flush=True,
            )
        else:
            _run(
                command,
                label=f"{task} materialize versioned Top-{task_top_k} archive and draws",
                log_path=out_run / f"logs/materialize_{task}.log",
                dry_run=False,
            )
        archive_payload = json.loads(archive_manifest.read_text(encoding="utf-8"))
        selected_ids = pd.read_csv(selection, keep_default_na=False)[
            "model_id"
        ].astype(str).tolist()
        if (
            int(archive_payload.get("models", -1)) != task_top_k
            or archive_payload.get("selected_model_ids") != selected_ids
            or set(selected_ids) & set(expected_exclusions)
            or int(archive_payload.get("n_draws", -1)) != generated_draws
            or int(archive_payload.get("archive_rows", -1))
            != _expected_selected_archive_rows(
                task,
                task_top_k,
                source_archive=source_archive,
                ledger=ledger,
                selected_model_ids=selected_ids,
            )
        ):
            raise RuntimeError(f"invalid materialized shared archive: {archive_manifest}")
        task_records[task] = {
            "top_k": task_top_k,
            "selected_model_ids": selected_ids,
            "archive_rows": int(archive_payload["archive_rows"]),
            "draw_rows": int(archive_payload["archive_rows"]) * generated_draws,
            "archive_sha256": str(archive_payload["archive_sha256"]),
            "draws_sha256": str(archive_payload["draws_sha256"]),
        }
        for artifact in (
            task_root / "event_ledger.csv",
            task_root / "candidate_selection_log.csv",
            task_root / "selection_fold_manifest.csv",
            task_root / "candidate_pool_eligibility.csv",
            task_root / "model_registry.csv",
            archive,
            draws,
            archive_manifest,
        ):
            artifacts[artifact.relative_to(input_run).as_posix()] = {
                "path": str(artifact.resolve()),
                "sha256": _sha256_file(artifact),
            }
                                                                         
                                                                        
        for name in (
            "candidate_ranking_all27.csv",
            "top10_selection.csv",
            "selection_context.json",
            "selection_context_validation.csv",
            "sequence_sketch.json",
            "task_embedding_manifest.json",
        ):
            artifact = selection_root / task / name
            if artifact.is_file():
                artifacts[f"external/selection/{task}/{name}"] = {
                    "path": str(artifact.resolve()),
                    "sha256": _sha256_file(artifact),
                }

    _safe_link(shared / "baseline", input_run / "baseline", directory=True)
    _safe_link(reference_run / "events", input_run / "events", directory=True)
    source_config = {
        "schema": "caster_rho_only_shared_selection_input_v3",
        "reference_run": str(reference_run),
        "shared_baseline_root": str(shared),
        "baseline_source_root": str((shared / "baseline").resolve()),
        "events_source_root": str((reference_run / "events").resolve()),
        "n_draws": generated_draws,
        "top_k_by_task": top_k_by_task,
        "candidate_cache_n_draws_label": cache_draws_label,
        "selection_profile": str(args.selection_profile),
        "selection_input_hash": selection_root.name,
        "selection_profile_status": str(selection_contract["status"]),
        "formal_preregistration_claimed": bool(
            selection_contract["formal_preregistration_claimed"]
        ),
        "selection_contract": selection_contract,
        "formal_caster_only": {
            "shared_selection_root": str(selection_root),
            "shared_candidate_cache_root": str(candidate_cache),
        },
        **agent_profile_record,
    }
    source_config_path = input_run / "run_config.json"
    source_config_path.write_text(
        json.dumps(source_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["run_config.json"] = {
        "path": str(source_config_path.resolve()),
        "sha256": _sha256_file(source_config_path),
    }
    pointer_path = selection_root.parent / "active_selection.json"
    shared_status_path = shared / "status.json"
    candidate_manifest_path = candidate_cache / "manifest.json"
    for label, artifact in (
        ("external/shared_status.json", shared_status_path),
        ("external/candidate_cache_manifest.json", candidate_manifest_path),
        ("external/selection_manifest.json", selection_root / "selection_manifest.json"),
        ("external/active_selection.json", pointer_path),
    ):
        artifacts[label] = {
            "path": str(artifact.resolve()),
            "sha256": _sha256_file(artifact),
        }
    agent_overall_manifest = str(
        agent_profile_record.get("agent_overall_manifest_path", "")
    ).strip()
    if agent_overall_manifest:
        artifact = Path(agent_overall_manifest)
        artifacts["external/agent_overall_manifest.json"] = {
            "path": str(artifact.resolve()),
            "sha256": _sha256_file(artifact),
        }
    input_manifest = {
        "schema": "caster_rho_only_shared_selection_input_manifest_v3",
        "selection_profile": str(args.selection_profile),
        "selection_input_hash": selection_root.name,
        "selection_profile_status": str(selection_contract["status"]),
        "formal_preregistration_claimed": bool(
            selection_contract["formal_preregistration_claimed"]
        ),
        "selection_contract": selection_contract,
        "selection_manifest_sha256": _sha256_file(
            selection_root / "selection_manifest.json"
        ),
        "active_selection_pointer_sha256": _sha256_file(pointer_path),
        "candidate_cache_manifest_sha256": _sha256_file(
            candidate_manifest_path
        ),
        "reference_run": str(reference_run),
        "shared_baseline_root": str(shared),
        "baseline_source_root": str((shared / "baseline").resolve()),
        "events_source_root": str((reference_run / "events").resolve()),
        "generated_draws_per_forecast": generated_draws,
        "top_k_by_task": top_k_by_task,
        "candidate_cache_n_draws_label": cache_draws_label,
        "seed": int(args.seed),
        **agent_profile_record,
        "tasks": task_records,
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.shared_selection_root = selection_root
    args.shared_candidate_cache_root = candidate_cache
    args.input_manifest_path = manifest_path
    print(f"[rho-only] shared input=created {input_run}", flush=True)
    return input_run


def _censoring_support_manifest_valid(
    path: Path,
    *,
    task: str,
    ledger: Path,
    archive: Path,
    eligibility: Path,
    candidate_manifest: Path,
    declared_archive_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        base_upper = payload["base_upper_raw_by_component"]
        eligible_model_ids = [
            str(value) for value in payload["eligible_model_ids"]
        ]
        values = [float(value) for value in base_upper.values()]
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    if (
        not isinstance(payload, dict)
        or not isinstance(base_upper, dict)
        or not values
        or any(not math.isfinite(value) or value <= 0.0 for value in values)
    ):
        return False
    try:
        expected_hashes = {
            "ledger_sha256": _sha256_file(ledger),
            "eligibility_sha256": _sha256_file(eligibility),
            "candidate_manifest_sha256": _sha256_file(candidate_manifest),
        }
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        return (
            payload.get("schema") == "caster_censoring_support_manifest_v1"
            and str(payload.get("task_id", "")) == task
            and str(payload.get("bound_scope", "")) == "eligible27_train"
            and str(payload.get("source_split", "")) == "train"
            and int(payload.get("eligible_model_count", -1)) == 27
            and len(eligible_model_ids) == 27
            and len(set(eligible_model_ids)) == 27
            and int(payload.get("archive_train_rows_used", -1))
            == int(payload.get("expected_archive_train_rows", -2))
            and int(payload.get("validation_rows_used", -1)) == 0
            and int(payload.get("test_rows_used", -1)) == 0
            and int(payload.get("test_targets_used", -1)) == 0
            and math.isclose(
                float(
                    payload.get(
                        "predictive_standard_deviations", math.nan
                    )
                ),
                4.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(payload.get("minimum_upper_raw", math.nan)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and Path(str(payload.get("ledger_path", ""))).resolve()
            == ledger.resolve()
            and Path(str(payload.get("archive_path", ""))).resolve()
            == archive.resolve()
            and Path(
                str(payload.get("eligibility_path", ""))
            ).resolve()
            == eligibility.resolve()
            and Path(
                str(payload.get("candidate_manifest_path", ""))
            ).resolve()
            == candidate_manifest.resolve()
            and str(payload.get("archive_sha256", ""))
            == str(declared_archive_sha256)
            and all(
                str(payload.get(key, "")) == expected
                for key, expected in expected_hashes.items()
            )
        )
    except (OSError, TypeError, ValueError):
        return False


def _prepare_censoring_support_manifests(
    args: argparse.Namespace,
    out_run: Path,
    selected_tasks: list[str],
) -> None:
    ""

    scope = _censoring_bound_scope(args)
    args.censoring_support_manifests = {}
    if scope == "selected_topk_train":
        return
    candidate_cache_raw = getattr(args, "shared_candidate_cache_root", None)
    selection_root_raw = getattr(args, "shared_selection_root", None)
    if candidate_cache_raw is None or selection_root_raw is None:
        source_config_path = Path(args.source_run).resolve() / "run_config.json"
        try:
            source_config = json.loads(
                source_config_path.read_text(encoding="utf-8")
            )
            formal_inputs = source_config["formal_caster_only"]
            candidate_cache_raw = formal_inputs[
                "shared_candidate_cache_root"
            ]
            selection_root_raw = formal_inputs["shared_selection_root"]
        except (
            OSError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            candidate_cache_raw = None
            selection_root_raw = None
    if candidate_cache_raw is None or selection_root_raw is None:
        raise SystemExit(
            "eligible27_train censoring support requires either --shared-root "
            "or a source bundle that freezes the active candidate/selection "
            "roots"
        )
    candidate_cache = Path(candidate_cache_raw).resolve()
    selection_root = Path(selection_root_raw).resolve()
    candidate_manifest = candidate_cache / "manifest.json"
    try:
        candidate_payload = json.loads(
            candidate_manifest.read_text(encoding="utf-8")
        )
        task_payloads = candidate_payload["tasks"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"invalid shared candidate manifest: {candidate_manifest}"
        ) from exc
    support_root = out_run / "inputs/censoring_support"
    records: dict[str, dict[str, object]] = {}
    for task in selected_tasks:
        ledger = _agent_input_ledger(args, candidate_cache, task)
        archive = candidate_cache / task / FULL_CANDIDATE_ARCHIVE_NAME
        eligibility = selection_root / task / "candidate_pool_eligibility.csv"
        out = support_root / f"{task}.eligible27_train.json"
        try:
            archive_sha256 = str(task_payloads[task]["archive_sha256"])
        except (KeyError, TypeError) as exc:
            raise SystemExit(
                f"candidate manifest has no archive identity for {task}"
            ) from exc
        records[task] = {
            "ledger": ledger,
            "archive": archive,
            "eligibility": eligibility,
            "out": out,
            "archive_sha256": archive_sha256,
        }

    def materialize(task: str) -> None:
        record = records[task]
        out = Path(record["out"])
        if _censoring_support_manifest_valid(
            out,
            task=task,
            ledger=Path(record["ledger"]),
            archive=Path(record["archive"]),
            eligibility=Path(record["eligibility"]),
            candidate_manifest=candidate_manifest,
            declared_archive_sha256=str(record["archive_sha256"]),
        ):
            print(
                f"[rho-only] {task} eligible27 train support=skip verified",
                flush=True,
            )
            return
        _run(
            [
                str(args.python),
                str(ROOT / "scripts/build_censoring_support_manifest.py"),
                "--ledger",
                str(record["ledger"]),
                "--archive",
                str(record["archive"]),
                "--eligibility",
                str(record["eligibility"]),
                "--task-id",
                task,
                "--out",
                str(out),
                "--declared-archive-sha256",
                str(record["archive_sha256"]),
                "--candidate-manifest",
                str(candidate_manifest),
            ],
            label=f"{task} freeze eligible27 train-only censoring support",
            log_path=out_run / f"logs/censoring_support_{task}.log",
            dry_run=bool(args.dry_run),
        )
        if not args.dry_run and not _censoring_support_manifest_valid(
            out,
            task=task,
            ledger=Path(record["ledger"]),
            archive=Path(record["archive"]),
            eligibility=Path(record["eligibility"]),
            candidate_manifest=candidate_manifest,
            declared_archive_sha256=str(record["archive_sha256"]),
        ):
            raise RuntimeError(
                f"invalid eligible27 train support manifest: {out}"
            )

    worker_count = min(
        max(1, _task_worker_count(args)),
        len(selected_tasks),
        4,
    )
    if worker_count == 1 or args.dry_run:
        for task in selected_tasks:
            materialize(task)
    else:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="censoring-support",
        ) as executor:
            futures = {
                task: executor.submit(materialize, task)
                for task in selected_tasks
            }
            for task, future in futures.items():
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"censoring support worker failed for {task}"
                    ) from exc
    args.censoring_support_manifests = {
        task: Path(records[task]["out"]).resolve()
        for task in selected_tasks
    }


def _run(
    command: Sequence[str],
    *,
    label: str,
    log_path: Path,
    dry_run: bool,
) -> None:
    rendered = shlex.join([str(value) for value in command])
    print(f"[rho-only] {label}: {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _task_args(task: str) -> list[str]:
    spec = TASKS[task]
    args = ["--task-id", task, "--target-components", str(spec["components"])]
    if spec["scope"]:
        args.extend(["--posterior-scope", str(spec["scope"])])
    return args


def _prepare_task_inputs(
    source_root: Path,
    out_root: Path,
    *,
    include_draws: bool = False,
) -> None:
    names = [
        "event_ledger.csv",
        "forecast_archive.csv",
        "forecast_archive_manifest.json",
        "selection_fold_manifest.csv",
    ]
    if include_draws:
        names.append("forecast_draws.csv")
    for name in names:
        _safe_link(source_root / name, out_root / name)


def _float_mapping_matches(
    observed: object,
    expected: dict[str, float],
) -> bool:
    if not isinstance(observed, dict) or set(observed) != set(expected):
        return False
    try:
        return all(
            math.isclose(
                float(observed[key]),
                float(value),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key, value in expected.items()
        )
    except (TypeError, ValueError):
        return False


def _isclose_optional_number(observed: object, expected: float) -> bool:
    try:
        return math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        return False


def _frozen_calibration_artifacts_valid(
    calibration_root: Path,
    freeze: dict[str, object],
    family: str,
) -> bool:
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    required = {
        "bridge_config.json",
        "bridge_config.one_layer.json",
        "bridge_config.hierarchical.json",
        f"bridge_config.one_layer.{family}.json",
        f"bridge_config.hierarchical.{family}.json",
        "bridge_component_calibration_report.csv",
        "parameter_selection_reference.json",
        "parameter_selection_trace.csv",
        "small_validation_manifest.csv",
    }
    if not required.issubset(artifacts):
        return False
    try:
        for name, entry in artifacts.items():
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not isinstance(entry, dict)
            ):
                return False
            expected_path = (calibration_root / name).resolve()
            recorded_path = Path(str(entry.get("path", ""))).resolve()
            if recorded_path != expected_path or not expected_path.is_file():
                return False
            if str(entry.get("sha256", "")) != _sha256_file(expected_path):
                return False
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _selected_state_matches_config(
    selected: object,
    config: dict[str, object],
    *,
    family: str,
    distribution: str,
    predictive_contract: str,
    fixed_gamma: float,
    fixed_nu: float | None,
    rho_bounds: list[float],
) -> bool:
    if not isinstance(selected, dict) or not isinstance(config, dict):
        return False
    selected_distribution = str(selected.get("distribution", "student_t"))
    try:
        selected_rho = float(selected["rho"])
        config_rho = float(config["rho"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        str(selected.get("family", "")) != family
        or selected_distribution != distribution
        or (
            _fixed_c_u_is_active(predictive_contract)
            and _recorded_predictive_contract(selected)
            != predictive_contract
        )
        or not math.isfinite(selected_rho)
        or not rho_bounds[0] <= selected_rho <= rho_bounds[1]
        or not math.isclose(selected_rho, config_rho, rel_tol=0.0, abs_tol=1e-12)
    ):
        return False
    scales = selected.get("scales")
    if not isinstance(scales, dict) or not scales:
        return False
    scale_field = "sigma_by_component" if family == "moment_t" else "tau_by_component"
    if not _float_mapping_matches(config.get(scale_field), scales):
        return False
    inactive_scale_field = (
        "tau_by_component" if family == "moment_t" else "sigma_by_component"
    )
    if config.get(inactive_scale_field, {}) != {}:
        return False

    gammas = selected.get("gammas")
    if not isinstance(gammas, dict):
        return False
    if family == "moment_t":
        try:
            gamma_mismatch = set(gammas) != set(scales) or any(
                not math.isclose(
                    float(value), fixed_gamma, rel_tol=0.0, abs_tol=1e-12
                )
                for value in gammas.values()
            )
        except (TypeError, ValueError):
            return False
        if gamma_mismatch:
            return False
    elif gammas:
        return False
    if not _float_mapping_matches(config.get("gamma_by_component", {}), gammas):
        return False

    nus = selected.get("nus")
    if not isinstance(nus, dict):
        return False
    if distribution == "student_t":
        try:
            if set(nus) != set(scales) or any(
                fixed_nu is None
                or not math.isclose(
                    float(value), float(fixed_nu), rel_tol=0.0, abs_tol=1e-12
                )
                for value in nus.values()
            ):
                return False
        except (TypeError, ValueError):
            return False
    elif nus:
        return False
    return _float_mapping_matches(config.get("nu_by_component", {}), nus)


def _calibration_identity_valid(
    args: argparse.Namespace,
    source_root: Path,
    calibration_root: Path,
    task: str,
    family: str,
) -> bool:
    freeze_path = calibration_root / "parameter_selection_freeze_manifest.json"
    config_paths = {
        variant: calibration_root / f"bridge_config.{variant}.json"
        for variant in ("one_layer", "hierarchical")
    }
    if not freeze_path.is_file() or any(
        not path.is_file() for path in config_paths.values()
    ):
        return False
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        configs = {
            variant: json.loads(path.read_text(encoding="utf-8"))
            for variant, path in config_paths.items()
        }
        fixed = freeze["fixed_parameters"]
        optimizer_settings = freeze["optimizer_settings"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if (
        not isinstance(freeze, dict)
        or not isinstance(fixed, dict)
        or not isinstance(optimizer_settings, dict)
        or any(not isinstance(config, dict) for config in configs.values())
    ):
        return False
    try:
        rho_bounds = [float(value) for value in _rho_bounds_for_task(args, task)]
        frozen_bounds = [float(value) for value in freeze.get("rho_bounds", [])]
        frozen_seed = int(freeze.get("small_validation_seed", -1))
        frozen_max_evaluations = int(
            optimizer_settings.get("max_evaluations", -1)
        )
    except (TypeError, ValueError):
        return False
    if frozen_bounds != rho_bounds:
        return False
    if str(freeze.get("rho_objective_rmse_mode", "short-long")) != (
        _rmse_objective_mode(args)
    ):
        return False
    if not _float_mapping_matches(
        freeze.get("objective_weights"),
        _objective_weights(args),
    ):
        return False
    distribution = _distribution(args)
    predictive_contract = _predictive_contract(args)
    fixed_c_u = _fixed_c_u(args)
    fixed_c_u_active = _fixed_c_u_is_active(predictive_contract)
    censoring_bound_scope = _censoring_bound_scope(args)
    censoring_support_sha256 = ""
    if fixed_c_u_active and censoring_bound_scope == "eligible27_train":
        support_path = getattr(
            args, "censoring_support_manifests", {}
        ).get(task)
        if support_path is None or not Path(support_path).is_file():
            return False
        try:
            censoring_support_sha256 = _sha256_file(Path(support_path))
        except (OSError, RuntimeError, ValueError):
            return False
    if (
        str(freeze.get("parameter_selection_protocol", ""))
        != "fixed_formula_rho_only_smallval_newton_v1"
        or str(freeze.get("task_id", "")) != task
        or str(fixed.get("family", "")) != family
        or str(fixed.get("distribution", "")) != distribution
        or _recorded_predictive_contract(fixed) != predictive_contract
        or str(fixed.get("sigma_formula", ""))
        != "alternate_log1p_transform_residual_rmse"
        or str(fixed.get("tau", "")) != "sigma"
        or frozen_seed != int(args.seed)
        or frozen_max_evaluations != int(args.max_newton_evaluations)
    ):
        return False
    if fixed_c_u_active:
        try:
            if not math.isclose(
                float(fixed.get("fixed_c_u")),
                fixed_c_u,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        except (TypeError, ValueError):
            return False
        if (
            str(
                fixed.get(
                    "censoring_bound_scope", "selected_topk_train"
                )
            )
            != censoring_bound_scope
            or (
                censoring_bound_scope == "eligible27_train"
                and str(
                    fixed.get("censoring_support_manifest_sha256", "")
                )
                != censoring_support_sha256
            )
        ):
            return False
    if family == "moment_t":
        try:
            if not math.isclose(
                float(fixed.get("gamma")),
                float(args.fixed_gamma),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        except (TypeError, ValueError):
            return False
    elif fixed.get("gamma") is not None:
        return False
    expected_nu = _fixed_nu(args)
    if fixed.get("nu") != expected_nu:
        return False
    if not _frozen_calibration_artifacts_valid(calibration_root, freeze, family):
        return False
    try:
        smallval_sha256 = _sha256_file(
            calibration_root / "small_validation_manifest.csv"
        )
    except (OSError, RuntimeError, ValueError):
        return False
    if str(freeze.get("small_validation_manifest_sha256", "")) != smallval_sha256:
        return False
    expected_hashes = {
        "ledger_sha256": source_root / "event_ledger.csv",
        "archive_sha256": source_root / "forecast_archive.csv",
        "registry_sha256": source_root / "model_registry.csv",
        "selection_sha256": source_root / "candidate_selection_log.csv",
    }
    if family == "draw_kernel_t":
        expected_hashes["draws_sha256"] = source_root / "forecast_draws.csv"
    try:
        expected_source_hashes = {
            key: _sha256_file(path) for key, path in expected_hashes.items()
        }
        selection_fold_manifest_sha256 = _selection_fold_manifest_sha256(
            source_root / "selection_fold_manifest.csv"
        )
    except (OSError, RuntimeError, ValueError):
        return False
    if str(freeze.get("selection_fold_manifest_sha256", "")) != (
        selection_fold_manifest_sha256
    ):
        return False
    selected_by_variant = freeze.get("selected")
    if not isinstance(selected_by_variant, dict):
        return False
    for variant, config in configs.items():
        try:
            metadata = config["calibration_metadata"]
            metadata_bounds = [float(value) for value in metadata.get("rho_bounds", [])]
            metadata_optimizer = metadata["optimizer"]
            metadata_seed = int(metadata.get("small_validation_seed", -1))
            metadata_max_evaluations = int(
                metadata_optimizer.get("max_evaluations", -1)
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        if not isinstance(metadata, dict) or not isinstance(metadata_optimizer, dict):
            return False
        if (
            str(config.get("distribution", "")) != distribution
            or str(config.get("kernel_distribution", "")) != distribution
            or _recorded_predictive_contract(config) != predictive_contract
            or str(metadata.get("distribution", "")) != distribution
            or str(metadata.get("kernel_distribution", "")) != distribution
            or _recorded_predictive_contract(metadata) != predictive_contract
            or (
                fixed_c_u_active
                and not _isclose_optional_number(
                    metadata.get("fixed_c_u"),
                    fixed_c_u,
                )
            )
            or (
                fixed_c_u_active
                and str(
                    metadata.get(
                        "censoring_bound_scope",
                        "selected_topk_train",
                    )
                )
                != censoring_bound_scope
            )
            or (
                fixed_c_u_active
                and censoring_bound_scope == "eligible27_train"
                and str(
                    metadata.get(
                        "censoring_support_manifest_sha256", ""
                    )
                )
                != censoring_support_sha256
            )
            or str(metadata.get("task_id", "")) != task
            or str(metadata.get("rho_selection_variant", "")) != variant
            or str(metadata.get("parameter_selection_protocol", ""))
            != "fixed_formula_rho_only_smallval_newton_v1"
            or metadata_seed != int(args.seed)
            or metadata_bounds != rho_bounds
            or str(metadata.get("rho_objective_rmse_mode", "short-long"))
            != _rmse_objective_mode(args)
            or not _float_mapping_matches(
                metadata.get("objective_weights"), _objective_weights(args)
            )
            or str(metadata_optimizer.get("name", ""))
            != "safeguarded_bounded_newton_log_rho"
            or metadata_max_evaluations != int(args.max_newton_evaluations)
            or str(metadata.get("small_validation_manifest_sha256", ""))
            != smallval_sha256
            or str(metadata.get("selection_fold_manifest_sha256", ""))
            != selection_fold_manifest_sha256
            or not _selected_state_matches_config(
                selected_by_variant.get(variant),
                config,
                family=family,
                distribution=distribution,
                predictive_contract=predictive_contract,
                fixed_gamma=float(args.fixed_gamma),
                fixed_nu=expected_nu,
                rho_bounds=rho_bounds,
            )
        ):
            return False
        if distribution == "student_t":
            try:
                if (
                    expected_nu is None
                    or not math.isclose(
                        float(config.get("nu")),
                        float(expected_nu),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        float(config.get("kernel_nu")),
                        float(expected_nu),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        float(metadata.get("nu")),
                        float(expected_nu),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or str(metadata.get("nu_grid", "")) != f"{expected_nu:g}"
                    or not math.isclose(
                        float(metadata.get("fixed_nu")),
                        float(expected_nu),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    return False
            except (TypeError, ValueError):
                return False
        try:
            if any(
                str(metadata.get(key, "")) != expected_sha256
                for key, expected_sha256 in expected_source_hashes.items()
            ):
                return False
        except (OSError, RuntimeError, ValueError):
            return False
    return True


def _filter_identity_valid(
    args: argparse.Namespace,
    marker: Path,
    bridge_config: Path,
    source_root: Path,
    family: str,
    variant: str,
) -> bool:
    if variant not in {"one_layer", "hierarchical"}:
        return False
    required_outputs = (
        (
            "posterior_path.csv",
            "forecast_readout.csv",
            "initial_prior.csv",
            "asof_posterior_readout_validation.csv",
        )
        if variant == "one_layer"
        else (
            "hierarchical_posterior_path.csv",
            "hierarchical_forecast_readout.csv",
            "family_posterior.csv",
            "asof_posterior_readout_validation.csv",
        )
    )
    if (
        not marker.is_file()
        or not bridge_config.is_file()
        or any(
            not (marker.parent / name).is_file()
            or (marker.parent / name).stat().st_size <= 0
            for name in required_outputs
        )
    ):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        config = json.loads(bridge_config.read_text(encoding="utf-8"))
        calibration_metadata = config["calibration_metadata"]
        expected_hashes = {
            "bridge_config_sha256": _sha256_file(bridge_config),
            "ledger_sha256": _sha256_file(source_root / "event_ledger.csv"),
            "archive_sha256": _sha256_file(source_root / "forecast_archive.csv"),
            "registry_sha256": _sha256_file(source_root / "model_registry.csv"),
            "selection_sha256": _sha256_file(
                source_root / "candidate_selection_log.csv"
            ),
        }
        if family == "draw_kernel_t":
            expected_hashes["draws_sha256"] = _sha256_file(
                source_root / "forecast_draws.csv"
            )
        future_violations = int(
            payload.get("readout_rows_future_snapshot_violation", -1)
        )
        self_violations = int(
            payload.get("readout_rows_self_target_update_violation", -1)
        )
        readout_rows = int(payload.get("readout_rows", 0))
        max_snapshot_days = float(payload.get("max_snapshot_after_origin_days", 1.0))
        recorded_bridge_config = Path(str(payload.get("bridge_config", ""))).resolve()
    except (
        AttributeError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    if not isinstance(payload, dict) or not isinstance(calibration_metadata, dict):
        return False
    expected_source = "draw_kernel" if family == "draw_kernel_t" else "archive_moment"
    expected_method = f"caster_{variant}"
    if family == "draw_kernel_t":
        expected_method += "_draw_kernel"
    expected_metadata_variant = (
        "one_layer_draw_kernel"
        if family == "draw_kernel_t" and variant == "one_layer"
        else variant
    )
    expected_distribution = _distribution(args)
    expected_predictive_contract = _predictive_contract(args)
    return (
        all(
            str(payload.get(key, "")) == value
            for key, value in expected_hashes.items()
        )
        and str(payload.get("bridge_distribution", "")) == expected_distribution
        and str(payload.get("kernel_distribution", "")) == expected_distribution
        and _recorded_predictive_contract(payload) == expected_predictive_contract
        and _recorded_predictive_contract(config) == expected_predictive_contract
        and _recorded_predictive_contract(calibration_metadata)
        == expected_predictive_contract
        and str(payload.get("score_source", "")) == expected_source
        and str(payload.get("result_method_id", "")) == expected_method
        and recorded_bridge_config == bridge_config.resolve()
        and str(payload.get("variant", "")) == expected_metadata_variant
        and str(payload.get("task_id", ""))
        == str(calibration_metadata.get("task_id", ""))
        and str(payload.get("posterior_update_policy", ""))
        == "prequential_asof_release_lte_origin"
        and str(payload.get("posterior_readout_policy", ""))
        == "asof_release_time_lte_forecast_origin"
        and str(payload.get("readout_split", "")) == "test"
        and str(payload.get("test_rows_used_for_posterior_update_policy", ""))
        == "released_evidence_only"
        and str(payload.get("embargo_posterior_update_policy", ""))
        == "released_evidence_only_asof"
        and payload.get("availability_provenance_required") is True
        and payload.get("gaussian_as_student_t_limit") is False
        and future_violations == 0
        and self_violations == 0
        and readout_rows > 0
        and max_snapshot_days <= 0.0
    )


def _calibrate_command(
    python: Path,
    source_root: Path,
    out_root: Path,
    task: str,
    args: argparse.Namespace,
    *,
    family: str,
) -> list[str]:
    rho_min, rho_max = _rho_bounds_for_task(args, task)
    weights = _objective_weights(args)
    distribution = _distribution(args)
    predictive_contract = _predictive_contract(args)
    command = [
        str(python),
        str(CODE_ROOT / "scripts/calibrate_likelihood_bridge.py"),
        "--ledger",
        str(source_root / "event_ledger.csv"),
        "--archive",
        str(source_root / "forecast_archive.csv"),
        "--registry",
        str(source_root / "model_registry.csv"),
        "--selection",
        str(source_root / "candidate_selection_log.csv"),
        "--selection-fold-manifest",
        str(source_root / "selection_fold_manifest.csv"),
        "--out-config",
        str(out_root / "bridge_config.json"),
        "--out-report",
        str(out_root / "temperature_report.csv"),
        "--calibration-mode",
        "fixed_bridge_rho_only",
        "--parameter-selection-protocol",
        "frozen_joint_multicriterion_causal_replay",
        "--distribution",
        distribution,
        "--predictive-contract",
        predictive_contract,
        "--transform",
        "log1p",
        "--fixed-gamma",
        f"{float(args.fixed_gamma):g}",
        "--fixed-bridge-family",
        family,
        "--min-sigma",
        "0.04",
        "--rho-grid",
        f"{rho_min:g},{rho_max:g}",
        "--allow-rho-grid-outside-result-range",
        "--rho-selection-variant",
        "both",
        "--small-validation-folds",
        "10",
        "--small-validation-seed",
        str(args.seed),
        "--rho-newton-max-evaluations",
        str(args.max_newton_evaluations),
        "--rho-objective-rmse-mode",
        _rmse_objective_mode(args),
        "--rho-objective-weight-nll",
        f"{weights['nll']:g}",
        "--rho-objective-weight-wis",
        f"{weights['wis']:g}",
        "--rho-objective-weight-short-rmse",
        f"{weights.get('short_rmse', 0.0):g}",
        "--rho-objective-weight-long-rmse",
        f"{weights.get('long_rmse', 0.0):g}",
        "--rho-objective-weight-overall-rmse",
        f"{weights.get('overall_rmse', 0.0):g}",
        "--rho-objective-weight-mae",
        f"{weights['mae']:g}",
        "--rho-objective-weight-coverage-penalty",
        f"{weights['coverage_penalty']:g}",
        "--seed",
        str(args.seed),
        *_task_args(task),
    ]
    if distribution == "student_t":
        command.extend(["--fixed-nu", f"{float(_fixed_nu(args)):g}"])
    if _fixed_c_u_is_active(predictive_contract):
        command.extend(["--fixed-c-u", f"{_fixed_c_u(args):g}"])
        if _censoring_bound_scope(args) == "eligible27_train":
            support_manifests = getattr(
                args, "censoring_support_manifests", {}
            )
            support_manifest = support_manifests.get(task)
            if support_manifest is None:
                raise RuntimeError(
                    f"missing eligible27 censoring support manifest for {task}"
                )
            command.extend(
                [
                    "--censoring-support-manifest",
                    str(Path(support_manifest).resolve()),
                ]
            )
    if family == "draw_kernel_t":
        command.extend(["--draws", str(source_root / "forecast_draws.csv")])
    return command


def _filter_command(
    python: Path,
    source_root: Path,
    calibration_root: Path,
    run_root: Path,
    task: str,
    *,
    hierarchical: bool,
    family: str,
    source_run: Path,
    seed: int,
    predictive_contract: str = alternate_PREDICTIVE_CONTRACT,
) -> list[str]:
    script = (
        "run_hierarchical_from_archive.py"
        if hierarchical
        else "run_caster_from_archive.py"
    )
    variant = "hierarchical" if hierarchical else "one_layer"
    is_draw = family == "draw_kernel_t"
    method = "caster_hierarchical" if hierarchical else "caster_one_layer"
    if is_draw:
        method += "_draw_kernel"
    command = [
        str(python),
        str(CODE_ROOT / "scripts" / script),
        "--ledger",
        str(calibration_root / "event_ledger.csv"),
        "--archive",
        str(calibration_root / "forecast_archive.csv"),
        "--registry",
        str(source_root / "model_registry.csv"),
        "--selection",
        str(source_root / "candidate_selection_log.csv"),
        "--bridge-config",
        str(calibration_root / f"bridge_config.{variant}.json"),
        "--out",
        str(run_root),
        "--update-splits",
        "train,val,embargo,test",
        "--readout-split",
        "test",
        "--posterior-update-policy",
        "prequential_asof",
        "--score-source",
        "draw_kernel" if is_draw else "archive_moment",
        "--predictive-contract",
        predictive_contract,
        "--method-id",
        method,
        *_task_args(task),
    ]
    if is_draw:
        command.extend(
            [
                "--draws",
                str(calibration_root / "forecast_draws.csv"),
                "--draw-kernel-bandwidth-source",
                "tau_equals_alternate_sigma_validation_frozen",
            ]
        )
    command.extend(["--seed", str(seed)])
    return command


def _write_run_config(args: argparse.Namespace, selected_tasks: list[str]) -> None:
    families = ["moment_t"]
    if args.include_draw_kernel:
        families.append("draw_kernel_t")
    distribution = _distribution(args)
    predictive_contract = _predictive_contract(args)
    fixed_c_u = _fixed_c_u(args)
    fixed_c_u_active = _fixed_c_u_is_active(predictive_contract)
    rho_bounds_by_task = {
        task: [float(value) for value in _rho_bounds_for_task(args, task)]
        for task in selected_tasks
    }
    input_manifest_path = getattr(args, "input_manifest_path", None)
    shared_selection_root = getattr(args, "shared_selection_root", None)
    censoring_bound_scope = _censoring_bound_scope(args)
    censoring_support_manifests = getattr(
        args, "censoring_support_manifests", {}
    )
    agent_profile_record = _bind_agent_run_profile(args)
    top_k_by_task = _top_k_by_task(args, selected_tasks)
    common_top_k = (
        next(iter(top_k_by_task.values()))
        if len(set(top_k_by_task.values())) == 1
        else None
    )
    payload = {
        "schema": "caster_rho_only_smallval_pilot_v5_fixed_nu_explicit",
        "command_line": [str(value) for value in sys.argv],
        "source_run": str(args.source_run.resolve()),
        "reference_source_run": str(
            getattr(args, "reference_source_run", args.source_run).resolve()
        ),
        "output_run": str(args.out_run.resolve()),
        "tasks": selected_tasks,
        "top_k": common_top_k,
        "top_k_by_task": top_k_by_task,
        "seed": int(args.seed),
        "task_workers": _task_worker_count(args),
        "export_workers": _export_worker_count(args),
        "n_draws": _input_draws(args),
        "generated_draws_per_forecast": _input_draws(args),
        "candidate_cache_n_draws_label": _candidate_cache_draws(args),
        "predictive_contract": predictive_contract,
        "censoring_bound_scope": (
            censoring_bound_scope
            if fixed_c_u_active
            else "not_applicable_for_predictive_contract"
        ),
        "censoring_support_manifests": {
            task: {
                "path": str(Path(path).resolve()),
                "sha256": (
                    _sha256_file(Path(path))
                    if Path(path).is_file()
                    else ""
                ),
            }
            for task, path in sorted(censoring_support_manifests.items())
        },
        "shared_input_root": (
            str(Path(args.shared_input_root).resolve())
            if getattr(args, "shared_input_root", None) is not None
            else ""
        ),
        "small_validation_endpoint_rows_per_task": 360,
        "small_validation_folds": 10,
        "fixed_parameters": {
            "families": families,
            "display_families": (
                families
                if distribution == "student_t"
                else [
                    "moment_gaussian"
                    if family == "moment_t"
                    else "draw_kernel_gaussian"
                    for family in families
                ]
            ),
            "sigma": "alternate_log1p_transform_residual_rmse_component_horizon",
            "draw_kernel_tau": "sigma",
            "moment_t_gamma": float(args.fixed_gamma),
            "draw_kernel_gamma": "not_applicable_inactive_for_draw_kernel",
            "distribution": distribution,
            "nu": _reported_nu(args),
            "predictive_contract": predictive_contract,
            "fixed_c_u": (
                fixed_c_u
                if fixed_c_u_active
                else "not_applicable_for_predictive_contract"
            ),
            "fixed_c_u_status": (
                "fixed_input"
                if fixed_c_u_active
                else "inactive_for_predictive_contract"
            ),
        },
        "optimized_parameter": "rho",
        "default_rho_bounds": [float(args.rho_min), float(args.rho_max)],
        "rho_bounds_by_task": rho_bounds_by_task,
        "optimizer": "safeguarded_bounded_newton_log_rho",
        "objective_weights": _objective_weights(args),
        "rho_objective_rmse_mode": _rmse_objective_mode(args),
        "selection_profile": str(args.selection_profile),
        **agent_profile_record,
        "selection_input_hash": (
            shared_selection_root.name if shared_selection_root is not None else "alternate_source"
        ),
        "input_manifest_path": (
            str(input_manifest_path.resolve()) if input_manifest_path is not None else ""
        ),
        "input_manifest_sha256": (
            _sha256_file(input_manifest_path)
            if input_manifest_path is not None and input_manifest_path.is_file()
            else ""
        ),
        "family_roots": {
            "moment_t": str(args.out_run.resolve()),
            **(
                {"draw_kernel_t": str(_draw_branch_root(args.out_run).resolve())}
                if args.include_draw_kernel
                else {}
            ),
        },
        "generated_outputs": ["result_metrics", "selected_parameters"],
        "formal_result_generated": False,
        "other_experiments_run": False,
    }
    args.out_run.mkdir(parents=True, exist_ok=True)
    (args.out_run / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.include_draw_kernel:
        draw_root = _draw_branch_root(args.out_run)
        draw_root.mkdir(parents=True, exist_ok=True)
        draw_payload = {
            **payload,
            "schema": "caster_rho_only_smallval_family_branch_v3_fixed_nu_explicit",
            "output_run": str(draw_root.resolve()),
            "parent_run": str(args.out_run.resolve()),
            "branch_family": "draw_kernel_t",
        }
        (draw_root / "run_config.json").write_text(
            json.dumps(draw_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _prepare_draw_figure_view(draw_root: Path, selected_tasks: list[str]) -> None:
    ""
    for task in selected_tasks:
        calibration_root = _task_root(draw_root, task)
        one_root = _filter_root(calibration_root, "draw_kernel_t", False)
        hierarchical_root = _filter_root(calibration_root, "draw_kernel_t", True)
        _safe_link(
            one_root / "posterior_path.csv",
            calibration_root / "posterior_path.csv",
        )
        for name in ("family_posterior.csv", "hierarchical_posterior_path.csv"):
            _safe_link(hierarchical_root / name, calibration_root / name)


def _write_selected_parameters(
    args: argparse.Namespace,
    selected_tasks: list[str],
) -> tuple[Path, Path]:
    import pandas as pd

    distribution = _distribution(args)
    predictive_contract = _predictive_contract(args)
    fixed_c_u = _fixed_c_u(args)
    fixed_c_u_active = _fixed_c_u_is_active(predictive_contract)
    rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    smallval_hashes: dict[str, set[str]] = {task: set() for task in selected_tasks}
    lanes = [("moment_t", args.out_run)]
    if args.include_draw_kernel:
        lanes.append(("draw_kernel_t", _draw_branch_root(args.out_run)))
    for family, lane_root in lanes:
        display_family = (
            family
            if distribution == "student_t"
            else (
                "moment_gaussian"
                if family == "moment_t"
                else "draw_kernel_gaussian"
            )
        )
        for task in selected_tasks:
            root = _task_root(lane_root, task)
            freeze = json.loads(
                (root / "parameter_selection_freeze_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            smallval_hashes[task].add(str(freeze["small_validation_manifest_sha256"]))
            report = pd.read_csv(root / "bridge_component_calibration_report.csv")
            for variant in ("one_layer", "hierarchical"):
                config_path = root / f"bridge_config.{variant}.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                metadata = config["calibration_metadata"]
                if (
                    str(config.get("distribution", "")) != distribution
                    or str(config.get("kernel_distribution", "")) != distribution
                    or _recorded_predictive_contract(config)
                    != predictive_contract
                    or str(metadata.get("distribution", "")) != distribution
                    or str(metadata.get("kernel_distribution", ""))
                    != distribution
                    or _recorded_predictive_contract(metadata)
                    != predictive_contract
                ):
                    raise RuntimeError(
                        f"bridge distribution differs from the run identity: {config_path}"
                    )
                if fixed_c_u_active and not _isclose_optional_number(
                    metadata.get("fixed_c_u"),
                    fixed_c_u,
                ):
                    raise RuntimeError(
                        f"bridge fixed c_U differs from the run identity: {config_path}"
                    )
                scales = (
                    config.get("tau_by_component", {})
                    if family == "draw_kernel_t"
                    else config.get("sigma_by_component", {})
                )
                rho_min, rho_max = _rho_bounds_for_task(args, task)
                if not rho_min <= float(config["rho"]) <= rho_max:
                    raise RuntimeError(f"selected rho is outside bounds: {config_path}")
                reported_bounds = [
                    float(value) for value in metadata.get("rho_bounds", [])
                ]
                if reported_bounds != [rho_min, rho_max]:
                    raise RuntimeError(
                        f"reported rho bounds do not match task bounds: {config_path}"
                    )
                if family == "draw_kernel_t":
                    selected_report = report[
                        report["rho_selection_variant"].astype(str).eq(variant)
                    ]
                    expected = dict(
                        zip(
                            selected_report["bridge_r_key"].astype(str),
                            selected_report["computed_sigma"].astype(float),
                            strict=False,
                        )
                    )
                    if set(scales) != set(expected) or any(
                        abs(float(scales[key]) - expected[key]) > 1e-12
                        for key in expected
                    ):
                        raise RuntimeError(f"draw tau != computed sigma: {config_path}")
                metrics = metadata.get("selected_validation_metrics", {})
                weights = metadata.get("objective_weights", {})
                rows.append(
                    {
                        "dataset": task,
                        "bridge_family": display_family,
                        "internal_bridge_family": family,
                        "method": f"caster_{variant}"
                        + ("_draw_kernel" if family == "draw_kernel_t" else ""),
                        "selected_rho": float(config["rho"]),
                        "rho_min": rho_min,
                        "rho_max": rho_max,
                        "scale_parameter": "tau" if family == "draw_kernel_t" else "sigma",
                        "scale_formula": "alternate_log1p_transform_residual_rmse_component_horizon",
                        "tau_equals_sigma": True,
                        "fixed_gamma": (
                            float(args.fixed_gamma)
                            if family == "moment_t"
                            else float("nan")
                        ),
                        "gamma_status": (
                            "fixed_input"
                            if family == "moment_t"
                            else "not_applicable_inactive_for_draw_kernel"
                        ),
                        "distribution": distribution,
                        "fixed_nu": _reported_nu(args),
                        "predictive_contract": predictive_contract,
                        "fixed_c_u": (
                            fixed_c_u if fixed_c_u_active else float("nan")
                        ),
                        "fixed_c_u_status": (
                            "fixed_input"
                            if fixed_c_u_active
                            else "inactive_for_predictive_contract"
                        ),
                        "validation_endpoint_rows": int(metadata["distinct_validation_endpoint_rows"]),
                        "validation_folds": int(metadata["validation_fold_count"]),
                        "rho_objective_rmse_mode": str(
                            metadata.get("rho_objective_rmse_mode", "short-long")
                        ),
                        "selected_joint_risk": float(metadata["selected_joint_risk"]),
                        **{f"validation_{key}": value for key, value in metrics.items()},
                        **{f"weight_{key}": value for key, value in weights.items()},
                        "bridge_config_path": str(config_path.resolve()),
                        "small_validation_manifest_sha256": metadata[
                            "small_validation_manifest_sha256"
                        ],
                    }
                )
                for key, value in sorted(scales.items()):
                    scale_rows.append(
                        {
                            "dataset": task,
                            "bridge_family": display_family,
                            "internal_bridge_family": family,
                            "variant": variant,
                            "scale_parameter": "tau" if family == "draw_kernel_t" else "sigma",
                            "component_horizon_key": key,
                            "fixed_scale": float(value),
                            "tau_equals_sigma": True,
                            "predictive_contract": predictive_contract,
                            "fixed_c_u": (
                                fixed_c_u if fixed_c_u_active else float("nan")
                            ),
                            "bridge_config_path": str(config_path.resolve()),
                        }
                    )
    for task, hashes in smallval_hashes.items():
        if len(hashes) != 1:
            raise RuntimeError(
                f"moment-t and draw-kernel used different small-validation rows for {task}"
            )
    comparison_root = args.out_run / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    selected_path = comparison_root / "selected_parameters.csv"
    scales_path = comparison_root / "selected_component_scales.csv"
    selected = pd.DataFrame(rows).sort_values(
        ["dataset", "bridge_family", "method"], kind="mergesort"
    )
    selected.to_csv(selected_path, index=False)
    pd.DataFrame(scale_rows).sort_values(
        ["dataset", "bridge_family", "variant", "component_horizon_key"],
        kind="mergesort",
    ).to_csv(scales_path, index=False)
    print("[rho-only] selected parameters:")
    print(
        selected[
            [
                "dataset",
                "bridge_family",
                "method",
                "selected_rho",
                "fixed_gamma",
                "fixed_nu",
                "fixed_c_u",
                "tau_equals_sigma",
            ]
        ].to_string(index=False)
    )
    return selected_path, scales_path


FamilyTaskJob = tuple[str, Path, str]


def _task_worker_count(args: argparse.Namespace) -> int:
    try:
        count = int(getattr(args, "task_workers", 1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("--task-workers must be an integer of at least 1") from exc
    if count < 1:
        raise SystemExit("--task-workers must be at least 1")
    return count


def _export_worker_count(args: argparse.Namespace) -> int:
    try:
        count = int(getattr(args, "export_workers", 1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("--export-workers must be an integer of at least 1") from exc
    if count < 1:
        raise SystemExit("--export-workers must be at least 1")
    return count


def _export_worker_cli_args(args: argparse.Namespace) -> list[str]:
    return ["--workers", str(_export_worker_count(args))]


def _family_task_job_specs(
    lanes: Sequence[tuple[str, Path]],
    selected_tasks: Sequence[str],
) -> list[FamilyTaskJob]:
    ""

    jobs: list[FamilyTaskJob] = []
    seen: set[tuple[str, str]] = set()
    for family, lane_root in lanes:
        for task in selected_tasks:
            identity = (str(family), str(task))
            if identity in seen:
                raise SystemExit(
                    "duplicate family/task job would share one artifact root: "
                    f"{family}/{task}"
                )
            seen.add(identity)
            jobs.append((str(family), Path(lane_root), str(task)))
    return jobs


@contextmanager
def _exclusive_directory_lock(
    directory: Path,
    *,
    owner: str,
) -> Iterator[None]:
    ""

    directory.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    locked = False
    try:
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another coordinator owns {owner}: {directory}"
            ) from exc
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _run_family_task_job(
    args: argparse.Namespace,
    *,
    source_run: Path,
    out_run: Path,
    job: FamilyTaskJob,
) -> None:
    ""

    family, lane_root, task = job
    source_root = _task_root(source_run, task)
    calibration_root = _task_root(lane_root, task)
    with _exclusive_directory_lock(
        calibration_root,
        owner=f"rho-only family/task job {family}/{task}",
    ):
        _prepare_task_inputs(
            source_root,
            calibration_root,
            include_draws=family == "draw_kernel_t",
        )
        calibration_complete = _calibration_identity_valid(
            args,
            source_root,
            calibration_root,
            task,
            family,
        )
        if not (args.resume and calibration_complete):
            _run(
                _calibrate_command(
                    args.python,
                    source_root,
                    calibration_root,
                    task,
                    args,
                    family=family,
                ),
                label=f"{task} fixed {family} rho-only calibration",
                log_path=out_run / f"logs/{family}_{task}_calibrate.log",
                dry_run=args.dry_run,
            )
        if bool(getattr(args, "stop_after_calibration", False)):
            return
        for hierarchical in (False, True):
            filter_root = _filter_root(calibration_root, family, hierarchical)
            filter_root.mkdir(parents=True, exist_ok=True)
            marker = filter_root / (
                "hierarchical_run_metadata.json"
                if hierarchical
                else "caster_run_metadata.json"
            )
            variant = "hierarchical" if hierarchical else "one_layer"
            bridge_config = calibration_root / f"bridge_config.{variant}.json"
            if args.resume and _filter_identity_valid(
                args,
                marker,
                bridge_config,
                source_root,
                family,
                variant,
            ):
                continue
            _run(
                _filter_command(
                    args.python,
                    source_root,
                    calibration_root,
                    filter_root,
                    task,
                    hierarchical=hierarchical,
                    family=family,
                    source_run=source_run,
                    seed=args.seed,
                    predictive_contract=_predictive_contract(args),
                ),
                label=f"{task} {variant} CASTER [{family}]",
                log_path=out_run / f"logs/{family}_{task}_{variant}.log",
                dry_run=args.dry_run,
            )


def _run_family_task_jobs(
    args: argparse.Namespace,
    *,
    source_run: Path,
    out_run: Path,
    jobs: Sequence[FamilyTaskJob],
    worker_count: int,
) -> None:
    ""

    if not jobs:
        return
    if worker_count == 1:
                                                                             
        for job in jobs:
            _run_family_task_job(
                args,
                source_run=source_run,
                out_run=out_run,
                job=job,
            )
        return

    max_workers = min(worker_count, len(jobs))
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="rho-only-task",
    ) as executor:
        futures = {
            job: executor.submit(
                _run_family_task_job,
                args,
                source_run=source_run,
                out_run=out_run,
                job=job,
            )
            for job in jobs
        }
        done, pending = wait(futures.values(), return_when=FIRST_EXCEPTION)
        failed = any(
            not future.cancelled() and future.exception() is not None
            for future in done
        )
        if failed:
            for future in pending:
                future.cancel()
                                                                             
                                                              
            wait(futures.values())

    failures = [
        (job, future.exception())
        for job, future in futures.items()
        if not future.cancelled() and future.exception() is not None
    ]
    if failures:
        (family, _lane_root, task), error = failures[0]
        raise RuntimeError(
            f"rho-only family/task worker failed: {family}/{task}"
        ) from error
    for family, _lane_root, task in jobs:
        future = futures[(family, _lane_root, task)]
        if not future.cancelled():
            future.result()
            print(
                f"[rho-only] completed task worker: {family}/{task}",
                flush=True,
            )


def _run_coordinator_body(args: argparse.Namespace) -> int:
    reference_source_run = args.source_run.resolve()
    out_run = args.out_run.resolve()
    if reference_source_run == out_run:
        raise SystemExit("source and output run roots must differ")
    selected_tasks = list(TASKS) if args.tasks == "all" else [
        value.strip() for value in args.tasks.split(",") if value.strip()
    ]
    unknown = sorted(set(selected_tasks) - set(TASKS))
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}")
    for task in selected_tasks:
        _rho_bounds_for_task(args, task)
    _objective_weights(args)
    _distribution(args)
    _predictive_contract(args)
    _fixed_c_u(args)
    _fixed_nu(args)
    _input_draws(args)
    _candidate_cache_draws(args)
    for task in selected_tasks:
        _task_top_k(args, task)
    _bind_agent_run_profile(args)
    worker_count = _task_worker_count(args)
    if not math.isfinite(float(args.fixed_gamma)) or float(args.fixed_gamma) <= 0.0:
        raise SystemExit("--fixed-gamma must be finite and positive")
    if set(selected_tasks) != set(RESULT_TASKS) and not (
        args.stop_after_calibration or args.stop_after_filters
    ):
        raise SystemExit("the complete result export requires all three result tasks")
    args.reference_source_run = reference_source_run
    args.out_run = out_run
    source_run = _prepare_shared_selection_input(
        args,
        reference_source_run,
        out_run,
        selected_tasks,
    )
    _prepare_censoring_support_manifests(args, out_run, selected_tasks)
    if args.materialize_input_only:
        if args.dry_run:
            print("[rho-only] input-only dry run complete", flush=True)
        else:
            print(f"[rho-only] input_bundle={source_run}", flush=True)
        return 0
    if args.dry_run and args.shared_root is not None:
        return 0
    args.source_run = source_run
    _write_run_config(args, selected_tasks)

    lanes = [("moment_t", out_run)]
    if args.include_draw_kernel:
        lanes.append(("draw_kernel_t", _draw_branch_root(out_run)))
    jobs = _family_task_job_specs(lanes, selected_tasks)
    _run_family_task_jobs(
        args,
        source_run=source_run,
        out_run=out_run,
        jobs=jobs,
        worker_count=worker_count,
    )

    if args.dry_run:
        return 0
    selected_parameters_path, selected_scales_path = _write_selected_parameters(
        args, selected_tasks
    )
    if args.stop_after_calibration:
        print(f"[rho-only] selected_parameters={selected_parameters_path}")
        return 0
    if args.include_draw_kernel:
        _prepare_draw_figure_view(_draw_branch_root(out_run), selected_tasks)
    if args.stop_after_filters:
        print(f"[rho-only] selected_parameters={selected_parameters_path}")
        return 0

    task_roots = {task: _task_root(out_run, task) for task in RESULT_TASKS}
    results_dir = out_run / "new_method/results/rho_only_inputs"
    export_command = [
        str(args.python),
        str(CODE_ROOT / "scripts/export_rho_only_results.py"),
        "--out-dir",
        str(results_dir),
        *_export_worker_cli_args(args),
    ]
    for task, task_root in task_roots.items():
        export_command.extend([f"--{task.replace('_', '-')}-root", str(task_root)])
    if args.include_draw_kernel:
        draw_root = _draw_branch_root(out_run)
        for task in RESULT_TASKS:
            export_command.extend(
                [
                    f"--draw-kernel-{task.replace('_', '-')}-root",
                    str(_task_root(draw_root, task)),
                ]
            )
    _run(
        export_command,
        label="export CASTER moment-t and draw-kernel result branches",
        log_path=out_run / "logs/export_results.log",
        dry_run=False,
    )
    print(f"[rho-only] run_root={out_run}")
    print(f"[rho-only] result_root={results_dir}")
    print(f"[rho-only] selected_parameters={selected_parameters_path}")
    print(f"[rho-only] selected_component_scales={selected_scales_path}")
    return 0


def run(args: argparse.Namespace) -> int:
    ""

    reference_source_run = args.source_run.resolve()
    out_run = args.out_run.resolve()
    if reference_source_run == out_run:
        raise SystemExit("source and output run roots must differ")
    _task_worker_count(args)
    _export_worker_count(args)
    with _exclusive_directory_lock(
        out_run,
        owner="rho-only output-run coordinator",
    ):
        return _run_coordinator_body(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-run", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--tasks", default="all")
    parser.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help=(
            "Maximum concurrent family/task subprocess pipelines. "
            "The default 1 preserves the fixed execution order."
        ),
    )
    parser.add_argument(
        "--export-workers",
        type=int,
        default=1,
        help=(
            "Maximum concurrent family/task worker processes in the final "
            "CASTER result exporter. The default 1 preserves alternate behavior."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Default frozen ranking-prefix length for every task. fixed "
            "experiments use 12."
        ),
    )
    parser.add_argument(
        "--benchmark-a-top-k",
        type=int,
        default=None,
        help="Optional Benchmark A override for --top-k.",
    )
    parser.add_argument(
        "--benchmark-b-top-k",
        type=int,
        default=None,
        help=(
            "Optional shared override for Benchmark B COVID, FLU, and pooled "
            "tasks."
        ),
    )
    parser.add_argument("--fixed-gamma", type=float, default=1.0)
    parser.add_argument(
        "--fixed-c-u",
        type=float,
        default=1.25,
        help=(
            "Frozen train-only censoring upper-bound multiplier. Active only "
            "for coherent_mean_preserving_censored_student_t."
        ),
    )
    parser.add_argument(
        "--censoring-bound-scope",
        choices=CENSORING_BOUND_SCOPES,
        default="selected_topk_train",
        help=(
            "Train-only forecast bank used to freeze Method-2 censoring "
            "bounds. The default preserves prior Top-10 runs; "
            "eligible27_train gives every displayed method one common "
            "pre-test support."
        ),
    )
    parser.add_argument(
        "--distribution",
        choices=["student_t", "gaussian"],
        default="student_t",
    )
    parser.add_argument(
        "--predictive-contract",
        choices=PREDICTIVE_CONTRACTS,
        default=alternate_PREDICTIVE_CONTRACT,
        help=(
            "Predictive readout contract. The default preserves the fixed "
            "archive-moment behavior."
        ),
    )
    parser.add_argument(
        "--fixed-nu",
        type=float,
        default=None,
        help="Frozen Student-t degrees of freedom; defaults to 5 for Student-t.",
    )
    parser.add_argument("--max-newton-evaluations", type=int, default=48)
    parser.add_argument("--rho-min", type=float, default=0.05)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--benchmark-a-rho-min", type=float)
    parser.add_argument("--benchmark-a-rho-max", type=float)
    parser.add_argument("--benchmark-b-rho-min", type=float)
    parser.add_argument("--benchmark-b-rho-max", type=float)
    for metric, default in CURRENT_OBJECTIVE_WEIGHTS.items():
        parser.add_argument(
            f"--weight-{metric.replace('_', '-')}",
            dest=f"weight_{metric}",
            type=float,
            default=float(default),
        )
    parser.add_argument(
        "--rmse-objective-mode",
        choices=["short-long", "overall"],
        default="short-long",
        help=(
            "Use separate Short/Long RMSE objective terms, or replace them "
            "with one Overall RMSE term."
        ),
    )
    parser.add_argument(
        "--weight-overall-rmse",
        type=float,
        default=None,
        help=(
            "Overall-RMSE weight in overall mode; defaults to Short+Long "
            "RMSE weights."
        ),
    )
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=None,
        help=(
            "Existing shared_baseline root; when set, materialize the active "
            "causal Top-K input bank."
        ),
    )
    parser.add_argument(
        "--agent-run-profile",
        type=_agent_run_profile_arg,
        default=DEFAULT_AGENT_RUN_PROFILE,
        help=(
            "Agent-output namespace under "
            "<shared-root>/baseline/runs/agentic."
        ),
    )
    parser.add_argument(
        "--selection-profile",
        default=CURRENT_SELECTION_PROFILE,
    )
    parser.add_argument("--selection-hash", default="active")
    parser.add_argument("--input-draws", type=int, default=10)
    parser.add_argument(
        "--candidate-cache-draws",
        type=int,
        default=None,
        help=(
            "Draw-count protocol label used to locate the shared candidate cache; "
            "defaults to --input-draws for backward compatibility."
        ),
    )
    parser.add_argument(
        "--shared-input-root",
        type=Path,
        default=None,
        help=(
            "Optional distribution-independent root for one reusable materialized "
            "task-specific Top-K archive/draw bundle per seed."
        ),
    )
    parser.add_argument("--materialize-input-only", action="store_true")
    parser.add_argument("--include-draw-kernel", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-after-calibration",
        action="store_true",
        help=(
            "Run validation-only bridge/rho calibration and write the selected "
            "parameter report without launching test filters or result export."
        ),
    )
    parser.add_argument("--stop-after-filters", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
