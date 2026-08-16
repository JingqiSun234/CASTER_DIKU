#!/usr/bin/env python3
""






from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from formal_candidate_bank import (
    FORMAL_ARCHIVE_NAME,
    FORMAL_CANDIDATE_COUNT,
    FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT,
    FORMAL_RESULT_EXCLUDED_MODEL_IDS,
    FORMAL_RESULT_TOP_K,
    FORMAL_POOLED_MANIFEST_NAME,
    FORMAL_RANKING_NAME,
    FORMAL_TEST_RANKING_NAME,
)


ROOT = Path(__file__).resolve().parents[1]
NEWMETHOD_ROOT = ROOT / "code/caster"
sys.path.insert(0, str(NEWMETHOD_ROOT / "src"))

from caster.models import (              
    FORMAL_RETRIEVAL_PROFILE,
    build_candidate_validation_scores,
    build_test_rmse_ranking,
    read_registry,
    select_top_k_candidates_formal,
)
from caster.tasks import (              
    build_selection_context,
    load_fold_declaration,
    load_task_specs,
    materialize_selection_folds,
    scientific_selection_text,
    selection_context_canonical_json,
)


TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
alternate_SELECTION_PROFILE = "formal_embedding_validation_full_history_v1"
CLEAN_FORMAL_27_PROFILE = "formal_27_country_macro_v1"
SUPPORTED_SELECTION_PROFILES = {
    alternate_SELECTION_PROFILE,
    CLEAN_FORMAL_27_PROFILE,
}
                                                               
PROFILE_ROOT_NAME = alternate_SELECTION_PROFILE
DEFAULT_CANDIDATE_POOL_POLICY = ROOT / "configs/caster_candidate_pool_policy_v21.yaml"
FROZEN_CONTEXT_MANIFEST_NAME = "frozen_caster_context_manifest.json"
CORE_TASK_ARTIFACT_NAMES = (
    FORMAL_RANKING_NAME,
    "selection_fold_manifest.csv",
    "selection_fold_validation.csv",
    "candidate_validation_by_fold.csv",
    "candidate_validation_summary.csv",
    "candidate_validation_validation.csv",
    "candidate_validation_normalization.json",
    "selection_context.json",
    "selection_context.txt",
    "scientific_selection_context.txt",
    "selection_context_validation.csv",
)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def _alternate_task_policies() -> dict[str, dict[str, object]]:
    return {
        task: {
            "reporting_macro_unit": "entity",
            "excluded_model_ids": (),
            "excluded_candidates": (),
        }
        for task in TASKS
    }


def _load_selection_profile(
    profile_name: str,
    policy_path: Path | None,
) -> dict[str, object]:
    ""

    profile = str(profile_name).strip()
    if profile == CLEAN_FORMAL_27_PROFILE:
        if policy_path is not None:
            raise ValueError(
                "--candidate-pool-policy is not needed for the configured "
                "27-candidate profile"
            )
        return {
            "profile_name": profile,
            "status": "formal_release",
            "schema": "caster_selection_profile_v1_formal_27_country_macro",
            "policy_path": None,
            "policy_sha256": "",
            "policy_payload": None,
            "task_policies": {
                task: {
                    "reporting_macro_unit": (
                        "country" if task == "benchmark_a" else "entity"
                    ),
                    "excluded_model_ids": (),
                    "excluded_candidates": (),
                }
                for task in TASKS
            },
            "formal_preregistration_claimed": False,
        }
    if profile == alternate_SELECTION_PROFILE:
        if policy_path is not None:
            raise ValueError(
                "--candidate-pool-policy is not supported by the "
                "configured selection profiles"
            )
        return {
            "profile_name": profile,
            "status": "formal_release",
            "schema": "caster_selection_profile_v1_alternate",
            "policy_path": None,
            "policy_sha256": "",
            "policy_payload": None,
            "task_policies": _alternate_task_policies(),
            "formal_preregistration_claimed": False,
        }
    raise ValueError(
        f"unsupported selection profile={profile!r}; "
        f"expected one of {sorted(SUPPORTED_SELECTION_PROFILES)}"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(encoded)


def _frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    normalized = frame[columns].fillna("").astype(str).sort_values(columns, kind="mergesort")
    return _sha256_bytes(normalized.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _enabled_model_ids(registry: pd.DataFrame) -> list[str]:
    rows = registry.copy()
    if "enabled" in rows.columns:
        rows = rows[rows["enabled"].astype(str).str.lower().isin({"true", "1", "t", "yes"})]
    return rows["model_id"].astype(str).drop_duplicates().tolist()


def _eligible_model_ids(
    all_model_ids: list[str],
    task_policy: dict[str, object],
) -> list[str]:
    excluded = {str(value) for value in task_policy["excluded_model_ids"]}
    if unknown := sorted(excluded - set(all_model_ids)):
        raise ValueError(f"candidate-pool policy excludes unknown enabled candidates: {unknown}")
    eligible = [model_id for model_id in all_model_ids if model_id not in excluded]
    if not eligible:
        raise ValueError("candidate-pool policy removed every enabled candidate")
    return eligible


def _eligibility_validation(
    registry: pd.DataFrame,
    task: str,
    task_policy: dict[str, object],
) -> pd.DataFrame:
    enabled_ids = _enabled_model_ids(registry)
    excluded_by_id = {
        str(row["model_id"]): row
        for row in task_policy["excluded_candidates"]
    }
    rows: list[dict[str, object]] = []
    for model_id in enabled_ids:
        exclusion = excluded_by_id.get(model_id)
        rows.append({
            "task_id": task,
            "model_id": model_id,
            "source_registry_enabled": True,
            "eligible_for_candidate_normalization": exclusion is None,
            "eligible_for_candidate_ranking": exclusion is None,
            "eligible_for_caster_top_k": exclusion is None,
            "retained_in_all_candidate_archive": True,
            "retained_as_independent_baseline": bool(
                exclusion and exclusion["retain_as_independent_baseline"]
            ),
            "exclusion_reason_code": "" if exclusion is None else str(exclusion["reason_code"]),
        })
    return pd.DataFrame(rows)


def _read_archive_projection(archive: Path, forecast_ids: set[str], model_ids: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {"forecast_id", "model_id", "particle_id", "pred_mean"}
    provenance = {
        "forecast_status",
        "forecast_fallback_used",
        "forecast_failure_reason",
        "forecast_fallback_method",
        "proxy_fallback_used",
        "unsafe_native_proxy_executed",
    }
    available_columns = set(pd.read_csv(archive, nrows=0).columns)
    missing = sorted((required | provenance) - available_columns)
    if missing:
        raise ValueError(
            f"{archive} missing formal native-forecast/provenance columns: {missing}"
        )
    for chunk in pd.read_csv(
        archive,
        usecols=lambda column: column in required | provenance,
        chunksize=250_000,
    ):
        missing = sorted(required - set(chunk.columns))
        if missing:
            raise ValueError(f"{archive} missing required archive columns: {missing}")
        keep = chunk[
            chunk["forecast_id"].astype(str).isin(forecast_ids)
            & chunk["model_id"].astype(str).isin(model_ids)
        ]
        if not keep.empty:
            frames.append(keep)
    if not frames:
        raise ValueError(f"archive projection is empty: {archive}")
    projection = pd.concat(frames, ignore_index=True)
    return projection


def _fallback_mask(frame: pd.DataFrame) -> pd.Series:
    values = frame["forecast_fallback_used"]
    if values.dtype == bool:
        fallback = values.fillna(False).astype(bool)
    else:
        fallback = values.astype(str).str.strip().str.lower().isin(
            {"true", "1", "t", "yes", "y"}
        )
    for column in ("forecast_status", "forecast_fallback_method"):
        if column in frame.columns:
            fallback |= frame[column].fillna("").astype(str).str.lower().str.contains(
                "fallback|last_value|unavailable", regex=True
            )
    for column in ("proxy_fallback_used", "unsafe_native_proxy_executed"):
        if column not in frame.columns:
            continue
        values = frame[column]
        if values.dtype == bool:
            fallback |= values.fillna(False).astype(bool)
        else:
            fallback |= values.astype(str).str.strip().str.lower().isin(
                {"true", "1", "t", "yes", "y"}
            )
    return fallback


def _core_selection_artifact_paths(
    root: Path,
    *,
    require_policy_artifacts: bool,
) -> list[Path]:
    required: list[Path] = []
    for task in TASKS:
        required.extend(root / task / name for name in CORE_TASK_ARTIFACT_NAMES)
        required.append(root / task / "candidate_pool_eligibility.csv")
    if require_policy_artifacts:
        required.append(root / "candidate_pool_policy.json")
    return required


def _ranking_matches_expected(
    path: Path,
    *,
    task_id: str,
    expected_model_ids: list[str],
) -> bool:
    try:
        ranking = pd.read_csv(path, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    required = {"rank", "model_id", "task_id"}
    if not required.issubset(ranking.columns):
        return False
    model_ids = ranking["model_id"].astype(str).tolist()
    ranks = pd.to_numeric(ranking["rank"], errors="coerce")
    return (
        len(model_ids) == len(expected_model_ids)
        and len(set(model_ids)) == len(model_ids)
        and set(model_ids) == set(expected_model_ids)
        and ranks.notna().all()
        and ranks.tolist() == list(range(1, len(model_ids) + 1))
        and set(ranking["task_id"].astype(str)) == {task_id}
    )


def _selection_complete(
    root: Path,
    *,
    expected_model_ids_by_task: dict[str, list[str]],
    require_policy_artifacts: bool = False,
) -> bool:
    manifest_path = root / "selection_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        return False
    required = _core_selection_artifact_paths(
        root,
        require_policy_artifacts=require_policy_artifacts,
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    recorded_hashes = manifest.get("selection_artifact_sha256", {})
    expected_relative_paths = {path.relative_to(root).as_posix() for path in required}
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != expected_relative_paths:
        return False
    if any(
        str(recorded_hashes.get(path.relative_to(root).as_posix(), ""))
        != _sha256_file(path)
        for path in required
    ):
        return False
    if set(expected_model_ids_by_task) != set(TASKS):
        return False
    return all(
        _ranking_matches_expected(
            root / task / FORMAL_RANKING_NAME,
            task_id=task,
            expected_model_ids=expected_model_ids_by_task[task],
        )
        for task in TASKS
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_paths(args: argparse.Namespace, task: str) -> tuple[Path, Path, Path]:
    if task == "benchmark_a":
        return args.panel_a, args.ledger_a, args.candidate_cache / task / FORMAL_ARCHIVE_NAME
    return args.panel_b, args.ledger_b, args.candidate_cache / task / FORMAL_ARCHIVE_NAME


def _task_archive_manifest_path(candidate_cache: Path, task: str) -> Path:
    name = (
        FORMAL_POOLED_MANIFEST_NAME
        if task == "benchmark_b_pooled"
        else "forecast_archive_manifest.json"
    )
    return candidate_cache / task / name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--ledger-a", type=Path, required=True)
    parser.add_argument("--panel-b", type=Path, required=True)
    parser.add_argument("--ledger-b", type=Path, required=True)
    parser.add_argument("--task-specs", type=Path, default=ROOT / "configs/caster_task_specs_v20.yaml")
    parser.add_argument("--fold-config", type=Path, default=ROOT / "configs/pretest_selection_folds_v20.yaml")
    parser.add_argument(
        "--selection-profile",
        choices=sorted(SUPPORTED_SELECTION_PROFILES),
        default=CLEAN_FORMAL_27_PROFILE,
        help=(
            "Versioned selection policy. The default preserves alternate entity-macro selection; "
            "the direct 27-model pool plus Benchmark A country-macro scoring requires an explicit profile."
        ),
    )
    parser.add_argument(
        "--candidate-pool-policy",
        type=Path,
        default=None,
        help=(
            "Policy YAML for the explicit v21 profile. If omitted under that profile, "
            "configs/caster_candidate_pool_policy_v21.yaml is used."
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=FORMAL_RESULT_TOP_K
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        profile = _load_selection_profile(args.selection_profile, args.candidate_pool_policy)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not 1 <= args.top_k <= FORMAL_CANDIDATE_COUNT:
        raise SystemExit(
            f"--top-k must be between 1 and {FORMAL_CANDIDATE_COUNT}"
        )
    registry_path = args.candidate_cache / "model_registry.formal.csv"
    embeddings_path = args.candidate_cache / "candidate_embeddings.csv"
    candidate_cache_manifest_path = args.candidate_cache / "manifest.json"
    required = [
        registry_path,
        embeddings_path,
        candidate_cache_manifest_path,
        args.panel_a,
        args.ledger_a,
        args.panel_b,
        args.ledger_b,
    ]
    required.extend(args.candidate_cache / task / FORMAL_ARCHIVE_NAME for task in TASKS)
    required.extend(_task_archive_manifest_path(args.candidate_cache, task) for task in TASKS)
    if missing := [path for path in required if not path.is_file()]:
        raise SystemExit("shared selection requires existing inputs; no models will be run; missing: " + ", ".join(map(str, missing)))

    candidate_cache_manifest = json.loads(
        candidate_cache_manifest_path.read_text(encoding="utf-8")
    )
    if str(candidate_cache_manifest.get("registry_sha256", "")) != _sha256_file(registry_path):
        raise SystemExit("shared candidate-cache manifest/registry hash disagreement")
    cache_task_records = candidate_cache_manifest.get("tasks", {})
    if not isinstance(cache_task_records, dict):
        raise SystemExit("shared candidate-cache manifest tasks must be a mapping")

    registry = read_registry(registry_path)
    embeddings = pd.read_csv(embeddings_path)
    all_model_ids = _enabled_model_ids(registry)
    if len(all_model_ids) != FORMAL_CANDIDATE_COUNT:
        raise SystemExit(
            "formal shared selection registry count mismatch: "
            f"expected={FORMAL_CANDIDATE_COUNT} found={len(all_model_ids)}"
        )
    specs = load_task_specs(args.task_specs)
    prepared: dict[str, dict[str, object]] = {}
    identity_tasks: dict[str, object] = {}

    for task in TASKS:
        spec = specs[task]
        task_policy = profile["task_policies"][task]
        reporting_macro_unit = str(task_policy["reporting_macro_unit"])
        excluded_model_ids = tuple(str(value) for value in task_policy["excluded_model_ids"])
        eligible_model_ids = _eligible_model_ids(all_model_ids, task_policy)
        panel_path, ledger_path, archive_path = _task_paths(args, task)
        archive_manifest_path = _task_archive_manifest_path(args.candidate_cache, task)
        archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        archive_sha256 = _sha256_file(archive_path)
        archive_manifest_sha256 = _sha256_file(archive_manifest_path)
        cache_task_record = cache_task_records.get(task, {})
        if (
            str(archive_manifest.get("archive_sha256", "")) != archive_sha256
            or (cache_task_record and not isinstance(cache_task_record, dict))
            or (
                isinstance(cache_task_record, dict)
                and cache_task_record
                and str(cache_task_record.get("archive_sha256", "")) != archive_sha256
            )
        ):
            raise ValueError(f"candidate archive/manifest hash disagreement: {task}")
        panel = pd.read_csv(panel_path)
        ledger = pd.read_csv(ledger_path)
        task_ledger = ledger[ledger["component"].astype(str).isin(spec.target_components)].copy()
        embargo = task_ledger[task_ledger["split"].astype(str).eq("embargo")].copy()
        if embargo.empty:
            raise ValueError(f"formal task ledger has no embargo forecast origins: {task}")
        if embargo["calibration_eligible"].astype(str).str.lower().isin({"1", "true", "yes"}).any():
            raise ValueError(f"embargo rows are calibration-eligible: {task}")
        declaration = load_fold_declaration(args.fold_config, task)
        folds, fold_validation = materialize_selection_folds(ledger, spec, declaration)
        if set(folds.get("split", pd.Series(dtype=str)).astype(str)) != {"val"}:
            raise ValueError(f"candidate selection accepts official validation folds only: {task}")
        if set(folds["forecast_id"].astype(str)) & set(embargo["forecast_id"].astype(str)):
            raise ValueError(f"embargo forecast scores leaked into selection folds: {task}")
        val_archive = _read_archive_projection(
            archive_path,
            set(folds["forecast_id"].astype(str)),
            set(eligible_model_ids),
        )
        fallback = _fallback_mask(val_archive)
        if fallback.any():
            counts = (
                val_archive.loc[fallback, "model_id"]
                .astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
            )
            raise ValueError(
                f"formal validation selection refuses fallback forecasts for {task}: {counts}"
            )
        by_fold, validation, normalization, validation_validation = build_candidate_validation_scores(
            folds,
            val_archive,
            registry,
            spec,
            reporting_macro_unit=reporting_macro_unit,
            excluded_model_ids=excluded_model_ids,
        )
        context, selection_text, context_validation = build_selection_context(panel, ledger, spec)
        context_json = selection_context_canonical_json(context)
        audit_context_sha = _sha256_bytes(context_json.encode("utf-8"))
        scientific_text = scientific_selection_text(context)
        context_sha = _sha256_bytes(scientific_text.encode("utf-8"))
        validation_sha = _frame_sha256(validation, sorted(validation.columns.tolist()))
        selection = select_top_k_candidates_formal(
            registry[registry["model_id"].astype(str).isin(set(eligible_model_ids))],
            embeddings[embeddings["model_id"].astype(str).isin(set(eligible_model_ids))],
            validation,
            selection_text=scientific_text,
            task_id=task,
            t_sel=spec.t_sel,
            task_spec_sha256=spec.task_spec_sha256,
            selection_context_sha256=context_sha,
            candidate_validation_sha256=validation_sha,
            top_k=len(eligible_model_ids),
        )
        val_point = val_archive.groupby(["forecast_id", "model_id"], as_index=False)["pred_mean"].mean()
        val_projection_sha = _frame_sha256(val_point, ["forecast_id", "model_id", "pred_mean"])
        identity_tasks[task] = {
            "task_spec_sha256": spec.task_spec_sha256,
            "selection_context_sha256": context_sha,
            "audit_context_sha256": audit_context_sha,
            "fold_manifest_sha256": str(folds["fold_manifest_sha256"].iloc[0]),
            "validation_projection_sha256": val_projection_sha,
            "embargo_forecast_rows": int(len(embargo)),
            "embargo_forecast_scores_used_for_selection": False,
            "embargo_forecast_scores_used_for_bridge": False,
            "reporting_macro_unit": reporting_macro_unit,
            "source_candidate_count": len(all_model_ids),
            "eligible_candidate_count": len(eligible_model_ids),
            "excluded_model_ids": list(excluded_model_ids),
            "excluded_before_normalization": True,
            "candidate_archive_sha256": archive_sha256,
            "candidate_archive_manifest_sha256": archive_manifest_sha256,
        }
        prepared[task] = {
            "selection": selection,
            "folds": folds,
            "fold_validation": fold_validation,
            "validation_by_fold": by_fold,
            "validation": validation,
            "normalization": normalization,
            "validation_validation": validation_validation,
            "context": context,
            "context_json": context_json,
            "selection_text": selection_text,
            "scientific_selection_text": scientific_text,
            "context_validation": context_validation,
            "archive": archive_path,
            "archive_sha256": archive_sha256,
            "archive_manifest": archive_manifest_path,
            "archive_manifest_sha256": archive_manifest_sha256,
            "ledger": ledger_path,
            "reporting_macro_unit": reporting_macro_unit,
            "excluded_model_ids": excluded_model_ids,
            "eligible_model_ids": eligible_model_ids,
            "eligibility_validation": _eligibility_validation(registry, task, task_policy),
        }
        del val_archive, val_point, panel, ledger

    implementation_paths = [
        NEWMETHOD_ROOT / "src/caster/models/retrieval.py",
        NEWMETHOD_ROOT / "src/caster/models/selection_validation.py",
        NEWMETHOD_ROOT / "src/caster/tasks/context.py",
        NEWMETHOD_ROOT / "src/caster/tasks/qwen_context.py",
        NEWMETHOD_ROOT / "src/caster/data/benchmark_a_mobility.py",
        NEWMETHOD_ROOT / "src/caster/data/benchmark_b_context.py",
        ROOT / "configs/benchmark_b_context_v26_1.yaml",
        NEWMETHOD_ROOT / "src/caster/tasks/folds.py",
        NEWMETHOD_ROOT / "src/caster/tasks/spec.py",
    ]
    if profile["status"] == "archived_nonrelease_profile":
        implementation_paths.extend([
            Path(__file__).resolve(),
            ROOT / "scripts/result_metric_contract.py",
        ])
    identity_payload = {
        "schema": (
            "caster_shared_selection_identity_v2_native_only_rmse_one_final_root"
            if profile["status"] == "formal_release"
            else "caster_shared_selection_identity_v21_all_tasks_formal_direct_a_country_macro"
        ),
        "retrieval_profile": FORMAL_RETRIEVAL_PROFILE,
        "beta_val": 1.0,
        "beta_runtime": 0.0,
        "priority_weight": 0.0,
        "family_diversity_bonus": 0.0,
        "runtime_functionality_implemented": False,
        "formal_preregistration_claimed": False,
        "registry_sha256": _sha256_file(registry_path),
        "embeddings_sha256": _sha256_file(embeddings_path),
        "candidate_cache_manifest_sha256": _sha256_file(candidate_cache_manifest_path),
        "task_specs_sha256": _sha256_file(args.task_specs),
        "fold_config_sha256": _sha256_file(args.fold_config),
        "selection_implementation_sha256": {
            str(path.relative_to(ROOT)): _sha256_file(path)
            for path in implementation_paths
        },
        "tasks": identity_tasks,
    }
    if profile["status"] == "archived_nonrelease_profile":
        identity_payload.update({
            "selection_profile": profile["profile_name"],
            "selection_profile_schema": profile["schema"],
            "selection_profile_status": profile["status"],
            "candidate_pool_policy_sha256": profile["policy_sha256"],
            "formal_preregistration_claimed": False,
            "test_metrics_used_for_rescoring": False,
        })
    selection_input_hash = _canonical_sha256(identity_payload)
    profile_root = args.out_root / str(profile["profile_name"])
    root = profile_root / selection_input_hash
    expected_model_ids_by_task = {
        task: list(prepared[task]["eligible_model_ids"])
        for task in TASKS
    }
    selection_was_complete = _selection_complete(
        root,
        expected_model_ids_by_task=expected_model_ids_by_task,
        require_policy_artifacts=profile["status"] != "formal_release",
    )
    if selection_was_complete:
        manifest = json.loads((root / "selection_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("selection_input_hash") != selection_input_hash:
            raise SystemExit(f"shared selection manifest/hash disagreement: {root}")
        print(f"shared_selection=skip existing {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            state = prepared[task]
            task_root = root / task
            task_root.mkdir(parents=True, exist_ok=True)
            state["selection"].to_csv(task_root / FORMAL_RANKING_NAME, index=False)
            state["folds"].to_csv(task_root / "selection_fold_manifest.csv", index=False)
            state["fold_validation"].to_csv(task_root / "selection_fold_validation.csv", index=False)
            state["validation_by_fold"].to_csv(task_root / "candidate_validation_by_fold.csv", index=False)
            state["validation"].to_csv(task_root / "candidate_validation_summary.csv", index=False)
            state["validation_validation"].to_csv(task_root / "candidate_validation_validation.csv", index=False)
            state["context_validation"].to_csv(task_root / "selection_context_validation.csv", index=False)
            state["eligibility_validation"].to_csv(
                task_root / "candidate_pool_eligibility.csv",
                index=False,
            )
            (task_root / "selection_context.json").write_text(str(state["context_json"]), encoding="utf-8")
            (task_root / "selection_context.txt").write_text(str(state["selection_text"]), encoding="utf-8")
            (task_root / "scientific_selection_context.txt").write_text(
                str(state["scientific_selection_text"]), encoding="utf-8"
            )
            _write_json(task_root / "candidate_validation_normalization.json", state["normalization"])
            if task == "benchmark_b_pooled":
                ranking_path = task_root / FORMAL_RANKING_NAME
                context_path = task_root / "selection_context.json"
                validation_path = task_root / "candidate_validation_summary.csv"
                context_payload = state["context"]
                _write_json(
                    task_root / FROZEN_CONTEXT_MANIFEST_NAME,
                    {
                        "schema": "caster_runtime_context_binding_manifest_v1",
                        "status": "frozen",
                        "task_id": "benchmark_b_pooled",
                        "t_sel": specs[task].t_sel,
                        "selection_context": context_payload,
                        "canonical_context_schema": str(context_payload.get("schema", "")),
                        "benchmark_b_context_contract_sha256": str(
                            context_payload.get("context_contract_sha256", "")
                        ),
                        "candidate_count": len(state["eligible_model_ids"]),
                        "candidate_fields": ["model_id", "family", "description"],
                        "candidate_rows_sha256": _sha256_file(ranking_path),
                        "source_sha256": {
                            "source_registry": _sha256_file(registry_path),
                            "selection_context": _sha256_file(context_path),
                            "validation_summary": _sha256_file(validation_path),
                        },
                        "validation_scores_in_runtime_context": False,
                        "test_information_in_runtime_context": False,
                        "input_change_invalidates_forecast_posterior_agent_results": True,
                    },
                )
        if profile["status"] != "formal_release":
            _write_json(root / "candidate_pool_policy.json", profile["policy_payload"])
        manifest = {
            "schema": (
                "caster_shared_selection_bank_v2_native_only_rmse_one_final_root"
                if profile["status"] == "formal_release"
                else "caster_shared_selection_bank_v21_all_tasks_formal_direct_a_country_macro"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "selection_input_hash": selection_input_hash,
            "identity": identity_payload,
            "candidate_cache": str(args.candidate_cache.resolve()),
            "candidate_cache_manifest_sha256": _sha256_file(candidate_cache_manifest_path),
            "candidate_archive_inputs": {
                task: {
                    "archive_sha256": str(prepared[task]["archive_sha256"]),
                    "archive_manifest_sha256": str(
                        prepared[task]["archive_manifest_sha256"]
                    ),
                }
                for task in TASKS
            },
            "candidate_model_count": FORMAL_CANDIDATE_COUNT,
            "task_ids": list(TASKS),
            "ranking_size_per_task": {
                task: len(prepared[task]["eligible_model_ids"])
                for task in TASKS
            } if profile["status"] != "formal_release" else FORMAL_CANDIDATE_COUNT,
            "tie_break": "retrieval_score_desc_model_id_asc",
            "validation_forecasts_native_only": True,
            "rmse_aggregation": {
                task: prepared[task]["normalization"]["rmse_aggregation"]
                for task in TASKS
            } if profile["status"] != "formal_release" else "sqrt(equal_fold_mean(component_strategy_horizon_entity_mean_squared_error))",
            "test_data_used_for_selection": False,
            "formal_preregistration_claimed": False,
            "archived_test_diagnostic_is_selection_input": False,
            "selection_artifact_sha256": {
                path.relative_to(root).as_posix(): _sha256_file(path)
                for path in _core_selection_artifact_paths(
                    root,
                    require_policy_artifacts=profile["status"] != "formal_release",
                )
            },
        }
        if profile["status"] != "formal_release":
            manifest.update({
                "selection_profile": profile["profile_name"],
                "selection_profile_status": profile["status"],
                "formal_preregistration_claimed": False,
                "candidate_pool_policy_sha256": profile["policy_sha256"],
                "materialized_candidate_pool_policy_sha256": _sha256_file(
                    root / "candidate_pool_policy.json"
                ),
                "source_candidate_count": FORMAL_CANDIDATE_COUNT,
                "eligible_candidate_count_by_task": {
                    task: len(prepared[task]["eligible_model_ids"])
                    for task in TASKS
                },
                "excluded_model_ids_by_task": {
                    task: list(prepared[task]["excluded_model_ids"])
                    for task in TASKS
                },
                "excluded_before_normalization": True,
                "all_candidate_archives_preserved": True,
                "excluded_candidates_retained_as_independent_baselines": True,
                "benchmark_b_reporting_macro_unit_unchanged": True,
            })
        _write_json(root / "selection_manifest.json", manifest)
        print(f"shared_selection=created {root}")

    if not _selection_complete(
        root,
        expected_model_ids_by_task=expected_model_ids_by_task,
        require_policy_artifacts=profile["status"] != "formal_release",
    ):
        raise SystemExit(f"shared selection failed artifact/ranking integrity validation: {root}")

    active_pointer = {
        "selection_input_hash": selection_input_hash,
        "selection_root": str(root.resolve()),
        "selection_manifest": str((root / "selection_manifest.json").resolve()),
        "selection_manifest_sha256": _sha256_file(root / "selection_manifest.json"),
        "formal_preregistration_claimed": False,
    }
    if profile["status"] != "formal_release":
        active_pointer.update({
            "selection_profile": profile["profile_name"],
            "selection_profile_status": profile["status"],
            "formal_preregistration_claimed": False,
            "formal_active_selection_replaced": False,
        })
    _write_json(profile_root / "active_selection.json", active_pointer)

    print(f"active_selection={profile_root / 'active_selection.json'}")
    print(f"selection_root={root}")
    print(f"selection_profile={profile['profile_name']}")
    for task in TASKS:
        selection = pd.read_csv(root / task / FORMAL_RANKING_NAME)
        print(f"{task}_top{args.top_k}=" + ",".join(selection.head(args.top_k)["model_id"].astype(str)))
        if prepared[task]["excluded_model_ids"]:
            print(f"{task}_excluded=" + ",".join(prepared[task]["excluded_model_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
