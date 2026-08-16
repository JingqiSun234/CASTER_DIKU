#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


RUNNER = Path("code/caster/scripts/run_incremental_ablation_from_archive.py")
CALIBRATOR = Path("code/caster/scripts/calibrate_incremental_bridge_from_archive.py")
DEFAULT_RHO_GRID = "0.05,0.1,0.2,0.35,0.5,0.75,1.0"
DEFAULT_BENCHMARK_A_RHO_GRID = "0.5,0.75,1.0"
STAGES = [
    "A1_top1_shared_bridge",
    "A2_topk_static_bridge",
    "A3_offline_one_layer_caster",
    "A4_one_layer_caster_without_selected_rho",
    "A5_causal_selected_rho",
]
OPTIONAL_STAGES = [
    "A0_top1_naive",                                                         
    "A3_native_fallback_frozen_diagnostic",
    "A3_strict_posterior_predictive_fallback_frozen",
]
BRIDGE_SCORE_STAGES = {
    "A1_top1_shared_bridge",
    "A2_topk_static_bridge",
    "A3_offline_one_layer_caster",
    "A3_native_fallback_frozen_diagnostic",
    "A3_strict_posterior_predictive_fallback_frozen",
    "A4_one_layer_caster_without_selected_rho",
    "A5_causal_selected_rho",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:                                                         
        raise SystemExit(f"PyYAML is required to read {path}: {exc}") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a mapping")
    return data


def _tasks(config: dict[str, Any], names: list[str]) -> list[dict[str, str]]:
    rows = config.get("tasks", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("spec must contain tasks for by-task incremental ablation")
    selected: list[dict[str, str]] = []
    wanted = set(names)
    for row in rows:
        task = str(row.get("task", "")).strip()
        if names and task not in wanted:
            continue
        item = {
            "task": task,
            "label": str(row.get("label", task)),
            "ledger": str(row.get("ledger", "")),
            "artifact_root": str(row.get("artifact_root", "")),
            "full_bridge_config": str(row.get("full_bridge_config", "")),
            "full_bridge_config_one_layer": str(row.get("full_bridge_config_one_layer", "")),
            "rho_grid": str(row.get("rho_grid", "")),
            "native_likelihood_scores": str(row.get("native_likelihood_scores", "")),
            "native_likelihood_availability": str(row.get("native_likelihood_availability", "")),
        }
        missing = [
            key
            for key, value in item.items()
            if key
            not in {
                "label",
                "full_bridge_config",
                "full_bridge_config_one_layer",
                "rho_grid",
                "native_likelihood_scores",
                "native_likelihood_availability",
            }
            and not value
        ]
        if not item["full_bridge_config"] and not item["full_bridge_config_one_layer"]:
            missing.append("full_bridge_config_one_layer")
        if missing:
            raise SystemExit(f"task {task!r} missing required fields: {missing}")
        selected.append(item)
    if names and len(selected) != len(wanted):
        found = {task["task"] for task in selected}
        raise SystemExit(f"requested tasks not found in spec: {sorted(wanted - found)}")
    return selected


def _require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"missing required file: {path}")


def _bridge_shape(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("distribution", "gaussian")), str(payload.get("transform", "log1p"))


def _bridge_score_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_metadata = payload.get("calibration_metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    family = str(metadata.get("selected_bridge_family", "moment_t"))
    score_source = str(metadata.get("score_source", "archive_moment"))
    if (family, score_source) not in {
        ("moment_t", "archive_moment"),
        ("draw_kernel_t", "draw_kernel"),
    }:
        raise SystemExit(
            f"inconsistent bridge family/score source in {path}: "
            f"family={family!r} source={score_source!r}"
        )
    return score_source


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _task_paths(task: dict[str, str], repo_root: Path) -> dict[str, Path]:
    artifact_root = _repo_path(repo_root, task["artifact_root"])
    paths = {
        "ledger": _repo_path(repo_root, task["ledger"]),
        "archive": artifact_root / "forecast_archive.csv",
        "draws": artifact_root / "forecast_draws.csv",
        "registry": artifact_root / "model_registry.csv",
        "selection": artifact_root / "candidate_selection_log.csv",
    }
                                                                             
                                                                             
                                                                              
    for key in ("ledger", "archive", "registry", "selection"):
        _require_file(paths[key])
    if task.get("native_likelihood_scores"):
        paths["native_scores"] = _repo_path(repo_root, task["native_likelihood_scores"])
    if task.get("native_likelihood_availability"):
        paths["native_availability"] = _repo_path(repo_root, task["native_likelihood_availability"])
    return paths


def _draws_args_for_stage(
    stage: str,
    paths: dict[str, Path],
    bridge_config: Path | None,
) -> list[str]:
    if stage not in BRIDGE_SCORE_STAGES or bridge_config is None:
        return []
    if _bridge_score_source(bridge_config) != "draw_kernel":
        return []
    draws = paths.get("draws")
    if draws is None:
        raise SystemExit(f"draw-kernel stage {stage} has no configured draw archive path")
    _require_file(draws)
    return ["--draws", str(draws)]


def _run(cmd: list[str], *, cwd: Path, skip_existing: Path | None = None) -> None:
    if skip_existing is not None and skip_existing.exists():
        print(f"skip existing {skip_existing}")
        return
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _stage_cmd(
    python: str,
    runner: Path,
    stage: str,
    paths: dict[str, Path],
    out: Path,
    *,
    bridge_config: Path | None = None,
    posterior_update_policy: str = "",
    offline_stage_root: Path | None = None,
    online_stage_root: Path | None = None,
    native_scores: Path | None = None,
    native_availability: Path | None = None,
    prior_policy: str = "uniform_model",
    seed: int,
) -> list[str]:
    draws_args = _draws_args_for_stage(stage, paths, bridge_config)
    cmd = [
        python,
        str(runner),
        "--stage",
        stage,
        "--ledger",
        str(paths["ledger"]),
        "--archive",
        str(paths["archive"]),
        *draws_args,
        "--registry",
        str(paths["registry"]),
        "--selection",
        str(paths["selection"]),
        "--out",
        str(out),
        "--seed",
        str(seed),
    ]
    if bridge_config is not None:
        cmd.extend(["--bridge-config", str(bridge_config)])
    if posterior_update_policy:
        cmd.extend(["--posterior-update-policy", posterior_update_policy])
    if offline_stage_root is not None:
        cmd.extend(["--offline-stage-root", str(offline_stage_root)])
    if online_stage_root is not None:
        cmd.extend(["--online-stage-root", str(online_stage_root)])
    if native_scores is not None:
        cmd.extend(["--native-scores", str(native_scores)])
    if native_availability is not None:
        cmd.extend(["--native-availability", str(native_availability)])
    if prior_policy != "uniform_model":
        cmd.extend(["--prior-policy", prior_policy])
    return cmd


def run_task(args: argparse.Namespace, task: dict[str, str]) -> None:
    print(f"== task {task['task']} ({task['label']}) ==", flush=True)
    paths = _task_paths(task, args.repo_root)
    task_root = args.out_root / task["task"]
    task_root.mkdir(parents=True, exist_ok=True)
    bridge_rho1 = task_root / "bridge_shared_rho1.json"
    bridge_tuned = task_root / "bridge_shared_tuned_rho.json"
    full_bridge = _repo_path(args.repo_root, task["full_bridge_config"]) if task.get("full_bridge_config") else bridge_tuned
    full_bridge_one_layer = (
        _repo_path(args.repo_root, task["full_bridge_config_one_layer"])
        if task.get("full_bridge_config_one_layer")
        else full_bridge
    )
    requested = set(args.stages or STAGES)
    if requested & {
        "A3_native_fallback_frozen_diagnostic",
        "A3_strict_posterior_predictive_fallback_frozen",
    }:
        requested.add("A3_offline_one_layer_caster")
    if "A4_one_layer_caster_without_selected_rho" in requested:
        requested.add("A3_offline_one_layer_caster")
    if "A5_causal_selected_rho" in requested:
        requested.add("A4_one_layer_caster_without_selected_rho")
        requested.add("A3_offline_one_layer_caster")
    if requested & {
        "A1_top1_shared_bridge",
        "A2_topk_static_bridge",
        "A3_offline_one_layer_caster",
        "A3_native_fallback_frozen_diagnostic",
        "A3_strict_posterior_predictive_fallback_frozen",
        "A4_one_layer_caster_without_selected_rho",
    }:
        _require_file(full_bridge_one_layer)
                                                                             
                                                                              
        preflight_stage = next(iter(requested & BRIDGE_SCORE_STAGES))
        _draws_args_for_stage(preflight_stage, paths, full_bridge_one_layer)
        distribution, transform = _bridge_shape(full_bridge_one_layer)
        rho1_extra = [
            "--base-bridge",
            str(full_bridge_one_layer),
            "--distribution",
            distribution,
            "--transform",
            transform,
        ]
        _run(
            [
                args.python,
                str(CALIBRATOR),
                "--ledger",
                str(paths["ledger"]),
                "--archive",
                str(paths["archive"]),
                "--registry",
                str(paths["registry"]),
                "--selection",
                str(paths["selection"]),
                "--rho",
                "1.0",
                "--out",
                str(bridge_rho1),
                "--seed",
                str(args.seed),
                *rho1_extra,
            ],
            cwd=args.repo_root,
            skip_existing=bridge_rho1 if args.skip_existing else None,
        )

    if "A0_top1_naive" in requested:
        _run(
            _stage_cmd(args.python, RUNNER, "A0_top1_naive", paths, task_root / "A0_top1_naive", seed=args.seed),
            cwd=args.repo_root,
            skip_existing=(task_root / "A0_top1_naive" / "stage_metrics.csv") if args.skip_existing else None,
        )
    for stage in ["A1_top1_shared_bridge", "A2_topk_static_bridge"]:
        if stage in requested:
            _run(
                _stage_cmd(args.python, RUNNER, stage, paths, task_root / stage, bridge_config=bridge_rho1, seed=args.seed),
                cwd=args.repo_root,
                skip_existing=(task_root / stage / "stage_metrics.csv") if args.skip_existing else None,
            )
    if "A3_offline_one_layer_caster" in requested:
        _run(
            _stage_cmd(args.python, RUNNER, "A3_offline_one_layer_caster", paths, task_root / "A3_offline_one_layer_caster", bridge_config=bridge_rho1, seed=args.seed),
            cwd=args.repo_root,
            skip_existing=(task_root / "A3_offline_one_layer_caster" / "stage_metrics.csv") if args.skip_existing else None,
        )
    if "A3_native_fallback_frozen_diagnostic" in requested:
        native_scores = paths.get("native_scores")
        native_availability = paths.get("native_availability")
        if native_scores is None or native_availability is None:
            raise SystemExit(
                f"task {task['task']!r} requires native_likelihood_scores and "
                "native_likelihood_availability for A3 native/fallback diagnostic"
            )
        _require_file(native_scores)
        _require_file(native_availability)
        _run(
            _stage_cmd(
                args.python,
                RUNNER,
                "A3_native_fallback_frozen_diagnostic",
                paths,
                task_root / "A3_native_fallback_frozen_diagnostic",
                bridge_config=bridge_rho1,
                native_scores=native_scores,
                native_availability=native_availability,
                seed=args.seed,
            ),
            cwd=args.repo_root,
            skip_existing=(task_root / "A3_native_fallback_frozen_diagnostic" / "stage_metrics.csv")
            if args.skip_existing
            else None,
        )
    if "A3_strict_posterior_predictive_fallback_frozen" in requested:
        native_scores = paths.get("native_scores")
        native_availability = paths.get("native_availability")
                                                                                
                                                                                
                                                                            
        if native_scores is not None and not native_scores.is_file():
            native_scores = None
        if native_availability is not None and not native_availability.is_file():
            native_availability = None
        strict_stage = "A3_strict_posterior_predictive_fallback_frozen"
        _run(
            _stage_cmd(
                args.python,
                RUNNER,
                strict_stage,
                paths,
                task_root / strict_stage,
                bridge_config=bridge_rho1,
                native_scores=native_scores,
                native_availability=native_availability,
                seed=args.seed,
            ),
            cwd=args.repo_root,
            skip_existing=(task_root / strict_stage / "stage_metrics.csv")
            if args.skip_existing
            else None,
        )
    if "A4_one_layer_caster_without_selected_rho" in requested:
        _run(
            _stage_cmd(
                args.python, RUNNER, "A4_one_layer_caster_without_selected_rho", paths, task_root / "A4_one_layer_caster_without_selected_rho",
                bridge_config=bridge_rho1, posterior_update_policy="prequential_asof",
                offline_stage_root=task_root / "A3_offline_one_layer_caster", seed=args.seed,
            ),
            cwd=args.repo_root,
            skip_existing=(task_root / "A4_one_layer_caster_without_selected_rho" / "stage_metrics.csv") if args.skip_existing else None,
        )
    for stage in ["A5_causal_selected_rho"]:
        if stage in requested:
            _run(
                _stage_cmd(
                    args.python,
                    RUNNER,
                    stage,
                    paths,
                    task_root / stage,
                    bridge_config=full_bridge_one_layer,
                    posterior_update_policy="prequential_asof",
                    online_stage_root=task_root / "A4_one_layer_caster_without_selected_rho",
                    seed=args.seed,
                ),
                cwd=args.repo_root,
                skip_existing=(task_root / stage / "stage_metrics.csv") if args.skip_existing else None,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the A1-A5 one-layer construction ablation."
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho-grid", default=DEFAULT_RHO_GRID)
    parser.add_argument("--target-ess-fraction", type=float, default=0.5)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--stages", nargs="*", choices=STAGES + OPTIONAL_STAGES, default=[])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if not args.out_root.is_absolute():
        args.out_root = args.repo_root / args.out_root
    config = _read_yaml(args.spec)
    for script in [RUNNER, CALIBRATOR]:
        _require_file(args.repo_root / script)
    for task in _tasks(config, args.tasks):
        run_task(args, task)
    print(f"incremental_ablation_by_task_root={args.out_root}")


if __name__ == "__main__":
    main()
