#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "code" / "caster" / "scripts"
CALIBRATOR = IMPL / "calibrate_incremental_bridge_from_archive.py"
ONE_LAYER = IMPL / "run_incremental_ablation_from_archive.py"
HIERARCHICAL = IMPL / "run_hierarchical_from_archive.py"
SCORER = ROOT / "scripts" / "score_hierarchical_ablation_stage.py"

TASKS = {
    "benchmark_a": ("benchmark_a", "cases"),
    "benchmark_b_covid": (
        "benchmark_b/benchmark_b_covid",
        "covid_adm_per100k",
    ),
    "benchmark_b_flu": (
        "benchmark_b/benchmark_b_flu",
        "flu_adm_per100k",
    ),
}

VARIANTS = {
    "one_layer_draw": (False, True, "caster_one_layer_draw_kernel"),
    "hierarchical_moment": (True, False, "caster_hierarchical"),
    "hierarchical_draw": (True, True, "caster_hierarchical_draw_kernel"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="*", choices=tuple(TASKS), default=[])
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS), default=[])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing inputs: " + ", ".join(missing))


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def execute(command: list[str], *, skip: Path | None = None) -> None:
    if skip is not None and skip.is_file():
        return
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def task_inputs(
    run_root: Path, task: str, *, draw: bool, hierarchical: bool
) -> tuple[dict[str, Path], Path]:
    relative, _component = TASKS[task]
    base = (
        run_root / "branches/draw_kernel/new_method/artifacts"
        if draw
        else run_root / "new_method/artifacts"
    )
    root = base / relative
    lane = "hierarchical" if hierarchical else "one_layer"
    moment_root = run_root / "new_method/artifacts" / relative
    inputs = {
        "ledger": root / "event_ledger.csv",
        "archive": root / "forecast_archive.csv",
        "registry": first_existing(
            root / "model_registry.csv",
            root / lane / "model_registry.csv",
            moment_root / "model_registry.csv",
        ),
        "selection": first_existing(
            root / "candidate_selection_log.csv",
            root / lane / "candidate_selection_log.csv",
            moment_root / "candidate_selection_log.csv",
        ),
    }
    if draw:
        inputs["draws"] = root / "forecast_draws.csv"
    bridge = root / (
        "bridge_config.hierarchical.json"
        if hierarchical
        else "bridge_config.one_layer.json"
    )
    require([*inputs.values(), bridge])
    return inputs, bridge


def make_rho_one(
    args: argparse.Namespace,
    inputs: dict[str, Path],
    base_bridge: Path,
    output: Path,
) -> None:
    payload = json.loads(base_bridge.read_text(encoding="utf-8"))
    command = [
        args.python,
        str(CALIBRATOR),
        "--ledger",
        str(inputs["ledger"]),
        "--archive",
        str(inputs["archive"]),
        "--registry",
        str(inputs["registry"]),
        "--selection",
        str(inputs["selection"]),
        "--rho",
        "1.0",
        "--base-bridge",
        str(base_bridge),
        "--distribution",
        str(payload.get("distribution", "student_t")),
        "--transform",
        str(payload.get("transform", "log1p")),
        "--out",
        str(output),
        "--seed",
        str(args.seed),
    ]
    execute(command, skip=output if args.skip_existing else None)


def one_layer_stage(
    args: argparse.Namespace,
    inputs: dict[str, Path],
    bridge: Path,
    stage: str,
    output: Path,
) -> None:
    command = [
        args.python,
        str(ONE_LAYER),
        "--stage",
        stage,
        "--ledger",
        str(inputs["ledger"]),
        "--archive",
        str(inputs["archive"]),
        "--registry",
        str(inputs["registry"]),
        "--selection",
        str(inputs["selection"]),
        "--bridge-config",
        str(bridge),
        "--out",
        str(output),
        "--seed",
        str(args.seed),
    ]
    if "draws" in inputs:
        command.extend(["--draws", str(inputs["draws"])])
    if stage == "A4_one_layer_caster_without_selected_rho":
        command.extend(
            [
                "--posterior-update-policy",
                "prequential_asof",
                "--offline-stage-root",
                str(output.parent / "A3_offline_one_layer_caster"),
            ]
        )
    skip = output / "stage_metrics.csv" if args.skip_existing else None
    execute(command, skip=skip)


def hierarchical_static(
    args: argparse.Namespace,
    inputs: dict[str, Path],
    bridge: Path,
    output: Path,
) -> None:
    command = [
        args.python,
        str(ONE_LAYER),
        "--stage",
        "A2_topk_static_bridge",
        "--ledger",
        str(inputs["ledger"]),
        "--archive",
        str(inputs["archive"]),
        "--registry",
        str(inputs["registry"]),
        "--selection",
        str(inputs["selection"]),
        "--bridge-config",
        str(bridge),
        "--prior-policy",
        "hierarchical_family_balanced",
        "--out",
        str(output),
        "--seed",
        str(args.seed),
    ]
    if "draws" in inputs:
        command.extend(["--draws", str(inputs["draws"])])
    skip = output / "stage_metrics.csv" if args.skip_existing else None
    execute(command, skip=skip)


def hierarchical_stage(
    args: argparse.Namespace,
    task: str,
    inputs: dict[str, Path],
    bridge: Path,
    method: str,
    frozen: bool,
    output: Path,
) -> None:
    _relative, component = TASKS[task]
    ledger = pd.read_csv(inputs["ledger"], usecols=["split", "forecast_origin"])
    cutoff = pd.to_datetime(
        ledger.loc[ledger["split"].astype(str).eq("test"), "forecast_origin"],
        errors="raise",
    ).min()
    stage = (
        "A3_hierarchical_frozen_pretest_rho1"
        if frozen
        else "A4_hierarchical_online_rho1"
    )
    command = [
        args.python,
        str(HIERARCHICAL),
        "--ledger",
        str(inputs["ledger"]),
        "--archive",
        str(inputs["archive"]),
        "--registry",
        str(inputs["registry"]),
        "--selection",
        str(inputs["selection"]),
        "--bridge-config",
        str(bridge),
        "--out",
        str(output),
        "--update-splits",
        "train,val,embargo" if frozen else "train,val,embargo,test",
        "--readout-split",
        "test",
        "--posterior-update-policy",
        "holdout_train_val" if frozen else "prequential_asof",
        "--seed",
        str(args.seed),
        "--task-id",
        task,
        "--target-components",
        component,
        "--posterior-scope",
        "component_stratified",
        "--score-source",
        "draw_kernel" if "draws" in inputs else "archive_moment",
        "--predictive-contract",
        "coherent_mean_preserving_censored_student_t",
        "--method-id",
        method,
    ]
    if frozen:
        command.extend(["--update-release-cutoff", pd.Timestamp(cutoff).isoformat()])
    if "draws" in inputs:
        command.extend(["--draws", str(inputs["draws"])])
    score = [
        args.python,
        str(SCORER),
        "--stage-root",
        str(output),
        "--stage",
        stage,
        "--method-id",
        stage.lower(),
        "--ledger",
        str(inputs["ledger"]),
        "--archive",
        str(inputs["archive"]),
        "--bridge-config",
        str(bridge),
    ]
    if "draws" in inputs:
        score.extend(["--draws", str(inputs["draws"])])
    skip = output / "stage_metrics.csv" if args.skip_existing else None
    if skip is not None and skip.is_file():
        return
    execute(command)
    execute(score)


def run_variant(
    args: argparse.Namespace, variant: str, task: str
) -> None:
    hierarchical, draw, method = VARIANTS[variant]
    inputs, base_bridge = task_inputs(
        args.run_root, task, draw=draw, hierarchical=hierarchical
    )
    task_root = args.out_root / variant / task
    task_root.mkdir(parents=True, exist_ok=True)
    bridge = task_root / "bridge_rho1.json"
    make_rho_one(args, inputs, base_bridge, bridge)
    if not hierarchical:
        for stage in (
            "A1_top1_shared_bridge",
            "A2_topk_static_bridge",
            "A3_offline_one_layer_caster",
            "A4_one_layer_caster_without_selected_rho",
        ):
            one_layer_stage(args, inputs, bridge, stage, task_root / stage)
        return
    hierarchical_static(args, inputs, bridge, task_root / "A2_family_balanced_static")
    hierarchical_stage(
        args,
        task,
        inputs,
        bridge,
        method,
        True,
        task_root / "A3_hierarchical_frozen_pretest_rho1",
    )
    hierarchical_stage(
        args,
        task,
        inputs,
        bridge,
        method,
        False,
        task_root / "A4_hierarchical_online_rho1",
    )


def main() -> int:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for variant in args.variants or tuple(VARIANTS):
        for task in args.tasks or tuple(TASKS):
            run_variant(args, variant, task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
