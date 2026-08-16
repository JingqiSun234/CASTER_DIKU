#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
PROFILE = "qwen25_multiscale_released_sequence_v1"
OUTPUTS = (
    "method_overview.svg",
    "calibration_cases.svg",
    "agent_comparison.svg",
    "moment_t_family_trajectories.svg",
    "draw_kernel_t_family_trajectories.svg",
    "moment_t_model_trajectories.svg",
    "draw_kernel_t_model_trajectories.svg",
    "event_aligned_trajectories.svg",
    "runtime_scaling.svg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the nine result figures from a completed run."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--events-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def execute(command: list[str], environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def full_recovery_forecast(shared_root: Path, task: str) -> Path:
    return (
        shared_root
        / "baseline/runs/agentic"
        / PROFILE
        / task
        / "agentic_full_recovery/forecast.csv"
    )


def render_commands(
    run_root: Path,
    shared_root: Path,
    summary_dir: Path,
    prepared: Path,
    destination: Path,
) -> list[list[str]]:
    calibration = [
        sys.executable,
        "visualization/render_calibration_cases.py",
        "--caster-intervals",
        str(run_root / "new_method/results/rho_only_inputs/forecast_intervals.csv"),
        "--forecast-metrics",
        str(summary_dir / "forecast_metrics.csv"),
        "--comparison-bridge-root",
        str(run_root / "new_method/artifacts"),
    ]
    for task in TASKS:
        calibration.extend(
            ["--comparison-forecast", str(full_recovery_forecast(shared_root, task))]
        )
    calibration.extend(["--output", str(destination / "calibration_cases.svg")])

    commands = [
        [
            sys.executable,
            "visualization/render_overview.py",
            "--output",
            str(destination / "method_overview.svg"),
        ],
        calibration,
        [
            sys.executable,
            "visualization/render_agent_comparison.py",
            "--forecast-summary",
            str(summary_dir / "forecast_metrics.csv"),
            "--selection-summary",
            str(summary_dir / "selection_metrics.csv"),
            "--output",
            str(destination / "agent_comparison.svg"),
        ],
    ]
    for lane in ("moment_t", "draw_kernel_t"):
        commands.extend(
            [
                [
                    sys.executable,
                    "visualization/render_family_trajectories.py",
                    "--posterior",
                    str(prepared / f"{lane}_family.csv"),
                    "--events",
                    str(prepared / "events.csv"),
                    "--lane",
                    lane,
                    "--output",
                    str(destination / f"{lane}_family_trajectories.svg"),
                ],
                [
                    sys.executable,
                    "visualization/render_model_trajectories.py",
                    "--models",
                    str(prepared / f"{lane}_models.csv"),
                    "--events",
                    str(prepared / "events.csv"),
                    "--lane",
                    lane,
                    "--output",
                    str(destination / f"{lane}_model_trajectories.svg"),
                ],
            ]
        )
    commands.extend(
        [
            [
                sys.executable,
                "visualization/render_event_aligned.py",
                "--source",
                str(prepared / "event_aligned.csv"),
                "--output",
                str(destination / "event_aligned_trajectories.svg"),
            ],
            [
                sys.executable,
                "visualization/render_runtime_scaling.py",
                "--source",
                str(run_root / "runtime/cells.csv"),
                "--output",
                str(destination / "runtime_scaling.svg"),
            ],
        ]
    )
    return commands


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    shared_root = args.shared_root.expanduser().resolve()
    events_root = args.events_root.expanduser().resolve()
    summary_dir = args.summary_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        unexpected = sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
            and path.relative_to(output_dir).as_posix() not in OUTPUTS
        )
        if unexpected:
            raise ValueError(
                "output directory contains unexpected files: " + ", ".join(unexpected)
            )

    with tempfile.TemporaryDirectory(prefix="caster-visuals-") as temporary:
        temporary_root = Path(temporary)
        prepared = temporary_root / "inputs"
        destination = temporary_root / "images"
        prepared.mkdir()
        destination.mkdir()
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = str(temporary_root / "matplotlib")
        execute(
            [
                sys.executable,
                "visualization/prepare_posterior_inputs.py",
                "--run-root",
                str(run_root),
                "--events-root",
                str(events_root),
                "--output-dir",
                str(prepared),
            ],
            environment,
        )
        for command in render_commands(
            run_root, shared_root, summary_dir, prepared, destination
        ):
            execute(command, environment)
        observed = tuple(
            sorted(path.name for path in destination.iterdir() if path.is_file())
        )
        expected = tuple(sorted(OUTPUTS))
        if observed != expected:
            raise ValueError(f"figure set differs: expected={expected}, observed={observed}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in OUTPUTS:
            shutil.copy2(destination / name, output_dir / name)
            print(output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
