from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
TASK_RELATIVE = {
    "benchmark_a": Path("benchmark_a"),
    "benchmark_b_covid": Path("benchmark_b/benchmark_b_covid"),
    "benchmark_b_flu": Path("benchmark_b/benchmark_b_flu"),
}
AGENTS = ("agentic_top_one", "react", "agentic_full_recovery")
PROFILE = "qwen25_multiscale_released_sequence_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the numeric result tables from a completed run."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--events-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def execute(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ledger(task: str) -> Path:
    if task == "benchmark_a":
        return ROOT / "data/benchmark_a/curated_full_v3_direct_rollout7/event_ledger.csv"
    return (
        ROOT
        / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/event_ledger.csv"
    )


def event_source(events_root: Path, task: str) -> Path:
    key = "benchmark_a" if task == "benchmark_a" else "benchmark_b"
    return events_root / key / "events.csv"


def task_root(run_root: Path, task: str, lane: str) -> Path:
    base = run_root if lane == "moment_t" else run_root / "branches/draw_kernel"
    return base / "new_method/artifacts" / TASK_RELATIVE[task]


def agent_root(shared_root: Path, task: str, method: str) -> Path:
    return shared_root / "baseline/runs/agentic" / PROFILE / task / method


def candidate_commands(
    run_root: Path,
    shared_root: Path,
    output_dir: Path,
) -> tuple[list[list[str]], list[Path]]:
    cache = shared_root / "caster_candidates/k27__draws10__seed42"
    registry = cache / "model_registry.formal.csv"
    commands: list[list[str]] = []
    outputs: list[Path] = []
    for task in TASKS:
        output = output_dir / f"candidate_{task}.csv"
        outputs.append(output)
        commands.append(
            [
                sys.executable,
                "summaries/score_candidate_pool.py",
                "--task",
                task,
                "--ledger",
                str(ledger(task)),
                "--archive",
                str(cache / task / "forecast_archive_all27.csv"),
                "--registry",
                str(registry),
                "--bridge",
                str(task_root(run_root, task, "moment_t") / "bridge_config.json"),
                "--output",
                str(output),
            ]
        )
    return commands, outputs


def score_inputs(
    run_root: Path,
    candidate_outputs: list[Path],
    agent_slices: Path,
) -> list[Path]:
    inputs = list(candidate_outputs)
    inputs.append(agent_slices)
    inputs.append(run_root / "new_method/results/rho_only_inputs/metric_slices.csv")
    return inputs


def agent_slice_command(
    run_root: Path,
    shared_root: Path,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "summaries/score_agent_forecasts.py",
        "--run-root",
        str(shared_root / "baseline/runs"),
        "--bridge-root",
        str(run_root / "new_method/artifacts"),
    ]
    for task in TASKS:
        for method in AGENTS:
            command.extend(
                ["--agent", task, method, str(agent_root(shared_root, task, method))]
            )
    command.extend(["--output", str(output)])
    return command


def summary_command(script: str, inputs: list[Path], output: Path) -> list[str]:
    command = [sys.executable, f"summaries/{script}"]
    for path in inputs:
        command.extend(["--input", str(path)])
    command.extend(["--output", str(output)])
    return command


def selection_command(
    run_root: Path,
    shared_root: Path,
    events_root: Path,
    forecast_summary: Path,
    output: Path,
) -> list[str]:
    command = [sys.executable, "summaries/summarize_selection_diagnostics.py"]
    for task in TASKS:
        moment = task_root(run_root, task, "moment_t")
        draw = task_root(run_root, task, "draw_kernel_t")
        lane_specs = (
            (
                "caster_one_layer",
                moment / "posterior_path.csv",
                moment / "asof_posterior_readout_validation.csv",
                "model_id",
                "weight",
            ),
            (
                "caster_hierarchical",
                moment / "family_posterior.csv",
                moment / "asof_posterior_readout_validation.csv",
                "family",
                "family_weight",
            ),
            (
                "caster_one_layer_draw_kernel",
                draw / "one_layer/posterior_path.csv",
                draw / "one_layer/asof_posterior_readout_validation.csv",
                "model_id",
                "weight",
            ),
            (
                "caster_hierarchical_draw_kernel",
                draw / "hierarchical/family_posterior.csv",
                draw / "hierarchical/asof_posterior_readout_validation.csv",
                "family",
                "family_weight",
            ),
        )
        for method, weights, readout, category, weight in lane_specs:
            command.extend(
                [
                    "--lane",
                    task,
                    method,
                    str(weights),
                    str(readout),
                    category,
                    weight,
                ]
            )
        for method in AGENTS:
            command.extend(["--one-hot", task, method])
            command.extend(
                [
                    "--shift-input",
                    task,
                    method,
                    str(agent_root(shared_root, task, method) / "forecast.csv"),
                    str(ledger(task)),
                    str(event_source(events_root, task)),
                    str(moment / "bridge_config.json"),
                ]
            )
        mixture_specs = (
            (
                "caster_one_layer",
                moment / "posterior_path.csv",
                moment / "evidence_log.csv",
                "model_id",
                "weight",
                "log_evidence",
            ),
            (
                "caster_hierarchical",
                moment / "family_posterior.csv",
                moment / "family_posterior.csv",
                "family",
                "family_weight",
                "family_log_evidence",
            ),
            (
                "caster_one_layer_draw_kernel",
                draw / "one_layer/posterior_path.csv",
                draw / "one_layer/evidence_log.csv",
                "model_id",
                "weight",
                "log_evidence",
            ),
            (
                "caster_hierarchical_draw_kernel",
                draw / "hierarchical/family_posterior.csv",
                draw / "hierarchical/family_posterior.csv",
                "family",
                "family_weight",
                "family_log_evidence",
            ),
        )
        for method, weights, evidence, category, weight, score in mixture_specs:
            command.extend(
                [
                    "--mixture-shift-input",
                    task,
                    method,
                    str(weights),
                    str(evidence),
                    str(event_source(events_root, task)),
                    category,
                    weight,
                    score,
                ]
            )
    command.extend(
        ["--forecast-summary", str(forecast_summary), "--output", str(output)]
    )
    return command


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    shared_root = args.shared_root.expanduser().resolve()
    events_root = args.events_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="caster_summary_") as temporary:
        candidate_dir = Path(temporary)
        commands, candidate_outputs = candidate_commands(
            run_root, shared_root, candidate_dir
        )
        for command in commands:
            execute(command)
        agent_slices = candidate_dir / "agent_metric_slices.csv"
        execute(agent_slice_command(run_root, shared_root, agent_slices))
        inputs = score_inputs(run_root, candidate_outputs, agent_slices)
        forecasts = output_dir / "forecast_metrics.csv"
        execute(summary_command("summarize_forecasts.py", inputs, forecasts))
        execute(
            summary_command(
                "summarize_endpoint_scores.py",
                inputs,
                output_dir / "endpoint_metrics.csv",
            )
        )
    execute(
        selection_command(
            run_root,
            shared_root,
            events_root,
            forecasts,
            output_dir / "selection_metrics.csv",
        )
    )
    execute(
        [
            sys.executable,
            "summaries/summarize_ablation.py",
            "--run-root",
            str(run_root),
            "--output",
            str(output_dir / "ablation_metrics.csv"),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
