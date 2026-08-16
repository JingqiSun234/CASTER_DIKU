#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "qwen25_multiscale_released_sequence_v1"
TASK = "benchmark_b_covid"
K_VALUES = (1, 2, 3, 5, 8, 10, 13, 16, 20, 27)
REPEATS = 3
METHODS = (
    "agentic_full_recovery",
    "caster_one_layer",
    "caster_hierarchical",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three-method runtime experiment."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--selection-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def required_inputs(
    run_root: Path,
    shared_root: Path,
    selection_root: Path,
) -> dict[str, Path]:
    cache = shared_root / "caster_candidates/k27__draws10__seed42"
    task_inputs = cache / "agent_inputs" / PROFILE / TASK
    paths = {
        "task_root": run_root / "new_method/artifacts/benchmark_b" / TASK,
        "ledger": task_inputs / "event_ledger.csv",
        "archive": cache / TASK / "forecast_archive_all27.csv",
        "registry": cache / "model_registry.formal.csv",
        "ranking": selection_root / TASK / "candidate_ranking_all27.csv",
        "manifest": task_inputs / f"{TASK}_manifest.csv",
    }
    missing = [
        str(path)
        for name, path in paths.items()
        if not (path.is_dir() if name == "task_root" else path.is_file())
    ]
    if missing:
        raise SystemExit("missing runtime inputs: " + ", ".join(missing))
    return paths


def base_command(
    run_root: Path,
    selection_root: Path,
    inputs: dict[str, Path],
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_runtime_scaling.py",
        "--run-root",
        str(run_root),
        "--k-values",
        ",".join(str(value) for value in K_VALUES),
        "--repeats",
        str(REPEATS),
        "--timing-task",
        TASK,
        "--task-artifact-root",
        str(inputs["task_root"]),
        "--ledger",
        str(inputs["ledger"]),
        "--forecast-archive",
        str(inputs["archive"]),
        "--model-registry",
        str(inputs["registry"]),
        "--formal-selection-ranking",
        str(inputs["ranking"]),
        "--full-recovery-manifest",
        str(inputs["manifest"]),
        "--candidate-pool-size",
        "27",
        "--candidate-pool-profile",
        "formal_27_country_macro_v1",
        "--selection-input-hash",
        selection_root.name,
        "--reuse-existing-selections",
    ]


def execute(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_and_sort(agent_path: Path, caster_path: Path) -> pd.DataFrame:
    agent = pd.read_csv(agent_path, low_memory=False)
    caster = pd.read_csv(caster_path, low_memory=False)
    if len(agent) != len(K_VALUES) * REPEATS:
        raise ValueError("agent runtime output has an unexpected row count")
    if len(caster) != 2 * len(K_VALUES) * REPEATS:
        raise ValueError("CASTER runtime output has an unexpected row count")
    combined = pd.concat([agent, caster], ignore_index=True, sort=False)
    keys = ["method", "candidate_count", "repeat"]
    missing = [column for column in keys if column not in combined.columns]
    if missing:
        raise ValueError("runtime output is missing columns: " + ", ".join(missing))
    if combined.duplicated(keys).any():
        raise ValueError("runtime output contains duplicate cells")
    observed = {
        (str(row.method), int(row.candidate_count), int(row.repeat))
        for row in combined.loc[:, keys].itertuples(index=False)
    }
    expected = {
        (method, candidate_count, repeat)
        for method in METHODS
        for candidate_count in K_VALUES
        for repeat in range(REPEATS)
    }
    if observed != expected or len(combined) != len(expected):
        raise ValueError("runtime output does not contain the expected 90 cells")
    order = {method: index for index, method in enumerate(METHODS)}
    combined["_method_order"] = combined["method"].map(order)
    combined = combined.sort_values(
        ["_method_order", "candidate_count", "repeat"], kind="stable"
    ).drop(columns="_method_order")
    return combined.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    shared_root = args.shared_root.expanduser().resolve()
    selection_root = args.selection_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    inputs = required_inputs(run_root, shared_root, selection_root)
    common = base_command(run_root, selection_root, inputs)
    with tempfile.TemporaryDirectory(prefix="caster-runtime-") as temporary:
        temporary_root = Path(temporary)
        agent_output = temporary_root / "agent.csv"
        caster_output = temporary_root / "caster.csv"
        execute(
            common
            + [
                "--methods",
                "agentic_full_recovery",
                "--out",
                str(agent_output),
                "--metadata-out",
                str(temporary_root / "agent.json"),
                "--persistent-agent-worker",
                "--agent-selection-mode",
                "true",
                "--agent-selection-engine",
                "qwen",
                "--charge-selection",
            ]
        )
        execute(
            common
            + [
                "--methods",
                "caster_one_layer,caster_hierarchical",
                "--out",
                str(caster_output),
                "--metadata-out",
                str(temporary_root / "caster.json"),
            ]
        )
        result = validate_and_sort(agent_output, caster_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(output.name + ".tmp")
        result.to_csv(temporary_output, index=False)
        os.replace(temporary_output, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
