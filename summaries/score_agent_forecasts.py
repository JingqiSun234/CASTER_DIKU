from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPTS = ROOT / "code/baseline/scripts"
if str(BASELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPTS))

from aggregate_baseline_results import _recompute_result_metrics_from_forecast


TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
METHODS = ("agentic_top_one", "react", "agentic_full_recovery")
DATASETS = {
    "benchmark_a": "benchmark_a",
    "benchmark_b_covid": "benchmark_b",
    "benchmark_b_flu": "benchmark_b",
}
COMPONENTS = {
    "benchmark_a": "cases",
    "benchmark_b_covid": "covid_adm_per100k",
    "benchmark_b_flu": "flu_adm_per100k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical test metric slices from agent forecasts."
    )
    parser.add_argument(
        "--agent",
        action="append",
        nargs=3,
        metavar=("TASK", "METHOD", "RUN_DIR"),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    bridge_root = args.bridge_root.expanduser().resolve()
    frames: list[pd.DataFrame] = []
    seen: set[tuple[str, str]] = set()
    for task, method, run_dir_name in args.agent:
        if task not in TASKS:
            raise SystemExit(f"unsupported task: {task}")
        if method not in METHODS:
            raise SystemExit(f"unsupported method: {method}")
        key = (task, method)
        if key in seen:
            raise SystemExit(f"duplicate agent entry: {key}")
        seen.add(key)
        run_dir = Path(run_dir_name).expanduser().resolve()
        if not (run_dir / "forecast.csv").is_file():
            raise SystemExit(f"missing forecast: {run_dir / 'forecast.csv'}")
        frame = _recompute_result_metrics_from_forecast(
            run_dir,
            run_root=run_root,
            bridge_config_root=bridge_root,
            strict_bridge_config=True,
            evaluation_split="test",
        )
        datasets = set(frame["dataset"].astype(str))
        components = set(frame["component"].astype(str))
        if datasets != {DATASETS[task]} or components != {COMPONENTS[task]}:
            raise SystemExit(
                f"unexpected task values for {key}: "
                f"datasets={sorted(datasets)}, components={sorted(components)}"
            )
        frame = frame.copy()
        frame["method"] = method
        frames.append(frame)
    expected = {(task, method) for task in TASKS for method in METHODS}
    if seen != expected:
        raise SystemExit(f"missing agent entries: {sorted(expected - seen)}")
    result = pd.concat(frames, ignore_index=True)
    sort_columns = [
        name
        for name in (
            "dataset",
            "method",
            "forecast_strategy",
            "entity_id",
            "component",
            "horizon",
        )
        if name in result.columns
    ]
    result = result.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise SystemExit("output path must end in .csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"rows={len(result)}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
