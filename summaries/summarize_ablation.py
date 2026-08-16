from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import TASKS, macro, prepare, read_csv, task_column, write_csv


LANE_METHODS = {
    "one_layer_moment_t": "caster_one_layer",
    "one_layer_draw_kernel_t": "caster_one_layer_draw_kernel",
    "hierarchical_moment_t": "caster_hierarchical",
    "hierarchical_draw_kernel_t": "caster_hierarchical_draw_kernel",
}
AUTO_STAGES = {
    "one_layer_moment_t": (
        ("A1", "ablation/one_layer_moment/{task}/A1_top1_shared_bridge/stage_metrics.csv"),
        ("A2", "ablation/one_layer_moment/{task}/A2_topk_static_bridge/stage_metrics.csv"),
        ("A3", "ablation/one_layer_moment/{task}/A3_offline_one_layer_caster/stage_metrics.csv"),
        ("A4", "ablation/one_layer_moment/{task}/A4_one_layer_caster_without_selected_rho/stage_metrics.csv"),
    ),
    "one_layer_draw_kernel_t": (
        ("A1", "ablation/additional/one_layer_draw/{task}/A1_top1_shared_bridge/stage_metrics.csv"),
        ("A2", "ablation/additional/one_layer_draw/{task}/A2_topk_static_bridge/stage_metrics.csv"),
        ("A3", "ablation/additional/one_layer_draw/{task}/A3_offline_one_layer_caster/stage_metrics.csv"),
        ("A4", "ablation/additional/one_layer_draw/{task}/A4_one_layer_caster_without_selected_rho/stage_metrics.csv"),
    ),
    "hierarchical_moment_t": (
        ("A1", "ablation/one_layer_moment/{task}/A1_top1_shared_bridge/stage_metrics.csv"),
        ("A2", "ablation/additional/hierarchical_moment/{task}/A2_family_balanced_static/stage_metrics.csv"),
        ("A3", "ablation/additional/hierarchical_moment/{task}/A3_hierarchical_frozen_pretest_rho1/stage_metrics.csv"),
        ("A4", "ablation/additional/hierarchical_moment/{task}/A4_hierarchical_online_rho1/stage_metrics.csv"),
    ),
    "hierarchical_draw_kernel_t": (
        ("A1", "ablation/additional/one_layer_draw/{task}/A1_top1_shared_bridge/stage_metrics.csv"),
        ("A2", "ablation/additional/hierarchical_draw/{task}/A2_family_balanced_static/stage_metrics.csv"),
        ("A3", "ablation/additional/hierarchical_draw/{task}/A3_hierarchical_frozen_pretest_rho1/stage_metrics.csv"),
        ("A4", "ablation/additional/hierarchical_draw/{task}/A4_hierarchical_online_rho1/stage_metrics.csv"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cumulative component variants.")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=4,
        metavar=("TASK", "LANE", "STAGE", "INPUT"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stage_specs(args: argparse.Namespace) -> list[list[str]]:
    specs = [list(values) for values in args.stage]
    if args.run_root is not None:
        full = args.run_root / "new_method/results/rho_only_inputs/metric_slices.csv"
        for task in TASKS:
            for lane, stages in AUTO_STAGES.items():
                for stage, relative in stages:
                    specs.append([task, lane, stage, str(args.run_root / relative.format(task=task))])
                specs.append([task, lane, "A5", str(full)])
    if not specs:
        raise SystemExit("--run-root or at least one --stage entry is required")
    return specs


def main() -> int:
    args = parse_args()
    output: list[dict[str, object]] = []
    for task, lane, stage, input_name in stage_specs(args):
        if task not in TASKS:
            raise SystemExit(f"unsupported task: {task}")
        source = read_csv(Path(input_name)).copy()
        source_task = task_column(source)
        available = set(source[source_task].astype(str))
        if task in available:
            source = source[source[source_task].astype(str).eq(task)].copy()
        elif not (
            task in {"benchmark_b_covid", "benchmark_b_flu"}
            and available == {"benchmark_b"}
        ):
            raise SystemExit(f"cannot select {task} from tasks {sorted(available)}: {input_name}")
        source["task"] = task
        rows = prepare(source)
        rows = rows[rows["task"].eq(task)].copy()
        methods = sorted(rows["method"].unique())
        lane_method = LANE_METHODS.get(lane, "")
        if lane_method in methods:
            rows = rows[rows["method"].eq(lane_method)].copy()
        elif len(methods) != 1:
            raise SystemExit(f"cannot select {lane} from methods {methods}: {input_name}")
        if rows.empty:
            raise SystemExit(f"no rows for {task}: {input_name}")
        values = macro(rows)
        output.append(
            {
                "task": task,
                "lane": lane,
                "stage": stage,
                "rmse": values["rmse"],
                "mae": values["mae"],
                "nll": values["nll"],
                "wis": values["wis"],
                "coverage_90": values["coverage_90"],
                "width_90": values["width_90"],
                "n_total": float(rows["n"].sum()),
                "metric_rows": int(len(rows)),
            }
        )
    result = pd.DataFrame(output)
    if result[["task", "lane", "stage"]].duplicated().any():
        raise SystemExit("duplicate task, lane, and stage entries")
    result = result.sort_values(["task", "lane", "stage"], kind="stable")
    write_csv(result, args.output)
    print(f"rows={len(result)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
