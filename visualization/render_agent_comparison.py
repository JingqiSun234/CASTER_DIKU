#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _style import TASK_LABEL, TASK_ORDER, configure_style, read_csv, save_figure


DATASETS = ("Benchmark A", "Benchmark B-COVID", "Benchmark B-FLU")
SHORT_LABEL = {
    "Benchmark A": "A",
    "Benchmark B-COVID": "B-COVID",
    "Benchmark B-FLU": "B-FLU",
}
BRIDGES = ("Moment-t", "Draw-kernel")
COLORS = {"Moment-t": "#2166ac", "Draw-kernel": "#b35806"}
HATCHES = {"Moment-t": "", "Draw-kernel": "////"}
METRICS = (
    ("rmse_reduction_percent", "RMSE reduction (%)", "(a) RMSE", True),
    ("wis_reduction_percent", "WIS reduction (%)", "(b) WIS", True),
    ("model_ess", "Model ESS", "(c) ESS", False),
)

BRIDGE_METHOD = {
    "Moment-t": "caster_one_layer",
    "Draw-kernel": "caster_one_layer_draw_kernel",
}
COMPARISON_METHOD = "agentic_full_recovery"


def one_value(
    frame: pd.DataFrame,
    *,
    task: str,
    method: str,
    column: str,
) -> float:
    rows = frame[
        frame["task"].astype(str).eq(task)
        & frame["method"].astype(str).eq(method)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one summary row for {task}/{method}, found {len(rows)}")
    value = float(rows.iloc[0][column])
    if not np.isfinite(value):
        raise ValueError(f"{task}/{method}/{column} is not finite")
    return value


def source_from_fresh_outputs(
    forecast_path: Path,
    selection_path: Path,
) -> pd.DataFrame:
    forecasts = read_csv(forecast_path, {"task", "method", "rmse", "wis"})
    selection = read_csv(selection_path, {"task", "method", "ess"})
    rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        comparison_rmse = one_value(
            forecasts,
            task=task,
            method=COMPARISON_METHOD,
            column="rmse",
        )
        comparison_wis = one_value(
            forecasts,
            task=task,
            method=COMPARISON_METHOD,
            column="wis",
        )
        if comparison_rmse <= 0.0 or comparison_wis <= 0.0:
            raise ValueError(f"comparison metrics must be positive for {task}")
        for bridge, method in BRIDGE_METHOD.items():
            caster_rmse = one_value(
                forecasts,
                task=task,
                method=method,
                column="rmse",
            )
            caster_wis = one_value(
                forecasts,
                task=task,
                method=method,
                column="wis",
            )
            rows.append(
                {
                    "dataset": TASK_LABEL[task].replace("B-COVID", "Benchmark B-COVID").replace("B-FLU", "Benchmark B-FLU"),
                    "bridge": bridge,
                    "rmse_reduction_percent": 100.0 * (comparison_rmse - caster_rmse) / comparison_rmse,
                    "wis_reduction_percent": 100.0 * (comparison_wis - caster_wis) / comparison_wis,
                    "model_ess": one_value(
                        selection,
                        task=task,
                        method=method,
                        column="ess",
                    ),
                }
            )
    return pd.DataFrame(rows)


def metric_matrix(source: pd.DataFrame, metric: str) -> np.ndarray:
    values = np.empty((len(BRIDGES), len(DATASETS)), dtype=float)
    for bridge_index, bridge in enumerate(BRIDGES):
        for dataset_index, dataset in enumerate(DATASETS):
            row = source[
                source["dataset"].astype(str).eq(dataset)
                & source["bridge"].astype(str).eq(bridge)
            ]
            if len(row) != 1:
                raise ValueError(f"expected one row for {dataset}/{bridge}, found {len(row)}")
            values[bridge_index, dataset_index] = float(row.iloc[0][metric])
    if not np.isfinite(values).all():
        raise ValueError(f"{metric} contains non-finite values")
    return values


def draw_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    zero_line: bool,
) -> None:
    values = metric_matrix(source, metric)
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.36
    for bridge_index, bridge in enumerate(BRIDGES):
        ax.bar(
            x + (bridge_index - 0.5) * width,
            values[bridge_index],
            width,
            color=COLORS[bridge],
            edgecolor="#222222",
            linewidth=0.55,
            hatch=HATCHES[bridge],
            label=bridge,
            zorder=3,
        )
    if zero_line:
        ax.axhline(0.0, color="#222222", linewidth=0.8, zorder=2)
    ax.set_xticks(x, [SHORT_LABEL[dataset] for dataset in DATASETS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=2.0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.55, alpha=0.75, zorder=0)
    low = min(0.0, float(values.min()))
    high = max(0.0, float(values.max()))
    span = max(1.0, high - low)
    ax.set_ylim(low - 0.18 * span, high + 0.25 * span)


def render(
    source_path: Path | None,
    forecast_summary: Path | None,
    selection_summary: Path | None,
    ess_column: str,
    output: Path,
) -> None:
    if source_path is not None:
        required = {"dataset", "bridge", "rmse_reduction_percent", "wis_reduction_percent", ess_column}
        source = read_csv(source_path, required)
        if ess_column != "model_ess":
            source = source.rename(columns={ess_column: "model_ess"})
    else:
        if forecast_summary is None or selection_summary is None:
            raise ValueError("provide --source or both fresh summary inputs")
        source = source_from_fresh_outputs(forecast_summary, selection_summary)
    expected_pairs = {(dataset, bridge) for dataset in DATASETS for bridge in BRIDGES}
    observed_pairs = set(zip(source["dataset"].astype(str), source["bridge"].astype(str)))
    if observed_pairs != expected_pairs or len(source) != 6:
        raise ValueError("comparison input must contain one row per dataset and bridge")
    configure_style(9.0)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.55))
    for ax, (metric, ylabel, title, zero_line) in zip(axes, METRICS):
        draw_panel(ax, source, metric, ylabel, title, zero_line)
    fig.legend(
        handles=[
            Patch(facecolor=COLORS[bridge], edgecolor="#222222", linewidth=0.55, hatch=HATCHES[bridge], label=bridge)
            for bridge in BRIDGES
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        ncol=2,
        handlelength=1.5,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.20, top=0.77, wspace=0.50)
    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the bridge comparison summary.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--forecast-summary", type=Path)
    parser.add_argument("--selection-summary", type=Path)
    parser.add_argument("--ess-column", default="model_ess")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(
        args.source,
        args.forecast_summary,
        args.selection_summary,
        args.ess_column,
        args.output,
    )
