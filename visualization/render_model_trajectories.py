#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import string
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _style import (
    EVENT_STYLE,
    FAMILY_STYLE,
    METHOD_LABEL,
    METHOD_ORDER,
    TASK_LABEL,
    TASK_ORDER,
    configure_style,
    read_csv,
    require_three_tasks,
    save_figure,
)


LANE_TITLE = {
    "moment_t": "Moment-t bridge posterior trajectories by model",
    "draw_kernel_t": "Draw-kernel-t bridge posterior trajectories by model",
}

STYLE_CYCLE: tuple[tuple[object, str | None], ...] = (
    ("-", None),
    ((0, (5.0, 1.5)), None),
    ((0, (1.2, 1.2)), None),
    ((0, (4.0, 1.2, 1.2, 1.2)), None),
    ("-", "o"),
    ((0, (5.0, 1.5)), "s"),
    ((0, (1.2, 1.2)), "^"),
    ((0, (4.0, 1.2, 1.2, 1.2)), "D"),
)


def validate_models(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["dataset"].astype(str).isin(TASK_ORDER)].copy()
    require_three_tasks(frame)
    if set(frame["method"].astype(str)) != set(METHOD_ORDER):
        raise ValueError("model input must contain both CASTER constructions")
    result = frame.copy()
    for column in ("dataset", "method", "model_id", "model_label", "family"):
        result[column] = result[column].astype(str)
    result["as_of_date"] = pd.to_datetime(result["as_of_date"], errors="raise")
    result["selection_rank"] = pd.to_numeric(result["selection_rank"], errors="raise").astype(int)
    result["posterior_mass"] = pd.to_numeric(result["posterior_mass"], errors="raise")
    unknown = sorted(set(result["family"]) - set(FAMILY_STYLE))
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")
    values = result["posterior_mass"].to_numpy(float)
    if not np.isfinite(values).all() or not result["posterior_mass"].between(0.0, 1.0).all():
        raise ValueError("posterior mass must be finite and in [0, 1]")
    key = ["dataset", "method", "as_of_date", "model_id"]
    if result.duplicated(key).any():
        raise ValueError("model input has duplicate task/method/date/model rows")
    sums = result.groupby(["dataset", "method", "as_of_date"], observed=True)["posterior_mass"].sum()
    error = float((sums - 1.0).abs().max())
    if not math.isfinite(error) or error > 1.0e-8:
        raise ValueError(f"posterior masses do not sum to one; maximum error={error:.6g}")
    expected = {(task, method) for task in TASK_ORDER for method in METHOD_ORDER}
    if set(zip(result["dataset"], result["method"])) != expected:
        raise ValueError("model panel coverage is incomplete")
    for task, method in expected:
        panel = result[result["dataset"].eq(task) & result["method"].eq(method)]
        models = panel[["selection_rank", "model_id"]].drop_duplicates()
        if len(models) != 10 or set(models["selection_rank"]) != set(range(1, 11)):
            raise ValueError(f"{task}/{method} must contain selection ranks 1 through 10")
    return result


def validate_events(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["dataset"].astype(str).isin(TASK_ORDER)].copy()
    require_three_tasks(frame)
    result = frame.copy()
    result["dataset"] = result["dataset"].astype(str)
    result["event_type"] = result["event_type"].astype(str)
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise")
    unknown = sorted(set(result["event_type"]) - set(EVENT_STYLE))
    if unknown:
        raise ValueError(f"unknown event types: {unknown}")
    return (
        result.sort_values(["dataset", "event_time", "event_type"], kind="stable")
        .drop_duplicates(["dataset", "event_type", "event_time"], keep="first")
        .reset_index(drop=True)
    )


def build_styles(frame: pd.DataFrame) -> dict[str, tuple[object, str | None]]:
    styles: dict[str, tuple[object, str | None]] = {}
    pairs = frame[["family", "model_id"]].drop_duplicates()
    for family, group in pairs.groupby("family", observed=True):
        model_ids = sorted(group["model_id"].astype(str))
        if len(model_ids) > len(STYLE_CYCLE):
            raise ValueError(f"family {family} has too many distinct models for the style cycle")
        for index, model_id in enumerate(model_ids):
            styles[model_id] = STYLE_CYCLE[index]
    return styles


def event_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=style["color"], linestyle=(0, (6.0, 3.0)), linewidth=1.05, label=style["label"])
        for style in EVENT_STYLE.values()
    ]


def model_handle(
    row: object,
    styles: Mapping[str, tuple[object, str | None]],
) -> Line2D:
    model_id = str(row.model_id)
    family = str(row.family)
    linestyle, marker = styles[model_id]
    return Line2D(
        [],
        [],
        color=FAMILY_STYLE[family]["color"],
        linestyle=linestyle,
        marker=marker,
        markersize=2.5,
        linewidth=1.15,
        label=f"{int(row.selection_rank)}  {row.model_label}",
    )


def render(models_path: Path, events_path: Path, lane: str, output: Path) -> None:
    models = validate_models(
        read_csv(
            models_path,
            {
                "dataset",
                "method",
                "as_of_date",
                "selection_rank",
                "model_id",
                "model_label",
                "family",
                "posterior_mass",
            },
        )
    )
    events = validate_events(read_csv(events_path, {"dataset", "event_type", "event_time"}))
    styles = build_styles(models)
    configure_style(9.1)
    fig = plt.figure(figsize=(7.4, 9.2))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 1.0, 0.86),
        left=0.085,
        right=0.995,
        top=0.80,
        bottom=0.06,
        wspace=0.14,
        hspace=0.30,
    )
    header_axes: dict[str, plt.Axes] = {}
    for row_index, task in enumerate(TASK_ORDER):
        left_axis: plt.Axes | None = None
        for column_index, method in enumerate(METHOD_ORDER):
            ax = fig.add_subplot(grid[row_index, column_index], sharey=left_axis)
            if row_index == 0:
                header_axes[method] = ax
            if left_axis is None:
                left_axis = ax
            else:
                ax.tick_params(labelleft=False)
            panel = models[models["dataset"].eq(task) & models["method"].eq(method)]
            task_events = events[events["dataset"].eq(task)]
            for event in task_events.itertuples(index=False):
                style = EVENT_STYLE[str(event.event_type)]
                ax.axvline(
                    pd.Timestamp(event.event_time),
                    color=style["color"],
                    linestyle=(0, (6.0, 3.0)),
                    linewidth=0.9,
                    alpha=0.28,
                    zorder=1,
                )
            ordered_models = (
                panel[["selection_rank", "model_id"]]
                .drop_duplicates()
                .sort_values("selection_rank", kind="stable")["model_id"]
            )
            for model_id in ordered_models:
                series = panel[panel["model_id"].eq(model_id)].sort_values("as_of_date", kind="stable")
                family = str(series["family"].iloc[0])
                linestyle, marker = styles[str(model_id)]
                ax.plot(
                    series["as_of_date"],
                    series["posterior_mass"],
                    color=FAMILY_STYLE[family]["color"],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=2.0,
                    markevery=max(1, len(series) // 8) if marker else None,
                    linewidth=1.05,
                    zorder=3,
                )
            ax.set_ylim(0.0, 1.03)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.margins(x=0.04)
            ax.grid(True, axis="y", color="#B0B0B0", linewidth=0.55, alpha=0.22, zorder=0)
            if task == "benchmark_a":
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU, interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            else:
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            letter = string.ascii_lowercase[row_index * 2 + column_index]
            ax.set_title(f"({letter}) {TASK_LABEL[task]}", loc="left", pad=2.5)

        legend_axis = fig.add_subplot(grid[row_index, 2])
        legend_axis.axis("off")
        key = (
            models[models["dataset"].eq(task)][["selection_rank", "model_id", "model_label", "family"]]
            .drop_duplicates()
            .sort_values("selection_rank", kind="stable")
        )
        legend_axis.legend(
            handles=[model_handle(row, styles) for row in key.itertuples(index=False)],
            loc="center left",
            bbox_to_anchor=(0.02, 0.5),
            ncol=1,
            frameon=False,
            title=f"{TASK_LABEL[task]} Top-10\n(selection rank)",
            fontsize=8.7,
            title_fontsize=9.1,
            handlelength=1.7,
            handletextpad=0.35,
            labelspacing=0.10,
            borderaxespad=0.0,
        )

    fig.suptitle(LANE_TITLE[lane], y=0.995, fontsize=12.0, fontweight="bold")
    for method in METHOD_ORDER:
        position = header_axes[method].get_position()
        fig.text(
            0.5 * (position.x0 + position.x1),
            0.817,
            METHOD_LABEL[method],
            ha="center",
            va="bottom",
            fontsize=10.3,
            fontweight="bold",
        )
    fig.supylabel("Posterior model mass", x=0.016, y=0.42, fontsize=9.1)
    fig.legend(
        handles=event_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        title="Diagnostic event markers",
        title_fontsize=9.1,
        handlelength=2.0,
        handletextpad=0.35,
        columnspacing=0.9,
        labelspacing=0.15,
    )
    fig.text(
        0.5,
        0.865,
        "Model color encodes family; line pattern and sparse markers identify individual models.",
        ha="center",
        va="center",
        fontsize=9.1,
    )
    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render model-level posterior trajectories.")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--lane", choices=tuple(LANE_TITLE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.models, args.events, args.lane, args.output)
