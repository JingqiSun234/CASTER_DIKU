#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import string

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
    "moment_t": "Moment-t bridge posterior family trajectories",
    "draw_kernel_t": "Draw-kernel-t bridge posterior family trajectories",
}


def validate_posterior(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["dataset"].astype(str).isin(TASK_ORDER)].copy()
    require_three_tasks(frame)
    if set(frame["method"].astype(str)) != set(METHOD_ORDER):
        raise ValueError("posterior input must contain both CASTER constructions")
    unknown = sorted(set(frame["family"].astype(str)) - set(FAMILY_STYLE))
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")
    result = frame.copy()
    result["dataset"] = result["dataset"].astype(str)
    result["method"] = result["method"].astype(str)
    result["family"] = result["family"].astype(str)
    result["as_of_date"] = pd.to_datetime(result["as_of_date"], errors="raise")
    result["posterior_mass"] = pd.to_numeric(result["posterior_mass"], errors="raise")
    if not np.isfinite(result["posterior_mass"].to_numpy(float)).all():
        raise ValueError("posterior mass must be finite")
    if not result["posterior_mass"].between(0.0, 1.0).all():
        raise ValueError("posterior mass must be in [0, 1]")
    key = ["dataset", "method", "as_of_date", "family"]
    if result.duplicated(key).any():
        raise ValueError("posterior input has duplicate task/method/date/family rows")
    sums = result.groupby(["dataset", "method", "as_of_date"], observed=True)["posterior_mass"].sum()
    error = float((sums - 1.0).abs().max())
    if not math.isfinite(error) or error > 1.0e-8:
        raise ValueError(f"posterior masses do not sum to one; maximum error={error:.6g}")
    expected = {(task, method) for task in TASK_ORDER for method in METHOD_ORDER}
    observed = set(zip(result["dataset"], result["method"]))
    if observed != expected:
        raise ValueError("posterior panel coverage is incomplete")
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


def family_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=style["color"], linestyle=style["linestyle"], linewidth=1.7, label=style["label"])
        for style in FAMILY_STYLE.values()
    ]


def event_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=style["color"], linestyle="--", linewidth=1.15, label=style["label"])
        for style in EVENT_STYLE.values()
    ]


def render(posterior_path: Path, events_path: Path, lane: str, output: Path) -> None:
    posterior = validate_posterior(
        read_csv(
            posterior_path,
            {"dataset", "method", "as_of_date", "family", "posterior_mass"},
        )
    )
    events = validate_events(read_csv(events_path, {"dataset", "event_type", "event_time"}))
    configure_style(8.0)
    fig, axes = plt.subplots(3, 2, figsize=(6.2, 8.0), sharey=True)
    fig.subplots_adjust(left=0.115, right=0.98, top=0.785, bottom=0.075, wspace=0.12, hspace=0.52)
    for row_index, task in enumerate(TASK_ORDER):
        task_events = events[events["dataset"].eq(task)]
        for column_index, method in enumerate(METHOD_ORDER):
            ax = axes[row_index, column_index]
            panel = posterior[posterior["dataset"].eq(task) & posterior["method"].eq(method)]
            for event in task_events.itertuples(index=False):
                style = EVENT_STYLE[str(event.event_type)]
                ax.axvline(
                    pd.Timestamp(event.event_time),
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.34,
                    zorder=1,
                )
            for family, style in FAMILY_STYLE.items():
                series = panel[panel["family"].eq(family)].sort_values("as_of_date", kind="stable")
                if series.empty:
                    continue
                ax.plot(
                    series["as_of_date"],
                    series["posterior_mass"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.35,
                    zorder=3,
                )
            ax.margins(x=0.05)
            ax.set_ylim(0.0, 1.03)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(True, axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.22, zorder=0)
            if task == "benchmark_a":
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU, interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            else:
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            letter = string.ascii_lowercase[row_index * 2 + column_index]
            ax.set_title(f"({letter}) {TASK_LABEL[task]}", loc="left", pad=3.5)

    fig.suptitle(LANE_TITLE[lane], x=0.5, y=0.993, fontsize=10.5, fontweight="bold")
    fig.text(0.315, 0.812, METHOD_LABEL[METHOD_ORDER[0]], ha="center", va="bottom", fontweight="bold")
    fig.text(0.755, 0.812, METHOD_LABEL[METHOD_ORDER[1]], ha="center", va="bottom", fontweight="bold")
    fig.supxlabel("Calendar time", x=0.55, y=0.018)
    fig.supylabel("Posterior family mass", x=0.018, y=0.43)
    fig.legend(
        handles=family_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.967),
        ncol=3,
        frameon=False,
        title="Model families",
        title_fontsize=8.0,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.20,
    )
    fig.legend(
        handles=event_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.882),
        ncol=4,
        frameon=False,
        title="Diagnostic event markers",
        title_fontsize=8.0,
        handlelength=1.55,
        handletextpad=0.32,
        columnspacing=0.72,
        labelspacing=0.15,
    )
    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render family-level posterior trajectories.")
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--lane", choices=tuple(LANE_TITLE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.posterior, args.events, args.lane, args.output)
