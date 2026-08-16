#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _style import (
    FAMILY_STYLE,
    METHOD_ORDER,
    TASK_ORDER,
    configure_style,
    read_csv,
    require_three_tasks,
    save_figure,
)


ROW_SPECS = (
    ("benchmark_a", "declared_wave_onset", "A-wave"),
    ("benchmark_b_covid", "variant_turnover", "B-variant"),
    ("benchmark_b_covid", "winter_onset", "B-COVID"),
    ("benchmark_b_flu", "winter_onset", "B-FLU"),
)


def family_handles(present: set[str]) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.70,
            linewidth=1.25,
            label=style["label"],
        )
        for family, style in FAMILY_STYLE.items()
        if family in present
    ]


def status_lookup(path: Path | None) -> dict[tuple[str, str, str], str]:
    if path is None:
        return {}
    frame = read_csv(path, {"dataset", "method", "event_type", "claim_status"})
    lookup: dict[tuple[str, str, str], str] = {}
    for keys, group in frame.groupby(["dataset", "method", "event_type"], dropna=False):
        values = sorted(set(group["claim_status"].astype(str)))
        lookup[tuple(str(value) for value in keys)] = "/".join(values)
    return lookup


def render(source_path: Path, status_path: Path | None, output: Path) -> None:
    source = read_csv(
        source_path,
        {
            "dataset",
            "method",
            "event_type",
            "family",
            "relative_week",
            "mean_posterior_mass",
            "se_posterior_mass",
            "n_events_total",
        },
    ).copy()
    source = source[source["dataset"].astype(str).isin(TASK_ORDER)].copy()
    require_three_tasks(source)
    for column in ("dataset", "method", "event_type", "family"):
        source[column] = source[column].astype(str)
    for column in ("relative_week", "mean_posterior_mass", "se_posterior_mass", "n_events_total"):
        source[column] = pd.to_numeric(source[column], errors="raise")
    unknown = sorted(set(source["family"]) - set(FAMILY_STYLE))
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")
    expected = {
        (dataset, method, event_type)
        for dataset, event_type, _label in ROW_SPECS
        for method in METHOD_ORDER
    }
    observed = set(zip(source["dataset"], source["method"], source["event_type"]))
    if observed != expected:
        raise ValueError(f"event-panel coverage differs: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    statuses = status_lookup(status_path)

    configure_style(9.0)
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 7.70), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.110, right=0.965, bottom=0.095, top=0.885, hspace=0.46, wspace=0.20)
    panel_index = 0
    for row_index, (dataset, event_type, row_label) in enumerate(ROW_SPECS):
        for column_index, method in enumerate(METHOD_ORDER):
            ax = axes[row_index, column_index]
            group = source[
                source["dataset"].eq(dataset)
                & source["event_type"].eq(event_type)
                & source["method"].eq(method)
            ]
            for family, style in FAMILY_STYLE.items():
                family_data = group[group["family"].eq(family)].sort_values("relative_week")
                if family_data.empty:
                    continue
                x = family_data["relative_week"].to_numpy(float)
                y = family_data["mean_posterior_mass"].to_numpy(float)
                se = family_data["se_posterior_mass"].fillna(0.0).to_numpy(float)
                lower = np.clip(y - se, 0.0, 1.0)
                upper = np.clip(y + se, 0.0, 1.0)
                if np.any(se > 0.0):
                    ax.fill_between(x, lower, upper, facecolor=style["color"], alpha=0.08, linewidth=0.0, zorder=0.6)
                for boundary in (lower, upper):
                    ax.plot(x, boundary, color=style["color"], linestyle=(0, (1.5, 1.5)), linewidth=0.55, zorder=1)
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markerfacecolor="white",
                    markeredgewidth=0.70,
                    markersize=3.7,
                    linewidth=1.25,
                )
            ax.axvline(0.0, color="#111111", linestyle="--", linewidth=0.75)
            letter = string.ascii_lowercase[panel_index]
            method_short = "one-layer" if method == "caster_one_layer" else "hierarchical"
            ax.set_title(f"({letter}) {row_label} · {method_short}", loc="left", pad=3)
            ax.set_xlim(-4, 8)
            ax.set_xticks([-4, -2, 0, 2, 4, 6, 8])
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(True, axis="y", linewidth=0.55, alpha=0.22)
            count = int(group["n_events_total"].max())
            status = statuses.get((dataset, method, event_type))
            label = f"n={count} events" if status is None else f"n={count} events; H3 {status}"
            ax.text(
                0.025,
                0.965,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.0,
                bbox={"facecolor": "white", "edgecolor": "#666666", "linewidth": 0.5, "alpha": 1.0, "pad": 1.5},
                zorder=5,
            )
            panel_index += 1
    handles = family_handles(set(source["family"]))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.15,
        labelspacing=0.45,
    )
    fig.supylabel("Posterior family mass", x=0.020, fontsize=9.5)
    fig.supxlabel("Weeks relative to predeclared event", y=0.018, fontsize=9.5)
    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render event-aligned posterior trajectories.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.source, args.status, args.output)
