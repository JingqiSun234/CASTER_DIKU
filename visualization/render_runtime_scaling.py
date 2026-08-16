#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import configure_style, save_figure


METHOD_ORDER = ("One-layer CASTER", "Hierarchical CASTER", "Agentic Full Recovery")
METHOD_ALIAS = {
    "caster_one_layer": "One-layer CASTER",
    "One-layer CASTER": "One-layer CASTER",
    "caster_hierarchical": "Hierarchical CASTER",
    "Hierarchical CASTER": "Hierarchical CASTER",
    "agentic_full_recovery": "Agentic Full Recovery",
    "Agentic Full Recovery": "Agentic Full Recovery",
}
STYLE = {
    "One-layer CASTER": {
        "color": "#0F5CAA",
        "linestyle": "-",
        "marker": "o",
        "band_color": "#A9C9E8",
        "markersize": 5.6,
    },
    "Hierarchical CASTER": {
        "color": "#883B1D",
        "linestyle": "--",
        "marker": "s",
        "band_color": "#D5B6A9",
        "markersize": 3.8,
    },
    "Agentic Full Recovery": {
        "color": "#4D4D4D",
        "linestyle": ":",
        "marker": "^",
        "band_color": "#C8C8C8",
        "markersize": 4.4,
    },
}


def choose_time_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    for candidate in ("runtime_update_sec", "update_sec", "algorithm_update_sec", "total_sec"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"runtime input has no usable time column; requested={requested}")


def summarize(
    frame: pd.DataFrame,
    requested_time_column: str,
    full_recovery_source_label: str,
) -> pd.DataFrame:
    if "method" not in frame.columns:
        raise ValueError("runtime input is missing method")
    work = frame.copy()
    if "status" in work.columns:
        work = work[work["status"].astype(str).eq("ok")]
    if "run_mode" in work.columns:
        work = work[work["run_mode"].astype(str).eq("formal")]
    aliases = dict(METHOD_ALIAS)
    aliases[full_recovery_source_label] = "Agentic Full Recovery"
    work["display_method"] = work["method"].astype(str).map(aliases)
    work = work[work["display_method"].notna()].copy()
    k_column = "K" if "K" in work.columns else "candidate_count"
    if k_column not in work.columns:
        raise ValueError("runtime input is missing candidate-bank size")
    work["K"] = pd.to_numeric(work[k_column], errors="raise").astype(int)

    summary_columns = {"median_sec", "q25_sec", "q75_sec"}
    if summary_columns <= set(work.columns):
        summary = work[["display_method", "K", *sorted(summary_columns)]].drop_duplicates()
        if summary.duplicated(["display_method", "K"]).any():
            raise ValueError("runtime input contains conflicting summaries")
    else:
        time_column = choose_time_column(work, requested_time_column)
        work["seconds"] = pd.to_numeric(work[time_column], errors="raise")
        if not np.isfinite(work["seconds"].to_numpy(float)).all() or not (work["seconds"] > 0.0).all():
            raise ValueError("runtime values must be positive and finite")
        counts = work.groupby(["display_method", "K"], observed=True).size()
        if not counts.ge(3).all():
            raise ValueError("each method and candidate-bank size needs at least three repeats")
        summary = (
            work.groupby(["display_method", "K"], observed=True)["seconds"]
            .agg(
                median_sec="median",
                q25_sec=lambda values: values.quantile(0.25),
                q75_sec=lambda values: values.quantile(0.75),
            )
            .reset_index()
        )
    expected = set(METHOD_ORDER)
    observed = set(summary["display_method"].astype(str))
    if observed != expected:
        raise ValueError(f"runtime method coverage differs: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    k_sets = {
        method: tuple(sorted(summary.loc[summary["display_method"].eq(method), "K"].astype(int)))
        for method in METHOD_ORDER
    }
    if len(set(k_sets.values())) != 1:
        raise ValueError("runtime methods use different candidate-bank grids")
    return summary


def render(
    source_path: Path,
    time_column: str,
    full_recovery_source_label: str,
    output: Path,
) -> None:
    source = pd.read_csv(source_path, low_memory=False)
    if source.empty:
        raise ValueError("runtime input has no rows")
    summary = summarize(source, time_column, full_recovery_source_label)
    configure_style(9.0)
    fig, ax = plt.subplots(figsize=(7.0, 2.75))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.76)
    for method in METHOD_ORDER:
        style = STYLE[method]
        group = summary[summary["display_method"].eq(method)].sort_values("K")
        x = pd.to_numeric(group["K"], errors="raise").to_numpy(float)
        median = pd.to_numeric(group["median_sec"], errors="raise").to_numpy(float)
        q25 = pd.to_numeric(group["q25_sec"], errors="raise").to_numpy(float)
        q75 = pd.to_numeric(group["q75_sec"], errors="raise").to_numpy(float)
        ax.fill_between(x, q25, q75, facecolor=style["band_color"], alpha=0.10, linewidth=0.0, zorder=0.6)
        ax.errorbar(
            x,
            median,
            yerr=np.vstack((median - q25, q75 - median)),
            fmt="none",
            ecolor=style["color"],
            elinewidth=0.80,
            capsize=3.0,
            capthick=0.80,
            zorder=1.2,
        )
        ax.plot(
            x,
            median,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            markersize=style["markersize"],
            linewidth=1.35,
            label=method,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Candidate-bank size K")
    ax.set_ylabel("Prewarmed online seconds (log scale)")
    ax.set_xticks(sorted(summary["K"].astype(int).unique()))
    ax.grid(True, axis="y", linewidth=0.55, alpha=0.24)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.43),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.0,
    )
    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render runtime scaling medians and interquartile ranges.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--time-column", default="runtime_update_sec")
    parser.add_argument("--full-recovery-source-label", default="Agentic Full Recovery")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.source, args.time_column, args.full_recovery_source_label, args.output)
