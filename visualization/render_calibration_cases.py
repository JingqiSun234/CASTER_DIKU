#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _style import configure_style, read_csv, save_figure


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPTS = ROOT / "code/baseline/scripts"
if str(BASELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPTS))

from aggregate_baseline_results import (  # noqa: E402
    _attach_bridge_scores,
    _filter_to_evaluation_split,
)


BLUE = "#2166ac"
ORANGE = "#b35806"
GREEN = "#087f5b"
RED = "#d73027"
PURPLE = "#6a3d9a"
DARK_GRAY = "#4d4d4d"

DATASET_STYLES = {
    "Benchmark A": {
        "caster_color": PURPLE,
        "caster_linestyle": "-",
        "caster_marker": "P",
        "comparison_color": DARK_GRAY,
        "comparison_linestyle": "--",
        "comparison_marker": "X",
        "short": "A",
    },
    "Benchmark B-COVID": {
        "caster_color": BLUE,
        "caster_linestyle": "-",
        "caster_marker": "o",
        "comparison_color": ORANGE,
        "comparison_linestyle": "--",
        "comparison_marker": "s",
        "short": "B-COVID",
    },
    "Benchmark B-FLU": {
        "caster_color": GREEN,
        "caster_linestyle": "-.",
        "caster_marker": "^",
        "comparison_color": RED,
        "comparison_linestyle": ":",
        "comparison_marker": "D",
        "short": "B-FLU",
    },
}

RELIABILITY_COLUMNS = {
    "row_order",
    "Dataset",
    "method_role",
    "Nominal coverage",
    "Empirical coverage",
    "caster_method",
    "caster_method_label",
}
CASE_COLUMNS = {
    "row_order",
    "case_rank",
    "target_time",
    "forecast_id",
    "lower_90",
    "upper_90",
    "comparator_lower_90",
    "comparator_upper_90",
    "observed_value",
    "predictive_mean",
    "comparator_pred_mean",
}

FORECAST_METRIC_COLUMNS = {
    "task",
    "method",
    "coverage_50",
    "width_50",
    "coverage_90",
    "width_90",
    "n_total",
}

METHOD_SPECS = (
    (0, "caster_one_layer_draw_kernel", "Draw-kernel t - one-layer"),
    (1, "caster_one_layer", "Moment t - one-layer"),
    (2, "caster_hierarchical", "Moment t - hierarchical"),
    (3, "caster_hierarchical_draw_kernel", "Draw-kernel t - hierarchical"),
)
DATASET_LABEL = {
    "benchmark_a": "Benchmark A",
    "benchmark_b_covid": "Benchmark B-COVID",
    "benchmark_b_flu": "Benchmark B-FLU",
}
COMPARISON_METHOD = "agentic_full_recovery"
CASE_SPECS = (
    (1, "benchmark_b_covid", "Georgia"),
    (2, "benchmark_b_flu", "West Virginia"),
)


def normalize_comparison(paths: list[Path], bridge_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame = _filter_to_evaluation_split(frame, "test")
        frame = _attach_bridge_scores(
            frame,
            run_dir=path.parent,
            bridge_config_root=bridge_root,
            strict_bridge_config=True,
        )
        if "dataset_key" in frame.columns:
            frame["dataset"] = frame["dataset_key"].astype(str)
        aliases = {
            "y_true": "observed_value",
            "bridge_predictive_mean": "predictive_mean",
            "bridge_lower_50": "lower_50",
            "bridge_upper_50": "upper_50",
            "bridge_lower_90": "lower_90",
            "bridge_upper_90": "upper_90",
        }
        for old, new in aliases.items():
            if old not in frame.columns:
                raise ValueError(f"{path} is missing formal readout column: {old}")
            frame[new] = pd.to_numeric(frame[old], errors="raise")
        if "jurisdiction" not in frame.columns and "entity_id" in frame.columns:
            frame["jurisdiction"] = frame["entity_id"].astype(str)
        required = {
            "dataset",
            "forecast_origin",
            "target_time",
            "horizon",
            "jurisdiction",
            "observed_value",
            "predictive_mean",
            "lower_50",
            "upper_50",
            "lower_90",
            "upper_90",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing comparison columns: {', '.join(missing)}")
        frame["method"] = COMPARISON_METHOD
        for lower in ("lower_50", "lower_90"):
            frame[lower] = pd.to_numeric(frame[lower], errors="raise").clip(lower=0.0)
        frames.append(frame)
    if not frames:
        raise ValueError("at least one comparison forecast is required")
    return pd.concat(frames, ignore_index=True)


def normalize_caster(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "dataset",
        "method",
        "forecast_id",
        "forecast_origin",
        "target_time",
        "horizon",
        "jurisdiction",
        "observed_value",
        "predictive_mean",
        "lower_50",
        "upper_50",
        "lower_90",
        "upper_90",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing CASTER interval columns: {', '.join(missing)}")
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).eq("test")].copy()
    allowed = {method for _row, method, _label in METHOD_SPECS}
    frame = frame[
        frame["dataset"].astype(str).isin(DATASET_LABEL)
        & frame["method"].astype(str).isin(allowed)
    ].copy()
    return frame


def normalize_forecast_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip", low_memory=False)
    missing = sorted(FORECAST_METRIC_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path} is missing macro forecast-summary columns: {', '.join(missing)}"
        )
    frame = frame.copy()
    frame["task"] = frame["task"].astype(str)
    frame["method"] = frame["method"].astype(str)
    for name in (
        "coverage_50",
        "width_50",
        "coverage_90",
        "width_90",
        "n_total",
    ):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(f"{path} contains non-finite {name}")
    if frame.duplicated(["task", "method"], keep=False).any():
        raise ValueError(f"{path} contains duplicate task/method summary rows")
    return frame


def build_macro_reliability(forecast_metrics: Path) -> pd.DataFrame:
    summary = normalize_forecast_metrics(forecast_metrics)
    expected_methods = {
        COMPARISON_METHOD,
        *[method for _row, method, _label in METHOD_SPECS],
    }
    selected = summary[
        summary["task"].astype(str).isin(DATASET_LABEL)
        & summary["method"].astype(str).isin(expected_methods)
    ].copy()
    rows: list[dict[str, object]] = []
    for row_order, caster_method, method_label in METHOD_SPECS:
        for task, dataset_label in DATASET_LABEL.items():
            for method_role, method in (
                ("caster", caster_method),
                ("comparator", COMPARISON_METHOD),
            ):
                selected = summary[
                    summary["task"].eq(task) & summary["method"].eq(method)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        "expected one macro forecast-summary row for "
                        f"task={task}, method={method}; found {len(selected)}"
                    )
                record = selected.iloc[0]
                endpoint_count = int(round(float(record["n_total"])))
                for level in (50, 90):
                    rows.append(
                        {
                            "row_order": row_order,
                            "caster_method": caster_method,
                            "caster_method_label": method_label,
                            "Dataset": dataset_label,
                            "Nominal coverage": level / 100.0,
                            "method_role": method_role,
                            "Empirical coverage": float(record[f"coverage_{level}"]),
                            "Mean interval width": float(record[f"width_{level}"]),
                            "Number of forecasts": endpoint_count,
                            "aggregation": "hierarchical_macro_contract",
                        }
                    )
    result = pd.DataFrame(rows).sort_values(
        ["row_order", "Dataset", "Nominal coverage", "method_role"],
        kind="stable",
    )
    if len(result) != 48:
        raise ValueError(f"expected 48 macro reliability rows, found {len(result)}")
    return result.reset_index(drop=True)


def build_fresh_inputs(
    caster_intervals: Path,
    comparison_forecasts: list[Path],
    forecast_metrics: Path,
    comparison_bridge_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    caster = normalize_caster(caster_intervals)
    comparison = normalize_comparison(comparison_forecasts, comparison_bridge_root)
    reliability = build_macro_reliability(forecast_metrics)

    case_rows: list[pd.DataFrame] = []
    start = pd.Timestamp("2025-09-20")
    stop = pd.Timestamp("2026-02-28")
    join_keys = ["dataset", "jurisdiction", "forecast_origin", "target_time", "horizon"]
    for row_order, method, method_label in METHOD_SPECS:
        for case_rank, dataset, jurisdiction in CASE_SPECS:
            left = caster[
                caster["dataset"].astype(str).eq(dataset)
                & caster["method"].astype(str).eq(method)
                & caster["jurisdiction"].astype(str).eq(jurisdiction)
                & pd.to_numeric(caster["horizon"], errors="raise").eq(1)
            ].copy()
            right = comparison[
                comparison["dataset"].astype(str).eq(dataset)
                & comparison["jurisdiction"].astype(str).eq(jurisdiction)
                & pd.to_numeric(comparison["horizon"], errors="raise").eq(1)
            ].copy()
            for frame in (left, right):
                frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"], errors="raise")
                frame["target_time"] = pd.to_datetime(frame["target_time"], errors="raise")
            left = left[left["forecast_origin"].between(start, stop)]
            right = right[right["forecast_origin"].between(start, stop)]
            right = right[
                join_keys
                + ["predictive_mean", "lower_90", "upper_90"]
            ].rename(
                columns={
                    "predictive_mean": "comparator_pred_mean",
                    "lower_90": "comparator_lower_90",
                    "upper_90": "comparator_upper_90",
                }
            )
            merged = left.merge(right, on=join_keys, how="inner", validate="one_to_one")
            if len(merged) != 24:
                raise ValueError(f"expected 24 case endpoints for {dataset}/{method}, found {len(merged)}")
            merged["row_order"] = row_order
            merged["case_rank"] = case_rank
            merged["caster_method"] = method
            merged["caster_method_label"] = method_label
            case_rows.append(merged)
    return reliability, pd.concat(case_rows, ignore_index=True)


def reliability_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    row_order: int,
    *,
    title: str | None,
    show_xlabel: bool,
) -> dict[str, Line2D]:
    rows = source[pd.to_numeric(source["row_order"], errors="raise").eq(row_order)]
    ideal = ax.plot([0.5, 0.9], [0.5, 0.9], color="#595959", label="Ideal", zorder=2)[0]
    handles: dict[str, Line2D] = {"Ideal": ideal}
    for dataset, style in DATASET_STYLES.items():
        for role in ("caster", "comparator"):
            group = rows[
                rows["Dataset"].astype(str).eq(dataset)
                & rows["method_role"].astype(str).eq(role)
            ].sort_values("Nominal coverage")
            if group.empty:
                raise ValueError(f"missing reliability rows for row={row_order}, dataset={dataset}, role={role}")
            is_caster = role == "caster"
            label = f"{style['short']} - CASTER" if is_caster else f"{style['short']} - Full Recovery"
            line = ax.plot(
                pd.to_numeric(group["Nominal coverage"], errors="raise"),
                pd.to_numeric(group["Empirical coverage"], errors="raise"),
                color=style["caster_color" if is_caster else "comparison_color"],
                linestyle=style["caster_linestyle" if is_caster else "comparison_linestyle"],
                marker=style["caster_marker" if is_caster else "comparison_marker"],
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.75,
                label=label,
                zorder=3,
            )[0]
            handles[label] = line
    if title:
        ax.set_title(title, pad=4.0)
    ax.set_xlim(0.47, 0.93)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.5, 0.9], ["50%", "90%"])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("Coverage", labelpad=2.0)
    if show_xlabel:
        ax.set_xlabel("Nominal", labelpad=1.5)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(color="#d9d9d9", linewidth=0.55, alpha=0.75)
    return handles


def common_ylim(source: pd.DataFrame, case_rank: int) -> tuple[float, float]:
    rows = source[pd.to_numeric(source["case_rank"], errors="raise").eq(case_rank)]
    values = pd.concat(
        [
            pd.to_numeric(rows[column], errors="coerce")
            for column in (
                "lower_90",
                "upper_90",
                "comparator_lower_90",
                "comparator_upper_90",
                "observed_value",
            )
        ],
        ignore_index=True,
    ).dropna()
    if values.empty:
        raise ValueError(f"case {case_rank} has no finite values")
    low = min(0.0, float(values.min()))
    high = max(1.0, float(values.max()))
    span = max(1.0, high - low)
    return low - 0.02 * span, high + 0.04 * span


def case_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    row_order: int,
    case_rank: int,
    limits: tuple[float, float],
    *,
    title: str | None,
    show_xlabel: bool,
) -> tuple[Line2D, Line2D, Patch, Line2D, Patch]:
    rows = source[
        pd.to_numeric(source["row_order"], errors="raise").eq(row_order)
        & pd.to_numeric(source["case_rank"], errors="raise").eq(case_rank)
    ].copy()
    if rows.empty:
        raise ValueError(f"missing case rows for row={row_order}, case={case_rank}")
    rows["target_time"] = pd.to_datetime(rows["target_time"], errors="raise")
    rows = rows.sort_values(["target_time", "forecast_id"])
    x = rows["target_time"].to_numpy()
    lower = pd.to_numeric(rows["lower_90"], errors="raise")
    upper = pd.to_numeric(rows["upper_90"], errors="raise")
    comparison_lower = pd.to_numeric(rows["comparator_lower_90"], errors="raise")
    comparison_upper = pd.to_numeric(rows["comparator_upper_90"], errors="raise")
    ax.fill_between(x, lower, upper, color="#67a9cf", alpha=0.24, edgecolor="none", zorder=1)
    ax.fill_between(x, comparison_lower, comparison_upper, color="#fdb863", alpha=0.20, edgecolor="none", hatch="///", zorder=1)
    for boundary in (lower, upper):
        ax.plot(x, boundary, color=BLUE, linewidth=0.55, zorder=2)
    for boundary in (comparison_lower, comparison_upper):
        ax.plot(x, boundary, color=ORANGE, linestyle="--", linewidth=0.55, zorder=2)
    observed = ax.plot(
        x,
        pd.to_numeric(rows["observed_value"], errors="raise"),
        color="#111111",
        marker="o",
        markevery=max(1, len(rows) // 6),
        markersize=3.0,
        zorder=4,
    )[0]
    caster = ax.plot(
        x,
        pd.to_numeric(rows["predictive_mean"], errors="raise"),
        color=BLUE,
        marker="o",
        markevery=max(1, len(rows) // 6),
        markersize=3.0,
        markerfacecolor="white",
        zorder=3,
    )[0]
    comparison = ax.plot(
        x,
        pd.to_numeric(rows["comparator_pred_mean"], errors="raise"),
        color=ORANGE,
        linestyle="--",
        zorder=3,
    )[0]
    if title:
        ax.set_title(title, pad=4.0)
    ax.set_ylim(*limits)
    ax.set_ylabel("Admissions / 100k", labelpad=2.0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    if show_xlabel:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_xlabel("Target date", labelpad=1.5)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(color="#d9d9d9", linewidth=0.55, alpha=0.72)
    caster_patch = Patch(facecolor=(103 / 255, 169 / 255, 207 / 255, 0.24), edgecolor=BLUE, linewidth=0.55)
    comparison_patch = Patch(facecolor=(253 / 255, 184 / 255, 99 / 255, 0.20), edgecolor=ORANGE, linewidth=0.55, hatch="///")
    return observed, caster, caster_patch, comparison, comparison_patch


def render(
    reliability_path: Path | None,
    cases_path: Path | None,
    caster_intervals: Path | None,
    comparison_forecasts: list[Path],
    forecast_metrics: Path | None,
    comparison_bridge_root: Path | None,
    output: Path,
) -> None:
    if caster_intervals is not None:
        if forecast_metrics is None or comparison_bridge_root is None:
            raise ValueError(
                "fresh rendering requires --forecast-metrics and "
                "--comparison-bridge-root so the fixed Full Recovery cases "
                "use the formal shared-bridge readout"
            )
        reliability, cases = build_fresh_inputs(
            caster_intervals,
            comparison_forecasts,
            forecast_metrics,
            comparison_bridge_root,
        )
    else:
        if reliability_path is None or cases_path is None:
            raise ValueError("provide prepared inputs or fresh forecast inputs")
        reliability = read_csv(reliability_path, RELIABILITY_COLUMNS)
        cases = read_csv(cases_path, CASE_COLUMNS)
    lane_rows = (
        reliability[["row_order", "caster_method", "caster_method_label"]]
        .drop_duplicates()
        .sort_values("row_order")
    )
    if len(lane_rows) != 4 or list(pd.to_numeric(lane_rows["row_order"], errors="raise")) != [0, 1, 2, 3]:
        raise ValueError("the calibration input must contain four ordered lanes")

    configure_style(8.0)
    fig, axes = plt.subplots(4, 3, figsize=(6.3, 8.25), squeeze=False)
    fig.subplots_adjust(left=0.118, right=0.992, bottom=0.070, top=0.824, wspace=0.46, hspace=0.30)
    limits = {1: common_ylim(cases, 1), 2: common_ylim(cases, 2)}
    reliability_handles: dict[str, Line2D] | None = None
    trajectory_handles: tuple[Line2D, Line2D, Patch, Line2D, Patch] | None = None
    for row in lane_rows.itertuples(index=False):
        row_order = int(row.row_order)
        show_xlabel = row_order == 3
        new_reliability_handles = reliability_panel(
            axes[row_order, 0],
            reliability,
            row_order,
            title="(a) Reliability" if row_order == 0 else None,
            show_xlabel=show_xlabel,
        )
        new_trajectory_handles = case_panel(
            axes[row_order, 1],
            cases,
            row_order,
            1,
            limits[1],
            title="(b) Georgia COVID" if row_order == 0 else None,
            show_xlabel=show_xlabel,
        )
        case_panel(
            axes[row_order, 2],
            cases,
            row_order,
            2,
            limits[2],
            title="(c) West Virginia Flu" if row_order == 0 else None,
            show_xlabel=show_xlabel,
        )
        if row_order == 0:
            reliability_handles = new_reliability_handles
            trajectory_handles = new_trajectory_handles
        position = axes[row_order, 0].get_position()
        fig.text(
            0.027,
            (position.y0 + position.y1) / 2.0,
            str(row.caster_method_label).replace(" - ", "\n"),
            rotation=90,
            ha="center",
            va="center",
            multialignment="center",
            fontsize=7.7,
            fontweight="semibold",
            color="#333333",
        )
    if reliability_handles is None or trajectory_handles is None:
        raise ValueError("the first calibration lane is missing")
    observed, caster, caster_patch, comparison, comparison_patch = trajectory_handles
    legend_handles: list[object] = [
        reliability_handles["Ideal"],
        observed,
        reliability_handles["A - CASTER"],
        reliability_handles["A - Full Recovery"],
        reliability_handles["B-COVID - CASTER"],
        reliability_handles["B-COVID - Full Recovery"],
        reliability_handles["B-FLU - CASTER"],
        reliability_handles["B-FLU - Full Recovery"],
        (caster, caster_patch),
        (comparison, comparison_patch),
    ]
    fig.legend(
        legend_handles,
        [
            "Ideal",
            "Observed",
            "A - CASTER",
            "A - Full Recovery",
            "B-COVID - CASTER",
            "B-COVID - Full Recovery",
            "B-FLU - CASTER",
            "B-FLU - Full Recovery",
            "CASTER + 90% PI",
            "Agentic Full Recovery + 90% PI",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.20)},
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        frameon=False,
        ncol=5,
        handlelength=1.05,
        handletextpad=0.25,
        columnspacing=0.48,
        labelspacing=0.12,
        fontsize=7.6,
    )
    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render calibration reliability and forecast cases.")
    parser.add_argument("--reliability", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--caster-intervals", type=Path)
    parser.add_argument("--comparison-forecast", type=Path, action="append", default=[])
    parser.add_argument("--forecast-metrics", type=Path)
    parser.add_argument("--comparison-bridge-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(
        args.reliability,
        args.cases,
        args.caster_intervals,
        args.comparison_forecast,
        args.forecast_metrics,
        args.comparison_bridge_root,
        args.output,
    )
