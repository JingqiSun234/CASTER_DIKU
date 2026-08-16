from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


TASK_ORDER = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
TASK_LABEL = {
    "benchmark_a": "Benchmark A",
    "benchmark_b_covid": "B-COVID",
    "benchmark_b_flu": "B-FLU",
}
METHOD_ORDER = ("caster_one_layer", "caster_hierarchical")
METHOD_LABEL = {
    "caster_one_layer": "One-layer CASTER",
    "caster_hierarchical": "Hierarchical CASTER",
}

FAMILY_STYLE = {
    "neural": {"label": "Neural", "color": "#0F5CAA", "linestyle": "-", "marker": "o"},
    "foundation_ts": {"label": "Foundation TS", "color": "#883B1D", "linestyle": "--", "marker": "s"},
    "statistical": {"label": "Statistical", "color": "#08754B", "linestyle": "-.", "marker": "^"},
    "state_space": {"label": "State space", "color": "#A80276", "linestyle": ":", "marker": "D"},
    "compartmental": {"label": "Compartmental", "color": "#786005", "linestyle": (0, (8, 2)), "marker": "v"},
    "renewal": {"label": "Renewal", "color": "#064B54", "linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "P"},
}

EVENT_STYLE = {
    "declared_wave_onset": {"label": "Declared wave", "color": "#54A24B"},
    "variant_turnover": {"label": "Variant turnover", "color": "#F58518"},
    "winter_onset": {"label": "Winter onset", "color": "#4C78A8"},
    "large_revision_spike": {"label": "Large revision spike", "color": "#B279A2"},
}


def configure_style(font_size: float = 9.0) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.titlesize": font_size + 1.0,
            "axes.titleweight": "bold",
            "axes.labelsize": font_size,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": max(7.5, font_size - 0.3),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "svg.fonttype": "none",
        }
    )


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{path} has no rows")
    return frame


def require_three_tasks(frame: pd.DataFrame, column: str = "dataset") -> None:
    observed = set(frame[column].astype(str))
    expected = set(TASK_ORDER)
    if observed != expected:
        raise ValueError(
            f"task coverage differs: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def save_figure(fig: plt.Figure, output: Path, *, dpi: int = 360) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

