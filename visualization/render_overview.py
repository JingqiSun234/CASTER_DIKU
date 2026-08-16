#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from _style import configure_style, save_figure


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    edge: str,
    face: str = "#FFFFFF",
    radius: float = 0.012,
    linewidth: float = 0.85,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.006,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=face,
        )
    )


def render(output: Path) -> None:
    configure_style(9.0)
    fig, ax = plt.subplots(figsize=(7.0, 3.95))
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.025, top=0.99)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    panels = [
        (0.008, 0.10, 0.190, 0.86, "task +\nevidence", "#0F5CAA"),
        (0.225, 0.10, 0.205, 0.86, "candidate\nmodels", "#4D4D4D"),
        (0.458, 0.10, 0.315, 0.86, "continual\nfiltering", "#142E73"),
        (0.805, 0.10, 0.187, 0.86, "outputs", "#542080"),
    ]
    for x, y, width, height, heading, edge in panels:
        rounded_box(ax, x, y, width, height, edge=edge, radius=0.014, linewidth=1.0)
        rounded_box(
            ax,
            x + 0.012,
            y + height - 0.080,
            width - 0.024,
            0.070,
            edge="#D8E3EF",
            face="#E8F1F8",
            radius=0.010,
            linewidth=0.65,
        )
        ax.text(
            x + width / 2,
            y + height - 0.055,
            heading,
            ha="center",
            va="center",
            fontsize=9.6,
            weight="bold",
            color="#07184D",
        )
    for left, right in zip(panels[:-1], panels[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left[0] + left[2] + 0.005, 0.54),
                (right[0] - 0.005, 0.54),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
                color="#2F2F2F",
            )
        )

    x, y, width, _height, _heading, edge = panels[0]
    rounded_box(ax, x + 0.014, y + 0.545, width - 0.028, 0.205, edge=edge, face="#F7FAFD")
    ax.text(x + width / 2, y + 0.715, "forecast task", ha="center", va="center", weight="bold")
    task_x = np.linspace(x + 0.038, x + width - 0.032, 60)
    task_y = y + 0.585 + 0.075 * np.exp(-((task_x - (x + width * 0.54)) / 0.035) ** 2)
    ax.plot(task_x, task_y, color="#0F5CAA", linewidth=1.4)
    ax.plot(task_x[-16:], task_y[-16:], color="#0F5CAA", linewidth=1.4, linestyle="--")
    ax.text(x + width / 2, y + 0.565, "time", ha="center", va="center")
    rounded_box(ax, x + 0.014, y + 0.080, width - 0.028, 0.435, edge=edge)
    ax.text(x + width / 2, y + 0.475, "released targets /\ncovariates", ha="center", va="center", weight="bold")
    for label, color, linestyle, offset in (
        ("targets", "#0F5CAA", "-", 0.405),
        ("mobility", "#08754B", "--", 0.290),
        ("wastewater", "#542080", "-.", 0.175),
    ):
        baseline = y + offset
        ax.text(x + 0.026, baseline, label, ha="left", va="center")
        stream_x = np.linspace(x + 0.105, x + width - 0.020, 15)
        stream_y = baseline + 0.018 * np.sin(np.linspace(0, 3.2 * np.pi, len(stream_x)))
        ax.plot(stream_x, stream_y, color=color, linestyle=linestyle, linewidth=1.15)

    x, y, width, _height, _heading, _edge = panels[1]
    for bottom, label, color in (
        (y + 0.545, "mechanistic", "#0F5CAA"),
        (y + 0.315, "time-series", "#08754B"),
        (y + 0.085, "covariate-aware", "#542080"),
    ):
        rounded_box(ax, x + 0.014, bottom, width - 0.028, 0.195, edge=color)
        ax.text(x + 0.025, bottom + 0.150, label, ha="left", va="center", weight="bold", color=color)
    beta_x = np.linspace(x + 0.045, x + width - 0.025, 40)
    beta_y = y + 0.350 + 0.030 * np.sin(np.linspace(0, 2.8 * np.pi, len(beta_x)))
    ax.plot(beta_x, beta_y, color="#08754B", linewidth=1.3)
    nodes = [(x + 0.050, y + 0.130), (x + 0.090, y + 0.170), (x + 0.128, y + 0.125), (x + 0.162, y + 0.170)]
    for first, second in zip(nodes[:-1], nodes[1:]):
        ax.plot([first[0], second[0]], [first[1], second[1]], color="#555555", linewidth=0.7)
    for node_x, node_y in nodes:
        ax.plot(node_x, node_y, marker="o", markersize=5.2, markerfacecolor="#8A4DB3", markeredgecolor="#2F1747")

    x, y, width, _height, _heading, _edge = panels[2]
    ax.text(x + width / 2, y + 0.745, "shared Student-t bridge", ha="center", va="center", style="italic", color="#142E73")
    rounded_box(ax, x + 0.016, y + 0.485, width - 0.032, 0.280, edge="#0F5CAA")
    ax.text(x + width / 2, y + 0.680, "distribution over\ncandidate models", ha="center", va="center", weight="bold")
    model_colors = ("#0F5CAA", "#08754B", "#542080", "#C04A00")
    masses = (0.74, 0.39, 0.61, 0.30)
    centers = np.linspace(x + 0.075, x + width - 0.055, 4)
    for index, (center, color, mass) in enumerate(zip(centers, model_colors, masses), start=1):
        ax.add_patch(plt.Rectangle((center - 0.012, y + 0.555), 0.024, 0.13 * mass, facecolor=color, edgecolor="#222222", linewidth=0.65))
        ax.text(center, y + 0.525, f"M{index}", ha="center", va="center")
    rounded_box(ax, x + 0.016, y + 0.105, width - 0.032, 0.310, edge="#0F5CAA")
    ax.text(x + width / 2, y + 0.380, "predictive components", ha="center", va="center", weight="bold")
    for center, color, linestyle in zip(centers, model_colors, ("-", "--", "-.", ":")):
        density_x = np.linspace(center - 0.024, center + 0.024, 50)
        density_y = y + 0.155 + 0.115 * np.exp(-((density_x - center) / 0.012) ** 2)
        ax.plot(density_x, density_y, color=color, linewidth=1.25, linestyle=linestyle)

    x, y, width, _height, _heading, edge = panels[3]
    ax.text(x + 0.020, y + 0.735, "(1) forecast +\nuncertainty", ha="left", va="center", weight="bold", color=edge)
    forecast_x = np.linspace(x + 0.035, x + width - 0.020, 60)
    split = 30
    history = y + 0.600 + 0.025 * np.sin(np.linspace(0, 2.8 * np.pi, split))
    future = history[-1] + 0.020 * np.sin(np.linspace(0, 2.5 * np.pi, len(forecast_x) - split))
    spread = np.linspace(0.020, 0.065, len(future))
    ax.fill_between(forecast_x[split:], future - spread, future + spread, color="#E7DDF2")
    ax.plot(forecast_x[:split], history, color="#222222", linewidth=1.25)
    ax.plot(forecast_x[split:], future, color="#542080", linewidth=1.4)
    ax.text(x + 0.020, y + 0.405, "(2) model / family\nposterior", ha="left", va="center", weight="bold", color=edge)
    for index, (center, color, mass) in enumerate(zip(np.linspace(x + 0.045, x + width - 0.035, 4), model_colors, masses), start=1):
        ax.add_patch(plt.Rectangle((center - 0.011, y + 0.170), 0.022, 0.13 * mass, facecolor=color, edgecolor="#222222", linewidth=0.65))
        ax.text(center, y + 0.145, f"M{index}", ha="center", va="center")

    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the CASTER method overview.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args().output)
