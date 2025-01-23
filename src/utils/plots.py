import logging
import pathlib
from typing import TypedDict

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


class AgentColors(TypedDict):
    base: str
    dark: str
    light: str


class PlotColorsParams(TypedDict):
    agent_aliases: dict[str, AgentColors]
    default: str


class PlotParams(TypedDict):
    colors: PlotColorsParams
    legend_order: list[str]


def format_time_label(seconds: float) -> str:
    hours = seconds / 3600
    if hours >= 24 * 8:
        return f"{int(hours / 24)}d"
    if hours >= 1:
        return f"{int(hours)} hr" + ("s" if int(hours) > 1 else "")
    if hours >= 1 / 60:
        return f"{int(hours * 60)} min"
    return f"{int(seconds)} sec"


possible_ticks = np.array(
    [
        1.875 / 60,
        3.75 / 60,
        7.5 / 60,
        15 / 60,
        30 / 60,
        1,
        2,
        4,
        8,
        15,
        30,
        60,
        120,
        240,
        480,
        960,
        40 * 60,
    ]
)


def log_x_axis(ax: matplotlib.axes.Axes, low_limit_seconds: int | None = None) -> None:
    ax.set_xscale("log")
    x_min, x_max = ax.get_xlim()

    if low_limit_seconds is not None:
        x_min = max(x_min, low_limit_seconds / 3600)
        ax.set_xlim(left=x_min)

    xticks = possible_ticks[(possible_ticks >= x_min) & (possible_ticks <= x_max)]
    labels = [format_time_label(tick * 3600) for tick in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks))  # type: ignore[reportArgumentType]
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def log_y_axis(ax: matplotlib.axes.Axes, unit: str = "minutes") -> None:
    ax.set_yscale("log")
    y_min, y_max = ax.get_ylim()
    multiplier = 60 if unit == "minutes" else 3600
    yticks = possible_ticks[(possible_ticks >= y_min) & (possible_ticks <= y_max)]
    labels = [format_time_label(tick * multiplier) for tick in yticks]

    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    ax.yaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([float(x) for x in yticks])
    )
    # ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def get_agent_color(colors: PlotColorsParams, agent: str, variant: str = "base") -> str:
    """Get color for agent, falling back to default if not found."""
    if "human" in agent.lower():
        agent = "human"

    return colors["agent_aliases"].get(agent, {}).get(variant, colors["default"])


def create_sorted_legend(ax: matplotlib.axes.Axes, legend_order: list[str]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = sorted(
        zip(handles, labels),
        key=lambda x: legend_order.index(x[1])
        if x[1] in legend_order
        else float("inf"),
    )
    handles, labels = zip(*legend_elements)

    ax.legend(handles=handles, labels=labels)


def save_or_open_plot(
    output_file: pathlib.Path | None = None, plot_format: str = "png"
) -> None:
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, format=plot_format, bbox_inches="tight")
        logging.info(f"Plot saved to {output_file}")
    else:
        plt.show()
    plt.close()
