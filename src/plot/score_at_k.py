import argparse
import logging
from pathlib import Path

import dvc.api
import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import src.utils.plots


def _plot_percentile_line(
    ax: matplotlib.axes.Axes,
    y_value: float,
    color: str,
    x_text: float,
    percentile_text: str,
    is_first: bool = False,
) -> matplotlib.lines.Line2D | None:
    line = ax.axhline(
        y=y_value,
        color=color,
        linestyle="--",
        alpha=0.8,
        label="Human 8h performance" if is_first else None,
    )

    ax.text(
        x_text,
        y_value + 0.01,
        f"{percentile_text} percentile",
        verticalalignment="bottom",
        horizontalalignment="left",
        alpha=0.8,
    )

    return line if is_first else None


def _add_human_mean_of_percentiles(
    ax: matplotlib.axes.Axes,
    human_mean_of_percentiles: pd.DataFrame,
    time_limit: int,
    plot_params: src.utils.plots.PlotParams,
) -> matplotlib.lines.Line2D | None:
    baseline_data = human_mean_of_percentiles[
        human_mean_of_percentiles["task_id"] == "average"
    ].iloc[0]

    percentiles = [
        ("p90", "90th"),
        ("p70", "70th"),
        ("p50", "50th"),
        ("p30", "30th"),
        ("p10", "10th"),
    ]

    human_color = src.utils.plots.get_agent_color(
        plot_params["colors"], "human", "base"
    )
    line = None
    x_text = 0.56 if time_limit == 7200 else 0.38

    for i, (db_key, percentile_text) in enumerate(percentiles):
        result = _plot_percentile_line(
            ax,
            baseline_data[db_key],
            human_color,
            x_text,
            percentile_text,
            is_first=(i == 0),
        )
        if result:
            line = result

    return line


def _plot_individual_agent_series(
    ax: matplotlib.axes.Axes, agent_data: pd.DataFrame, color: str, label: str
) -> matplotlib.lines.Line2D:
    line = ax.semilogx(
        agent_data["samples"],
        agent_data["point_estimate"],
        color=color,
        label=label,
        zorder=4,
    )[0]

    ax.scatter(
        agent_data["samples"],
        agent_data["point_estimate"],
        color=color,
        s=30,
        zorder=5,
    )

    ax.fill_between(
        agent_data["samples"],
        agent_data["ci_lower"],
        agent_data["ci_upper"],
        color=color,
        alpha=0.2,
    )

    return line


def _add_agent_series(
    ax: matplotlib.axes.Axes,
    data: pd.DataFrame,
    time_limit_label: str,
    plot_params: src.utils.plots.PlotParams,
) -> list[matplotlib.lines.Line2D]:
    lines = []
    for agent_id in data["agent"].unique():
        agent_data = data[data["agent"] == agent_id]

        color = src.utils.plots.get_agent_color(plot_params["colors"], agent_id, "base")
        label = f"{agent_id} {time_limit_label}"

        line = _plot_individual_agent_series(ax, agent_data, color, label)
        lines.append(line)

    return lines


def _configure_plot_axes(
    ax: matplotlib.axes.Axes,
    data: pd.DataFrame,
    handles: list[matplotlib.lines.Line2D],
    labels: list[str],
    time_limit_label: str,
    time_limit: int,
) -> None:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    max_samples = data["samples"].max()
    ax.set_xticks([2**i for i in range(0, int(max_samples).bit_length())])
    left_limit = 1 / 2**0.9 if time_limit == 7200 else 1 / 2**1.5
    ax.set_xlim(left=left_limit, right=max_samples * 1.1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Average Normalized Score")
    ax.set_xlabel("Number of samples (k)")
    ax.set_title(f"Score@k (95% CI): {time_limit_label} Time Limit", pad=10)
    ax.grid(True, which="both", axis="y", linestyle="-", alpha=0.2, zorder=0)
    ax.legend(handles=handles, labels=labels, loc="upper right", framealpha=1.0)


def plot_score_at_k(
    data: pd.DataFrame,
    human_mean_of_percentiles: pd.DataFrame,
    plot_params: src.utils.plots.PlotParams,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    time_limit = data["time_limit"].iloc[0]
    human_line = _add_human_mean_of_percentiles(
        ax, human_mean_of_percentiles, time_limit, plot_params
    )
    time_limit_label = src.utils.plots.format_time_label(time_limit)
    _add_agent_series(ax, data, time_limit_label, plot_params)

    legend_elements = []

    for agent_id in data["agent"].unique():
        color = src.utils.plots.get_agent_color(plot_params["colors"], agent_id, "base")
        label = f"{agent_id} {time_limit_label}"

        line = matplotlib.lines.Line2D([0], [0], color=color)
        legend_elements.append((line, label, agent_id))

    legend_elements.append((human_line, "Human 8h performance", "human"))

    legend_order = plot_params["legend_order"]
    legend_elements = sorted(
        legend_elements,
        key=lambda x: legend_order.index(x[2])
        if x[2] in legend_order
        else float("inf"),
    )

    handles, labels = zip(*[(elem[0], elem[1]) for elem in legend_elements])

    _configure_plot_axes(ax, data, handles, labels, time_limit_label, time_limit)

    plt.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-score-at-k",
        type=Path,
        required=True,
        help="Path to the processed data file (JSONL)",
    )
    parser.add_argument(
        "--input-human-mean-of-percentiles",
        type=Path,
        required=True,
        help="Path to the human mean of percentiles file (JSONL)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefix path for output files (will append _<time_limit>.png)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()
    params = dvc.api.params_show(stages="plot_score_at_k")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    data = pd.read_json(args.input_score_at_k, lines=True)
    logging.info(f"Loaded data from {args.input_score_at_k}")

    human_mean_of_percentiles = pd.read_json(
        args.input_human_mean_of_percentiles, lines=True
    )
    logging.info(
        f"Loaded human mean of percentiles from {args.input_human_mean_of_percentiles}"
    )

    for time_limit in sorted(data["time_limit"].unique()):
        time_limit_data = data[data["time_limit"] == time_limit]
        plot_score_at_k(time_limit_data, human_mean_of_percentiles, params["plots"])

        plot_format = params["plot_format"]
        output_path = (
            args.output_prefix.parent
            / f"{args.output_prefix.name}_{time_limit}.{plot_format}"
        )
        src.utils.plots.save_or_open_plot(output_path, plot_format)


if __name__ == "__main__":
    main()
