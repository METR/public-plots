import argparse
import logging

import dvc.api
import matplotlib.lines
import matplotlib.pyplot as plt
import pandas as pd

import src.utils.plots


def _calculate_bar_positions(
    agent_data: pd.DataFrame,
    time_limits: list[int],
    bar_width: float = 0.2,
    bar_spacing: float = 0,
    group_padding: float = 0.15,
) -> tuple[list[int], list[float], list[float]]:
    group_sizes = [len(agent_data[agent_data["time_limit"] == t]) for t in time_limits]
    group_widths = [
        size * (bar_width + bar_spacing) + group_padding for size in group_sizes
    ]
    group_starts = [sum(group_widths[:i]) for i in range(len(time_limits))]

    return group_sizes, group_widths, group_starts


def plot_bar_by_time_allocation(
    plot_params: src.utils.plots.PlotParams,
    data: pd.DataFrame,
    output_path: str,
    plot_format: str = "png",
) -> None:
    fig, ax = plt.subplots()

    # Filter out human data
    agent_data = data[~data["agent"].str.contains("human", case=False)]

    # Create sort key based on legend order
    agent_data["sort_key"] = agent_data["agent"].apply(
        lambda x: plot_params["legend_order"].index(x)
        if x in plot_params["legend_order"]
        else float("inf")
    )

    logging.info(agent_data)

    time_limits = sorted(agent_data["time_limit"].unique())
    bar_width = 0.2
    bar_spacing = 0
    group_padding = 0.15
    group_sizes, group_widths, group_starts = _calculate_bar_positions(
        agent_data,
        time_limits,
        bar_width=bar_width,
        bar_spacing=bar_spacing,
        group_padding=group_padding,
    )

    for i, time_limit in enumerate(time_limits):
        time_data = agent_data[agent_data["time_limit"] == time_limit]
        time_data = time_data.sort_values("sort_key")
        print(time_data)

        for j, (_, row) in enumerate(time_data.iterrows()):
            x = group_starts[i] + j * (bar_width + bar_spacing)
            y_err = (
                [row["point_estimate"] - row["ci_lower"]],
                [row["ci_upper"] - row["point_estimate"]],
            )
            if not all(e >= 0 for es in y_err for e in es):
                logging.warning(f"Negative error bars: {y_err} at {row['agent']}")
                y_err = ([max(0, e) for e in y_err[0]], [max(0, e) for e in y_err[1]])

            color = src.utils.plots.get_agent_color(
                plot_params["colors"], row["agent"], "base"
            )
            ax.bar(
                x,
                row["point_estimate"],
                width=bar_width,
                yerr=y_err,
                error_kw={
                    "capsize": 4,
                    "capthick": 0.9,
                    "elinewidth": 0.9,
                    "zorder": 3,
                },
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=row["agent"],
                zorder=2,
            )

    ax.grid(True, which="both", axis="y", linestyle="-", alpha=0.2, zorder=0)
    ax.set_ylabel("Average Normalized Score")
    ax.set_xlabel("Time Allocation")
    ax.set_ylim(bottom=0, top=1.25)

    group_centers = [
        start + ((size - 1) * (bar_width + bar_spacing)) / 2
        for start, size in zip(group_starts, group_sizes)
    ]

    ax.set_xticks(group_centers)
    ax.set_xlim(left=-0.75, right=group_starts[-1] + group_widths[-1] - group_padding)

    time_labels = [
        agent_data[agent_data["time_limit"] == t].iloc[0]["time_label"]
        for t in time_limits
    ]
    ax.set_xticklabels(time_labels, ha="center")
    ax.set_title(
        "Score@k by Time Allocation (95% CI)",
        pad=10,
    )

    # Simplify legend creation
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Add human reference line if needed
    human_color = src.utils.plots.get_agent_color(
        plot_params["colors"], "human", "dark"
    )
    unique_handles.append(
        matplotlib.lines.Line2D([0], [0], color=human_color, linestyle="--", alpha=0.8)
    )
    unique_labels.append("Human 8-hour score")

    # Sort legend elements
    legend_order = plot_params["legend_order"]
    legend_elements = sorted(
        zip(unique_handles, unique_labels),
        key=lambda x: legend_order.index(x[1])
        if x[1] in legend_order
        else float("inf"),
    )
    handles, labels = zip(*legend_elements)

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        framealpha=1.0,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format=plot_format, bbox_inches="tight")
        logging.info(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean scores by width over time.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the processed data file (JSONL).",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the plot."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--plot-format",
        default="png",
        choices=["png", "svg", "jpg"],
        help="Format to save plots in",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    logging.info(f"Loading data from {args.input}")
    data = pd.read_json(args.input, lines=True)

    logging.info(f"Plotting to {args.output}")
    params = dvc.api.params_show(stages="plot_bar_by_time_allocation")
    plot_bar_by_time_allocation(params["plots"], data, args.output, args.plot_format)


if __name__ == "__main__":
    main()
