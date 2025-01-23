import argparse
import datetime
import itertools
import logging
import pathlib
from typing import Any, Sequence

import dvc.api
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import yaml
from adjustText import adjust_text
from matplotlib import pyplot as plt

import src.utils.plots
from src.plot.logistic import plot_trendline


def plot_points_and_many_lines(
    plot_params: src.utils.plots.PlotParams,
    ax: matplotlib.axes.Axes,
    agent_summaries: pd.DataFrame,
    focus_agents: Sequence[str],
    weighting: str,
    regularization: str,
) -> list[dict[str, Any]]:
    # exclude agents with nan 50%
    logging.info(
        f"Excluded {len(agent_summaries) - len(agent_summaries[agent_summaries['50%'].notna()])} agents with nan 50%"
    )
    agent_summaries = agent_summaries[agent_summaries["50%"].notna()]
    agent_summaries = agent_summaries[agent_summaries["agent"].isin(focus_agents)]
    logging.debug(agent_summaries)

    agent_summaries.loc[:, "50%_clipped"] = agent_summaries["50%"].clip(
        # np.finfo(float).eps, np.inf
        0.5,
        np.inf,
    )  # clip because log scale makes 0 -> -inf
    agent_summaries.loc[:, "50_low"].clip(1, np.inf)
    agent_summaries.loc[:, "50_high"].clip(1, np.inf)
    y = agent_summaries["50%"]
    yerr = np.array(
        [y - agent_summaries.loc[:, "50_low"], agent_summaries.loc[:, "50_high"] - y]
    )
    yerr = np.clip(yerr, 0, np.inf)
    ax.errorbar(
        agent_summaries["release_date"],
        agent_summaries["50%_clipped"],
        yerr=yerr,
        fmt="o",
        markersize=0,  # invisible
        capsize=5,
        ecolor="gray",
    )

    logging.info(agent_summaries["50%"])
    ax.scatter(
        agent_summaries["release_date"],
        agent_summaries["50%_clipped"],
        marker="o",
        edgecolor=[
            (
                src.utils.plots.get_agent_color(
                    plot_params["colors"], agent_summaries["agent"].iloc[i]
                )
                if agent_summaries["50%"].iloc[i] > 1
                else "black"
            )
            for i in range(len(agent_summaries))
        ],
        facecolor=[
            (
                src.utils.plots.get_agent_color(
                    plot_params["colors"], agent_summaries["agent"].iloc[i]
                )
                if agent_summaries["50%"].iloc[i] > 1
                else "white"
            )
            for i in range(len(agent_summaries))
        ],
    )

    # Add arrows for out-of-range points
    mask_out_range = agent_summaries["50%_clipped"] != agent_summaries["50%"]
    ax.scatter(
        agent_summaries.loc[mask_out_range, "release_date"],
        [0.4] * mask_out_range.sum(),  # Place at bottom of visible range
        marker="v",  # Downward pointing triangle
        color="black",
    )

    texts = []

    records = []
    for method in ["OLS", "WLS"]:
        dates_colors = [
            # ("2022-06-01", "blue"),
            ("2023-01-01", "blue"),
            ("2024-01-01", "red"),
        ]
        for after_date, color in dates_colors:
            records.append(
                {
                    "weighting": weighting,
                    "regularization": regularization,
                    "after_date": after_date,
                    "method": method,
                    "doubling_time": plot_trendline(
                        ax,
                        agent_summaries,
                        after=after_date,
                        log_scale=True,
                        annotate=False,
                        fit_color=color,
                        method=method,
                    ),
                }
            )

    # Adjust text positions to avoid overlaps
    adjust_text(
        texts,
        max_move=7,
    )

    ax.set_yscale("log")
    ax.set_yticks([1 / 60, 1, 4, 15, 60, 60 * 4, 60 * 40])
    ax.set_yticklabels(
        [
            "1 sec",
            "1 min",
            "4 min",
            "15 min",
            "1 hr",
            "4 hrs",
            "40 hrs",
        ]
    )
    ax.set_ylim(1 / 60, 60 * 40)
    ax.set_xlabel("Model release date")
    ax.set_xticks(
        [
            min(agent_summaries["release_date"]) + pd.Timedelta(days=365 * n)
            for n in range(0, 6)
        ]
    )
    ax.set_xbound(
        datetime.date(2022, 11, 11),  # type: ignore[reportArgumentType]
        max(agent_summaries["release_date"]) + pd.Timedelta(weeks=52),
    )

    ax.set_ylabel("Human time-to-complete @ 50% chance of success")
    return records


def record_metrics(records: list[dict[str, Any]], metrics_file: pathlib.Path) -> None:
    df = pd.DataFrame(records)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "overall": {
            "10%": float(df["doubling_time"].quantile(0.1)),
            "50%": float(df["doubling_time"].quantile(0.5)),
            "90%": float(df["doubling_time"].quantile(0.9)),
        },
        "marginal_medians": {
            "weighting": {
                weight: float(df[df["weighting"] == weight]["doubling_time"].median())
                for weight in df["weighting"].unique()
            },
            "regularization": {
                str(reg): float(
                    df[df["regularization"] == reg]["doubling_time"].median()
                )
                for reg in df["regularization"].unique()
            },
            "after_date": {
                date: float(df[df["after_date"] == date]["doubling_time"].median())
                for date in df["after_date"].unique()
            },
            "method": {
                method: float(df[df["method"] == method]["doubling_time"].median())
                for method in df["method"].unique()
            },
        },
    }
    metrics_file.write_text(yaml.dump(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-prefix", type=str, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--weightings", type=str, required=True)
    parser.add_argument("--regularizations", type=str, required=True)
    parser.add_argument("--metrics-file", type=pathlib.Path, required=True)
    parser.add_argument("--categories", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show(stages="plot_logistic_multiverse")

    weightings = args.weightings.split(",")
    regularizations = args.regularizations.split(",")
    fig, axs = plt.subplots(1, 2, width_ratios=[6, 1], figsize=(10, 7))
    ax0: matplotlib.axes.Axes = axs[0]
    ax1: matplotlib.axes.Axes = axs[1]
    records = []

    for weighting, regularization in itertools.product(weightings, regularizations):
        agent_summaries = pd.read_csv(
            args.input_file_prefix
            + f"{weighting}_{regularization}_{args.categories}.csv"
        )
        logging.info(agent_summaries)
        release_dates = yaml.safe_load(args.release_dates.read_text())
        agent_summaries["release_date"] = agent_summaries["agent"].map(
            release_dates["date"]
        )

        focus_agents = [
            "Claude 3 Opus",
            "Claude 3.5 Sonnet (New)",
            "Claude 3.5 Sonnet (Old)",
            "GPT-4 0314",
            "GPT-4 Turbo",
            "GPT-4o",
            "davinci-002",
            "gpt-3.5-turbo-instruct",
            "o1",
            "o1-preview",
        ]

        logging.info("Loaded input data")

        records.extend(
            plot_points_and_many_lines(
                params["plots"],
                ax0,
                agent_summaries,
                focus_agents,
                weighting,
                regularization,
            )
        )
    doubling_times = [record["doubling_time"] for record in records]
    ax1.boxplot(doubling_times, vert=True)
    fig.suptitle(
        f"Multiverse analysis of time horizon, {len(records)} fits\nParameters: weighting, WLS vs OLS, date range, regularization"
    )
    ax1.set_xticklabels(["Doubling times\n(days)"])

    # Annotate top left with range
    ax0.annotate(
        f"Range: {min(doubling_times):.0f} - {max(doubling_times):.0f} days\nRed lines are 2024 data only",
        (0.05, 0.9),
        xycoords="axes fraction",
        fontsize=10,
    )

    record_metrics(records, args.metrics_file)
    logging.info(f"Saved metrics to {args.metrics_file}")

    src.utils.plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
