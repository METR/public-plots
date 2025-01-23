import argparse
import logging
import pathlib
from datetime import date
from typing import List

import dvc.api
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num
from matplotlib.figure import Figure

import src.utils.plots
from src.plot.logistic import plot_trendline

logger = logging.getLogger(__name__)


def plot_bootstrap_ci(
    fig: Figure,
    ax: Axes,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, dict[str, date]],
    focus_agents: List[str],
    weighting: str,
    categories: str,
    regularization: str,
    n_samples: int = 10,
    after_date: str = "2023-03-01",
) -> List[float]:
    """Plot bootstrap confidence intervals with sampled lines.

    Args:
        fig: matplotlib figure
        ax: matplotlib axes
        bootstrap_results: DataFrame with columns for each agent containing p50s
        release_dates: Dictionary mapping agent names to release dates
        focus_agents: List of agents to plot
        n_samples: Number of sample lines to plot

    Returns:
        List of doubling times from the trendlines
    """
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Convert release dates to timestamps
    dates = release_dates["date"]

    # Sort agents by release date
    focus_agents = sorted(focus_agents, key=lambda x: dates[x])

    # First plot the confidence intervals and medians
    for i, agent in enumerate(focus_agents):
        logging.info(agent)
        if agent not in bootstrap_results.columns:
            logger.warning(f"Agent {agent} not found in bootstrap results")
            continue

        color = colors[i % len(colors)]
        p50s = pd.to_numeric(bootstrap_results[agent], errors="coerce")
        p50s = p50s[~p50s.isna()]

        if len(p50s) == 0:
            logger.warning(f"No valid p50s for {agent}")
            continue

        # Calculate confidence intervals
        ci_low, ci_high = np.percentile(p50s, [10, 90])
        median = np.median(p50s)

        # Plot confidence interval as filled region
        x = np.array([dates[agent]])
        ax.fill_between(x, [ci_low], [ci_high], alpha=0.2, color=color)

        # Plot median point
        ax.scatter([x], [median], color=color, label=agent, zorder=10)

    min_date = pd.to_datetime(min(dates[agent] for agent in focus_agents))
    max_date = pd.to_datetime(max(dates[agent] for agent in focus_agents))
    ax.set_xlim(
        float(date2num(min_date - pd.Timedelta(days=50))),
        float(date2num(max_date + pd.Timedelta(days=365))),
    )

    doubling_times = []
    # Now plot trendlines for each bootstrap sample
    n_bootstraps = len(bootstrap_results)
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_bootstraps, size=n_samples, replace=False)

    for sample_idx in sample_indices:
        # Create a DataFrame for this bootstrap sample
        sample_data = []
        for agent in focus_agents:
            if agent not in bootstrap_results.columns:
                continue
            p50 = pd.to_numeric(
                bootstrap_results[agent].iloc[sample_idx], errors="coerce"
            )
            if pd.isna(p50) or np.isinf(p50) or p50 < 1e-3:
                continue
            sample_data.append(
                {
                    "agent": agent,
                    "release_date": dates[agent],
                    "50%": p50,
                    "50_low": p50,  # TODO fix this if using WLS
                    "50_high": p50,  # Not used by plot_trendline
                }
            )

        sample_df = pd.DataFrame(sample_data)
        logging.debug(sample_df)
        # Plot trendline for this sample with low alpha
        doubling_time = plot_trendline(
            ax,
            sample_df,
            after=after_date,
            log_scale=True,
            annotate=False,
            method="OLS",
            fit_color="gray",
            plot_kwargs={"alpha": 0.25},
        )
        if doubling_time is not None:
            doubling_times.append(doubling_time)

    p10, p50, p90 = np.percentile(doubling_times, [10, 50, 90])
    ax.text(
        0.95,
        0.95,
        f"{after_date}+ data:\nDoubling time 80% CI:\n {p10:.0f} to {p90:.0f} days (median {p50:.0f})\nPoint CIs are 80%",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
    )

    # Format axes
    ax.set_yscale("log")
    ax.set_yticks([5 / 60, 1, 2, 4, 8, 15, 30, 60, 120, 240, 20 * 60])
    ax.set_yticklabels(
        [
            "5 sec",
            "1 min",
            "2 min",
            "4 min",
            "8 min",
            "15 min",
            "30 min",
            "1 hr",
            "2 hrs",
            "4 hrs",
            "20 hrs",
        ]
    )
    ax.set_xlabel("Model release date")
    ax.set_ylabel("Human time-to-complete @ 50% chance of success")

    # Set reasonable y bounds
    ax.set_ylim(5 / 60, 40 * 60)  # 5 seconds to 40 hours

    ax.legend(loc="upper left")

    category_names = {
        "b": "time buckets",
        "f": "families",
        "t": "tasks",
        "r": "runs",
    }
    category_str = ", ".join(category_names[c] for c in categories)
    ax.set_title(
        f"Bootstrapping results across {category_str} ({n_samples} bootstraps):\n {weighting} reg={regularization}"
    )

    return doubling_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--weighting", type=str, required=True)
    parser.add_argument("--categories", type=str, required=True)
    parser.add_argument("--regularization", type=str, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--n-samples", type=int, default=10, help="Number of sample lines to plot"
    )
    parser.add_argument(
        "--after-date", type=str, default="2023-03-01", help="Start date for trendline"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show(stages=["plot_bar_chart"])

    # Load data
    bootstrap_results = pd.read_csv(args.input_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())
    logging.info(f"Loaded {len(bootstrap_results)} bootstrap results")
    print(bootstrap_results)

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

    # Create plot with two subplots
    fig, axs = plt.subplots(1, 2, width_ratios=[6, 1], figsize=(12, 6))
    doubling_times = plot_bootstrap_ci(
        fig=fig,
        ax=axs[0],
        bootstrap_results=bootstrap_results,
        release_dates=release_dates,
        focus_agents=focus_agents,
        categories=args.categories,
        weighting=args.weighting,
        regularization=args.regularization,
        n_samples=args.n_samples,
        after_date=args.after_date,
    )

    # Plot doubling times boxplot
    axs[1].boxplot([doubling_times], vert=True)
    axs[1].set_xticklabels(["Doubling times\n(days)"])
    axs[1].set_ylim(0, None)

    # Save plot
    src.utils.plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
