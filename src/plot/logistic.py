from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any, Sequence, cast

import dvc.api
import matplotlib.axes
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num, num2date
from numpy.typing import NDArray
from scipy.optimize import Bounds, curve_fit
from sklearn.linear_model import LinearRegression

import src.utils.plots


def plot_task_distribution(
    ax: matplotlib.axes.Axes,
    runs_df: pd.DataFrame,
    weight_key: str | None = None,
) -> None:
    """Plot a vertical histogram of the human time estimates for each run."""

    # Because we're plotting the distribution of tasks, equal_task_weight is equivalent to no weight
    use_weighting = weight_key == "invsqrt_task_weight"

    data = runs_df.groupby("task_id")["human_minutes"].first().to_numpy()

    # Make sure we use the same size bins regardless of the range we're plotting
    log_bins = np.arange(np.log10(0.5), np.log10(data.max()), 0.2)
    bins = list(10**log_bins)

    if use_weighting:
        weights = runs_df.groupby("task_id")[weight_key].sum().to_numpy()
        ax.hist(data, bins=bins, weights=weights, orientation="horizontal")
        ax.set_xlabel("Number of tasks\n(Weighted)")
        ax.set_xticks([])  # With weighting, absolute number of runs isn't meaningful
    else:
        ax.hist(data, bins=bins, orientation="horizontal")
        ax.set_xlabel("Number of tasks")

    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Human time-to-complete")
    ax.set_yscale("log")
    ax.set_yticks([])  # y ticks will be shown in main plot

    ax.set_title("Task distribution", pad=10)


def plot_horizon_graph(
    plot_params: src.utils.plots.PlotParams,
    agent_summaries: pd.DataFrame,
    runs_df: pd.DataFrame | None,
    output_file: pathlib.Path,
    subtitle: str,
    focus_agents: Sequence[str],
    trendlines: bool = True,
    after_date: str = "2022-06-01",
    include_task_distribution: str = "none",
    weight_key: str | None = None,
) -> None:
    agent_summaries = agent_summaries[
        agent_summaries["agent"].isin(focus_agents)
    ].copy()
    if include_task_distribution != "none":
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 5, wspace=0.5)
        ax = fig.add_subplot(gs[0, :4])
        ax_hist = fig.add_subplot(gs[0, 4])
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_hist = None

    agent_summaries["50%_clipped"] = agent_summaries["50%"].clip(
        # np.finfo(float).eps, np.inf
        0.5,
        np.inf,
    )  # clip because log scale makes 0 -> -inf

    agent_summaries.loc[:, "50_low"].clip(1 / 60, np.inf)
    agent_summaries.loc[:, "50_high"].clip(1 / 60, np.inf)
    y = agent_summaries["50%"]
    yerr = np.array(
        [y - agent_summaries.loc[:, "50_low"], agent_summaries.loc[:, "50_high"] - y]
    )
    yerr = np.clip(yerr, 0, np.inf)

    for i, agent in enumerate(agent_summaries["agent"]):
        agent_color = src.utils.plots.get_agent_color(plot_params["colors"], agent)
        ax.errorbar(
            agent_summaries["release_date"].iloc[i],
            agent_summaries["50%_clipped"].iloc[i],
            yerr=[[yerr[0, i]], [yerr[1, i]]],
            fmt="o",
            capsize=5,
            color=agent_color if y.iloc[i] > 0.5 else "black",
            markerfacecolor=agent_color if y.iloc[i] > 0.5 else "white",
            ecolor="gray",
            label=agent,
        )

    # Add arrows for out-of-range points
    mask_out_range = agent_summaries["50%_clipped"] != y
    logging.info(f"masking out {mask_out_range.sum()} points")
    ax.scatter(
        agent_summaries.loc[mask_out_range, "release_date"],
        [0.4] * mask_out_range.sum(),  # Place at bottom of visible range
        marker="v",  # Downward pointing triangle
        color="black",
    )

    if after_date == "2024-01-01":
        ax.set_xlim(
            float(mdates.date2num(pd.Timestamp("2023-01-01"))),
            float(mdates.date2num(pd.Timestamp("2025-03-01"))),
        )

    annotations = []
    if trendlines:
        annotations.append(
            plot_trendline(ax, agent_summaries, after=after_date, log_scale=True)
        )
        if after_date != "2024-01-01":
            annotations.append(
                plot_trendline(ax, agent_summaries, after=after_date, log_scale=False)
            )
            annotations.append(
                plot_trendline_hyperbolic(ax, agent_summaries, after=after_date)
            )

    src.utils.plots.log_y_axis(ax)
    ax.set_ylim(0.25, 8 * 60)
    ax.set_xlabel("Model release date")
    xticks = np.array(
        [pd.Timestamp(num2date(x)) for x in np.linspace(*ax.get_xlim(), 6)]
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(map(lambda x: x.strftime("%Y-%m-%d"), xticks))
    if after_date == "2024-01-01":
        ax.set_xbound(
            pd.Timestamp("2023-01-01"),  # type: ignore[reportArgumentType]
            pd.Timestamp("2025-03-01"),  # type: ignore[reportArgumentType]
        )
    ax.set_yscale("log")
    yticks = [1, 2, 4, 8, 15, 30, 60, 120]
    ylabels = ["1 min", "2 min", "4 min", "8 min", "15 min", "30 min", "1 hr", "2 hrs"]

    ax.set_xlabel("Release Date")
    # ax.set_xticks(
    #     [
    #         pd.Timestamp(after_date) + pd.Timedelta(weeks=- 52 + 13*n)
    #         for n in range(0, 5)
    #     ]
    # )

    ax.set_ylabel("Human time-to-complete @ 50% chance of AI success")
    ax.set_title(f"Time horizon on well-defined tasks\n{subtitle}")

    # The graph is too busy if we have both trendlines and legend
    if not trendlines:
        src.utils.plots.create_sorted_legend(ax, plot_params["legend_order"])

    if (
        include_task_distribution != "none"
        and runs_df is not None
        and ax_hist is not None
    ):
        plot_task_distribution(ax_hist, runs_df, weight_key)

    if include_task_distribution == "full" and ax_hist is not None:
        # y limits are determined by the histogram
        low, high = ax_hist.get_ylim()
        ax.set_ylim(low, high)

        # Add ticks for the rest of the distribution
        hours = 4
        while hours * 60 < high:
            yticks.append(hours * 60)
            ylabels.append(f"{hours} hrs")
            hours *= 2
    elif include_task_distribution == "clipped" and ax_hist is not None:
        # y limits are determined by the main plot
        ax_hist.set_ylim(ax.get_ylim())

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    # Lay the annotations we collected earlier, ensuring they don't overlap
    padding = 10
    line_height = 12
    bbox = ax.get_window_extent()
    y = (
        (bbox.y1 - bbox.y0) * 72 / fig.dpi
    )  # start at top-left corner of the plot, in axes points
    y -= padding
    for a in annotations:
        ax.annotate(xy=(padding, y), xycoords="axes points", ha="left", va="top", **a)
        n_lines = len(a["text"].split("\n"))
        # next annotation will go below this one
        y -= line_height * n_lines + padding


def fit_trendline(
    agent_summaries: pd.DataFrame,
    after: str,
    log_scale: bool = False,
    method: str = "OLS",
) -> tuple[LinearRegression, float]:
    """Plot a trendline showing the relationship between release date and time horizon."""
    agent_summaries = agent_summaries.sort_values("release_date")
    mask = agent_summaries["release_date"] >= pd.Timestamp(after).date()

    dates = pd.to_datetime(agent_summaries["release_date"])
    X = np.array([date2num(d) for d in dates]).reshape(-1, 1)
    y_raw = agent_summaries["50%"].clip(1e-3, np.inf)
    y = np.log(y_raw) if log_scale else y_raw

    X = X[mask]
    y = y[mask]
    if method == "WLS":
        spread = np.log(agent_summaries["50_high"]) - np.log(agent_summaries["50_low"])
        weights = 1 / (spread**2 + 1e-6)
        # replace nan with 0
        weights = np.nan_to_num(weights)[mask]
    else:
        weights = None
    reg = LinearRegression().fit(X, y, sample_weight=weights)

    score = float(reg.score(X, y))

    return reg, score


def plot_trendline(
    ax: Axes,
    agent_summaries: pd.DataFrame,
    after: str,
    log_scale: bool = False,
    method: str = "OLS",
    annotate: bool = True,
    fit_color: str | None = None,
    plot_kwargs: dict[str, Any] = {},
) -> dict[str, Any] | float | None:
    """Plot a trendline showing the relationship between release date and time horizon."""
    reg, score = fit_trendline(agent_summaries, after, log_scale, method)

    # trendline goes to the end of the x-axis
    x_range = np.linspace(date2num(pd.Timestamp(after)), ax.get_xlim()[1], 20)
    y_pred = reg.predict(x_range.reshape(-1, 1))

    x_dates = [num2date(x) for x in x_range]
    y_values = np.exp(y_pred) if log_scale else y_pred  # Convert back from log scale

    fit_color = fit_color or ("blue" if log_scale else "red")
    # Plot trendline
    pk = {"color": fit_color, "alpha": 0.5} | plot_kwargs
    ax.plot(x_dates, y_values, "--", **pk)

    doubling_time = 1 / reg.coef_[0] * np.log(2) if log_scale else None
    if not annotate:
        return doubling_time

    fit_type = "Exponential" if log_scale else "Linear"
    annotation = f"{fit_type} fit\nr^2: {score:.2f}"
    if log_scale and doubling_time is not None:
        annotation += f"\n(Doubling time: {doubling_time:.0f} days)"
        annotation += "\n" + ("All data" if after == "0000-00-00" else f"{after}+ data")
    else:
        annotation += f"\n(Rate: +{reg.coef_[0]*365:.0f} minutes per year)"

    return dict(
        text=annotation,
        color=fit_color,
        transform=ax.get_xaxis_transform(),
    )


def plot_trendline_hyperbolic(
    ax: matplotlib.axes.Axes, agent_summaries: pd.DataFrame, after: str
) -> dict[str, Any]:
    """Plot a trendline showing the relationship between release date and time horizon."""

    def hyperbolic_func(
        x: NDArray[Any], a: NDArray[Any], t_agi: NDArray[Any]
    ) -> NDArray[Any]:
        """
        Tries to fit the y value in log space
        """
        return np.log(a / (t_agi - x))

    log_scale = True
    agent_summaries = agent_summaries.sort_values("release_date")
    mask = agent_summaries["release_date"] >= pd.Timestamp(after).date()
    # Convert release dates to numeric (days since first release)
    dates = pd.to_datetime(agent_summaries["release_date"])
    first_release = min(dates)
    days_since_release = (dates - first_release).dt.days

    # Fit linear regression
    X = (cast(NDArray[Any], days_since_release.values)).reshape(-1, 1)
    y_raw = agent_summaries["50%"].clip(np.finfo(float).eps.astype(float), np.inf)
    y = np.log(y_raw) if log_scale else y_raw
    X = X[mask]
    y = y[mask]

    bounds = Bounds([0, max(X)[0]], [np.inf, np.inf])  # type: ignore[reportArgumentType]
    params, _ = curve_fit(
        hyperbolic_func, X.flatten(), y, bounds=bounds, p0=[5, 365 * 5.0]
    )

    # get r^2
    residuals = y - hyperbolic_func(X.flatten(), *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # type: ignore[reportOperatorIssue]
    r_squared = 1 - (ss_res / ss_tot)

    # Generate points for trendline, 20 equally spaced
    # x_range = np.array([min(X)[0], max(X)[0]])
    x_range = np.linspace(min(X)[0], max(X)[0], 20)
    y_pred = hyperbolic_func(x_range, params[0], params[1])

    # Convert back to datetime for plotting
    x_dates = [first_release + pd.Timedelta(days=x) for x in x_range]
    y_values = np.exp(y_pred) if log_scale else y_pred  # Convert back from log scale

    # print(f"x_dates: {x_dates}")
    # print(f"y_values: {y_values}")

    fit_color = "green"

    # Plot trendline
    ax.plot(x_dates, y_values, "--", color=fit_color, alpha=0.5)  # type: ignore[reportArgumentType]
    fit_type = "Hyperbolic"
    annotation = f"{fit_type} fit\nr^2: {r_squared:.2f}"
    t_agi = params[1]
    # convert to date
    t_agi = first_release + pd.Timedelta(days=t_agi)
    annotation += f"\n(T_agi: {t_agi:%Y-%m-%d})"

    return dict(
        text=annotation,
        color=fit_color,
        transform=ax.get_xaxis_transform(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--weighting", type=str)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--trendlines", type=str, default="true")
    parser.add_argument("--after-date", type=str, default="2022-06-01")
    parser.add_argument("--include-task-distribution", type=str, default="none")
    parser.add_argument("--runs-file", type=pathlib.Path, required=False)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    agent_summaries = pd.read_csv(args.input_file)
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
        # "davinci-002",
        "gpt-3.5-turbo-instruct",
        "o1",
        "o1-preview",
    ]

    logging.info("Loaded input data")
    trendlines = args.trendlines.lower() == "true"

    params = dvc.api.params_show(stages="plot_logistic_regression")

    weighting_list = params["weighting"]
    weighting = next(w for w in weighting_list if w["weight_col"] == args.weighting)
    subtitle = weighting["graph_snippet"]

    if args.runs_file:
        runs_df = pd.read_json(args.runs_file, lines=True)
    else:
        runs_df = None

    plot_horizon_graph(
        params["plots"],
        agent_summaries,
        runs_df=runs_df,
        output_file=args.output_file,
        focus_agents=focus_agents,
        trendlines=trendlines,
        after_date=args.after_date,
        subtitle=subtitle,
        include_task_distribution=args.include_task_distribution,
        weight_key=weighting["weight_col"],
    )

    src.utils.plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
