import argparse
import logging
import pathlib
import textwrap
from typing import Any

import dvc.api
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

import src.utils.plots

logger = logging.getLogger(__name__)


def overall_bar_chart_weighted(
    plot_params: src.utils.plots.PlotParams,
    df: pd.DataFrame,
    score_col: str = "score",
    title: str = "",
    focus_agents: list[str] | None = None,
    show: bool = False,
    agent_labels: dict[str, str] | None = None,
    ylabel: str | None = None,
    colors: dict[str, str] | None = None,
    plot_human_line: bool = True,
    pass_at_k_sampling: int | None = None,
    order_agents: list[str] | None = None,
    abbreviated_subtitle: bool = False,
) -> Figure:
    df = df.copy()

    # If focus_agents is provided, only plot those agents
    if focus_agents:
        df = df[df["alias"].isin(focus_agents)]
        for agent in focus_agents:
            assert agent in df["alias"].unique(), f"Agent {agent} not found in df"

    print("df when printing:")
    print(df)

    # Add model family column
    df["model_family"] = df["alias"].map(
        lambda x: (
            "Claude"
            if "Claude" in x
            else "GPT"
            if "GPT" in x
            else "Human"
            if "human" in x
            else "Other"
        )
    )

    # Extract human data to separate df
    human_df = df[df["model_family"] == "Human"]
    human_df = human_df.groupby("alias")[score_col].mean().reset_index()
    if agent_labels:
        human_df = human_df.replace({"alias": agent_labels})
    df = df[df["model_family"] != "Human"]

    # Order agents by ordering if exists, otherwise sort by model family, then by score
    agent_ordering = plot_params["legend_order"]
    agent_ordering = [
        agent for agent in agent_ordering if agent in df["alias"].unique()
    ]
    print(f"Plotting agents {agent_ordering}")

    agent_colors = [
        src.utils.plots.get_agent_color(plot_params["colors"], agent)
        for agent in agent_ordering
    ]
    df["color"] = df["alias"].map(lambda x: agent_colors[agent_ordering.index(x)])

    # Plotting time
    # Ensure the plot is not too narrow when few agents are provided
    fig, ax = plt.subplots(
        figsize=(max(7, 1.2 * len(agent_ordering)), 5),
        tight_layout=True,
    )

    # If agent_labels, replace the alias values with the agent_labels
    if agent_labels:
        df["alias"] = df["alias"].map(agent_labels).fillna(df["alias"])
        agent_ordering = [agent_labels.get(agent, agent) for agent in agent_ordering]

    # Plot human lines as annotated dotted horizontal lines
    agent_bar_offset = 0.66

    print(df)
    # Plot agent performance
    for idx_agent, agent in enumerate(agent_ordering):
        print(f"Plotting agent {agent}")
        color = df[df["alias"] == agent]["color"].iloc[0]
        df_agent = df[df["alias"] == agent]
        score = df_agent[score_col].item()
        print(f"{score}: {agent}")
        ax.bar(
            idx_agent + agent_bar_offset,
            score,
            color=color,
            edgecolor="black" if "Human" in agent else color,
            lw=1.5,
            yerr=(df_agent[["ci_low", "ci_high"]] - score).abs().values.T,
            error_kw=dict(capsize=7.5, lw=1.5, capthick=1.5),
            zorder=4,
        )

    ax.set_axisbelow(True)

    # X labels
    wrapped_labels = [textwrap.fill(label, width=12) for label in agent_ordering]
    plt.xticks(
        [x + agent_bar_offset for x in range(0, len(agent_ordering))], wrapped_labels
    )
    # ax.set_xlim(0, len(agent_ordering) + human_text_offset)
    if not plot_human_line:
        ax.grid(axis="y", linestyle="--", linewidth=1, which="both", alpha=0.5)

    # Y label
    if ylabel:
        ax.set_ylabel(ylabel)

    # Y limit
    ax.set_ylim(min(df["score"]) / 2, max(df["score"]) * 2)
    src.utils.plots.log_y_axis(ax, unit="minutes")
    # Title
    if title:
        title = f"{title} (95% CI)"
        ax.set_title(title)

    # Subtitle
    subtitle = f"{f' - pass@{pass_at_k_sampling}' if pass_at_k_sampling else ''})"
    # if abbreviated_subtitle:
    # subtitle = f"({bootstrapping['graph_snippet']}{f' - pass@{pass_at_k_sampling}' if pass_at_k_sampling else ''})"
    # else:
    #     subtitle = f"({scoring['graph_snippet']} - {weighting['graph_snippet']} - {bootstrapping['graph_snippet']}{f' - pass@{pass_at_k_sampling}' if pass_at_k_sampling else ''})"
    ax.text(
        0.5,
        0.95,
        subtitle,
        size=plt.rcParams["axes.labelsize"] * 0.8,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )

    if show:
        plt.show()
    return fig


def main(
    metrics_file: pathlib.Path,
    output_file: pathlib.Path,
    params: dict[str, Any],
    boot_set: str,
    log_level: str,
    weighting: str,
    pass_at_k_sampling: int | None = None,
) -> None:
    logging.basicConfig(level=log_level.upper())
    params = dvc.api.params_show(stages=["plot_bar_chart"])
    with metrics_file.open() as file:
        df = pd.read_csv(file)

    # Extract required columns and rename for compatibility
    df = df[["agent", "50%", "50_low", "50_high"]].copy()
    df = df.rename(
        columns={
            "agent": "alias",
            "50%": "score",
            "50_low": "ci_low",
            "50_high": "ci_high",
        }
    )

    print(df)

    overall_bar_chart_weighted(
        params["plots"],
        df,
        focus_agents=[
            "Claude 3 Opus",
            "Claude 3.5 Sonnet (New)",
            "GPT-4o",
            "GPT-4 Turbo",
        ],
        show=False,
    )

    src.utils.plots.save_or_open_plot(output_file, params["plot_format"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--params", type=str, required=True)
    parser.add_argument("--boot-set", type=str, required=True)
    parser.add_argument("--pass-at-k-sampling", required=False)
    parser.add_argument("--log-level", type=str, required=False)
    parser.add_argument("--weighting", type=str, required=False)
    args = parser.parse_args()
    if args.pass_at_k_sampling == "None":
        args.pass_at_k_sampling = None
    else:
        raise NotImplementedError(
            "Pass at k sampling is not implemented for bar charts; bug tkwa if you want this"
        )
        args.pass_at_k_sampling = int(args.pass_at_k_sampling)

    if args.boot_set != "None":
        raise NotImplementedError(
            "Bootstrapping is not implemented for bar charts; bug tkwa if you want this"
        )

    main(**vars(args))
