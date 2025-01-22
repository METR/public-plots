import argparse
import logging
import pathlib

import dvc.api
import numpy as np
import pandas as pd

import src.stats.statistics
import src.utils.plots

logger = logging.getLogger("eval_pipeline.wrangle_bar_by_time_allocation")


_MAX_TIME_LIMIT_IN_SECONDS = 8 * 60 * 60


def wrangle_bar_by_time_allocation(
    runs_file: pathlib.Path,
    wrangled_file: pathlib.Path,
) -> None:
    """Wrangle mean scores and confidence intervals for different time limits.

    For each time limit, calculates best-of-k performance where k is the number
    of times that time limit fits into 8 hours.
    """
    df_runs = pd.read_json(runs_file, lines=True)

    params = dvc.api.params_show(stages="wrangle_bar_by_time_allocation")
    n_bootstrap = params["n_bootstrap"]
    time_limits = params["stages"]["wrangle_bar_by_time_allocation"]["time_limits"]

    data_wrangled = []
    for time_limit in time_limits:
        k = _MAX_TIME_LIMIT_IN_SECONDS // time_limit
        df_time_limit = df_runs.loc[df_runs["time_limit"] == time_limit, :]
        time_limit_label = src.utils.plots.format_time_label(time_limit)
        time_label = f"{time_limit_label} @ {k}"

        for agent, df_agent in df_time_limit.groupby("alias"):
            valid_runs_by_task = []
            valid_task_ids = []
            for task_id, task_group in df_agent.groupby("task_id"):
                task_scores = np.array(task_group["score"])
                if len(task_scores) < k:
                    logger.warning(
                        f"Skipping task {task_id} for agent {agent} at time limit {time_limit}: "
                        f"insufficient runs (needed {k}, found {len(task_scores)})"
                    )
                    continue

                valid_runs_by_task.append(task_scores)
                valid_task_ids.append(task_id)

            if not valid_runs_by_task:
                logger.warning(
                    f"No valid tasks for agent {agent} at time limit {time_limit}"
                )
                continue

            point_estimate, mean_score, ci_lower, ci_upper = (
                src.stats.statistics.get_cross_task_summary_statistics(
                    valid_runs_by_task, valid_task_ids, k, n_bootstrap
                )
            )

            data_wrangled.append(
                {
                    "agent": agent,
                    "time_label": time_label,
                    "time_limit": time_limit,
                    "mean_score": float(mean_score),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "point_estimate": float(point_estimate),
                }
            )

    df_wrangled = pd.DataFrame(data_wrangled)
    wrangled_file.parent.mkdir(parents=True, exist_ok=True)
    df_wrangled.to_json(wrangled_file, orient="records", lines=True)
    logger.info(f"Data saved to {wrangled_file}")


parser = argparse.ArgumentParser(description="Compute mean scores by time limit.")
parser.add_argument(
    "--runs-file",
    type=pathlib.Path,
    required=True,
    help="Path to the runs JSONL file.",
)
parser.add_argument(
    "--wrangled-file",
    type=pathlib.Path,
    required=True,
    help="Path to save the wrangled data.",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity.")
if __name__ == "__main__":
    args = vars(parser.parse_args())

    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s")
    logging.getLogger("eval_pipeline").setLevel(
        logging.DEBUG if args.pop("verbose") else logging.INFO
    )

    wrangle_bar_by_time_allocation(**args)
