import argparse
import logging
from pathlib import Path

import dvc.api
import numpy as np
import numpy.typing as npt
import pandas as pd

import src.stats.statistics as statistics


def _calculate_statistics_across_tasks(
    runs_grouped_by_task: list[npt.NDArray[np.float64]],
    task_ids: list[str],
    ks: list[int],
    time_limit: int,
    agent: str,
    n_bootstrap: int = 5_000,
) -> list[dict[str, float | int | str]]:
    """Calculate confidence intervals and mean scores across all tasks.

    For each task, generates bootstrap samples and processes them differently:
    - For small_scaling_law: takes mean of k samples (average performance)
    - For other tasks: takes max of k samples (best-of-k performance)
    For each sample, the same bootstrap samples are used for all k values.
    Then averages these processed scores across all tasks and computes statistics.
    """
    results = []
    for k in ks:
        point_estimate, mean, ci_lower, ci_upper = (
            statistics.get_cross_task_summary_statistics(
                runs_grouped_by_task, task_ids, k, n_bootstrap
            )
        )
        results.append(
            {
                "samples": k,
                "time_limit": time_limit,
                "agent": agent,
                "ci_upper": ci_upper,
                "ci_lower": ci_lower,
                "mean_score": float(mean),
                "point_estimate": float(point_estimate),
            }
        )
    return results


def prepare_plot_data(
    df: pd.DataFrame,
    samples: int,
    time_limits: list[int],
    n_bootstrap: int = 5_000,
) -> pd.DataFrame:
    if samples <= 0:
        raise ValueError("Number of samples must be positive")

    results = []
    sample_sizes = [2**i for i in range(int(np.log2(samples)) + 1)]

    for time_limit in time_limits:
        time_limit_df = df.loc[df["time_limit"] == time_limit]

        for agent_alias, task_data in time_limit_df.groupby("alias"):
            logging.info(f"Calculating for {agent_alias} at time limit {time_limit}")
            runs_grouped_by_task = []
            task_ids = []
            min_runs_available = float("inf")

            for task_id, run_data in task_data.groupby("task_id"):
                scores = run_data["score"]
                num_runs = len(scores)
                if num_runs < samples:
                    logging.warning(
                        f"Task {task_id} has {num_runs} samples, less than requested {samples}"
                    )
                if (
                    task_id != "ai_rd_small_scaling_law/main"
                    and num_runs < min_runs_available
                ):
                    min_runs_available = num_runs
                runs_grouped_by_task.append(scores.values)
                task_ids.append(task_id)

            if not runs_grouped_by_task:
                logging.warning(
                    f"No runs for any tasks after processing for {agent_alias}"
                )
                continue

            valid_sample_sizes = [k for k in sample_sizes if k <= min_runs_available]

            if not valid_sample_sizes:
                logging.warning(f"No valid sample sizes for {agent_alias}")
                continue

            logging.info(
                f"Calculating data points for {len(valid_sample_sizes)} sample sizes"
            )
            results.extend(
                _calculate_statistics_across_tasks(
                    runs_grouped_by_task,
                    task_ids,
                    valid_sample_sizes,
                    time_limit,
                    str(agent_alias),
                    n_bootstrap,
                )
            )

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrangle data for performance over samples plot"
    )
    parser.add_argument(
        "--input-score-at-k",
        required=True,
        type=Path,
        help="Path to the input final scores JSONL file",
    )
    parser.add_argument(
        "--output-score-at-k",
        required=True,
        type=Path,
        help="Path to save the wrangled JSONL data",
    )
    parser.add_argument(
        "--samples", type=int, default=128, help="Number of samples to plot to"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        help="Number of bootstrap samples",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )

    logging.info(f"Loading data from {args.input_score_at_k}")
    df = pd.read_json(args.input_score_at_k, lines=True)

    logging.info("Preparing plot data")
    params = dvc.api.params_show()
    wrangled_data = prepare_plot_data(
        df,
        args.samples,
        params["rebench_best_of_k_parameters"]["time_limits"],
        args.n_bootstrap,
    )

    logging.info(f"Saving wrangled data to {args.output_score_at_k}")
    wrangled_data.to_json(args.output_score_at_k, orient="records", lines=True)


if __name__ == "__main__":
    main()
