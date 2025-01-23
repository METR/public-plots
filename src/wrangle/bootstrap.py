import argparse
import logging
import pathlib
from typing import List, Dict
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from src.wrangle.logistic import get_x_for_quantile, unscaled_regression

logger = logging.getLogger(__name__)


def bootstrap_sample(
    data: pd.DataFrame,
    categories: List[str],
) -> pd.DataFrame:
    """
    Perform hierarchical bootstrapping of the data.
    We don't reweight points-- if we sample a task twice, it should be counted twice.

    Note: bootstrapping implementations are notorious for being buggy, so
    check with someone before making changes.
    Args:
        data: DataFrame containing the runs data
        categories: List of categories to bootstrap over (e.g. ["task_family", "task_id"])
        bootstrap_time_buckets: Whether to bootstrap over time buckets
    """
    sampled_data = data.copy()

    bootstrap_runs = "runs" in categories
    categories = [c for c in categories if c not in ["runs"]]

    # Assert that later categories are subcategories of earlier categories
    # Deliberately don't check buckets bc a few families span multiple buckets
    for i in range(len(categories) - 1):
        parent = categories[i]
        child = categories[i + 1]
        assert (
            data.groupby(child)[parent].nunique().max() == 1
        ), f"{child} is not a subcategory of {parent}"

    categories = [("bucket" if c == "time_buckets" else c) for c in categories]

    if "bucket" in categories:
        buckets = np.geomspace(
            data["human_minutes"].min(), data["human_minutes"].max(), 10
        )
        sampled_data["bucket"] = pd.cut(
            sampled_data["human_minutes"], bins=buckets
        ).astype(str)

    sampled_data["split_id"] = 0
    new_split_id = 0
    # Bootstrap over each category hierarchically
    for category in categories:
        # For each split_id, resample its category groups
        new_split_data = []
        for group_id, group in sampled_data.groupby("split_id"):
            values = group[category].unique()
            sampled_values = np.random.choice(values, size=len(values), replace=True)
            for value in sampled_values:
                sampled_rows = group[group[category] == value].copy()
                sampled_rows["split_id"] = new_split_id
                new_split_data.append(sampled_rows)
                new_split_id += 1

        sampled_data = pd.concat(new_split_data)

    if bootstrap_runs:
        # Bootstrap individual runs within each task
        sampled_runs = []
        for (task_id, agent), group in sampled_data.groupby(["task_id", "agent"]):
            n_runs = len(group)
            sampled_indices = np.random.choice(n_runs, size=n_runs, replace=True)
            sampled_runs.append(group.iloc[sampled_indices])
        sampled_data = pd.concat(sampled_runs)

    if "bucket" in categories:
        sampled_data.drop(columns=["bucket"], inplace=True)

    sampled_data.drop(columns=["split_id"], inplace=True)

    return sampled_data


def _process_bootstrap(
    bootstrap_idx: int,
    data: pd.DataFrame,
    categories: List[str],
    weights_col: str,
    regularization: float,
) -> Dict[str, float]:
    """Helper function to process a single bootstrap iteration."""
    bootstrap_results = {}

    # Perform hierarchical bootstrap
    bootstrap_data = bootstrap_sample(data, categories)

    for agent_name in bootstrap_data["agent"].unique():
        agent_data = bootstrap_data[bootstrap_data["agent"] == agent_name]

        # Prepare data for regression
        x = np.log2(agent_data["human_minutes"].values).reshape(-1, 1)
        y = np.clip(agent_data["score"].values, 0, 1)

        # Get weights if specified
        weights = agent_data[weights_col].values

        # Fit regression and get p50
        if len(np.unique(y)) < 2:
            continue

        model = unscaled_regression(x, y, sample_weight=weights)
        p50 = np.exp2(get_x_for_quantile(model, 0.5)).item()
        logger.debug(f"{agent_name} p50: {p50}")
        if np.isnan(p50):
            logger.warning(
                f"{agent_name} has nan p50 on bootstrap {bootstrap_idx}; params: {model.coef_}, {model.intercept_}"
            )

        bootstrap_results[agent_name] = p50

    return bootstrap_results


def compute_bootstrap_regressions(
    data: pd.DataFrame,
    categories: List[str],
    n_bootstrap: int,
    regularization: float,
    weights_col: str,
) -> pd.DataFrame:
    """
    Compute bootstrapped logistic regressions and extract the 50% points.

    Args:
        data: DataFrame containing the runs data
        categories: List of categories to bootstrap over
        bootstrap_time_buckets: Whether to bootstrap over time buckets
        n_bootstrap: Number of bootstrap iterations
        method: Either "scaled" or "unscaled"
        weights_col: Column to use for weights, if any
    """
    # Use all available CPU cores except one
    n_jobs = max(1, Parallel(n_jobs=-1)._effective_n_jobs())  # type: ignore

    # Run parallel computations with progress bar
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_process_bootstrap)(
            bootstrap_idx=i,
            data=data,
            categories=categories,
            weights_col=weights_col,
            regularization=regularization,
        )
        for i in range(n_bootstrap)
    )

    return pd.DataFrame(results)  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument(
        "--categories",
        type=str,
        required=True,
        help="Categories to bootstrap over (e.g. `ftr` for task_family,task_id,runs)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations"
    )
    parser.add_argument(
        "--weights-col", type=str, required=True, help="Column to use for weights"
    )
    parser.add_argument(
        "--regularization", type=float, default=0.01, help="Regularization parameter"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load data
    data = pd.read_json(args.input_file, lines=True, orient="records")
    logging.info(f"Loaded {len(data)} runs from {args.input_file}")
    data.rename(columns={"alias": "agent"}, inplace=True)

    category_dict = {
        "f": "task_family",
        "t": "task_id",
        "r": "runs",
        "b": "time_buckets",
    }
    categories = [category_dict[c] for c in args.categories]
    logging.info(f"Bootstrapping over categories: {categories}")

    # Compute bootstrapped regressions
    results = compute_bootstrap_regressions(
        data=data,
        categories=categories,
        n_bootstrap=args.n_bootstrap,
        weights_col=args.weights_col,
        regularization=args.regularization,
    )

    # Save results
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_file, index=False)
    logging.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
