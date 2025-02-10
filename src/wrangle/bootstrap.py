import argparse
import logging
import pathlib
from typing import List, Dict, Any
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from src.wrangle.logistic import get_x_for_quantile, unscaled_regression

logger = logging.getLogger(__name__)


def bootstrap_runs_by_task_agent(
    task_col: np.ndarray[Any, np.dtype[Any]],
    agent_col: np.ndarray[Any, np.dtype[Any]],
    indices: np.ndarray[Any, np.dtype[np.int_]],
    rng: np.random.Generator,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Bootstrap runs within each task-agent group.

    Args:
        task_col: Array of task IDs
        agent_col: Array of agent names
        indices: Current indices to sample from
        rng: Random number generator

    Returns:
        Array of resampled indices
    """
    assert len(task_col) == len(
        agent_col
    ), "Task and agent arrays must be the same length"
    task_agent = np.char.add(
        np.char.add(task_col.astype(str), "|||"), agent_col.astype(str)
    )
    task_agents = task_agent[indices]
    unique_task_agents, task_agent_indices, counts = np.unique(
        task_agents, return_inverse=True, return_counts=True
    )
    # Generate random numbers and sample with replacement
    random_nums = rng.random(len(indices))
    offsets = np.cumsum([0] + list(counts)[:-1])
    all_new_indices = [
        indices[task_agent_indices == j][
            (random_nums[offset : offset + count] * count).astype(np.int64)
        ]
        for j, (count, offset) in enumerate(zip(counts, offsets))
    ]
    return np.concatenate(all_new_indices)


def bootstrap_sample(
    data: pd.DataFrame,
    categories: List[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Perform hierarchical bootstrapping of the data.
    We don't reweight points-- if we sample a task twice, it should be counted twice.

    Note: bootstrapping implementations are notorious for being buggy, so
    check with someone before making changes.
    Args:
        data: DataFrame containing the runs data
        categories: List of categories to bootstrap over (e.g. ["task_family", "task_id"])
        rng: NumPy random number generator for reproducibility
    """
    # Handle run_id separately at the end
    has_run_id = "run_id" in categories
    has_bucket = "time_buckets" in categories
    categories = [c for c in categories if c != "run_id" and c != "time_buckets"]

    # Assert that later categories are subcategories of earlier categories
    # Deliberately don't check buckets bc a few families span multiple buckets
    # Don't check run_id for performance reasons
    for i in range(len(categories) - 1):
        parent = categories[i]
        child = categories[i + 1]
        assert (
            data.groupby(child)[parent].nunique().max() == 1
        ), f"{child} is not a subcategory of {parent}"

    if has_bucket:
        buckets = np.geomspace(
            data["human_minutes"].min(), data["human_minutes"].max(), 10
        )
        data["bucket"] = pd.cut(data["human_minutes"], bins=buckets).astype(str)
        categories.insert(0, "bucket")

    # Pre-compute numpy arrays for each category, only storing unique values
    category_arrays = {category: data[category].to_numpy() for category in categories}

    # Start with all indices
    indices = np.arange(len(data))
    split_ids = np.zeros(len(data), dtype=np.int32)
    new_split_id = 0

    # Bootstrap over each category hierarchically
    for i, category in enumerate(categories):
        is_last_category = i == len(categories) - 1
        all_new_indices = []
        all_new_split_ids = []

        # For each split_id, resample its category groups
        for group_id in np.unique(split_ids):
            group_indices = indices[split_ids == group_id]
            category_values = category_arrays[category][group_indices]

            # Get unique values
            # e.g. [family0, family1], [0, 0, 1, 0, 1, 1]
            values, value_indices = np.unique(category_values, return_inverse=True)
            n_values = len(values)
            # Sample values with replacement: e.g. [0, 0]
            sampled_values = rng.choice(n_values, size=n_values, replace=True)

            # For each sampled value, append corresponding indices to all_new_indices
            # e.g. [0, 0] -> [0, 1, 3, 0, 1, 3]
            for j, sampled_value in enumerate(sampled_values):
                sampled_indices = group_indices[value_indices == sampled_value]
                all_new_indices.append(sampled_indices)

                if not is_last_category:
                    all_new_split_ids.append(
                        np.full(len(sampled_indices), new_split_id + j)
                    )

            new_split_id += n_values

        # Update indices and split_ids for next iteration
        indices = np.concatenate(all_new_indices)
        if not is_last_category:
            split_ids = np.concatenate(all_new_split_ids)

    # Handle run_id sampling at the end if needed
    if has_run_id:
        task_col = data["task_id"].to_numpy()
        agent_col = data["agent"].to_numpy()
        indices = bootstrap_runs_by_task_agent(task_col, agent_col, indices, rng)

    if has_bucket:
        data.drop(columns=["bucket"], inplace=True)  # Clean up temporary column

    # Create final result using the sampled indices
    result = data.iloc[indices].copy()

    return result


def _process_bootstrap(
    bootstrap_idx: int,
    data: pd.DataFrame,
    categories: List[str],
    weights_col: str,
    regularization: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Helper function to process a single bootstrap iteration."""
    bootstrap_results = {}

    bootstrap_data = bootstrap_sample(data, categories, rng)

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

        model = unscaled_regression(
            x, y, sample_weight=weights, regularization=regularization
        )
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
        n_bootstrap: Number of bootstrap iterations
        regularization: Regularization parameter
        weights_col: Column to use for weights
    """
    # Use all available CPU cores except one
    n_jobs = max(1, Parallel(n_jobs=-1)._effective_n_jobs())  # type: ignore

    # Batch size of 10 bootstraps per worker
    batch_size = 10
    n_batches = (n_bootstrap + batch_size - 1) // batch_size  # Round up division

    def process_batch(batch_idx: int) -> List[Dict[str, float]]:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_bootstrap)
        batch_results = []
        for i in range(start_idx, end_idx):
            # Create a new random state for each bootstrap iteration
            rng = np.random.default_rng(42 + i)
            result = _process_bootstrap(
                i, data, categories, weights_col, regularization, rng
            )
            batch_results.append(result)
        return batch_results

    # Run parallel computations in batches
    batched_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_batch)(i) for i in range(n_batches)
    )

    # Flatten results
    results = [result for batch in batched_results for result in batch]  # type: ignore
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
        "r": "run_id",
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
