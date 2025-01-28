import numpy as np
import pandas as pd


def compute_sample_weights(
    df_agent: pd.DataFrame,
) -> pd.DataFrame:
    """
    Input: df_agent, runs for a single agent
    Output: two weighting columns
        - equal_task_weight: each run is 1 / n_run_in_task
        - invsqrt_task_weight: each run is 1 / (n_run_in_task * sqrt(n_task_in_family))
    """
    df_agent_tasks = (
        df_agent.groupby("task_id")
        .agg(
            {
                "task_family": "first",
                "alias": "count",
            }
        )
        .rename(columns={"alias": "num_runs_in_task"})
    )

    nans = df_agent.score.isna().sum()
    assert nans == 0, f"NaNs in df_agent: {nans}, {df_agent.agent_id.unique()}"

    equal_task_weight = 1 / df_agent_tasks["num_runs_in_task"]
    equal_task_weight = (
        df_agent["task_id"].map(equal_task_weight).rename("equal_task_weight")
    )
    assert np.allclose(equal_task_weight.sum(), df_agent["task_id"].nunique())

    family_sizes = (
        df_agent_tasks.reset_index().groupby("task_family")["task_id"].count()
    )
    invsqrt_tasks_in_family = 1 / np.sqrt(df_agent["task_family"].map(family_sizes))
    invsqrt_weight = (equal_task_weight * invsqrt_tasks_in_family).rename(
        "invsqrt_task_weight"
    )
    assert np.allclose(invsqrt_weight.sum(), np.sqrt(family_sizes).sum())

    equal_task_weight = equal_task_weight / equal_task_weight.sum()
    invsqrt_weight = invsqrt_weight / invsqrt_weight.sum()

    return pd.concat([equal_task_weight, invsqrt_weight], axis=1)
