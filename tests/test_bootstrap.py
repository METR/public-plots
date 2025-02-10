# %%
import pytest
import numpy as np
import pandas as pd
from src.wrangle.bootstrap import bootstrap_sample, bootstrap_runs_by_task_agent
from collections import Counter, defaultdict


def generate_synthetic_data(n_samples: int, seed: int) -> pd.DataFrame:
    """
    Generate synthetic data with a mean of 0.
    """
    id = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n_samples)
    y -= y.mean()
    family = rng.integers(0, 10, size=n_samples)
    task = family * 100 + rng.integers(0, 10, size=n_samples)
    agent = rng.integers(0, 10, size=n_samples)
    run = task * 100 + id
    df = pd.DataFrame(
        {
            "id": id,
            "family": family,
            "task_id": task,
            "run": run,
            "agent": agent,
            "y": y,
        },
        index=id,
    )
    return df


def test_bootstrap_sample_stats() -> None:
    df = generate_synthetic_data(100, seed=0)

    sampled_ids = set()
    sampling_means = []
    sampling_stds = []
    for i in range(100):
        sample_df = bootstrap_sample(
            df, categories=["family", "task_id", "run"], rng=np.random.default_rng(i)
        )
        ids = set(sample_df["id"].tolist())
        assert len(ids) < len(df), "Sampling each row once is astronomically unlikely"
        sampled_ids.update(ids)
        sampling_means.append(sample_df["y"].mean())
        sampling_stds.append(sample_df["y"].std())

    np.testing.assert_allclose(np.mean(sampling_means), 0, atol=0.1)
    np.testing.assert_allclose(np.mean(sampling_stds), 0.95, atol=0.1)


def test_bootstrap_expected_copies() -> None:
    """
    The expected copies of each row after bootstrapping is 1.
    """
    rows = [{"family": 0, "task_id": 0, "run": 0}]
    rows += [{"family": 1, "task_id": 1, "run": 1 + i} for i in range(10)]
    rows += [{"family": 2, "task_id": 2 + i, "run": 12 + i} for i in range(10)]
    df = pd.DataFrame(rows)
    result = pd.concat(
        [
            bootstrap_sample(
                df,
                categories=["family", "task_id", "run"],
                rng=np.random.default_rng(i),
            )
            for i in range(1000)
        ]
    )
    frequencies = result["run"].value_counts() / 1000

    assert set(df["run"].tolist()) == set(
        result["run"].tolist()
    ), "All rows should be sampled at least once"
    assert np.allclose(
        frequencies, 1, atol=0.2
    ), f"Expected each row to be sampled once on average; actual frequencies: {frequencies}"


def test_bootstrap_runs_expected_copies() -> None:
    """
    The expected copies of each row after bootstrapping is 1.
    """
    rows = [{"agent": 0, "task_id": 0, "run": 0}]
    rows += [{"agent": 1, "task_id": 1, "run": 1 + i} for i in range(10)]
    rows += [{"agent": 2, "task_id": 2 + i, "run": 12 + i} for i in range(10)]
    df = pd.DataFrame(rows)
    result = pd.concat(
        [
            df.iloc[
                bootstrap_runs_by_task_agent(
                    df["task_id"].to_numpy(),
                    df["agent"].to_numpy(),
                    np.arange(len(df)),
                    rng=np.random.default_rng(i),
                )
            ]
            for i in range(1000)
        ]
    )
    frequencies = result["run"].value_counts() / 1000

    assert set(df["run"].tolist()) == set(
        result["run"].tolist()
    ), "All rows should be sampled at least once"
    assert np.allclose(
        frequencies, 1, atol=0.2
    ), f"Expected each row to be sampled once on average; actual frequencies: {frequencies}"


def test_num_distinct_tasks_per_family() -> None:
    """
    If a family is sampled n times, the expected number of distinct tasks in the sample should be:
    1 - (1/e) ** n
    """
    df = generate_synthetic_data(100, seed=0)
    # keep only one row per task so we can count copies of each family
    df = df.groupby("task_id").head(1)

    frac_distinct_tasks = defaultdict(list)
    for i in range(100):
        sampled_df = bootstrap_sample(
            df, categories=["family", "task_id"], rng=np.random.default_rng(i)
        )
        n_family_copies = (
            sampled_df.groupby("family")["task_id"].count()
            / df.groupby("family")["task_id"].count()
        )
        distinct_tasks_frac = (
            sampled_df.groupby("family")["task_id"].nunique()
            / df.groupby("family")["task_id"].nunique()
        )
        for family in sampled_df["family"].unique():
            frac_distinct_tasks[n_family_copies[family].item()].append(
                distinct_tasks_frac[family].item()
            )

    for n_copies, frac_distinct in frac_distinct_tasks.items():
        assert np.allclose(
            np.mean(frac_distinct), 1 - (1 / np.e) ** n_copies, atol=0.05
        ), f"Expected families with {n_copies} copies to keep {1 - (1 / np.e) ** n_copies:.2f} of their distinct tasks on average; actual {np.mean(frac_distinct):.2f}"


def test_bootstrap_runs_by_task_agent() -> None:
    """
    Test the subroutine that bootstraps runs by task and agent.
    """
    data = pd.DataFrame(
        {
            "run_id": np.arange(300),
            "task_id": np.arange(300) % 10,
            "agent": np.random.randint(0, 10, size=300),
            "schmagent": np.random.randint(0, 10, size=300),
        }
    )

    indices = np.arange(len(data))
    new_indices_42 = bootstrap_runs_by_task_agent(
        data["task_id"].to_numpy(),
        data["agent"].to_numpy(),
        np.arange(len(data)),
        np.random.default_rng(42),
    )

    for i in range(5):
        new_indices_A = bootstrap_runs_by_task_agent(
            data["task_id"].to_numpy(),
            data["agent"].to_numpy(),
            indices,
            rng=np.random.default_rng(i),
        )
        new_indices_B = bootstrap_runs_by_task_agent(
            data["task_id"].to_numpy(),
            data["agent"].to_numpy(),
            indices,
            rng=np.random.default_rng(i),
        )

        assert set(new_indices_A) == set(
            new_indices_B
        ), "Should return the same set of indices for the same random seed"
        assert set(new_indices_A) != set(
            new_indices_42
        ), "Should return different sets of indices for different random seeds"

        result = data.iloc[new_indices_A]

        assert len(result) == len(data), "Should return the same number of rows"
        assert np.all(
            result.groupby(["task_id", "agent"])["run_id"].count()
            == data.groupby(["task_id", "agent"])["run_id"].count()
        ), "All task_id, agent groups should have the same number of runs"
        assert Counter(
            result.groupby(["task_id", "schmagent"])["run_id"].count()
        ) != Counter(
            data.groupby(["task_id", "schmagent"])["run_id"].count()
        ), "Sizes of any groups other than task_id, agent should be different"


def test_bootstrap_runs_and_other_categories() -> None:
    """Test that bootstrapping runs works correctly.
    "runs" groups by task and agent, so the group lengths should be the same.

    Due to the other categories, the number of runs per task_id, agent group should be different,
    but ratios should be preserved.
    """
    random_ints = np.random.randint(0, 100, size=300)
    data = pd.DataFrame(
        {
            "run_id": np.arange(300),
            "family": random_ints % 5,
            "task_id": random_ints % 10,
            "agent": np.random.randint(0, 10, size=300),
            "schmagent": np.random.randint(0, 10, size=300),
            "human_minutes": np.exp(random_ints % 10 * 0.1) * 42,
        }
    )

    for i in range(2):
        result_with_run_id = bootstrap_sample(
            data,
            categories=["time_buckets", "family", "task_id", "run_id"],
            rng=np.random.default_rng(i),
        )
        result_without_run_id = bootstrap_sample(
            data,
            categories=["time_buckets", "family", "task_id"],
            rng=np.random.default_rng(i),
        )

        assert len(result_with_run_id) == len(
            result_without_run_id
        ), "Should return the same number of rows"
        assert np.all(
            result_with_run_id.groupby(["task_id", "agent"])["run_id"].count()
            == result_without_run_id.groupby(["task_id", "agent"])["run_id"].count()
        ), "Task_id, agent groups should have their sizes preserved by bootstrapping runs"

        assert Counter(
            result_with_run_id.groupby(["task_id", "schmagent"])["run_id"].count()
        ) != Counter(
            result_without_run_id.groupby(["task_id", "schmagent"])["run_id"].count()
        ), "Sizes of any groups other than task_id, agent should be different"

        # Check that for each task, the ratio of agent runs is preserved
        for task_id in data["task_id"].unique():
            task_data = data[data["task_id"] == task_id]
            task_result = result_with_run_id[result_with_run_id["task_id"] == task_id]

            # Get run counts per agent for this task
            data_agent_counts = task_data.groupby("agent")["run_id"].count()
            result_agent_counts = task_result.groupby("agent")["run_id"].count()

            # Calculate ratios
            data_ratios = data_agent_counts / data_agent_counts.sum()
            result_ratios = result_agent_counts / result_agent_counts.sum()

            # Compare ratios for agents that appear in both
            common_agents = list(set(data_ratios.index) & set(result_ratios.index))
            assert np.allclose(
                data_ratios[common_agents], result_ratios[common_agents]
            ), f"Agent run ratios should be preserved within task {task_id}"


def test_random_seed_used() -> None:
    """
    The set of ids should be the same for the same random seed, and different for different random seeds.
    """
    data = generate_synthetic_data(100, seed=0)

    result_0A = bootstrap_sample(
        data, categories=["family", "task_id"], rng=np.random.default_rng(0)
    )
    result_0B = bootstrap_sample(
        data, categories=["family", "task_id"], rng=np.random.default_rng(0)
    )
    result_1 = bootstrap_sample(
        data, categories=["family", "task_id"], rng=np.random.default_rng(1)
    )

    assert set(result_0A["id"].tolist()) == set(
        result_0B["id"].tolist()
    ), "Should return the same set of ids for the same random seed"
    assert set(result_0A["id"].tolist()) != set(
        result_1["id"].tolist()
    ), "Should return different sets of ids for different random seeds"


def test_bootstrap_time_buckets() -> None:
    """Test that bootstrapping time buckets works correctly"""
    data = pd.DataFrame(
        {"human_minutes": np.geomspace(1, 100, 20), "score": np.random.random(20)}
    )
    result = bootstrap_sample(
        data, categories=["time_buckets"], rng=np.random.default_rng(0)
    )

    assert len(result) > 0, "Should return non-empty result"
    assert "bucket" not in result.columns, "Should not include bucket column in result"
    assert set(result.columns) == set(data.columns), "Should maintain same columns"


def test_invalid_hierarchy() -> None:
    """Test that invalid hierarchical relationships raise an error"""
    data = pd.DataFrame(
        {
            "task_family": ["family1", "family2", "family1"],
            "task_id": ["task1", "task2", "task2"],  # task2 belongs to two families
            "score": [0.5, 0.6, 0.7],
        }
    )

    with pytest.raises(AssertionError):
        bootstrap_sample(
            data, categories=["task_family", "task_id"], rng=np.random.default_rng(0)
        )


def test_num_dropped_groups() -> None:
    """
    After the first level, we shouldn't drop any top-level groups by dropping their subcategories.
    So the fraction dropped should be 1/e.
    """

    size = 4000
    fudge_factor = 100

    data = pd.DataFrame(
        {
            "l1": np.arange(size),
            "l2": np.arange(size),
            "l3": np.arange(size),
        }
    )

    one_level_result = bootstrap_sample(
        data, categories=["l1"], rng=np.random.default_rng(0)
    )
    three_level_result = bootstrap_sample(
        data, categories=["l1", "l2", "l3"], rng=np.random.default_rng(0)
    )
    one_level_uniques = one_level_result.l1.nunique()
    three_level_uniques = three_level_result.l3.nunique()

    reference_uniques = size * (1 - 1 / np.e)

    assert (
        reference_uniques - fudge_factor
        < one_level_uniques
        < reference_uniques + fudge_factor
    ), f"1-level bootstrapping should keep about {reference_uniques} groups; actual {one_level_uniques}"

    assert (
        reference_uniques - fudge_factor
        < three_level_uniques
        < reference_uniques + fudge_factor
    ), f"3-level bootstrapping should keep about {reference_uniques} groups; actual {three_level_uniques}"


# %%
