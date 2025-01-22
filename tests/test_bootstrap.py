# %%
import pytest
import numpy as np
import pandas as pd
from src.wrangle.bootstrap import bootstrap_sample
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
    run = task * 100 + id
    df = pd.DataFrame(
        {
            "id": id,
            "family": family,
            "task": task,
            "run": run,
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
    for _ in range(100):
        sample_df = bootstrap_sample(df, categories=["family", "task", "run"])
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
    rows = [{"family": 0, "task": 0, "run_id": 0}]
    rows += [{"family": 1, "task": 1, "run_id": 1 + i} for i in range(10)]
    rows += [{"family": 2, "task": 2 + i, "run_id": 12 + i} for i in range(10)]
    df = pd.DataFrame(rows)
    result = pd.concat(
        [
            bootstrap_sample(df, categories=["family", "task", "run_id"])
            for _ in range(1000)
        ]
    )
    frequencies = result["run_id"].value_counts() / 1000

    assert set(df["run_id"].tolist()) == set(
        result["run_id"].tolist()
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
    df = df.groupby("task").head(1)

    frac_distinct_tasks = defaultdict(list)
    for _ in range(100):
        sampled_df = bootstrap_sample(df, categories=["family", "task"])
        n_family_copies = (
            sampled_df.groupby("family")["task"].count()
            / df.groupby("family")["task"].count()
        )
        distinct_tasks_frac = (
            sampled_df.groupby("family")["task"].nunique()
            / df.groupby("family")["task"].nunique()
        )
        for family in sampled_df["family"].unique():
            frac_distinct_tasks[n_family_copies[family].item()].append(
                distinct_tasks_frac[family].item()
            )

    for n_copies, frac_distinct in frac_distinct_tasks.items():
        assert np.allclose(
            np.mean(frac_distinct), 1 - (1 / np.e) ** n_copies, atol=0.05
        ), f"Expected families with {n_copies} copies to keep {1 - (1 / np.e) ** n_copies:.2f} of their distinct tasks on average; actual {np.mean(frac_distinct):.2f}"


def test_bootstrap_sample_runs() -> None:
    """Test that bootstrapping runs works correctly.
    "runs" groups by task and agent, so the group lengths should be the same.
    """
    data = pd.DataFrame(
        {
            "run_id": np.arange(300),
            "task_id": np.arange(300) % 10,
            "agent": np.random.randint(0, 10, size=300),
            "schmagent": np.random.randint(0, 10, size=300),
            "score": np.random.random(300),
        }
    )

    result = bootstrap_sample(data, categories=["runs"])

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


def test_bootstrap_time_buckets() -> None:
    """Test that bootstrapping time buckets works correctly"""
    data = pd.DataFrame(
        {"human_minutes": np.geomspace(1, 100, 20), "score": np.random.random(20)}
    )

    result = bootstrap_sample(data, categories=["time_buckets"])

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
        bootstrap_sample(data, categories=["task_family", "task_id"])


def test_num_dropped_groups() -> None:
    """
    After the first level, we shouldn't drop any top-level groups by dropping their subcategories.
    So the fraction dropped should be 1/e.
    """
    data = pd.DataFrame(
        {
            "l1": np.arange(1000),
            "l2": np.arange(1000),
            "l3": np.arange(1000),
        }
    )

    one_level_result = bootstrap_sample(data, categories=["l1"])
    three_level_result = bootstrap_sample(data, categories=["l1", "l2", "l3"])
    one_level_uniques = one_level_result.l1.nunique()
    three_level_uniques = three_level_result.l3.nunique()

    reference_uniques = 1000 * (1 - 1 / np.e)

    assert (
        reference_uniques - 30 < one_level_uniques < reference_uniques + 30
    ), f"1-level bootstrapping should keep about {reference_uniques} groups; actual {one_level_uniques}"

    assert (
        reference_uniques - 30 < three_level_uniques < reference_uniques + 30
    ), f"3-level bootstrapping should keep about {reference_uniques} groups; actual {three_level_uniques}"


# %%
