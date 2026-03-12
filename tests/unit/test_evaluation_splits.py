from __future__ import annotations

import pytest

from pc_build_recommender.evaluation import (
    assert_group_disjoint,
    deterministic_group_split,
    split_indices,
)


def test_group_split_is_deterministic_balanced_and_disjoint() -> None:
    groups = [f"family-{index}" for index in range(30)]

    first = deterministic_group_split(groups, seed=17)
    reordered = deterministic_group_split(reversed(groups), seed=17)

    assert first.assignments == reordered.assignments
    assert first.group_counts() == {"train": 18, "validation": 6, "test": 6}

    row_groups = [group for group in groups for _ in range(2)]
    indices = split_indices(row_groups, first)
    assert sum(len(rows) for rows in indices.values()) == 60
    for group in groups:
        row_splits = {
            split_name
            for split_name, rows in indices.items()
            if any(row_groups[index] == group for index in rows)
        }
        assert len(row_splits) == 1


def test_group_split_supports_group_level_strata() -> None:
    groups = [f"cpu-{index}" for index in range(6)] + [f"gpu-{index}" for index in range(6)]
    strata = {group: group.split("-", maxsplit=1)[0] for group in groups}

    split = deterministic_group_split(groups, seed=8, strata=strata)

    for category in ("cpu", "gpu"):
        category_splits = {split.split_for(group) for group in groups if group.startswith(category)}
        assert category_splits == {"train", "validation", "test"}


def test_group_disjoint_check_rejects_leakage() -> None:
    with pytest.raises(ValueError, match="leaks across splits"):
        assert_group_disjoint(["family-a", "family-a"], ["train", "test"])


def test_group_split_rejects_empty_explicit_weights() -> None:
    with pytest.raises(ValueError, match="at least one split weight"):
        deterministic_group_split(["family-a"], weights={})


def test_different_seed_changes_assignment() -> None:
    groups = [f"family-{index}" for index in range(20)]

    assert (
        deterministic_group_split(groups, seed=1).assignments
        != deterministic_group_split(groups, seed=2).assignments
    )
