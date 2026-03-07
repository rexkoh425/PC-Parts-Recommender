"""Deterministic group-level data splitting utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

DEFAULT_SPLIT_WEIGHTS: dict[str, float] = {
    "train": 0.6,
    "validation": 0.2,
    "test": 0.2,
}


def _normalise_for_hash(value: Hashable) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("group identifiers cannot contain non-finite floats")
        return ["float", value]
    if isinstance(value, tuple):
        return ["tuple", [_normalise_for_hash(item) for item in value]]
    raise TypeError(
        f"group identifiers must be JSON-stable scalars or tuples; received {type(value).__name__}"
    )


def _stable_digest(value: Hashable, *, seed: int) -> str:
    payload = json.dumps(
        [seed, _normalise_for_hash(value)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("at least one split weight is required")
    result: dict[str, float] = {}
    for name, weight in weights.items():
        if not name:
            raise ValueError("split names must not be empty")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("split weights must be finite and positive")
        result[name] = float(weight)
    return result


def _allocate_counts(size: int, weights: Mapping[str, float]) -> dict[str, int]:
    names = list(weights)
    total_weight = sum(weights.values())
    exact = {name: size * weights[name] / total_weight for name in names}
    counts = {name: math.floor(exact[name]) for name in names}
    remainder = size - sum(counts.values())
    order = sorted(names, key=lambda name: (-(exact[name] - counts[name]), names.index(name)))
    for name in order[:remainder]:
        counts[name] += 1

    if size >= len(names):
        empty_names = [name for name in names if counts[name] == 0]
        for empty_name in empty_names:
            donors = sorted(names, key=lambda name: (-counts[name], names.index(name)))
            donor = next(name for name in donors if counts[name] > 1)
            counts[donor] -= 1
            counts[empty_name] += 1
    return counts


@dataclass(frozen=True, slots=True)
class GroupSplit:
    """A stable assignment from leakage unit to split name."""

    assignments: dict[Hashable, str]
    weights: dict[str, float]
    seed: int

    def split_for(self, group_id: Hashable) -> str:
        try:
            return self.assignments[group_id]
        except KeyError as exc:
            raise KeyError(f"unknown group_id: {group_id!r}") from exc

    def row_assignments(self, group_ids: Sequence[Hashable]) -> list[str]:
        return [self.split_for(group_id) for group_id in group_ids]

    def group_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.weights}
        for split_name in self.assignments.values():
            counts[split_name] += 1
        return counts


def deterministic_group_split(
    group_ids: Iterable[Hashable],
    *,
    weights: Mapping[str, float] | None = None,
    seed: int = 20260722,
    strata: Mapping[Hashable, Hashable] | None = None,
) -> GroupSplit:
    """Assign every unique group to exactly one deterministic split.

    ``strata`` is a group-level mapping, not a row-level label. When supplied,
    allocation happens independently within each stratum while preserving group
    integrity. Callers should use model family, query-intent family, or another
    true leakage unit as ``group_ids``.
    """

    split_weights = _validate_weights(DEFAULT_SPLIT_WEIGHTS if weights is None else weights)
    unique_groups = list(dict.fromkeys(group_ids))
    if not unique_groups:
        raise ValueError("at least one group identifier is required")

    if strata is not None:
        missing = [group_id for group_id in unique_groups if group_id not in strata]
        if missing:
            raise ValueError(f"missing strata for {len(missing)} group identifiers")

    by_stratum: dict[Hashable | None, list[Hashable]] = defaultdict(list)
    for group_id in unique_groups:
        _normalise_for_hash(group_id)
        stratum = strata[group_id] if strata is not None else None
        if stratum is not None:
            _normalise_for_hash(stratum)
        by_stratum[stratum].append(group_id)

    assignments: dict[Hashable, str] = {}
    for stratum_groups in by_stratum.values():
        ordered = sorted(stratum_groups, key=lambda value: _stable_digest(value, seed=seed))
        counts = _allocate_counts(len(ordered), split_weights)
        cursor = 0
        for split_name in split_weights:
            next_cursor = cursor + counts[split_name]
            for group_id in ordered[cursor:next_cursor]:
                assignments[group_id] = split_name
            cursor = next_cursor

    return GroupSplit(assignments=assignments, weights=split_weights, seed=seed)


def assert_group_disjoint(group_ids: Sequence[Hashable], split_names: Sequence[str]) -> None:
    """Raise when one group appears in more than one supplied split."""

    if len(group_ids) != len(split_names):
        raise ValueError("group_ids and split_names must have the same length")
    observed: dict[Hashable, str] = {}
    for group_id, split_name in zip(group_ids, split_names, strict=True):
        previous = observed.setdefault(group_id, split_name)
        if previous != split_name:
            raise ValueError(
                f"group {group_id!r} leaks across splits {previous!r} and {split_name!r}"
            )


def split_indices(
    group_ids: Sequence[Hashable],
    split: GroupSplit,
) -> dict[str, list[int]]:
    """Return row indices for each split without separating group members."""

    result: dict[str, list[int]] = {name: [] for name in split.weights}
    assignments = split.row_assignments(group_ids)
    assert_group_disjoint(group_ids, assignments)
    for index, split_name in enumerate(assignments):
        result[split_name].append(index)
    return result
