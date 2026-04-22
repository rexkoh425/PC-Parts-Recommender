"""Deterministic feature construction for heuristic and learned rankers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pc_build_recommender.retrieval.text import tokenize

from .models import ScoredCandidate, RankingContext

FloatFeatureMatrix = NDArray[np.float64]


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _unit(value: Any, default: float = 0.0) -> float:
    number = _number(value, default)
    if number > 1.0:
        number /= 100.0
    return min(1.0, max(0.0, number))


def _first(candidate: ScoredCandidate, *names: str) -> Any:
    for name in names:
        value = candidate.get(name)
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    query_id: str
    product_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: FloatFeatureMatrix

# TODO: rest of this module still to come.
