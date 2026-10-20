"""Rank-only fusion for heterogeneous retrieval systems."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .models import FusedHit, SearchHit


def _product_id(item: str | SearchHit) -> str:
    return item if isinstance(item, str) else item.product_id


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str | SearchHit]],
    *,
    k: int = 60,
    limit: int | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse ranked lists using ``sum(1 / (k + rank))``.

    Scores from BM25 and vector retrieval are intentionally ignored because
    their scales are not comparable. Optional positive source weights let a
    held-out-tuned policy rebalance sources without mixing raw score scales.
    Duplicate IDs within a source count only at their first rank. Ties are
    resolved by best source rank then product ID.
    """

    if k < 1:
        raise ValueError("k must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    active_weights = dict(weights or {})
    unknown_sources = set(active_weights) - set(rankings)
    if unknown_sources:
        raise ValueError(f"weights reference unknown ranking sources: {sorted(unknown_sources)}")
    if any(
        not math.isfinite(float(weight)) or float(weight) <= 0 for weight in active_weights.values()
    ):
        raise ValueError("source weights must be finite and positive")

    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    for source, items in rankings.items():
        seen: set[str] = set()
        for rank, item in enumerate(items, start=1):
            product_id = _product_id(item)
            if product_id in seen:
                continue
            seen.add(product_id)
            weight = float(active_weights.get(source, 1.0))
            scores[product_id] = scores.get(product_id, 0.0) + weight / (k + rank)
            source_ranks.setdefault(product_id, {})[source] = rank

    ordered = sorted(
        scores,
        key=lambda product_id: (
            -scores[product_id],
            min(source_ranks[product_id].values()),
            product_id,
        ),
    )
    if limit is not None:
        ordered = ordered[:limit]
    return [
        FusedHit(
            product_id=product_id,
            score=scores[product_id],
            rank=rank,
            source_ranks=source_ranks[product_id],
        )
        for rank, product_id in enumerate(ordered, start=1)
    ]
