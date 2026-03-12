"""Deterministic active-learning sampling for pending entity-resolution reviews."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .review import ReviewQueueItem, ReviewState


@dataclass(frozen=True, slots=True)
class ActiveLearningSelection:
    queue_item_id: str
    listing_id: str
    product_id: str
    category: str
    model_probability: float
    priority_score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "queue_item_id": self.queue_item_id,
            "listing_id": self.listing_id,
            "product_id": self.product_id,
            "category": self.category,
            "model_probability": self.model_probability,
            "priority_score": self.priority_score,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ActiveLearningBatch:
    selections: tuple[ActiveLearningSelection, ...]
    model_version: str
    data_version: str
    operating_threshold: float
    candidate_count: int
    selection_policy: str = "uncertainty_boundary_conflict_diversity_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "data_version": self.data_version,
            "operating_threshold": self.operating_threshold,
            "candidate_count": self.candidate_count,
            "selected_count": len(self.selections),
            "selection_policy": self.selection_policy,
            "label_policy": "unlabelled review selection; no labels are inferred",
            "selections": [selection.to_dict() for selection in self.selections],
        }


def _entropy(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    return -(
        probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability)
    )


def _boundary_proximity(probability: float, threshold: float) -> float:
    scale = max(threshold, 1.0 - threshold, 1e-9)
    return max(0.0, 1.0 - abs(probability - threshold) / scale)


def _stable_tie_breaker(queue_item_id: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{queue_item_id}".encode()).hexdigest()


def _score_item(
    item: ReviewQueueItem,
    probability: float,
    *,
    operating_threshold: float,
    manual_review_threshold: float,
) -> tuple[float, tuple[str, ...]]:
    uncertainty = _entropy(probability)
    boundary = _boundary_proximity(probability, operating_threshold)
    conflict_disagreement = float(bool(item.conflicts) and probability >= manual_review_threshold)
    # Blocking ambiguity supplements model uncertainty before a robust model exists.
    blocking_ambiguity = 1.0 - abs(item.blocking_score - 0.5) * 2.0
    priority = (
        0.40 * uncertainty
        + 0.30 * boundary
        + 0.20 * conflict_disagreement
        + 0.10 * max(0.0, blocking_ambiguity)
    )
    reasons: list[str] = []
    if conflict_disagreement:
        reasons.append("conflict_model_disagreement")
    if abs(probability - operating_threshold) <= 0.10:
        reasons.append("operating_threshold_boundary")
    if uncertainty >= 0.90:
        reasons.append("high_predictive_entropy")
    if 0.35 <= item.blocking_score <= 0.65:
        reasons.append("ambiguous_blocking_score")
    if not reasons:
        reasons.append("diversity_exploration")
    return min(1.0, priority), tuple(reasons)


def sample_active_learning(
    items: Sequence[ReviewQueueItem],
    probabilities: Mapping[str, float],
    *,
    limit: int,
    model_version: str,
    data_version: str,
    operating_threshold: float = 0.98,
    manual_review_threshold: float = 0.80,
    max_per_listing: int = 2,
    seed: int = 20260722,
) -> ActiveLearningBatch:
    """Select pending candidates without assigning or inferring any match labels.

    Greedy diversity penalties spread the batch across categories and listings while the
    base score prioritises model uncertainty, the precision-first operating boundary, and
    disagreements with deterministic conflict gates.
    """

    if limit < 1 or max_per_listing < 1:
        raise ValueError("limit and max_per_listing must be positive")
    if not model_version.strip() or not data_version.strip():
        raise ValueError("model_version and data_version must not be empty")
    if not 0.0 <= manual_review_threshold <= operating_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= manual <= operating <= 1")
    pending = [item for item in items if item.state is ReviewState.PENDING]
    missing = [item.queue_item_id for item in pending if item.queue_item_id not in probabilities]
    if missing:
        raise ValueError(f"missing probabilities for {len(missing)} pending candidates")

    base: dict[str, tuple[float, tuple[str, ...]]] = {}
    for item in pending:
        probability = float(probabilities[item.queue_item_id])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid probability for {item.queue_item_id}")
        base[item.queue_item_id] = _score_item(
            item,
            probability,
            operating_threshold=operating_threshold,
            manual_review_threshold=manual_review_threshold,
        )

    remaining = list(pending)
    selected: list[ActiveLearningSelection] = []
    category_counts: Counter[str] = Counter()
    listing_counts: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        eligible = [
            item for item in remaining if listing_counts[item.listing.listing_id] < max_per_listing
        ]
        if not eligible:
            break

        def ordering(item: ReviewQueueItem) -> tuple[float, str]:
            base_score = base[item.queue_item_id][0]
            diversity_penalty = min(0.25, 0.05 * category_counts[item.listing.category])
            adjusted = base_score - diversity_penalty
            return (-adjusted, _stable_tie_breaker(item.queue_item_id, seed=seed))

        chosen = min(eligible, key=ordering)
        probability = float(probabilities[chosen.queue_item_id])
        base_score, reasons = base[chosen.queue_item_id]
        diversity_penalty = min(0.25, 0.05 * category_counts[chosen.listing.category])
        selected.append(
            ActiveLearningSelection(
                queue_item_id=chosen.queue_item_id,
                listing_id=chosen.listing.listing_id,
                product_id=chosen.product.product_id,
                category=chosen.listing.category,
                model_probability=probability,
                priority_score=max(0.0, base_score - diversity_penalty),
                reasons=reasons,
            )
        )
        listing_counts[chosen.listing.listing_id] += 1
        category_counts[chosen.listing.category] += 1
        remaining.remove(chosen)

    return ActiveLearningBatch(
        selections=tuple(selected),
        model_version=model_version,
        data_version=data_version,
        operating_threshold=operating_threshold,
        candidate_count=len(pending),
    )
