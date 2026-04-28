"""Precision-first calibration, grouped splitting, and human-label evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pc_build_recommender.evaluation.contracts import EvaluationResult
from pc_build_recommender.evaluation.metrics import (
    evaluate_entity_resolution,
    wilson_confidence_interval,
)
from pc_build_recommender.evaluation.splits import deterministic_group_split

from .models import BaseEntityResolver
from .records import PairExample
from .review import ReviewQueue, ReviewQueueItem


@dataclass(frozen=True, slots=True)
class PrecisionFirstOperatingPoint:
    """A validation-selected threshold, or a fail-closed result when evidence is weak."""

    threshold: float | None
    precision: float
    recall: float
    f1: float
    precision_ci_lower: float | None
    true_positives: int
    false_positives: int
    false_negatives: int
    predicted_matches: int
    minimum_precision: float
    minimum_predicted_matches: int
    require_precision_ci_lower_bound: bool
    target_met: bool
    selection_policy: str = "max_recall_subject_to_precision_and_support_v1"

    @property
    def deployment_eligible(self) -> bool:
        return self.target_met and self.threshold is not None

    def require_deployable_threshold(self) -> float:
        if not self.deployment_eligible or self.threshold is None:
            raise RuntimeError("validation evidence did not satisfy the precision-first gate")
        return self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "precision_ci_lower": self.precision_ci_lower,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "predicted_matches": self.predicted_matches,
            "minimum_precision": self.minimum_precision,
            "minimum_predicted_matches": self.minimum_predicted_matches,
            "require_precision_ci_lower_bound": self.require_precision_ci_lower_bound,
            "target_met": self.target_met,
            "deployment_eligible": self.deployment_eligible,
            "selection_policy": self.selection_policy,
        }


@dataclass(frozen=True, slots=True)
class GroupedHumanEvaluation:
    evaluation: EvaluationResult
    listing_group_count: int
    human_label_count: int
    source_policy_reportable: bool
    evidence_scope: str

    @property
    def reportable(self) -> bool:
        return (
            self.source_policy_reportable and self.evaluation.data_use.eligible_for_reported_metrics
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation": self.evaluation.to_dict(),
            "listing_group_count": self.listing_group_count,
            "human_label_count": self.human_label_count,
            "source_policy_reportable": self.source_policy_reportable,
            "evidence_scope": self.evidence_scope,
            "reportable": self.reportable,
            "label_source": "attributable_human_reviews",
        }


def _binary_counts(
    labels: NDArray[np.int64], scores: NDArray[np.float64], threshold: float
) -> tuple[int, int, int]:
    predictions = scores >= threshold
    true_positives = int(np.sum((labels == 1) & predictions))
    false_positives = int(np.sum((labels == 0) & predictions))
    false_negatives = int(np.sum((labels == 1) & ~predictions))
    return true_positives, false_positives, false_negatives


def _candidate_operating_point(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    threshold: float,
    *,
    minimum_precision: float,
    minimum_predicted_matches: int,
    require_precision_ci_lower_bound: bool,
    confidence_level: float,
) -> PrecisionFirstOperatingPoint:
    true_positives, false_positives, false_negatives = _binary_counts(labels, scores, threshold)
    predicted_matches = true_positives + false_positives
    actual_matches = true_positives + false_negatives
    precision = true_positives / predicted_matches if predicted_matches else 0.0
    recall = true_positives / actual_matches if actual_matches else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    interval = wilson_confidence_interval(
        true_positives, predicted_matches, confidence_level=confidence_level
    )
    lower = interval[0] if interval is not None else None
    precision_gate = (
        lower is not None and lower >= minimum_precision
        if require_precision_ci_lower_bound
        else precision >= minimum_precision
    )
    target_met = predicted_matches >= minimum_predicted_matches and precision_gate
    return PrecisionFirstOperatingPoint(
        threshold=threshold if target_met else None,
        precision=precision,
        recall=recall,
        f1=f1,
        precision_ci_lower=lower,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        predicted_matches=predicted_matches,
        minimum_precision=minimum_precision,
        minimum_predicted_matches=minimum_predicted_matches,
        require_precision_ci_lower_bound=require_precision_ci_lower_bound,
        target_met=target_met,
    )


def select_precision_first_threshold(
    labels: Sequence[int] | NDArray[np.int64],
    probabilities: Sequence[float] | NDArray[np.float64],
    *,
    minimum_precision: float = 0.99,
    minimum_predicted_matches: int = 25,
    require_precision_ci_lower_bound: bool = False,
    confidence_level: float = 0.95,
) -> PrecisionFirstOperatingPoint:
    """Maximise validation recall subject to precision and evidence-support gates.

    If no threshold meets the gate, ``threshold`` is ``None``. The best diagnostic point
    remains visible, but callers must not silently deploy it.
    """

    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if targets.shape != scores.shape or not targets.size:
        raise ValueError("labels and probabilities must have equal non-zero length")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("threshold selection requires both binary labels")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    if minimum_predicted_matches < 1:
        raise ValueError("minimum_predicted_matches must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    candidates = [
        _candidate_operating_point(
            targets,
            scores,
            float(threshold),
            minimum_precision=minimum_precision,
            minimum_predicted_matches=minimum_predicted_matches,
            require_precision_ci_lower_bound=require_precision_ci_lower_bound,
            confidence_level=confidence_level,
        )
        for threshold in sorted(set(scores.tolist()), reverse=True)
    ]
    eligible = [candidate for candidate in candidates if candidate.target_met]
    if eligible:
        return max(
            eligible,
            key=lambda candidate: (
                candidate.recall,
                candidate.precision,
                candidate.threshold or 0.0,
            ),
        )

    # A non-deployable diagnostic explains how close validation came to the gate.
    diagnostic = max(
        candidates,
        key=lambda candidate: (
            candidate.precision_ci_lower
            if require_precision_ci_lower_bound and candidate.precision_ci_lower is not None
            else candidate.precision,
            candidate.predicted_matches >= minimum_predicted_matches,
            candidate.recall,
        ),
    )
    return PrecisionFirstOperatingPoint(
        threshold=None,
        precision=diagnostic.precision,
        recall=diagnostic.recall,
        f1=diagnostic.f1,
        precision_ci_lower=diagnostic.precision_ci_lower,
        true_positives=diagnostic.true_positives,
        false_positives=diagnostic.false_positives,
        false_negatives=diagnostic.false_negatives,
        predicted_matches=diagnostic.predicted_matches,
        minimum_precision=minimum_precision,
        minimum_predicted_matches=minimum_predicted_matches,
        require_precision_ci_lower_bound=require_precision_ci_lower_bound,
        target_met=False,
    )


def _human_pairs(items: Sequence[ReviewQueueItem]) -> tuple[PairExample, ...]:
    if not items:
        raise ValueError("at least one reviewed item is required")
    return tuple(item.to_pair_example() for item in items)


def calibrate_and_select_from_human_reviews(
    resolver: BaseEntityResolver,
    calibration_items: Sequence[ReviewQueueItem],
    threshold_selection_items: Sequence[ReviewQueueItem],
    *,
    minimum_precision: float = 0.99,
    minimum_predicted_matches: int = 25,
    require_precision_ci_lower_bound: bool = False,
) -> PrecisionFirstOperatingPoint:
    """Fit calibration and select the operating point on disjoint listing groups."""

    calibration_pairs = _human_pairs(calibration_items)
    selection_pairs = _human_pairs(threshold_selection_items)
    calibration_groups = {pair.listing.listing_id for pair in calibration_pairs}
    selection_groups = {pair.listing.listing_id for pair in selection_pairs}
    overlap = calibration_groups & selection_groups
    if overlap:
        raise ValueError(
            f"calibration and threshold selection overlap {len(overlap)} listing groups"
        )
    resolver.fit_calibrator(calibration_pairs)
    probabilities = resolver.predict_proba(selection_pairs)
    return select_precision_first_threshold(
        [pair.label for pair in selection_pairs],
        probabilities,
        minimum_precision=minimum_precision,
        minimum_predicted_matches=minimum_predicted_matches,
        require_precision_ci_lower_bound=require_precision_ci_lower_bound,
    )


def split_human_review_queue(
    queue: ReviewQueue,
    *,
    weights: Mapping[str, float] | None = None,
    seed: int = 20260722,
) -> dict[str, tuple[ReviewQueueItem, ...]]:
    """Split attributable binary labels without separating one listing's candidates."""

    if not queue.source_policy.training_eligible:
        raise PermissionError("queue source policy forbids model training")
    items = tuple(item for item in queue.items if item.is_binary_human_label)
    if not items:
        raise ValueError("queue has no binary human labels")
    group_ids = [item.listing.listing_id for item in items]
    split = deterministic_group_split(
        group_ids,
        weights=weights or {"train": 0.60, "calibration": 0.15, "threshold": 0.10, "test": 0.15},
        seed=seed,
    )
    result: dict[str, list[ReviewQueueItem]] = {name: [] for name in split.weights}
    for item in items:
        result[split.split_for(item.listing.listing_id)].append(item)
    return {name: tuple(rows) for name, rows in result.items()}


def evaluate_grouped_human_reviews(
    items: Sequence[ReviewQueueItem],
    probabilities: Sequence[float] | NDArray[np.float64],
    *,
    threshold: float,
    source_policy_reportable: bool,
    evidence_scope: str,
    n_resamples: int = 1_000,
    seed: int = 20260722,
) -> GroupedHumanEvaluation:
    """Evaluate with listing-level bootstrap groups and explicit source reportability."""

    pairs = _human_pairs(items)
    if len(probabilities) != len(pairs):
        raise ValueError("probabilities must contain one score per reviewed item")
    groups = [pair.listing.listing_id for pair in pairs]
    result = evaluate_entity_resolution(
        [pair.label for pair in pairs],
        [float(value) for value in probabilities],
        threshold=threshold,
        is_synthetic=[pair.is_synthetic for pair in pairs],
        groups=groups,
        n_resamples=n_resamples,
        seed=seed,
    )
    metadata = {
        **result.metadata,
        "label_source": "attributable_human_reviews",
        "bootstrap_unit": "listing_id",
        "listing_group_count": len(set(groups)),
        "source_policy_reportable": source_policy_reportable,
        "evidence_scope": evidence_scope,
    }
    return GroupedHumanEvaluation(
        evaluation=EvaluationResult(
            metrics=result.metrics,
            data_use=result.data_use,
            metadata=metadata,
        ),
        listing_group_count=len(set(groups)),
        human_label_count=len(pairs),
        source_policy_reportable=source_policy_reportable,
        evidence_scope=evidence_scope,
    )


def evaluate_grouped_review_queue(
    queue: ReviewQueue,
    probabilities: Sequence[float] | NDArray[np.float64],
    *,
    threshold: float,
    n_resamples: int = 1_000,
    seed: int = 20260722,
) -> GroupedHumanEvaluation:
    """Evaluate a queue without allowing callers to override its source-use policy."""

    items = tuple(item for item in queue.items if item.is_binary_human_label)
    return evaluate_grouped_human_reviews(
        items,
        probabilities,
        threshold=threshold,
        source_policy_reportable=queue.source_policy.published_metrics_eligible,
        evidence_scope=queue.source_policy.scope_note,
        n_resamples=n_resamples,
        seed=seed,
    )
