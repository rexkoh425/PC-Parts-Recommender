"""Entity-resolution metrics with explicit synthetic-data eligibility."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pc_build_recommender.evaluation.contracts import DataUseDeclaration

from .decision import MatchThresholds


@dataclass(frozen=True, slots=True)
class EntityResolutionEvaluation:
    """Precision-first offline metrics and their evidence eligibility."""

    sample_count: int
    positive_count: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    auto_match_precision: float
    auto_match_recall: float
    auto_match_count: int
    manual_review_count: int
    rejected_count: int
    brier_score: float
    roc_auc: float | None
    average_precision: float | None
    classification_threshold: float
    thresholds: MatchThresholds
    data_use: DataUseDeclaration

    @property
    def eligible_for_promotion(self) -> bool:
        """Whether these metrics can support a measured model claim."""

        return self.data_use.eligible_for_reported_metrics

    @property
    def is_promotable(self) -> bool:
        return self.eligible_for_promotion

    @property
    def non_promotable_reason(self) -> str | None:
        return self.data_use.reporting_block_reason

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "auto_match_precision": self.auto_match_precision,
            "auto_match_recall": self.auto_match_recall,
            "auto_match_count": self.auto_match_count,
            "manual_review_count": self.manual_review_count,
            "rejected_count": self.rejected_count,
            "brier_score": self.brier_score,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "classification_threshold": self.classification_threshold,
            "thresholds": self.thresholds.to_dict(),
            "synthetic_data": self.data_use.to_dict(),
            "eligible_for_promotion": self.eligible_for_promotion,
            "non_promotable_reason": self.non_promotable_reason,
        }


def _safe_positive_metric(
    function: object,
    labels: NDArray[np.int64],
    predictions: NDArray[np.bool_],
) -> float:
    return float(function(labels, predictions, zero_division=0))  # type: ignore[operator]


def evaluate_binary_predictions(
    labels: Sequence[int] | NDArray[np.int_],
    probabilities: Sequence[float] | NDArray[np.float64],
    *,
    hard_conflicts: Sequence[bool] | NDArray[np.bool_] | None = None,
    is_synthetic: Sequence[bool] | NDArray[np.bool_] | None = None,
    include_synthetic: bool = False,
    classification_threshold: float = 0.5,
    thresholds: MatchThresholds | None = None,
) -> EntityResolutionEvaluation:
    """Evaluate probabilities after applying the same hard gate used in serving.

    Synthetic rows are excluded by default.  Callers may include them for engineering
    smoke tests, but the returned result is then explicitly ineligible for promotion or
    résumé claims.
    """

    if not 0.0 <= classification_threshold <= 1.0:
        raise ValueError("classification_threshold must be between zero and one")
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if targets.shape != scores.shape or targets.size == 0:
        raise ValueError("labels and probabilities must have equal non-zero length")
    if not set(np.unique(targets)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")

    conflicts = (
        np.zeros(targets.shape, dtype=np.bool_)
        if hard_conflicts is None
        else np.asarray(hard_conflicts, dtype=np.bool_).reshape(-1)
    )
    synthetic = (
        np.zeros(targets.shape, dtype=np.bool_)
        if is_synthetic is None
        else np.asarray(is_synthetic, dtype=np.bool_).reshape(-1)
    )
    if conflicts.shape != targets.shape or synthetic.shape != targets.shape:
        raise ValueError("hard_conflicts and is_synthetic must align with labels")

    data_use = DataUseDeclaration.from_flags(
        synthetic.tolist(), include_synthetic=include_synthetic
    )
    mask = np.ones(targets.shape, dtype=np.bool_) if include_synthetic else ~synthetic
    if not np.any(mask):
        raise ValueError(
            "no evaluable non-synthetic rows; set include_synthetic=True for a "
            "non-promotable engineering metric"
        )
    targets = targets[mask]
    scores = scores[mask]
    conflicts = conflicts[mask]

    # Hard conflicts are rejected regardless of a model's calibrated confidence.
    effective_scores = np.where(conflicts, 0.0, scores)
    positive_predictions = effective_scores >= classification_threshold
    policy = thresholds or MatchThresholds()
    automatic = effective_scores >= policy.auto_match
    manual = (
        (effective_scores >= policy.manual_review) & (effective_scores < policy.auto_match)
    )
    rejected = ~(automatic | manual)

    has_both_classes = len(np.unique(targets)) == 2
    return EntityResolutionEvaluation(
        sample_count=int(targets.size),
        positive_count=int(targets.sum()),
        precision=_safe_positive_metric(precision_score, targets, positive_predictions),
        recall=_safe_positive_metric(recall_score, targets, positive_predictions),
        f1=_safe_positive_metric(f1_score, targets, positive_predictions),
        accuracy=float(accuracy_score(targets, positive_predictions)),
        auto_match_precision=_safe_positive_metric(precision_score, targets, automatic),
        auto_match_recall=_safe_positive_metric(recall_score, targets, automatic),
        auto_match_count=int(automatic.sum()),
        manual_review_count=int(manual.sum()),
        rejected_count=int(rejected.sum()),
        brier_score=float(brier_score_loss(targets, effective_scores)),
        roc_auc=float(roc_auc_score(targets, effective_scores)) if has_both_classes else None,
        average_precision=(
            float(average_precision_score(targets, effective_scores))
            if has_both_classes
            else None
        ),
        classification_threshold=classification_threshold,
        thresholds=policy,
        data_use=data_use,
    )
