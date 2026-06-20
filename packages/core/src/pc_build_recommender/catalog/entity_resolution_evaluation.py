"""Replay the deployed catalogue matcher on frozen listing-level human labels."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from pc_build_recommender.entity_resolution import (
    ER_CATALOG_MATCHER_DECISION_VERSION,
    EntityResolutionPolicy,
    EntityResolutionRuntime,
    FrozenListingLabelSet,
)

from .entity_matcher import CatalogEntityMatcher
from .mapping_review import MappingOutcome

ER_LISTING_EVALUATION_EVIDENCE_SCHEMA_VERSION = (
    "pc-build-recommender.er-listing-evaluation-evidence.v1"
)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _wilson_interval(
    numerator: int,
    denominator: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if denominator <= 0:
        return 0.0, 0.0
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    proportion = numerator / denominator
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z_score * z_score
    denominator_term = 1.0 + z_squared / denominator
    centre = (proportion + z_squared / (2.0 * denominator)) / denominator_term
    spread = (
        z_score
        * math.sqrt(
            proportion * (1.0 - proportion) / denominator
            + z_squared / (4.0 * denominator * denominator)
        )
        / denominator_term
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def frozen_listing_groups_sha256(labels: FrozenListingLabelSet) -> str:
    """Match the training command's stable held-out listing-group commitment."""

    listing_ids = sorted(group.listing.listing_id for group in labels.listing_groups)
    return hashlib.sha256("\n".join(listing_ids).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ListingMatcherEvaluation:
    dataset_version: str
    label_dataset_sha256: str
    source_review_queue_sha256: str
    frozen_listing_groups_sha256: str
    model_release_sha256: str
    model_version: str
    matcher_decision_version: str
    labelled_pair_count: int
    listing_count: int
    independent_reviewer_count: int
    canonical_catalogue_version: str
    canonical_catalogue_sha256: str
    canonical_catalogue_file_sha256: str
    canonical_catalogue_product_count: int
    in_catalogue_listing_count: int
    unmatched_listing_count: int
    candidate_blocking_hits: int
    candidate_blocking_denominator: int
    candidate_blocking_recall: float
    winner_selection_correct: int
    winner_selection_denominator: int
    winner_selection_accuracy: float
    auto_match_correct: int
    auto_match_count: int
    auto_match_precision: float
    auto_match_precision_ci_lower: float
    auto_match_precision_ci_upper: float
    auto_match_recall: float
    auto_match_f1: float
    anchor_auto_match_count: int
    model_route_listing_count: int
    model_in_catalogue_listing_count: int
    model_unmatched_listing_count: int
    model_hard_negative_pair_count: int
    model_hard_negative_listing_count: int
    model_auto_match_correct: int
    model_auto_match_count: int
    model_auto_match_precision: float
    model_auto_match_precision_ci_lower: float
    model_auto_match_precision_ci_upper: float
    model_auto_match_recall: float
    model_auto_match_f1: float
    ambiguity_case_count: int
    ambiguity_deferred_count: int
    ambiguity_false_auto_match_count: int
    outcome_counts: Mapping[str, int]
    decision_rows: tuple[Mapping[str, Any], ...]
    promotion_blockers: tuple[str, ...]

    @property
    def promotion_eligible(self) -> bool:
        return not self.promotion_blockers

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": ER_LISTING_EVALUATION_EVIDENCE_SCHEMA_VERSION,
            "dataset_version": self.dataset_version,
            "label_dataset_sha256": self.label_dataset_sha256,
            "source_review_queue_sha256": self.source_review_queue_sha256,
            "frozen_listing_groups_sha256": self.frozen_listing_groups_sha256,
            "model_release_sha256": self.model_release_sha256,
            "model_version": self.model_version,
            "matcher_decision_version": self.matcher_decision_version,
            "labelled_pair_count": self.labelled_pair_count,
            "listing_count": self.listing_count,
            "independent_reviewer_count": self.independent_reviewer_count,
            "canonical_catalogue": {
                "version": self.canonical_catalogue_version,
                "sha256": self.canonical_catalogue_sha256,
                "file_sha256": self.canonical_catalogue_file_sha256,
                "product_count": self.canonical_catalogue_product_count,
            },
            "listing_coverage": {
                "in_catalogue": self.in_catalogue_listing_count,
                "unmatched": self.unmatched_listing_count,
            },
            "candidate_blocking": {
                "hits": self.candidate_blocking_hits,
                "denominator": self.candidate_blocking_denominator,
                "recall": self.candidate_blocking_recall,
            },
            "winner_selection": {
                "correct": self.winner_selection_correct,
                "denominator": self.winner_selection_denominator,
                "accuracy": self.winner_selection_accuracy,
            },
            "automatic_matches": {
                "correct": self.auto_match_correct,
                "count": self.auto_match_count,
                "precision": self.auto_match_precision,
                "precision_ci_lower": self.auto_match_precision_ci_lower,
                "precision_ci_upper": self.auto_match_precision_ci_upper,
                "recall": self.auto_match_recall,
                "f1": self.auto_match_f1,
            },
            "model_only": {
                "route_listing_count": self.model_route_listing_count,
                "in_catalogue_listing_count": self.model_in_catalogue_listing_count,
                "unmatched_listing_count": self.model_unmatched_listing_count,
                "hard_negative_pair_count": self.model_hard_negative_pair_count,
                "hard_negative_listing_count": self.model_hard_negative_listing_count,
                "automatic_match_correct": self.model_auto_match_correct,
                "automatic_match_count": self.model_auto_match_count,
                "precision": self.model_auto_match_precision,
                "precision_ci_lower": self.model_auto_match_precision_ci_lower,
                "precision_ci_upper": self.model_auto_match_precision_ci_upper,
                "recall": self.model_auto_match_recall,
                "f1": self.model_auto_match_f1,
            },
            "anchor_auto_match_count": self.anchor_auto_match_count,
            "ambiguity_margin": {
                "case_count": self.ambiguity_case_count,
                "deferred_count": self.ambiguity_deferred_count,
                "false_auto_match_count": self.ambiguity_false_auto_match_count,
            },
            "outcome_counts": dict(self.outcome_counts),
            "promotion_eligible": self.promotion_eligible,
            "promotion_blockers": list(self.promotion_blockers),
        }


def evaluate_listing_matcher(
    labels: FrozenListingLabelSet,
    runtime: EntityResolutionRuntime,
    policy: EntityResolutionPolicy,
) -> ListingMatcherEvaluation:
    """Evaluate blocking, winner choice, and ambiguity handling on frozen labels.

    The runtime may be a human-trained diagnostic artifact.  The matcher exposes its raw
    authorization-independent decision trace, while still refusing to persist an automatic
    mapping in shadow mode.  This evaluator never changes the model or creates labels.
    """

    matcher = CatalogEntityMatcher(labels.products, runtime=runtime, **policy.matcher_kwargs())
    decision_rows: list[Mapping[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    blocking_hits = 0
    winner_correct = 0
    auto_match_correct = 0
    auto_match_count = 0
    ambiguity_cases = 0
    ambiguity_deferred = 0
    ambiguity_false_auto = 0
    in_catalogue_count = 0
    unmatched_count = 0
    anchor_auto_count = 0
    model_route_count = 0
    model_in_catalogue_count = 0
    model_unmatched_count = 0
    model_hard_negative_pairs = 0
    model_hard_negative_listings = 0
    model_auto_correct = 0
    model_auto_count = 0

    for group in sorted(labels.listing_groups, key=lambda item: item.listing.listing_id):
        listing = group.listing
        gold_product_id = group.gold_product_id
        in_catalogue = gold_product_id is not None
        in_catalogue_count += int(in_catalogue)
        unmatched_count += int(not in_catalogue)
        blocked = matcher.blocker.candidates(listing, labels.products)
        blocked_ids = tuple(candidate.product.product_id for candidate in blocked)
        blocking_hit = gold_product_id is not None and gold_product_id in blocked_ids
        blocking_hits += int(blocking_hit)

        result = matcher.match(listing)
        evidence = dict(result.evidence)
        raw_winner = evidence.get("winner_product_id")
        winner_product_id = (
            result.matched_product_id
            if result.matched_product_id is not None
            else str(raw_winner)
            if isinstance(raw_winner, str) and raw_winner
            else None
        )
        winner_is_correct = winner_product_id == gold_product_id
        winner_correct += int(in_catalogue and winner_is_correct)
        model_routed = result.method == "lightgbm"
        model_route_count += int(model_routed)
        model_in_catalogue_count += int(model_routed and in_catalogue)
        model_unmatched_count += int(model_routed and not in_catalogue)
        labelled_non_matches = {
            pair.product_id
            for pair in group.pair_labels
            if pair.resolved_label.value == "NON_MATCH"
        }
        blocked_hard_negatives = tuple(
            product_id for product_id in blocked_ids if product_id in labelled_non_matches
        )
        if model_routed:
            model_hard_negative_pairs += len(blocked_hard_negatives)
            model_hard_negative_listings += int(bool(blocked_hard_negatives))
        would_auto_match = result.outcome is MappingOutcome.AUTO_MATCHED or bool(
            evidence.get("would_auto_match_if_authorized", False)
        )
        if would_auto_match:
            auto_match_count += 1
            auto_match_correct += int(winner_is_correct)
            if model_routed:
                model_auto_count += 1
                model_auto_correct += int(winner_is_correct)
            else:
                anchor_auto_count += 1

        raw_outcome = evidence.get("raw_threshold_outcome")
        margin = evidence.get("winner_margin")
        ambiguity_case = bool(
            raw_outcome == "AUTO_MATCH"
            and isinstance(margin, int | float)
            and not isinstance(margin, bool)
            and float(margin) < policy.minimum_auto_margin
        )
        if ambiguity_case:
            ambiguity_cases += 1
            deferred = not would_auto_match and result.outcome is MappingOutcome.MANUAL_REVIEW
            ambiguity_deferred += int(deferred)
            ambiguity_false_auto += int(would_auto_match)
        outcome_counts[result.outcome.value] += 1
        decision_rows.append(
            {
                "listing_id": listing.listing_id,
                "category": listing.category,
                "gold_product_id": gold_product_id,
                "match_disposition": group.match_disposition,
                "blocked_candidate_product_ids": list(blocked_ids),
                "blocking_hit": blocking_hit,
                "winner_product_id": winner_product_id,
                "winner_correct": winner_is_correct,
                "method": result.method,
                "model_routed": model_routed,
                "reviewed_blocked_hard_negative_product_ids": list(
                    blocked_hard_negatives
                ),
                "would_auto_match_if_authorized": would_auto_match,
                "outcome": result.outcome.value,
                "probability": result.probability,
                "winner_margin": margin,
                "ambiguity_margin_case": ambiguity_case,
                "candidate_product_ids": list(result.candidate_product_ids),
                "matcher_evidence": evidence,
            }
        )

    listing_count = len(labels.listing_groups)
    blocking_recall = _ratio(blocking_hits, in_catalogue_count)
    winner_accuracy = _ratio(winner_correct, in_catalogue_count)
    precision = _ratio(auto_match_correct, auto_match_count)
    recall = _ratio(auto_match_correct, in_catalogue_count)
    f1 = _f1(precision, recall)
    ci_lower, ci_upper = _wilson_interval(auto_match_correct, auto_match_count)
    model_precision = _ratio(model_auto_correct, model_auto_count)
    model_recall = _ratio(model_auto_correct, model_in_catalogue_count)
    model_f1 = _f1(model_precision, model_recall)
    model_ci_lower, model_ci_upper = _wilson_interval(
        model_auto_correct,
        model_auto_count,
    )
    groups_sha256 = frozen_listing_groups_sha256(labels)

    blockers: list[str] = []
    source_policy = labels.source_policy
    if labels.dataset_version != runtime.evidence.dataset_version:
        blockers.append("label dataset version does not match the trained model evidence")
    if labels.source_review_queue_sha256 != runtime.evidence.review_queue_sha256:
        blockers.append("label dataset does not bind the model's reviewed source queue")
    if groups_sha256 != runtime.evidence.frozen_test_groups_sha256:
        blockers.append("label listing groups do not match the model's frozen test groups")
    if not source_policy.training_eligible:
        blockers.append("label source policy forbids derived-model training")
    if not source_policy.published_metrics_eligible:
        blockers.append("label source policy forbids published metrics")
    if not source_policy.model_serving_eligible:
        blockers.append("label source policy forbids serving the derived model")
    if labels.labelled_pair_count < policy.minimum_labelled_pairs:
        blockers.append(
            f"independently reviewed pair count={labels.labelled_pair_count} below "
            f"production minimum={policy.minimum_labelled_pairs}"
        )
    if blocking_recall < policy.minimum_recall:
        blockers.append("candidate-blocking recall is below the production policy")
    if winner_accuracy < policy.minimum_recall:
        blockers.append("winner-selection accuracy is below the production policy")
    if auto_match_count < policy.minimum_auto_matches:
        blockers.append("automatic-match support is below the production policy")
    if precision < policy.minimum_precision:
        blockers.append("automatic-match precision is below the production policy")
    if ci_lower < policy.minimum_precision:
        blockers.append("automatic-match precision confidence lower bound is below policy")
    if recall < policy.minimum_recall:
        blockers.append("automatic-match recall is below the production policy")
    if f1 < policy.minimum_f1:
        blockers.append("automatic-match F1 is below the production policy")
    if model_auto_count < policy.minimum_auto_matches:
        blockers.append("model-only automatic-match support is below the production policy")
    if model_precision < policy.minimum_precision:
        blockers.append("model-only precision is below the production policy")
    if model_ci_lower < policy.minimum_precision:
        blockers.append("model-only precision confidence lower bound is below policy")
    if model_recall < policy.minimum_recall:
        blockers.append("model-only recall is below the production policy")
    if model_f1 < policy.minimum_f1:
        blockers.append("model-only F1 is below the production policy")
    if model_hard_negative_listings < policy.minimum_auto_matches:
        blockers.append("model-only reviewed hard-negative coverage is below policy")
    if model_unmatched_count < policy.minimum_auto_matches:
        blockers.append("model-only unmatched-listing coverage is below policy")
    if ambiguity_deferred != ambiguity_cases:
        blockers.append("not every below-margin automatic candidate was deferred")
    if ambiguity_false_auto:
        blockers.append("below-margin ambiguity produced an automatic match")

    return ListingMatcherEvaluation(
        dataset_version=labels.dataset_version,
        label_dataset_sha256=labels.dataset_sha256,
        source_review_queue_sha256=labels.source_review_queue_sha256,
        frozen_listing_groups_sha256=groups_sha256,
        model_release_sha256=runtime.release_sha256,
        model_version=runtime.model_version,
        matcher_decision_version=ER_CATALOG_MATCHER_DECISION_VERSION,
        labelled_pair_count=labels.labelled_pair_count,
        listing_count=listing_count,
        independent_reviewer_count=labels.independent_reviewer_count,
        canonical_catalogue_version=labels.canonical_catalogue_version,
        canonical_catalogue_sha256=labels.canonical_catalogue_sha256,
        canonical_catalogue_file_sha256=labels.canonical_catalogue_file_sha256,
        canonical_catalogue_product_count=len(labels.products),
        in_catalogue_listing_count=in_catalogue_count,
        unmatched_listing_count=unmatched_count,
        candidate_blocking_hits=blocking_hits,
        candidate_blocking_denominator=in_catalogue_count,
        candidate_blocking_recall=blocking_recall,
        winner_selection_correct=winner_correct,
        winner_selection_denominator=in_catalogue_count,
        winner_selection_accuracy=winner_accuracy,
        auto_match_correct=auto_match_correct,
        auto_match_count=auto_match_count,
        auto_match_precision=precision,
        auto_match_precision_ci_lower=ci_lower,
        auto_match_precision_ci_upper=ci_upper,
        auto_match_recall=recall,
        auto_match_f1=f1,
        anchor_auto_match_count=anchor_auto_count,
        model_route_listing_count=model_route_count,
        model_in_catalogue_listing_count=model_in_catalogue_count,
        model_unmatched_listing_count=model_unmatched_count,
        model_hard_negative_pair_count=model_hard_negative_pairs,
        model_hard_negative_listing_count=model_hard_negative_listings,
        model_auto_match_correct=model_auto_correct,
        model_auto_match_count=model_auto_count,
        model_auto_match_precision=model_precision,
        model_auto_match_precision_ci_lower=model_ci_lower,
        model_auto_match_precision_ci_upper=model_ci_upper,
        model_auto_match_recall=model_recall,
        model_auto_match_f1=model_f1,
        ambiguity_case_count=ambiguity_cases,
        ambiguity_deferred_count=ambiguity_deferred,
        ambiguity_false_auto_match_count=ambiguity_false_auto,
        outcome_counts=dict(sorted(outcome_counts.items())),
        decision_rows=tuple(decision_rows),
        promotion_blockers=tuple(blockers),
    )
