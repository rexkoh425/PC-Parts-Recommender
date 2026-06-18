"""Shared precision-first entity matching for processed catalogue ingestion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from pc_build_recommender.entity_resolution import (
    CanonicalProductRecord,
    EntityResolutionRuntime,
    ListingRecord,
    MatchOutcome,
    PairFeatureExtractor,
    PCDomainCandidateBlocker,
    find_numeric_conflicts,
    normalize_identifier,
    normalize_text,
    tokenize,
)
from pc_build_recommender.entity_resolution.serving import (
    ER_CATALOG_MATCHER_DECISION_VERSION,
)

from .mapping_review import MappingOutcome

ENTITY_MATCHING_DECISION_VERSION = ER_CATALOG_MATCHER_DECISION_VERSION
_COLOUR_TOKENS = frozenset(
    {
        "black",
        "white",
        "silver",
        "gray",
        "grey",
        "red",
        "blue",
        "green",
        "orange",
        "pink",
        "purple",
        "brown",
    }
)


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _metadata_mapping(metadata: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = metadata.get(field)
    return value if isinstance(value, Mapping) else {}


def listing_record_from_offer(
    *,
    listing_id: str,
    title: str,
    category: str,
    retailer: str,
    current_price_sgd: float,
    metadata: Mapping[str, Any],
) -> ListingRecord:
    """Project governed offer metadata into the stable pair-feature contract."""

    identifiers = _metadata_mapping(metadata, "identifiers")
    attributes = _metadata_mapping(metadata, "attributes")
    return ListingRecord(
        listing_id=listing_id,
        title=title,
        category=category,
        brand=_optional_text(metadata.get("brand") or identifiers.get("brand")) or "",
        manufacturer_part_number=_optional_text(
            metadata.get("manufacturer_part_number")
            or metadata.get("mpn")
            or identifiers.get("mpn")
        ),
        gtin=_optional_text(metadata.get("gtin") or identifiers.get("gtin")),
        attributes=dict(attributes),
        current_price_sgd=current_price_sgd,
        retailer=retailer,
        is_synthetic=False,
    )


def _brand_matches(listing: ListingRecord, product: CanonicalProductRecord) -> bool:
    listing_brand = normalize_text(listing.brand)
    product_brand = normalize_text(product.brand)
    if listing_brand:
        return bool(product_brand and listing_brand == product_brand)
    title_tokens = tokenize(listing.title)
    brand_tokens = tokenize(product.brand)
    return bool(brand_tokens and title_tokens[: len(brand_tokens)] == brand_tokens)


def _colour_conflict(
    listing: ListingRecord,
    product: CanonicalProductRecord,
) -> bool:
    title_colours = _COLOUR_TOKENS.intersection(tokenize(listing.title))
    if len(title_colours) > 1:
        return True
    product_colour = _optional_text(product.attributes.get("colour"))
    if not title_colours or not product_colour:
        return False
    product_colours = _COLOUR_TOKENS.intersection(tokenize(product_colour))
    return bool(product_colours and title_colours.isdisjoint(product_colours))


def _hard_conflict_reasons(
    listing: ListingRecord,
    product: CanonicalProductRecord,
) -> tuple[str, ...]:
    reasons = [
        f"numeric_conflict:{conflict.field}"
        for conflict in find_numeric_conflicts(listing, product)
    ]
    if _colour_conflict(listing, product):
        reasons.append("colour_variant_conflict")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class CatalogMatchResult:
    """One listing-level decision with bounded, JSON-safe audit evidence."""

    outcome: MappingOutcome
    matched_product_id: str | None
    candidate_product_ids: tuple[str, ...]
    method: str | None
    reason: str | None
    probability: float | None
    model_version: str | None
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("entity-resolution probability must be between zero and one")
        if self.matched_product_id is not None and self.outcome not in {
            MappingOutcome.AUTO_MATCHED,
            MappingOutcome.REVIEWED_MATCHED,
        }:
            raise ValueError("only matched outcomes may select a canonical product")
        object.__setattr__(self, "evidence", _freeze_json_mapping(self.evidence))


class CatalogEntityMatcher:
    """Apply deterministic anchors, hard gates, and optional LightGBM scoring."""

    def __init__(
        self,
        products: Sequence[CanonicalProductRecord],
        *,
        runtime: EntityResolutionRuntime | None = None,
        max_candidates: int = 50,
        minimum_text_score: float = 0.12,
        minimum_auto_margin: float = 0.02,
        evidence_candidate_limit: int = 5,
    ) -> None:
        if not 0.0 <= minimum_auto_margin <= 1.0:
            raise ValueError("minimum_auto_margin must be between zero and one")
        if evidence_candidate_limit < 1:
            raise ValueError("evidence_candidate_limit must be positive")
        self.runtime = runtime
        self.minimum_auto_margin = minimum_auto_margin
        self.evidence_candidate_limit = evidence_candidate_limit
        if isinstance(runtime, EntityResolutionRuntime) and runtime.production_authorized:
            policy = runtime.release_policy
            if policy is None:
                raise ValueError("authorized entity-resolution runtime is missing its policy")
            configured = {
                "max_candidates": max_candidates,
                "minimum_text_score": minimum_text_score,
                "minimum_auto_margin": minimum_auto_margin,
                "evidence_candidate_limit": evidence_candidate_limit,
            }
            if configured != policy.matcher_kwargs():
                raise ValueError("catalogue matcher settings do not match the authorized ER policy")
        self.feature_extractor = PairFeatureExtractor()
        self.blocker = PCDomainCandidateBlocker(
            max_candidates=max_candidates,
            minimum_text_score=minimum_text_score,
        )
        by_category: dict[str, list[CanonicalProductRecord]] = defaultdict(list)
        for product in products:
            by_category[normalize_text(product.category).replace(" ", "_")].append(product)
        self.products_by_category = {
            category: tuple(sorted(items, key=lambda item: item.product_id))
            for category, items in by_category.items()
        }

    @property
    def model_version(self) -> str | None:
        return self.runtime.model_version if self.runtime is not None else None

    def _products(self, listing: ListingRecord) -> tuple[CanonicalProductRecord, ...]:
        category = normalize_text(listing.category).replace(" ", "_")
        return self.products_by_category.get(category, ())

    def _anchor_candidates(
        self,
        listing: ListingRecord,
    ) -> tuple[str | None, tuple[CanonicalProductRecord, ...], tuple[str, ...]]:
        products = self._products(listing)
        listing_gtin = normalize_identifier(listing.gtin)
        gtin_matches = tuple(
            product
            for product in products
            if listing_gtin
            and normalize_identifier(product.gtin)
            and listing_gtin == normalize_identifier(product.gtin)
        )

        listing_mpn = normalize_identifier(listing.manufacturer_part_number)
        title_identifier = normalize_identifier(listing.title)
        mpn_matches = tuple(
            product
            for product in products
            if _brand_matches(listing, product)
            and len(normalize_identifier(product.manufacturer_part_number)) >= 3
            and (
                (
                    bool(listing_mpn)
                    and listing_mpn == normalize_identifier(product.manufacturer_part_number)
                )
                or (
                    not listing_mpn
                    and len(normalize_identifier(product.manufacturer_part_number)) >= 6
                    and normalize_identifier(product.manufacturer_part_number) in title_identifier
                )
            )
        )
        gtin_ids = {product.product_id for product in gtin_matches}
        mpn_ids = {product.product_id for product in mpn_matches}
        if gtin_ids and mpn_ids and gtin_ids != mpn_ids:
            distinct = {product.product_id: product for product in gtin_matches + mpn_matches}
            candidates = tuple(
                sorted(
                    distinct.values(),
                    key=lambda item: item.product_id,
                )
            )
            return None, candidates, ("conflicting_exact_identifiers",)
        if gtin_matches:
            return "exact_gtin", gtin_matches, ()
        if mpn_matches:
            return "exact_mpn_brand", mpn_matches, ()
        return None, (), ()

    def _anchor_result(self, listing: ListingRecord) -> CatalogMatchResult | None:
        method, anchored, identifier_conflicts = self._anchor_candidates(listing)
        if identifier_conflicts:
            return CatalogMatchResult(
                outcome=MappingOutcome.AMBIGUOUS,
                matched_product_id=None,
                candidate_product_ids=tuple(item.product_id for item in anchored),
                method=None,
                reason="GTIN and brand plus MPN identify different canonical products",
                probability=None,
                model_version=None,
                evidence={
                    "decision_version": ENTITY_MATCHING_DECISION_VERSION,
                    "anchor_conflicts": list(identifier_conflicts),
                },
            )
        if not anchored or method is None:
            return None
        conflicts = {item.product_id: _hard_conflict_reasons(listing, item) for item in anchored}
        safe = tuple(item for item in anchored if not conflicts[item.product_id])
        evidence = {
            "decision_version": ENTITY_MATCHING_DECISION_VERSION,
            "anchor": method,
            "hard_conflicts": {
                product_id: list(reasons) for product_id, reasons in conflicts.items() if reasons
            },
        }
        if len(safe) == 1 and len(anchored) == 1:
            return CatalogMatchResult(
                outcome=MappingOutcome.AUTO_MATCHED,
                matched_product_id=safe[0].product_id,
                candidate_product_ids=(safe[0].product_id,),
                method=method,
                reason=None,
                probability=None,
                model_version=None,
                evidence=evidence,
            )
        if not safe:
            return CatalogMatchResult(
                outcome=MappingOutcome.HARD_CONFLICT,
                matched_product_id=None,
                candidate_product_ids=tuple(item.product_id for item in anchored),
                method=None,
                reason="exact identifier candidate has a hard variant conflict",
                probability=None,
                model_version=None,
                evidence=evidence,
            )
        return CatalogMatchResult(
            outcome=MappingOutcome.AMBIGUOUS,
            matched_product_id=None,
            candidate_product_ids=tuple(item.product_id for item in safe),
            method=None,
            reason=f"multiple safe {method} candidates",
            probability=None,
            model_version=None,
            evidence=evidence,
        )

    def match(self, listing: ListingRecord) -> CatalogMatchResult:
        """Resolve one offer without allowing ML to override exact or hard evidence."""

        anchor_result = self._anchor_result(listing)
        if anchor_result is not None:
            return anchor_result
        if self.runtime is None:
            return CatalogMatchResult(
                outcome=MappingOutcome.UNMATCHED,
                matched_product_id=None,
                candidate_product_ids=(),
                method=None,
                reason="no exact identifier anchor and no entity-resolution model configured",
                probability=None,
                model_version=None,
                evidence={"decision_version": ENTITY_MATCHING_DECISION_VERSION},
            )

        candidates = self.blocker.candidates(listing, self._products(listing))
        if not candidates:
            return CatalogMatchResult(
                outcome=MappingOutcome.MODEL_REJECTED,
                matched_product_id=None,
                candidate_product_ids=(),
                method="lightgbm",
                reason="candidate blocking returned no eligible canonical products",
                probability=None,
                model_version=self.runtime.model_version,
                evidence={
                    "decision_version": ENTITY_MATCHING_DECISION_VERSION,
                    "candidate_count": 0,
                },
            )

        safe_candidates = [
            candidate
            for candidate in candidates
            if not candidate.has_hard_conflict
            and not _colour_conflict(candidate.listing, candidate.product)
        ]
        if not safe_candidates:
            return CatalogMatchResult(
                outcome=MappingOutcome.HARD_CONFLICT,
                matched_product_id=None,
                candidate_product_ids=tuple(item.product.product_id for item in candidates),
                method="lightgbm",
                reason="every blocked candidate has a hard variant conflict",
                probability=0.0,
                model_version=self.runtime.model_version,
                evidence={
                    "decision_version": ENTITY_MATCHING_DECISION_VERSION,
                    "candidate_count": len(candidates),
                    "hard_conflicts": {
                        item.product.product_id: list(
                            _hard_conflict_reasons(item.listing, item.product)
                        )
                        for item in candidates[: self.evidence_candidate_limit]
                    },
                },
            )

        features = [
            self.feature_extractor.extract(item.listing, item.product) for item in safe_candidates
        ]
        matrix = np.vstack([item.as_array() for item in features])
        probabilities = self.runtime.resolver.predict_proba(matrix)
        scored = sorted(
            zip(safe_candidates, features, probabilities, strict=True),
            key=lambda item: (-float(item[2]), item[0].product.product_id),
        )
        winner, _, raw_probability = scored[0]
        probability = float(raw_probability)
        decision = self.runtime.resolver.thresholds.decide(probability)
        runner_up_probability = float(scored[1][2]) if len(scored) > 1 else None
        margin = probability - runner_up_probability if runner_up_probability is not None else None
        if (
            decision.outcome is MatchOutcome.AUTO_MATCH
            and margin is not None
            and margin < self.minimum_auto_margin
        ):
            outcome = MappingOutcome.MANUAL_REVIEW
            reason = "automatic candidate margin is below the precision-first safety threshold"
        elif decision.outcome is MatchOutcome.AUTO_MATCH and not (
            isinstance(self.runtime, EntityResolutionRuntime) and self.runtime.production_authorized
        ):
            outcome = MappingOutcome.MANUAL_REVIEW
            reason = "model is running in shadow mode and cannot persist automatic mappings"
        elif decision.outcome is MatchOutcome.AUTO_MATCH:
            outcome = MappingOutcome.AUTO_MATCHED
            reason = None
        elif decision.outcome is MatchOutcome.MANUAL_REVIEW:
            outcome = MappingOutcome.MANUAL_REVIEW
            reason = "model probability is within the manual-review band"
        else:
            outcome = MappingOutcome.MODEL_REJECTED
            reason = "model probability is below the manual-review threshold"

        evidence_rows = []
        for candidate, pair_features, score in scored[: self.evidence_candidate_limit]:
            evidence_rows.append(
                {
                    "product_id": candidate.product.product_id,
                    "probability": float(score),
                    "blocking_score": candidate.blocking_score,
                    "blocking_reasons": list(candidate.reasons),
                    "features": pair_features.to_dict(),
                }
            )
        production_authorized = (
            isinstance(self.runtime, EntityResolutionRuntime) and self.runtime.production_authorized
        )
        release_identity = (
            self.runtime.release_identity.to_dict()
            if production_authorized and self.runtime.release_identity is not None
            else None
        )
        evidence: dict[str, Any] = {
            "decision_version": ENTITY_MATCHING_DECISION_VERSION,
            "model_version": self.runtime.model_version,
            "production_authorized": production_authorized,
            "release_identity": release_identity,
            "thresholds": self.runtime.resolver.thresholds.to_dict(),
            "minimum_auto_margin": self.minimum_auto_margin,
            "winner_product_id": winner.product.product_id,
            "winner_margin": margin,
            "raw_threshold_outcome": decision.outcome.value,
            "would_auto_match_if_authorized": bool(
                decision.outcome is MatchOutcome.AUTO_MATCH
                and (margin is None or margin >= self.minimum_auto_margin)
            ),
            "candidate_count": len(candidates),
            "safe_candidate_count": len(safe_candidates),
            "evidence_candidates": evidence_rows,
        }
        candidate_ids = tuple(item[0].product.product_id for item in scored)
        return CatalogMatchResult(
            outcome=outcome,
            matched_product_id=(
                winner.product.product_id if outcome is MappingOutcome.AUTO_MATCHED else None
            ),
            candidate_product_ids=candidate_ids,
            method="lightgbm",
            reason=reason,
            probability=probability,
            model_version=self.runtime.model_version,
            evidence=evidence,
        )
