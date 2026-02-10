"""PC-domain blocking and unlabeled hard-negative candidate discovery.

This module deliberately does not create :class:`LabelledPair` objects.  A hard
conflict is useful annotation evidence, but it is not a ground-truth label: a
retailer title, canonical record, or extracted specification may be wrong.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from enum import StrEnum

from .conflicts import NumericConflict, find_numeric_conflicts
from .normalization import normalize_identifier, normalize_text, unique_tokens
from .records import CanonicalProductRecord, ListingRow

_GENERIC_PC_TOKENS = {
    "card",
    "case",
    "cooler",
    "cpu",
    "desktop",
    "gpu",
    "graphics",
    "hdd",
    "memory",
    "motherboard",
    "new",
    "power",
    "psu",
    "ram",
    "ssd",
    "supply",
}

_CATEGORY_ALIASES = {
    "cpu": "cpu",
    "processor": "cpu",
    "gpu": "gpu",
    "graphics card": "gpu",
    "video card": "gpu",
    "motherboard": "motherboard",
    "mainboard": "motherboard",
    "memory": "memory",
    "ram": "memory",
    "storage": "storage",
    "ssd": "storage",
    "hdd": "storage",
    "power supply": "power_supply",
    "psu": "power_supply",
    "cooler": "cooler",
    "cpu cooler": "cooler",
    "case": "case",
    "pc case": "case",
    "chassis": "case",
}


class CandidateSupervision(StrEnum):
    """The only supervision state produced by candidate generation."""

    UNLABELED = "UNLABELED"


@dataclass(frozen=True, slots=True)
class BlockingScoreComponent:
    """An auditable contribution to a discovery score, not a match probability."""

    reason: str
    contribution: float

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("blocking score reason must not be empty")
        if not 0.0 <= self.contribution <= 1.0:
            raise ValueError("blocking score contribution must be between zero and one")


@dataclass(frozen=True, slots=True)
class PCBlockingCandidate:
    """One PC listing/product pair retained for scoring or human annotation.

    ``blocking_score`` measures candidate-discovery salience.  It must never be
    interpreted as a duplicate probability.  Numeric conflicts are retained as
    evidence so extraction mistakes and close variants remain auditable.
    """

    listing: ListingRow
    product: CanonicalProductRecord
    blocking_score: float
    score_components: tuple[BlockingScoreComponent, ...]
    conflicts: tuple[NumericConflict, ...] = ()
    supervision_status: CandidateSupervision = field(
        default=CandidateSupervision.UNLABELED,
        init=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.blocking_score <= 1.0:
            raise ValueError("blocking_score must be between zero and one")
        if not self.score_components:
            raise ValueError("score_components must contain blocking evidence")

    @property
    def reasons(self) -> tuple[str, ...]:
        """Stable reason codes, including every retained conflict field."""

        reasons = [component.reason for component in self.score_components]
        reasons.extend(f"numeric_conflict:{conflict.field}" for conflict in self.conflicts)
        return tuple(dict.fromkeys(reasons))

    @property
    def has_hard_conflict(self) -> bool:
        return bool(self.conflicts)

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-safe discovery metadata with no inferred label."""

        return {
            "listing_id": self.listing.listing_id,
            "product_id": self.product.product_id,
            "category": canonical_pc_category(self.listing.category),
            "blocking_score": self.blocking_score,
            "blocking_reasons": list(self.reasons),
            "score_components": [
                {
                    "reason": component.reason,
                    "contribution": component.contribution,
                }
                for component in self.score_components
            ],
            "conflicts": [
                {
                    "field": conflict.field,
                    "listing_value": conflict.listing_value,
                    "product_value": conflict.product_value,
                    "message": conflict.message,
                }
                for conflict in self.conflicts
            ],
            "supervision_status": self.supervision_status.value,
        }


@dataclass(frozen=True, slots=True)
class UnlabeledHardNegativeCandidate:
    """Metadata for a difficult pair that still requires an annotation decision.

    The name describes sampling intent only.  There is intentionally no ``label``
    field and no conversion to ``LabelledPair`` on this type.
    """

    candidate: PCBlockingCandidate
    hardness_score: float
    selection_reasons: tuple[str, ...]
    supervision_status: CandidateSupervision = field(
        default=CandidateSupervision.UNLABELED,
        init=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.hardness_score <= 1.0:
            raise ValueError("hardness_score must be between zero and one")
        if not self.selection_reasons:
            raise ValueError("selection_reasons must not be empty")

    @property
    def listing(self) -> ListingRow:
        return self.candidate.listing

    @property
    def product(self) -> CanonicalProductRecord:
        return self.candidate.product

    def to_metadata(self) -> dict[str, object]:
        """Return review-candidate metadata; absence of ``label`` is intentional."""

        return {
            **self.candidate.to_metadata(),
            "hardness_score": self.hardness_score,
            "hard_negative_selection_reasons": list(self.selection_reasons),
            "supervision_status": self.supervision_status.value,
        }


def canonical_pc_category(category: str) -> str | None:
    """Return the canonical category for one of the eight supported PC categories."""

    normalized = normalize_text(category).replace("_", " ")
    return _CATEGORY_ALIASES.get(normalized)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(unique_tokens(left)) - _GENERIC_PC_TOKENS
    right_tokens = set(unique_tokens(right)) - _GENERIC_PC_TOKENS
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _pc_numeric_conflicts(
    listing: ListingRow,
    product: CanonicalProductRecord,
    category: str,
) -> tuple[NumericConflict, ...]:
    # The conflict engine intentionally compares category strings exactly.  Here
    # aliases such as RAM/memory and PSU/power supply are known to be equivalent.
    conflict_category = {
        "power_supply": "power supply",
        "cooler": "cooler",
    }.get(category, category)
    return find_numeric_conflicts(
        replace(listing, category=conflict_category),
        replace(product, category=conflict_category),
    )


class PCDomainCandidateBlocker:
    """Generate high-recall, conflict-preserving PC component candidates."""

    def __init__(self, *, max_candidates: int = 50, minimum_text_score: float = 0.12) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least one")
        if not 0.0 <= minimum_text_score <= 1.0:
            raise ValueError("minimum_text_score must be between zero and one")
        self.max_candidates = max_candidates
        self.minimum_text_score = minimum_text_score

    def candidates(
        self,
        listing: ListingRow,
        products: Iterable[CanonicalProductRecord],
    ) -> tuple[PCBlockingCandidate, ...]:
        listing_category = canonical_pc_category(listing.category)
        if listing_category is None:
            return ()

        listing_brand = normalize_text(listing.brand)
        listing_mpn = normalize_identifier(listing.manufacturer_part_number)
        listing_gtin = normalize_identifier(listing.gtin)
        listing_text = normalize_text(listing.text)
        result: list[PCBlockingCandidate] = []

        for product in products:
            if canonical_pc_category(product.category) != listing_category:
                continue

            product_mpn = normalize_identifier(product.manufacturer_part_number)
            product_gtin = normalize_identifier(product.gtin)
            exact_mpn = bool(listing_mpn and product_mpn and listing_mpn == product_mpn)
            exact_gtin = bool(listing_gtin and product_gtin and listing_gtin == product_gtin)
            product_brand = normalize_text(product.brand)
            brand_match = bool(listing_brand and product_brand and listing_brand == product_brand)
            if (
                listing_brand
                and product_brand
                and not brand_match
                and not (exact_mpn or exact_gtin)
            ):
                continue

            product_text = normalize_text(product.text)
            token_score = _token_overlap(listing_text, product_text)
            character_score = SequenceMatcher(None, listing_text, product_text).ratio()
            text_score = 0.7 * token_score + 0.3 * character_score
            if not (exact_mpn or exact_gtin) and text_score < self.minimum_text_score:
                continue

            conflicts = _pc_numeric_conflicts(listing, product, listing_category)
            components: list[BlockingScoreComponent] = []
            if exact_mpn:
                components.append(BlockingScoreComponent("exact_manufacturer_part_number", 0.60))
            elif listing_mpn and product_mpn:
                components.append(BlockingScoreComponent("manufacturer_part_number_mismatch", 0.0))
            if exact_gtin:
                components.append(BlockingScoreComponent("exact_gtin", 0.65))
            elif listing_gtin and product_gtin:
                components.append(BlockingScoreComponent("gtin_mismatch", 0.0))
            if brand_match:
                components.append(BlockingScoreComponent("brand", 0.10))
            if token_score > 0.0:
                components.append(BlockingScoreComponent("model_tokens", 0.21 * token_score))
            if character_score > 0.0:
                components.append(
                    BlockingScoreComponent("character_similarity", 0.09 * character_score)
                )
            if conflicts:
                # A small review bonus prevents a close numeric variant from being
                # lost at the candidate cap.  It does not imply a negative label.
                components.append(BlockingScoreComponent("numeric_conflict_review", 0.05))

            score = min(1.0, sum(component.contribution for component in components))
            result.append(
                PCBlockingCandidate(
                    listing=listing,
                    product=product,
                    blocking_score=score,
                    score_components=tuple(components),
                    conflicts=conflicts,
                )
            )

        result.sort(
            key=lambda item: (
                -item.blocking_score,
                -int(item.has_hard_conflict),
                item.product.product_id,
            )
        )
        return tuple(result[: self.max_candidates])

    def generate(
        self,
        listings: Sequence[ListingRow],
        products: Sequence[CanonicalProductRecord],
    ) -> tuple[PCBlockingCandidate, ...]:
        """Generate candidates independent of caller input ordering."""

        result: list[PCBlockingCandidate] = []
        for listing in sorted(listings, key=lambda item: item.listing_id):
            result.extend(self.candidates(listing, products))
        return tuple(result)


class DeterministicHardNegativeSampler:
    """Select difficult, unlabeled annotation candidates per listing."""

    def __init__(self, *, max_per_listing: int = 5, minimum_blocking_score: float = 0.12) -> None:
        if max_per_listing < 1:
            raise ValueError("max_per_listing must be at least one")
        if not 0.0 <= minimum_blocking_score <= 1.0:
            raise ValueError("minimum_blocking_score must be between zero and one")
        self.max_per_listing = max_per_listing
        self.minimum_blocking_score = minimum_blocking_score

    def _score(
        self,
        candidate: PCBlockingCandidate,
    ) -> tuple[float, tuple[str, ...]] | None:
        reasons = set(candidate.reasons)
        exact_identifier = bool({"exact_manufacturer_part_number", "exact_gtin"} & reasons)
        mismatch_reasons = tuple(
            reason
            for reason in ("manufacturer_part_number_mismatch", "gtin_mismatch")
            if reason in reasons
        )

        selection_reasons: list[str] = []
        bonus = 0.0
        if candidate.conflicts:
            selection_reasons.extend(
                f"hard_numeric_conflict:{conflict.field}" for conflict in candidate.conflicts
            )
            bonus += min(0.45, 0.35 + 0.05 * (len(candidate.conflicts) - 1))
        if mismatch_reasons:
            selection_reasons.extend(mismatch_reasons)
            bonus += 0.20
        if not exact_identifier and "model_tokens" in reasons:
            selection_reasons.append("near_model_variant")
            bonus += 0.10

        # A clean exact-identifier pair is positive-looking evidence, not a hard-
        # negative annotation candidate.  Conflicting exact identifiers remain.
        if not selection_reasons or (exact_identifier and not candidate.conflicts):
            return None
        hardness = min(1.0, 0.55 * candidate.blocking_score + bonus)
        if candidate.blocking_score < self.minimum_blocking_score and not candidate.conflicts:
            return None
        return hardness, tuple(dict.fromkeys(selection_reasons))

    def sample(
        self,
        candidates: Iterable[PCBlockingCandidate],
    ) -> tuple[UnlabeledHardNegativeCandidate, ...]:
        grouped: dict[str, list[UnlabeledHardNegativeCandidate]] = defaultdict(list)
        for candidate in candidates:
            scored = self._score(candidate)
            if scored is None:
                continue
            hardness, reasons = scored
            grouped[candidate.listing.listing_id].append(
                UnlabeledHardNegativeCandidate(
                    candidate=candidate,
                    hardness_score=hardness,
                    selection_reasons=reasons,
                )
            )

        result: list[UnlabeledHardNegativeCandidate] = []
        for listing_id in sorted(grouped):
            ranked = sorted(
                grouped[listing_id],
                key=lambda item: (-item.hardness_score, item.product.product_id),
            )
            result.extend(ranked[: self.max_per_listing])
        return tuple(result)

    def generate(
        self,
        listings: Sequence[ListingRow],
        products: Sequence[CanonicalProductRecord],
        *,
        blocker: PCDomainCandidateBlocker | None = None,
    ) -> tuple[UnlabeledHardNegativeCandidate, ...]:
        """Block and sample in one deterministic, still-unlabeled operation."""

        candidate_blocker = blocker or PCDomainCandidateBlocker()
        return self.sample(candidate_blocker.generate(listings, products))
