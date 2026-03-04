"""Deterministic candidate generation for listing-to-product matching."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from .conflicts import find_numeric_conflicts
from .normalization import normalize_identifier, normalize_text, unique_tokens
from .records import CanonicalProductRecord, ListingRow

_GENERIC_TOKENS = {
    "cpu",
    "gpu",
    "graphics",
    "card",
    "memory",
    "ram",
    "ssd",
    "hdd",
    "motherboard",
    "power",
    "supply",
    "psu",
    "cooler",
    "case",
    "desktop",
    "new",
}


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """One candidate pair and the auditable reasons it survived blocking."""

    listing: ListingRow
    product: CanonicalProductRecord
    blocking_score: float
    reasons: tuple[str, ...]


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(unique_tokens(left)) - _GENERIC_TOKENS
    right_tokens = set(unique_tokens(right)) - _GENERIC_TOKENS
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class CandidateBlocker:
    """Generate high-recall candidates without permitting known variant conflicts."""

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
    ) -> tuple[CandidatePair, ...]:
        result: list[CandidatePair] = []
        listing_category = normalize_text(listing.category)
        listing_brand = normalize_text(listing.brand)
        listing_mpn = normalize_identifier(listing.manufacturer_part_number)
        listing_gtin = normalize_identifier(listing.gtin)
        listing_text = normalize_text(listing.text)

        for product in products:
            if listing_category != normalize_text(product.category):
                continue
            if find_numeric_conflicts(listing, product):
                continue

            product_mpn = normalize_identifier(product.manufacturer_part_number)
            product_gtin = normalize_identifier(product.gtin)
            exact_mpn = bool(listing_mpn and product_mpn and listing_mpn == product_mpn)
            exact_gtin = bool(listing_gtin and product_gtin and listing_gtin == product_gtin)
            reasons: list[str] = []
            if exact_mpn:
                reasons.append("exact_manufacturer_part_number")
            if exact_gtin:
                reasons.append("exact_gtin")

            product_brand = normalize_text(product.brand)
            brand_match = bool(
                listing_brand and product_brand and listing_brand == product_brand
            )
            if (
                listing_brand
                and product_brand
                and not brand_match
                and not (exact_mpn or exact_gtin)
            ):
                continue
            if brand_match:
                reasons.append("brand")

            product_text = normalize_text(product.text)
            token_score = _token_overlap(listing_text, product_text)
            character_score = SequenceMatcher(None, listing_text, product_text).ratio()
            text_score = 0.7 * token_score + 0.3 * character_score
            if token_score > 0:
                reasons.append("model_tokens")

            if not (exact_mpn or exact_gtin) and text_score < self.minimum_text_score:
                continue

            # Exact identifiers dominate but do not bypass numeric variant gates above.
            score = min(
                1.0,
                (0.60 if exact_mpn else 0.0)
                + (0.65 if exact_gtin else 0.0)
                + (0.10 if brand_match else 0.0)
                + 0.30 * text_score,
            )
            result.append(
                CandidatePair(
                    listing=listing,
                    product=product,
                    blocking_score=score,
                    reasons=tuple(reasons),
                )
            )

        result.sort(key=lambda item: (-item.blocking_score, item.product.product_id))
        return tuple(result[: self.max_candidates])

    def generate(
        self,
        listings: Sequence[ListingRow],
        products: Sequence[CanonicalProductRecord],
    ) -> tuple[CandidatePair, ...]:
        """Generate candidates for multiple listings in stable listing order."""

        result: list[CandidatePair] = []
        for listing in listings:
            result.extend(self.candidates(listing, products))
        return tuple(result)
