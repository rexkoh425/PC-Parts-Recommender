"""Deterministic pairwise features for duplicate-listing classification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .conflicts import find_numeric_conflicts
from .normalization import (
    extract_numeric_facts,
    normalize_identifier,
    normalize_text,
    numeric_tokens,
    unique_tokens,
)
from .records import CanonicalProductRecord, ListingRow, LabelledPair

FEATURE_NAMES: tuple[str, ...] = (
    "exact_mpn_match",
    "exact_gtin_match",
    "brand_match",
    "category_match",
    "model_token_overlap",
    "character_similarity",
    "numeric_token_agreement",
    "capacity_agreement",
    "form_factor_agreement",
    "specification_overlap",
    "embedding_cosine_similarity",
    "relative_price_difference",
    "price_missing",
    "numeric_conflict",
    "mpn_mismatch",
    "gtin_mismatch",
    "brand_mismatch",
    "numeric_conflict_count",
    "numeric_conflict_severity",
    "capacity_conflict",
    "module_count_conflict",
    "power_conflict",
    "radiator_conflict",
)

NUMERIC_CONFLICT_FEATURE_INDEX = FEATURE_NAMES.index("numeric_conflict")

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
    "new",
}


def _jaccard(left: set[Any], right: set[Any], *, missing_value: float = 0.0) -> float:
    if not left or not right:
        return missing_value
    return len(left & right) / len(left | right)


def _identifier_match(left: str | None, right: str | None) -> float:
    left_id = normalize_identifier(left)
    right_id = normalize_identifier(right)
    return float(bool(left_id and right_id and left_id == right_id))


def _identifier_mismatch(left: str | None, right: str | None) -> float:
    left_id = normalize_identifier(left)
    right_id = normalize_identifier(right)
    return float(bool(left_id and right_id and left_id != right_id))


def _normalised_attributes(attributes: Mapping[str, Any]) -> dict[str, str]:
    return {
        normalize_text(key).replace(" ", "_"): normalize_text(value)
        for key, value in attributes.items()
        if value is not None and normalize_text(value)
    }


def _attribute_value(attributes: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    normalised = _normalised_attributes(attributes)
    for alias in aliases:
        if value := normalised.get(alias):
            return value
    return None


def _capacity_values(text: str) -> set[float]:
    facts = extract_numeric_facts(text)
    totals = {fact.value for fact in facts if fact.kind == "total_capacity"}
    if totals:
        return totals
    return {fact.value for fact in facts if fact.kind == "capacity"}


def _capacity_agreement(listing: ListingRow, product: CanonicalProductRecord) -> float:
    aliases = (
        "capacity_gb",
        "memory_capacity_gb",
        "storage_capacity_gb",
        "total_capacity_gb",
        "vram_gb",
        "vram_capacity_gb",
    )
    listing_value = _attribute_value(listing.attributes, aliases)
    product_value = _attribute_value(product.attributes, aliases)
    if listing_value is not None and product_value is not None:
        return float(listing_value == product_value)
    listing_values = _capacity_values(listing.text)
    product_values = _capacity_values(product.text)
    if not listing_values or not product_values:
        return 0.5
    return float(bool(listing_values & product_values))


def _form_factor_agreement(listing: ListingRow, product: CanonicalProductRecord) -> float:
    aliases = ("form_factor", "motherboard_form_factor", "psu_form_factor", "storage_form_factor")
    listing_value = _attribute_value(listing.attributes, aliases)
    product_value = _attribute_value(product.attributes, aliases)
    if listing_value is None or product_value is None:
        return 0.5
    return float(listing_value == product_value)


def _cosine(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float:
    if left is None or right is None or len(left) != len(right):
        return 0.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _relative_price_difference(listing_price: float | None, product_price: float | None) -> float:
    if listing_price is None or product_price is None:
        return 0.0
    denominator = max(abs(listing_price), abs(product_price), 1.0)
    return min(abs(listing_price - product_price) / denominator, 5.0)


@dataclass(frozen=True, slots=True)
class PairFeatures:
    """Named pairwise features with a stable vector contract."""

    exact_mpn_match: float
    exact_gtin_match: float
    brand_match: float
    category_match: float
    model_token_overlap: float
    character_similarity: float
    numeric_token_agreement: float
    capacity_agreement: float
    form_factor_agreement: float
    specification_overlap: float
    embedding_cosine_similarity: float
    relative_price_difference: float
    price_missing: float
    numeric_conflict: float
    mpn_mismatch: float
    gtin_mismatch: float
    brand_mismatch: float
    numeric_conflict_count: float
    numeric_conflict_severity: float
    capacity_conflict: float
    module_count_conflict: float
    power_conflict: float
    radiator_conflict: float

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(tuple(getattr(self, name) for name in FEATURE_NAMES), dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

# TODO: rest of this module still to come.
