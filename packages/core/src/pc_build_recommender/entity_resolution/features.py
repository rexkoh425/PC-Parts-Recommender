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


class PairFeatureExtractor:
    """Extract reproducible features from typed pair records."""

    feature_names = FEATURE_NAMES

    def extract(
        self,
        listing: ListingRow,
        product: CanonicalProductRecord,
    ) -> PairFeatures:
        listing_text = normalize_text(listing.text)
        product_text = normalize_text(product.text)
        listing_model_tokens = set(unique_tokens(listing_text)) - _GENERIC_TOKENS
        product_model_tokens = set(unique_tokens(product_text)) - _GENERIC_TOKENS
        listing_numbers = set(numeric_tokens(listing_text))
        product_numbers = set(numeric_tokens(product_text))
        listing_specs = set(_normalised_attributes(listing.attributes).items())
        product_specs = set(_normalised_attributes(product.attributes).items())
        conflicts = find_numeric_conflicts(listing, product)
        conflict_fields = {conflict.field for conflict in conflicts}
        numeric_conflict_severity = max(
            (
                abs(conflict.listing_value - conflict.product_value)
                / max(abs(conflict.listing_value), abs(conflict.product_value), 1.0)
                for conflict in conflicts
            ),
            default=0.0,
        )
        listing_brand = normalize_text(listing.brand)
        product_brand = normalize_text(product.brand)

        return PairFeatures(
            exact_mpn_match=_identifier_match(
                listing.manufacturer_part_number, product.manufacturer_part_number
            ),
            exact_gtin_match=_identifier_match(listing.gtin, product.gtin),
            brand_match=float(bool(listing_brand) and listing_brand == product_brand),
            category_match=float(
                normalize_text(listing.category) == normalize_text(product.category)
            ),
            model_token_overlap=_jaccard(listing_model_tokens, product_model_tokens),
            character_similarity=SequenceMatcher(None, listing_text, product_text).ratio(),
            numeric_token_agreement=_jaccard(listing_numbers, product_numbers, missing_value=0.5),
            capacity_agreement=_capacity_agreement(listing, product),
            form_factor_agreement=_form_factor_agreement(listing, product),
            specification_overlap=_jaccard(listing_specs, product_specs, missing_value=0.0),
            embedding_cosine_similarity=_cosine(listing.embedding, product.embedding),
            relative_price_difference=_relative_price_difference(
                listing.current_price_sgd, product.price_sgd
            ),
            price_missing=float(listing.current_price_sgd is None or product.price_sgd is None),
            numeric_conflict=float(bool(conflicts)),
            mpn_mismatch=_identifier_mismatch(
                listing.manufacturer_part_number, product.manufacturer_part_number
            ),
            gtin_mismatch=_identifier_mismatch(listing.gtin, product.gtin),
            brand_mismatch=float(
                bool(listing_brand) and bool(product_brand) and listing_brand != product_brand
            ),
            numeric_conflict_count=float(len(conflicts)),
            numeric_conflict_severity=numeric_conflict_severity,
            capacity_conflict=float(
                "capacity_gb" in conflict_fields or "vram_gb" in conflict_fields
            ),
            module_count_conflict=float("module_count" in conflict_fields),
            power_conflict=float("wattage_w" in conflict_fields),
            radiator_conflict=float("radiator_size_mm" in conflict_fields),
        )

    def transform(
        self,
        examples: Iterable[LabelledPair],
    ) -> NDArray[np.float64]:
        rows = [self.extract(example.listing, example.product).as_array() for example in examples]
        if not rows:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        return np.vstack(rows)

    def hard_conflict_mask(self, examples: Iterable[LabelledPair]) -> NDArray[np.bool_]:
        return np.asarray(
            [bool(find_numeric_conflicts(item.listing, item.product)) for item in examples],
            dtype=np.bool_,
        )


def validate_feature_matrix(values: Any) -> NDArray[np.float64]:
    """Validate external feature matrices against the stable feature contract."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"feature matrix must have shape (n, {len(FEATURE_NAMES)}), got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains NaN or infinite values")
    return matrix
