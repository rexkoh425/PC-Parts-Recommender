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

# TODO: rest of this module still to come.
