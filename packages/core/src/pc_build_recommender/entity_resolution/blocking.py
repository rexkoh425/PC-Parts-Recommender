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

# TODO: rest of this module still to come.
