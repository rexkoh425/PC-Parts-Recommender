"""Deterministic text and identifier normalisation for entity resolution.

The functions in this module deliberately avoid locale-dependent behaviour.  The same
listing title therefore produces the same tokens in ingestion, training, and serving.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")
_IDENTIFIER_SEPARATORS = re.compile(r"[^a-z0-9]+")
_NUMBER = r"(?P<value>\d+(?:\.\d+)?)"
_CAPACITY_RE = re.compile(rf"(?<![a-z0-9]){_NUMBER}\s*(?P<unit>tb|gb|mb)(?![a-z])")
_POWER_RE = re.compile(rf"(?<![a-z0-9]){_NUMBER}\s*(?P<unit>kw|w)(?![a-z])")
_LENGTH_RE = re.compile(rf"(?<![a-z0-9]){_NUMBER}\s*(?P<unit>cm|mm)(?![a-z])")
_FREQUENCY_RE = re.compile(rf"(?<![a-z0-9]){_NUMBER}\s*(?P<unit>ghz|mhz)(?![a-z])")
_MULTIPACK_RE = re.compile(
    r"(?<![a-z0-9])(?P<count>\d+)\s*[xX]\s*(?P<size>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>tb|gb|mb)(?![a-z])"
)


@dataclass(frozen=True, slots=True, order=True)
class NumericFact:
    """A unit-normalised numeric fact extracted from unstructured text."""

    kind: str
    value: float
    unit: str


def normalize_text(value: object | None) -> str:
    """Return a stable, searchable ASCII-ish representation of ``value``.

    NFKC handles compatibility characters (for example full-width digits), while
    casefolding makes brand and model comparison independent of case.  Punctuation is
    treated as a token boundary rather than deleted, avoiding accidental concatenation.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    # NFKD removes accents deterministically without relying on the process locale.
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    text = _NON_ALPHANUMERIC.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_identifier(value: object | None) -> str:
    """Normalise manufacturer identifiers for exact comparison.

    Hyphens, spaces, and punctuation are presentation details in many retailer feeds, so
    identifiers are compared as a compact alphanumeric string.
    """

    return _IDENTIFIER_SEPARATORS.sub("", normalize_text(value))


def tokenize(value: object | None) -> tuple[str, ...]:
    """Tokenise a value with deterministic ordering and no empty tokens."""

    normalised = normalize_text(value)
    return tuple(normalised.split()) if normalised else ()


def unique_tokens(value: object | None) -> tuple[str, ...]:
    """Return sorted unique tokens, useful for order-independent feature calculation."""

    return tuple(sorted(set(tokenize(value))))


def numeric_tokens(value: object | None) -> tuple[str, ...]:
    """Return sorted unique tokens containing at least one digit."""

    return tuple(
        sorted({token for token in tokenize(value) if any(char.isdigit() for char in token)})
    )


def _decimal(raw: str) -> Decimal:
    try:
        return Decimal(raw)
    except InvalidOperation as error:  # pragma: no cover - regex guarantees numeric input
        raise ValueError(f"invalid numeric value: {raw!r}") from error


def _scaled(raw: str, unit: str, scales: dict[str, Decimal]) -> float:
    return float(_decimal(raw) * scales[unit])


def extract_numeric_facts(value: object | None) -> tuple[NumericFact, ...]:
    """Extract dimensions with canonical units from a product title.

    Bare model numbers are intentionally excluded.  A GPU model number such as ``4070``
    is useful as a similarity feature, but it is not a safe hard-conflict signal.  Units
    make capacity, wattage, length, and frequency comparisons much less ambiguous.
    """

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    facts: set[NumericFact] = set()

    for match in _CAPACITY_RE.finditer(text):
        facts.add(
            NumericFact(
                "capacity",
                _scaled(match.group("value"), match.group("unit"), {
                    "tb": Decimal(1024),
                    "gb": Decimal(1),
                    "mb": Decimal(1) / Decimal(1024),
                }),
                "gb",
            )
        )
    for match in _MULTIPACK_RE.finditer(text):
        per_module = _scaled(match.group("size"), match.group("unit"), {
            "tb": Decimal(1024),
            "gb": Decimal(1),
            "mb": Decimal(1) / Decimal(1024),
        })
        count = int(match.group("count"))
        facts.add(NumericFact("module_count", float(count), "count"))
        facts.add(NumericFact("total_capacity", per_module * count, "gb"))
    for match in _POWER_RE.finditer(text):
        facts.add(
            NumericFact(
                "power",
                _scaled(match.group("value"), match.group("unit"), {
                    "kw": Decimal(1000),
                    "w": Decimal(1),
                }),
                "w",
            )
        )
    for match in _LENGTH_RE.finditer(text):
        facts.add(
            NumericFact(
                "length",
                _scaled(match.group("value"), match.group("unit"), {
                    "cm": Decimal(10),
                    "mm": Decimal(1),
                }),
                "mm",
            )
        )
    for match in _FREQUENCY_RE.finditer(text):
        facts.add(
            NumericFact(
                "frequency",
                _scaled(match.group("value"), match.group("unit"), {
                    "ghz": Decimal(1000),
                    "mhz": Decimal(1),
                }),
                "mhz",
            )
        )

    return tuple(sorted(facts))
