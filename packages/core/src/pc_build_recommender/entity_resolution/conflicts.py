"""High-precision hard gates for conflicting product variants."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .normalization import extract_numeric_facts, normalize_text


class _MatchRecord(Protocol):
    @property
    def category(self) -> str: ...

    @property
    def attributes(self) -> Mapping[str, Any]: ...

    @property
    def text(self) -> str: ...


@dataclass(frozen=True, slots=True)
class NumericConflict:
    """Evidence that a pair represents incompatible numeric variants."""

    field: str
    listing_value: float
    product_value: float
    message: str


_ALIASES: dict[str, tuple[str, ...]] = {
    "capacity_gb": (
        "capacity_gb",
        "memory_capacity_gb",
        "storage_capacity_gb",
        "total_capacity_gb",
        "capacity",
    ),
    "vram_gb": ("vram_gb", "vram_capacity_gb", "gpu_memory_gb", "memory_gb"),
    "module_count": ("module_count", "modules", "stick_count"),
    "wattage_w": ("wattage_w", "wattage", "power_w"),
    "radiator_size_mm": ("radiator_size_mm", "radiator_mm", "radiator_size"),
}

_FIELDS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "memory": ("capacity_gb", "module_count"),
    "ram": ("capacity_gb", "module_count"),
    "storage": ("capacity_gb",),
    "ssd": ("capacity_gb",),
    "hdd": ("capacity_gb",),
    "gpu": ("vram_gb",),
    "graphics card": ("vram_gb",),
    "power supply": ("wattage_w",),
    "psu": ("wattage_w",),
    "cooler": ("radiator_size_mm",),
    "cpu cooler": ("radiator_size_mm",),
}

_NUMERIC_VALUE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(tb|gb|mb|kw|w|cm|mm)?\s*$",
    re.IGNORECASE,
)


def _category(value: str) -> str:
    normalised = normalize_text(value).replace("_", " ")
    return {
        "ram": "memory",
        "ssd": "storage",
        "hdd": "storage",
        "graphics card": "gpu",
        "video card": "gpu",
        "psu": "power supply",
        "cpu cooler": "cooler",
    }.get(normalised, normalised)


def _coerce_number(value: Any, *, field: str) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        match = _NUMERIC_VALUE.fullmatch(str(value))
        if match is None:
            return None
        numeric = float(match.group(1))
        lowered = str(value).casefold()
        if field in {"capacity_gb", "vram_gb"} and "tb" in lowered:
            numeric *= 1024.0
        elif field in {"capacity_gb", "vram_gb"} and "mb" in lowered:
            numeric /= 1024.0
        elif field == "wattage_w" and "kw" in lowered:
            numeric *= 1000.0
        elif field == "radiator_size_mm" and "cm" in lowered:
            numeric *= 10.0
    return numeric


def _attribute_number(record: _MatchRecord, field: str) -> float | None:
    normalised = {
        normalize_text(key).replace(" ", "_"): value for key, value in record.attributes.items()
    }
    for alias in _ALIASES[field]:
        if alias in normalised:
            if value := _coerce_number(normalised[alias], field=field):
                return value
            if normalised[alias] in (0, 0.0, "0"):
                return 0.0
    return None


def _title_variant_value(record: _MatchRecord, field: str) -> float | None:
    facts = extract_numeric_facts(record.text)
    if field in {"capacity_gb", "vram_gb"}:
        totals = [fact.value for fact in facts if fact.kind == "total_capacity"]
        if totals:
            return max(totals)
        capacities = [fact.value for fact in facts if fact.kind == "capacity"]
        return max(capacities) if capacities else None
    if field == "module_count":
        counts = [fact.value for fact in facts if fact.kind == "module_count"]
        return max(counts) if counts else None
    if field == "wattage_w":
        powers = [fact.value for fact in facts if fact.kind == "power"]
        return max(powers) if powers else None
    # Free-text length is too ambiguous (product dimensions versus radiator size) for a gate.
    return None


def _variant_value(record: _MatchRecord, field: str) -> float | None:
    return _attribute_number(record, field) or _title_variant_value(record, field)


def find_numeric_conflicts(
    listing: _MatchRecord,
    product: _MatchRecord,
) -> tuple[NumericConflict, ...]:
    """Return category-specific variant conflicts.

    Only values with strong semantics are hard gates.  Bare numeric model tokens remain
    soft similarity features, preventing a model name such as RTX 4070 from being confused
    with memory capacity or a physical dimension.
    """

    listing_category = _category(listing.category)
    product_category = _category(product.category)
    if listing_category != product_category:
        return ()

    conflicts: list[NumericConflict] = []
    for field in _FIELDS_BY_CATEGORY.get(listing_category, ()):
        listing_value = _variant_value(listing, field)
        product_value = _variant_value(product, field)
        if listing_value is None or product_value is None:
            continue
        if abs(listing_value - product_value) <= max(1e-9, abs(product_value) * 1e-6):
            continue
        conflicts.append(
            NumericConflict(
                field=field,
                listing_value=listing_value,
                product_value=product_value,
                message=(
                    f"{field} conflicts: listing={listing_value:g}, "
                    f"canonical_product={product_value:g}"
                ),
            )
        )
    return tuple(conflicts)


def has_numeric_conflict(listing: _MatchRecord, product: _MatchRecord) -> bool:
    """Whether a known numeric variant mismatch must reject the pair."""

    return bool(find_numeric_conflicts(listing, product))
