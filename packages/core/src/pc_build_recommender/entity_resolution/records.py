"""Typed, serialisable records used by the entity-resolution pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Self


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _freeze_embedding(value: Sequence[float] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    embedding = tuple(float(item) for item in value)
    if not embedding:
        return None
    return embedding


def _optional_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalised = str(value).strip().casefold()
    if normalised in {"true", "1", "yes", "y"}:
        return True
    if normalised in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _json_mapping(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise TypeError("attributes must be a JSON object")
    return dict(decoded)


def _json_embedding(value: Any) -> tuple[float, ...] | None:
    if value is None or value == "":
        return None
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes, bytearray)):
        raise TypeError("embedding must be a JSON array")
    return _freeze_embedding(decoded)


@dataclass(frozen=True, slots=True)
class CanonicalProductRecord:
    """Minimal canonical-product representation needed for matching."""

    product_id: str
    category: str
    brand: str
    model: str
    canonical_name: str
    manufacturer_part_number: str | None = None
    gtin: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    price_sgd: float | None = None
    embedding: tuple[float, ...] | None = None
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if not self.product_id.strip():
            raise ValueError("product_id must not be empty")
        if not self.category.strip():
            raise ValueError("category must not be empty")
        if not self.canonical_name.strip():
            raise ValueError("canonical_name must not be empty")
        if self.price_sgd is not None and self.price_sgd < 0:
            raise ValueError("price_sgd must be non-negative")
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))
        object.__setattr__(self, "embedding", _freeze_embedding(self.embedding))

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.brand, self.model, self.canonical_name) if part)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "category": self.category,
            "brand": self.brand,
            "model": self.model,
            "canonical_name": self.canonical_name,
            "manufacturer_part_number": self.manufacturer_part_number,
            "gtin": self.gtin,
            "attributes": dict(self.attributes),
            "price_sgd": self.price_sgd,
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "is_synthetic": self.is_synthetic,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Self:
        return cls(
            product_id=str(row["product_id"]),
            category=str(row["category"]),
            brand=str(row.get("brand", "")),
            model=str(row.get("model", "")),
            canonical_name=str(row.get("canonical_name") or row.get("model") or ""),
            manufacturer_part_number=_optional_string(row.get("manufacturer_part_number")),
            gtin=_optional_string(row.get("gtin")),
            attributes=_json_mapping(row.get("attributes")),
            price_sgd=_optional_float(row.get("price_sgd")),
            embedding=_json_embedding(row.get("embedding")),
            is_synthetic=_bool(row.get("is_synthetic", False)),
        )


@dataclass(frozen=True, slots=True)
class ListingRow:
    """Retailer listing representation used during blocking and matching."""

    listing_id: str
    title: str
    category: str
    brand: str = ""
    manufacturer_part_number: str | None = None
    gtin: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    current_price_sgd: float | None = None
    embedding: tuple[float, ...] | None = None
    retailer: str | None = None
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if not self.listing_id.strip():
            raise ValueError("listing_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.category.strip():
            raise ValueError("category must not be empty")
        if self.current_price_sgd is not None and self.current_price_sgd < 0:
            raise ValueError("current_price_sgd must be non-negative")
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))
        object.__setattr__(self, "embedding", _freeze_embedding(self.embedding))

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.brand, self.title) if part)

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "title": self.title,
            "category": self.category,
            "brand": self.brand,
            "manufacturer_part_number": self.manufacturer_part_number,
            "gtin": self.gtin,
            "attributes": dict(self.attributes),
            "current_price_sgd": self.current_price_sgd,
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "retailer": self.retailer,
            "is_synthetic": self.is_synthetic,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Self:
        return cls(
            listing_id=str(row["listing_id"]),
            title=str(row["title"]),
            category=str(row["category"]),
            brand=str(row.get("brand", "")),
            manufacturer_part_number=_optional_string(row.get("manufacturer_part_number")),
            gtin=_optional_string(row.get("gtin")),
            attributes=_json_mapping(row.get("attributes")),
            current_price_sgd=_optional_float(row.get("current_price_sgd")),
            embedding=_json_embedding(row.get("embedding")),
            retailer=_optional_string(row.get("retailer")),
            is_synthetic=_bool(row.get("is_synthetic", False)),
        )


@dataclass(frozen=True, slots=True)
class LabelledPair:
    """A labelled listing-to-product pair with explicit data provenance."""

    pair_id: str
    listing: ListingRow
    product: CanonicalProductRecord
    label: int
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if not self.pair_id.strip():
            raise ValueError("pair_id must not be empty")
        if self.label not in (0, 1):
            raise ValueError("label must be either 0 or 1")
        # Provenance is contagious: a pair cannot hide a synthetic input record.
        object.__setattr__(
            self,
            "is_synthetic",
            self.is_synthetic or self.listing.is_synthetic or self.product.is_synthetic,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "listing": self.listing.to_dict(),
            "product": self.product.to_dict(),
            "label": self.label,
            "is_synthetic": self.is_synthetic,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Return one CSV/JSONL-safe row with stable prefixed field names."""

        listing = self.listing.to_dict()
        product = self.product.to_dict()
        row: dict[str, Any] = {
            "pair_id": self.pair_id,
            "label": self.label,
            "is_synthetic": self.is_synthetic,
        }
        for prefix, record in (("listing", listing), ("product", product)):
            for key, value in record.items():
                if key == "is_synthetic":
                    row[f"{prefix}_{key}"] = value
                elif key == "attributes":
                    row[f"{prefix}_attributes_json"] = json.dumps(
                        value, sort_keys=True, separators=(",", ":")
                    )
                elif key == "embedding":
                    row[f"{prefix}_embedding_json"] = (
                        json.dumps(value, separators=(",", ":")) if value is not None else ""
                    )
                else:
                    row[f"{prefix}_{key}"] = value
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Self:
        """Load either the nested JSON form or the flat CSV/JSONL form."""

        if isinstance(row.get("listing"), Mapping) and isinstance(row.get("product"), Mapping):
            return cls(
                pair_id=str(row["pair_id"]),
                listing=ListingRow.from_dict(row["listing"]),
                product=CanonicalProductRecord.from_dict(row["product"]),
                label=int(row["label"]),
                is_synthetic=_bool(row.get("is_synthetic", False)),
            )
        return cls.from_flat_dict(row)

    @classmethod
    def from_flat_dict(cls, row: Mapping[str, Any]) -> Self:
        def collect(prefix: str) -> dict[str, Any]:
            result: dict[str, Any] = {}
            marker = f"{prefix}_"
            for key, value in row.items():
                if not key.startswith(marker):
                    continue
                field_name = key[len(marker) :]
                if field_name == "attributes_json":
                    result["attributes"] = value
                elif field_name == "embedding_json":
                    result["embedding"] = value
                else:
                    result[field_name] = value
            return result

        return cls(
            pair_id=str(row["pair_id"]),
            listing=ListingRow.from_dict(collect("listing")),
            product=CanonicalProductRecord.from_dict(collect("product")),
            label=int(row["label"]),
            is_synthetic=_bool(row.get("is_synthetic", False)),
        )


def pair_example_from_dict(row: Mapping[str, Any]) -> LabelledPair:
    """Functional loader for CLI code that should not need schema-specific branching."""

    return LabelledPair.from_dict(row)
