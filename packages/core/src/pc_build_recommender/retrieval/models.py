"""Small, storage-agnostic contracts used by product retrieval.

The retrieval package deliberately does not import database models.  Pipeline,
API, and test callers can construct :class:`ProductDocument` instances from
plain mappings while a database adapter can implement the same conversion at
its boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _normalise_text(value: object) -> str:
    return " ".join(str(value).strip().split())


@dataclass(frozen=True, slots=True)
class ProductDocument:
    """Canonical product text and filterable fields.

    ``attributes`` may contain either flat keys or nested category attributes.
    ``get`` understands dotted paths and a small set of canonical top-level
    fields, which keeps this contract usable before a final ORM model exists.
    """

    product_id: str
    category: str
    text: str
    brand: str | None = None
    price_sgd: float | None = None
    stock_status: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        product_id = _normalise_text(self.product_id)
        category = _normalise_text(self.category).casefold()
        text = _normalise_text(self.text)
        if not product_id:
            raise ValueError("product_id must not be empty")
        if not category:
            raise ValueError("category must not be empty")
        if not text:
            raise ValueError("text must not be empty")
        if self.price_sgd is not None and (
            not math.isfinite(self.price_sgd) or self.price_sgd < 0
        ):
            raise ValueError("price_sgd must be finite and non-negative")
        object.__setattr__(self, "product_id", product_id)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "attributes", dict(self.attributes))

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> ProductDocument:
        """Build a document from a canonical-product shaped mapping."""

        product_id = record.get("product_id", record.get("id"))
        category = record.get("category")
        if product_id is None or category is None:
            raise ValueError("record must contain product_id (or id) and category")

        attributes: dict[str, Any] = {}
        for key in ("common_attributes", "category_attributes", "attributes"):
            value = record.get(key)
            if isinstance(value, Mapping):
                attributes.update(value)

        excluded = {
            "product_id",
            "id",
            "category",
            "search_text",
            "text",
            "canonical_name",
            "brand",
            "price_sgd",
            "current_price_sgd",
            "stock_status",
            "common_attributes",
            "category_attributes",
            "attributes",
        }
        attributes.update({key: value for key, value in record.items() if key not in excluded})

        text = record.get("search_text") or record.get("text")
        if not text:
            name_parts = (
                category,
                record.get("brand"),
                record.get("canonical_name"),
                record.get("model"),
                record.get("manufacturer_part_number"),
            )
            attr_parts = [
                f"{key.replace('_', ' ')} {value}"
                for key, value in sorted(attributes.items())
                if value is not None and not isinstance(value, Mapping)
            ]
            text = " ".join(str(part) for part in (*name_parts, *attr_parts) if part)

        price = record.get("price_sgd", record.get("current_price_sgd"))
        return cls(
            product_id=str(product_id),
            category=str(category),
            text=str(text),
            brand=str(record["brand"]) if record.get("brand") is not None else None,
            price_sgd=float(price) if price is not None else None,
            stock_status=(
                str(record["stock_status"]) if record.get("stock_status") is not None else None
            ),
            attributes=attributes,
        )

    def get(self, field_name: str, default: Any = None) -> Any:
        """Read a top-level field or a dotted path from ``attributes``."""

        top_level: dict[str, Any] = {
            "product_id": self.product_id,
            "category": self.category,
            "brand": self.brand,
            "price_sgd": self.price_sgd,
            "stock_status": self.stock_status,
            "text": self.text,
        }
        if field_name in top_level:
            return top_level[field_name]

        current: Any = self.attributes
        for part in field_name.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked result from an individual retrieval system."""

    product_id: str
    score: float
    rank: int
    source: str

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must start at one")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A result after Reciprocal Rank Fusion."""

    product_id: str
    score: float
    rank: int
    source_ranks: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    """A product with transparent per-retriever and fused evidence."""

    product: ProductDocument
    rank: int
    rrf_score: float
    lexical_score: float = 0.0
    lexical_rank: int | None = None
    lexical_model: str | None = None
    bm25_score: float = 0.0
    vector_similarity: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None

    @property
    def product_id(self) -> str:
        return self.product.product_id


@dataclass(frozen=True, slots=True)
class StructuredFilters:
    """Direct product requirements applied before scored retrieval.

    ``allowed_product_ids`` is intended for IDs approved by an external
    compatibility engine.  Retrieval never attempts to infer compatibility,
    and a ranker never receives products excluded by this allow-list.
    """

    maximum_price_sgd: float | None = None
    minimum_gpu_vram_gb: float | None = None
    minimum_memory_gb: float | None = None
    required_memory_type: str | None = None
    required_form_factor: str | None = None
    wifi_required: bool = False
    excluded_brands: frozenset[str] = frozenset()
    in_stock_only: bool = True
    allowed_product_ids: frozenset[str] | None = None
    attribute_equals: Mapping[str, Any] = field(default_factory=dict)
    attribute_minimums: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("maximum_price_sgd", "minimum_gpu_vram_gb", "minimum_memory_gb"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in self.attribute_minimums.items():
            if not math.isfinite(value):
                raise ValueError(f"minimum for {name!r} must be finite")
        object.__setattr__(
            self,
            "excluded_brands",
            frozenset(brand.casefold() for brand in self.excluded_brands),
        )
        if self.allowed_product_ids is not None:
            object.__setattr__(self, "allowed_product_ids", frozenset(self.allowed_product_ids))
        object.__setattr__(self, "attribute_equals", dict(self.attribute_equals))
        object.__setattr__(self, "attribute_minimums", dict(self.attribute_minimums))
