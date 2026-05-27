"""Strict contracts for cited explanations.

These types prevent an explanation from silently losing the distinction between a
measured benchmark, a model estimate, and a relative score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import Any


class EvidenceBasis(StrEnum):
    OBSERVED = "observed"
    PREDICTED = "predicted"
    RELATIVE = "relative"


class ReasonKind(StrEnum):
    REQUIREMENT = "requirement"
    SPECIFICATION = "specification"
    BENCHMARK = "benchmark"
    PRICE = "price"
    COMPATIBILITY = "compatibility"
    MODEL_OUTPUT = "model_output"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class StoredSourceCitation:
    """Reference to evidence persisted by the ingestion pipeline."""

    source_id: str
    source_url: str
    title: str
    source_type: str
    retrieved_at: datetime | None = None

    def __post_init__(self) -> None:
        source_id = self.source_id.strip()
        source_url = self.source_url.strip()
        title = self.title.strip()
        source_type = self.source_type.strip().casefold()
        if not source_id or "]" in source_id or "\n" in source_id:
            raise ValueError("source_id must be a safe, non-empty citation marker")
        if not source_url:
            raise ValueError("source_url must not be empty")
        if not title:
            raise ValueError("citation title must not be empty")
        if not source_type:
            raise ValueError("source_type must not be empty")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_url", source_url)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "source_type", source_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_url": self.source_url,
            "title": self.title,
            "source_type": self.source_type,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
        }


def normalise_citations(
    citations: tuple[StoredSourceCitation, ...],
) -> tuple[StoredSourceCitation, ...]:
    by_id: dict[str, StoredSourceCitation] = {}
    for citation in citations:
        incumbent = by_id.get(citation.source_id)
        if incumbent is not None and incumbent != citation:
            raise ValueError(f"conflicting citations use source_id {citation.source_id!r}")
        by_id[citation.source_id] = citation
    return tuple(by_id[source_id] for source_id in sorted(by_id))


def _validate_numeric(value: Decimal | int | float) -> None:
    valid = value.is_finite() if isinstance(value, Decimal) else isfinite(float(value))
    if not valid:
        raise ValueError("metric values must be finite")


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    label: str
    value: Decimal | int | float
    unit: str
    basis: EvidenceBasis | str
    citations: tuple[StoredSourceCitation, ...]
    model_version: str | None = None
    confidence: str | None = None
    relative_to: str | None = None

    def __post_init__(self) -> None:
        label = self.label.strip()
        unit = self.unit.strip()
        basis = EvidenceBasis(self.basis)
        citations = normalise_citations(self.citations)
        if not label:
            raise ValueError("metric label must not be empty")
        if not unit:
            raise ValueError("metric unit must not be empty")
        _validate_numeric(self.value)
        if not citations:
            raise ValueError("metric evidence must cite at least one stored source")
        model_version = self.model_version.strip() if self.model_version else None
        confidence = self.confidence.strip().casefold() if self.confidence else None
        relative_to = self.relative_to.strip() if self.relative_to else None
        if basis is EvidenceBasis.PREDICTED and not model_version:
            raise ValueError("predicted metrics require model_version")
        if basis is EvidenceBasis.RELATIVE and not relative_to:
            raise ValueError("relative metrics require a comparison basis")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "relative_to", relative_to)


@dataclass(frozen=True, slots=True)
class SelectionReason:
    statement: str
    kind: ReasonKind | str
    citations: tuple[StoredSourceCitation, ...]

    def __post_init__(self) -> None:
        statement = self.statement.strip().rstrip(".")
        kind = ReasonKind(self.kind)
        citations = normalise_citations(self.citations)
        if not statement:
            raise ValueError("selection reason must not be empty")
        if not citations:
            raise ValueError("selection reasons must cite stored evidence")
        if kind is ReasonKind.REVIEW and any(
            citation.source_type != "review" for citation in citations
        ):
            raise ValueError("review claims may cite only stored review evidence")
        object.__setattr__(self, "statement", statement)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "citations", citations)


@dataclass(frozen=True, slots=True)
class ComponentSelection:
    category: str
    product_id: str
    product_name: str
    reasons: tuple[SelectionReason, ...]
    metrics: tuple[MetricEvidence, ...] = ()

    def __post_init__(self) -> None:
        identifiers = (self.category, self.product_id, self.product_name)
        if any(not value.strip() for value in identifiers):
            raise ValueError("component category, product_id, and product_name are required")
        if not self.reasons:
            raise ValueError("a selected component requires at least one cited reason")


@dataclass(frozen=True, slots=True)
class MetricDelta:
    before: MetricEvidence
    after: MetricEvidence

    def __post_init__(self) -> None:
        if self.before.label != self.after.label or self.before.unit != self.after.unit:
            raise ValueError("metric delta endpoints must use the same label and unit")


@dataclass(frozen=True, slots=True)
class ConstraintDelta:
    name: str
    before: str
    after: str
    citations: tuple[StoredSourceCitation, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.before.strip() or not self.after.strip():
            raise ValueError("constraint delta name and states are required")
        citations = normalise_citations(self.citations)
        if not citations:
            raise ValueError("constraint changes must cite a stored rule or specification")
        object.__setattr__(self, "citations", citations)


@dataclass(frozen=True, slots=True)
class ReplacementComparison:
    old_product_name: str
    new_product_name: str
    old_delivered_price: Decimal | int | float
    new_delivered_price: Decimal | int | float
    currency: str
    price_citations: tuple[StoredSourceCitation, ...]
    old_compatibility: str
    new_compatibility: str
    compatibility_citations: tuple[StoredSourceCitation, ...]
    metric_deltas: tuple[MetricDelta, ...] = ()
    constraint_deltas: tuple[ConstraintDelta, ...] = ()

    def __post_init__(self) -> None:
        if not self.old_product_name.strip() or not self.new_product_name.strip():
            raise ValueError("replacement product names are required")
        _validate_numeric(self.old_delivered_price)
        _validate_numeric(self.new_delivered_price)
        if not self.currency.strip():
            raise ValueError("replacement currency is required")
        price_citations = normalise_citations(self.price_citations)
        compatibility_citations = normalise_citations(self.compatibility_citations)
        if not price_citations:
            raise ValueError("replacement price delta requires stored price citations")
        if not compatibility_citations:
            raise ValueError("replacement compatibility requires stored rule citations")
        object.__setattr__(self, "price_citations", price_citations)
        object.__setattr__(self, "compatibility_citations", compatibility_citations)


@dataclass(frozen=True, slots=True)
class ReviewNote:
    evidence_id: str
    aspect: str
    sentiment: str
    evidence_text: str
    confidence: float
    citation: StoredSourceCitation

    def __post_init__(self) -> None:
        required_text = (self.evidence_id, self.aspect, self.evidence_text)
        if any(not value.strip() for value in required_text):
            raise ValueError("review evidence id, aspect, and stored text are required")
        sentiment = self.sentiment.strip().casefold()
        if sentiment not in {"positive", "negative", "neutral", "mixed"}:
            raise ValueError("review sentiment must be positive, negative, neutral, or mixed")
        if not 0 <= self.confidence <= 1:
            raise ValueError("review evidence confidence must be between zero and one")
        if self.citation.source_type != "review":
            raise ValueError("review evidence must reference a stored review source")
        object.__setattr__(self, "sentiment", sentiment)


@dataclass(frozen=True, slots=True)
class ExplanationStatement:
    text: str
    citations: tuple[StoredSourceCitation, ...]

    def __post_init__(self) -> None:
        text = self.text.strip()
        citations = normalise_citations(self.citations)
        if not text:
            raise ValueError("explanation text must not be empty")
        if not citations:
            raise ValueError("every explanation statement must have a citation")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "citations", citations)

    @property
    def rendered_text(self) -> str:
        markers = "".join(f"[{citation.source_id}]" for citation in self.citations)
        return f"{self.text} {markers}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "rendered_text": self.rendered_text,
            "citation_ids": [citation.source_id for citation in self.citations],
        }


@dataclass(frozen=True, slots=True)
class Explanation:
    statements: tuple[ExplanationStatement, ...]

    @property
    def citations(self) -> tuple[StoredSourceCitation, ...]:
        return normalise_citations(
            tuple(citation for item in self.statements for citation in item.citations)
        )

    def render(self) -> str:
        return "\n".join(statement.rendered_text for statement in self.statements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statements": [statement.to_dict() for statement in self.statements],
            "sources": [citation.to_dict() for citation in self.citations],
        }
