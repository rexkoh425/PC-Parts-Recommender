"""Deterministic, evidence-backed recommendation explanations."""

from .service import (
    explain_build_selection,
    explain_component_selection,
    explain_replacement,
    summarize_review_evidence,
)
from .types import (
    ComponentSelection,
    ConstraintDelta,
    EvidenceBasis,
    Explanation,
    ExplanationStatement,
    MetricDelta,
    MetricEvidence,
    ReasonKind,
    ReplacementComparison,
    ReviewEvidence,
    SelectionReason,
    StoredSourceCitation,
)

__all__ = [
    "ComponentSelection",
    "ConstraintDelta",
    "EvidenceBasis",
    "Explanation",
    "ExplanationStatement",
    "MetricDelta",
    "MetricEvidence",
    "ReasonKind",
    "ReplacementComparison",
    "ReviewEvidence",
    "SelectionReason",
    "StoredSourceCitation",
    "explain_build_selection",
    "explain_component_selection",
    "explain_replacement",
    "summarize_review_evidence",
]
