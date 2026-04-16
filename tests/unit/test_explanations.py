from datetime import UTC, datetime
from decimal import Decimal

import pytest

from pc_build_recommender.explanations import (
    ComponentSelection,
    ConstraintDelta,
    EvidenceBasis,
    MetricDelta,
    MetricEvidence,
    ReasonKind,
    ReplacementComparison,
    ReviewNote,
    SelectionReason,
    StoredSourceCitation,
    explain_build_selection,
    explain_component_selection,
    explain_replacement,
    summarize_review_evidence,
)


def source(source_id: str, source_type: str = "benchmark") -> StoredSourceCitation:
    return StoredSourceCitation(
        source_id=source_id,
        source_url=f"https://evidence.test/{source_id}",
        title=f"Evidence {source_id}",
        source_type=source_type,
        retrieved_at=datetime(2026, 7, 22, tzinfo=UTC),
    )


def metric(
    label: str,
    value: int,
    basis: EvidenceBasis,
    citation: StoredSourceCitation,
) -> MetricEvidence:
    return MetricEvidence(
        label=label,
        value=value,
        unit="points",
        basis=basis,
        citations=(citation,),
        model_version="gpu-ai-v2" if basis is EvidenceBasis.PREDICTED else None,
        confidence="medium" if basis is EvidenceBasis.PREDICTED else None,
        relative_to="catalogue median" if basis is EvidenceBasis.RELATIVE else None,
    )


def selection(name: str, category: str, source_id: str) -> ComponentSelection:
    citation = source(source_id, "manufacturer")
    return ComponentSelection(
        category=category,
        product_id=name.casefold(),
        product_name=name,
        reasons=(
            SelectionReason(
                statement="meets the stored capacity requirement",
                kind=ReasonKind.REQUIREMENT,
                citations=(citation,),
            ),
        ),
    )


def test_selection_explanation_is_cited_and_labels_evidence_basis() -> None:
    manufacturer = source("mfr", "manufacturer")
    benchmark = source("bench")
    model = source("model", "model")
    relative = source("relative", "benchmark")
    component = ComponentSelection(
        category="GPU",
        product_id="gpu-1",
        product_name="Example GPU 16 GB",
        reasons=(
            SelectionReason(
                statement="its verified specification provides 16 GB of VRAM",
                kind="specification",
                citations=(manufacturer,),
            ),
        ),
        metrics=(
            metric("Gaming score", 90, EvidenceBasis.OBSERVED, benchmark),
            metric("AI score", 84, EvidenceBasis.PREDICTED, model),
            metric("Value score", 75, EvidenceBasis.RELATIVE, relative),
        ),
    )

    explanation = explain_component_selection(component)
    rendered = explanation.render()

    assert "Selected Example GPU 16 GB for GPU" in rendered
    assert "(observed)" in rendered
    assert "predicted by gpu-ai-v2, medium confidence" in rendered
    assert "relative score versus catalogue median" in rendered
    assert "[mfr]" in rendered
    assert {item.source_id for item in explanation.citations} == {
        "mfr",
        "bench",
        "model",
        "relative",
    }


def test_predicted_and_relative_metrics_cannot_hide_their_basis() -> None:
    citation = source("model", "model")
    with pytest.raises(ValueError, match="model_version"):
        MetricEvidence(
            label="AI score",
            value=80,
            unit="points",
            basis="predicted",
            citations=(citation,),
        )
    with pytest.raises(ValueError, match="comparison basis"):
        MetricEvidence(
            label="Value score",
            value=80,
            unit="points",
            basis="relative",
            citations=(citation,),
        )


def test_every_reason_and_statement_requires_stored_evidence() -> None:
    with pytest.raises(ValueError, match="cite stored evidence"):
        SelectionReason(statement="is excellent", kind="model_output", citations=())


def test_review_reason_rejects_non_review_citation() -> None:
    with pytest.raises(ValueError, match="review evidence"):
        SelectionReason(
            statement="reviewers describe low noise",
            kind="review",
            citations=(source("manufacturer", "manufacturer"),),
        )


def test_build_explanation_order_is_deterministic() -> None:
    gpu = selection("GPU B", "GPU", "gpu-source")
    cpu = selection("CPU A", "CPU", "cpu-source")
    forward = explain_build_selection([gpu, cpu]).render()
    reverse = explain_build_selection([cpu, gpu]).render()

    assert forward == reverse
    assert forward.index("CPU A") < forward.index("GPU B")


def test_replacement_explains_price_metric_compatibility_and_constraint_deltas() -> None:
    old_benchmark = source("old-benchmark")
    new_model = source("new-model", "model")
    old_price = source("old-price", "price_snapshot")
    new_price = source("new-price", "price_snapshot")
    rule = source("compat-v7", "compatibility_rule")
    spec = source("gpu-spec", "manufacturer")
    comparison = ReplacementComparison(
        old_product_name="GPU A",
        new_product_name="GPU B",
        old_delivered_price=Decimal("799"),
        new_delivered_price=Decimal("849"),
        currency="SGD",
        price_citations=(old_price, new_price),
        old_compatibility="pass",
        new_compatibility="warning",
        compatibility_citations=(rule,),
        metric_deltas=(
            MetricDelta(
                before=metric("AI score", 80, EvidenceBasis.OBSERVED, old_benchmark),
                after=metric("AI score", 85, EvidenceBasis.PREDICTED, new_model),
            ),
        ),
        constraint_deltas=(
            ConstraintDelta(
                name="Minimum VRAM",
                before="not met",
                after="met",
                citations=(spec,),
            ),
        ),
    )

    rendered = explain_replacement(comparison).render()

    assert "changes delivered price by +S$50.00" in rendered
    assert "AI score changes by +5 points" in rendered
    assert "80 observed to 85 predicted by gpu-ai-v2, medium confidence" in rendered
    assert "Compatibility changes from PASS to WARNING" in rendered
    assert "Minimum VRAM changes from not met to met" in rendered
    assert "[new-price][old-price]" in rendered


def test_review_summary_uses_only_stored_confident_evidence_and_never_echoes_prose() -> None:
    review_a = source("review-a", "review")
    review_b = source("review-b", "review")
    evidence = [
        ReviewNote(
            evidence_id="e1",
            aspect="Noise",
            sentiment="negative",
            evidence_text="UNSAFE UNSUPPORTED-SOUNDING SOURCE TEXT",
            confidence=0.9,
            citation=review_a,
        ),
        ReviewNote(
            evidence_id="e2",
            aspect="Noise",
            sentiment="positive",
            evidence_text="A measured noise passage stored from the review.",
            confidence=0.8,
            citation=review_b,
        ),
        ReviewNote(
            evidence_id="weak",
            aspect="Thermals",
            sentiment="negative",
            evidence_text="Low confidence extraction.",
            confidence=0.2,
            citation=review_a,
        ),
    ]

    explanation = summarize_review_evidence(evidence)
    rendered = explanation.render()

    assert "Noise: stored review evidence contains 1 positive finding, 1 concern" in rendered
    assert "across 2 sources" in rendered
    assert "Thermals" not in rendered
    assert "UNSAFE" not in rendered
    assert "[review-a][review-b]" in rendered


def test_duplicate_review_evidence_ids_are_counted_once() -> None:
    review = source("review", "review")
    item = ReviewNote(
        evidence_id="e1",
        aspect="Build quality",
        sentiment="positive",
        evidence_text="Stored passage.",
        confidence=0.9,
        citation=review,
    )
    rendered = summarize_review_evidence([item, item]).render()
    assert "1 positive finding" in rendered
    assert "2 positive" not in rendered
