"""Template-based explanations that never invent evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal

from .types import (
    ComponentSelection,
    EvidenceBasis,
    Explanation,
    ExplanationStatement,
    MetricDelta,
    MetricEvidence,
    ReplacementComparison,
    ReviewEvidence,
    normalise_citations,
)


def _format_number(value: Decimal | int | float) -> str:
    if isinstance(value, Decimal):
        rendered = format(value, "f")
    elif isinstance(value, int):
        return str(value)
    else:
        rendered = f"{value:.2f}"
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _format_money(value: Decimal | int | float, currency: str) -> str:
    amount = Decimal(str(value))
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    symbol = "S$" if currency.strip().upper() == "SGD" else f"{currency.strip().upper()} "
    return f"{sign}{symbol}{amount:.2f}"


def _signed(value: Decimal | int | float, *, money_currency: str | None = None) -> str:
    numeric = Decimal(str(value))
    sign = "+" if numeric > 0 else ""
    rendered = (
        _format_money(numeric, money_currency)
        if money_currency is not None
        else _format_number(numeric)
    )
    return f"{sign}{rendered}"


def _metric_basis(metric: MetricEvidence) -> str:
    if metric.basis is EvidenceBasis.OBSERVED:
        return "observed"
    if metric.basis is EvidenceBasis.PREDICTED:
        confidence = f", {metric.confidence} confidence" if metric.confidence else ""
        return f"predicted by {metric.model_version}{confidence}"
    return f"relative score versus {metric.relative_to}"


def _metric_statement(metric: MetricEvidence) -> ExplanationStatement:
    text = (
        f"{metric.label}: {_format_number(metric.value)} {metric.unit} "
        f"({_metric_basis(metric)})."
    )
    return ExplanationStatement(text=text, citations=metric.citations)


def explain_component_selection(selection: ComponentSelection) -> Explanation:
    """Explain one selected component from only caller-supplied cited facts."""

    statements: list[ExplanationStatement] = []
    for reason in selection.reasons:
        statements.append(
            ExplanationStatement(
                text=(
                    f"Selected {selection.product_name} for {selection.category}: "
                    f"{reason.statement}."
                ),
                citations=reason.citations,
            )
        )
    statements.extend(_metric_statement(metric) for metric in selection.metrics)
    return Explanation(statements=tuple(statements))


def explain_build_selection(selections: Iterable[ComponentSelection]) -> Explanation:
    """Create stable output independent of database iteration order."""

    ordered = sorted(
        selections,
        key=lambda item: (item.category.casefold(), item.product_name.casefold(), item.product_id),
    )
    statements = tuple(
        statement
        for selection in ordered
        for statement in explain_component_selection(selection).statements
    )
    return Explanation(statements=statements)


def _delta_statement(delta: MetricDelta) -> ExplanationStatement:
    difference = Decimal(str(delta.after.value)) - Decimal(str(delta.before.value))
    citations = normalise_citations(delta.before.citations + delta.after.citations)
    text = (
        f"{delta.before.label} changes by {_signed(difference)} {delta.before.unit} "
        f"({_format_number(delta.before.value)} {_metric_basis(delta.before)} to "
        f"{_format_number(delta.after.value)} {_metric_basis(delta.after)})."
    )
    return ExplanationStatement(text=text, citations=citations)


def explain_replacement(comparison: ReplacementComparison) -> Explanation:
    """Explain price, performance, compatibility, and constraint replacement deltas."""

    old_price = Decimal(str(comparison.old_delivered_price))
    new_price = Decimal(str(comparison.new_delivered_price))
    price_delta = new_price - old_price
    statements: list[ExplanationStatement] = [
        ExplanationStatement(
            text=(
                f"Replacing {comparison.old_product_name} with {comparison.new_product_name} "
                "changes delivered price by "
                f"{_signed(price_delta, money_currency=comparison.currency)} "
                f"({_format_money(old_price, comparison.currency)} to "
                f"{_format_money(new_price, comparison.currency)})."
            ),
            citations=comparison.price_citations,
        )
    ]
    statements.extend(_delta_statement(delta) for delta in comparison.metric_deltas)
    statements.append(
        ExplanationStatement(
            text=(
                f"Compatibility changes from {comparison.old_compatibility.upper()} to "
                f"{comparison.new_compatibility.upper()} under the stored compatibility rules."
            ),
            citations=comparison.compatibility_citations,
        )
    )
    for constraint in comparison.constraint_deltas:
        statements.append(
            ExplanationStatement(
                text=(
                    f"{constraint.name} changes from {constraint.before} to {constraint.after}."
                ),
                citations=constraint.citations,
            )
        )
    return Explanation(statements=tuple(statements))


def _count_phrase(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def summarize_review_evidence(
    evidence: Iterable[ReviewEvidence],
    *,
    minimum_confidence: float = 0.6,
) -> Explanation:
    """Summarise only stored, sufficiently confident review evidence.

    The function reports evidence counts and direction by aspect. It intentionally
    does not generate a product claim from review prose, so unsupported details cannot
    be introduced by this layer.
    """

    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between zero and one")
    unique: dict[str, ReviewEvidence] = {}
    for item in evidence:
        if item.confidence >= minimum_confidence:
            unique.setdefault(item.evidence_id, item)
    grouped: dict[str, list[ReviewEvidence]] = defaultdict(list)
    for item in unique.values():
        grouped[item.aspect.strip()].append(item)

    statements: list[ExplanationStatement] = []
    for aspect in sorted(grouped, key=str.casefold):
        items = grouped[aspect]
        positive = sum(item.sentiment == "positive" for item in items)
        concerns = sum(item.sentiment == "negative" for item in items)
        mixed = sum(item.sentiment == "mixed" for item in items)
        neutral = sum(item.sentiment == "neutral" for item in items)
        source_count = len({item.citation.source_id for item in items})
        parts: list[str] = []
        if positive:
            parts.append(_count_phrase(positive, "positive finding", "positive findings"))
        if concerns:
            parts.append(_count_phrase(concerns, "concern", "concerns"))
        if mixed:
            parts.append(_count_phrase(mixed, "mixed finding", "mixed findings"))
        if neutral:
            parts.append(_count_phrase(neutral, "neutral finding", "neutral findings"))
        findings = ", ".join(parts)
        text = (
            f"{aspect}: stored review evidence contains {findings} across "
            f"{_count_phrase(source_count, 'source', 'sources')}."
        )
        statements.append(
            ExplanationStatement(
                text=text,
                citations=normalise_citations(tuple(item.citation for item in items)),
            )
        )
    return Explanation(statements=tuple(statements))
