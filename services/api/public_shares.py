"""Allow-listed, immutable projections for publicly shared build links."""

from __future__ import annotations

from services.api.models import BuildSummary, PublicBuildComponent, PublicBuildSnapshot

_MAX_PUBLIC_EXPLANATIONS = 4
_MAX_PUBLIC_WARNINGS = 4


def _bounded_text(value: str | None, maximum: int = 500) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized[:maximum] if normalized else None


def public_build_snapshot(build: BuildSummary) -> PublicBuildSnapshot:
    """Project a full build response into data safe for a public, durable link.

    This function intentionally does not copy build/request/product/listing IDs, retailer or
    listing URLs, ownership flags, replacement options, compatibility evidence sources, or
    benchmark-source URLs. A retained component has no per-component price in the public
    projection so the share cannot disclose that it belongs to the originating user.
    """

    return PublicBuildSnapshot(
        profile=build.profile,
        total_price_sgd=build.total_price_sgd,
        overall_score=build.overall_score,
        value_score=build.value_score,
        upgradeability_score=build.upgradeability_score,
        efficiency_score=build.efficiency_score,
        estimated_peak_power_w=build.estimated_peak_power_w,
        workload_scores=build.workload_scores,
        compatibility_status=build.compatibility_status,
        components=[
            PublicBuildComponent(
                category=component.category,
                canonical_name=component.canonical_name,
                brand=component.brand,
                price_sgd=None if component.already_owned else component.price_sgd,
                component_score=component.component_score,
                selection_reason=_bounded_text(
                    component.selection_reasons[0] if component.selection_reasons else None
                ),
            )
            for component in build.components
        ],
        explanations=[
            text
            for item in (build.explanation or [])
            if (text := _bounded_text(item if isinstance(item, str) else item.text)) is not None
        ][:_MAX_PUBLIC_EXPLANATIONS],
        warnings=[
            text
            for item in (build.warnings or [])
            if item.status.value == "warning"
            and (text := _bounded_text(item.message)) is not None
        ][:_MAX_PUBLIC_WARNINGS],
        generated_at=build.generated_at,
        data_version=build.data_version,
        ranking_model=build.ranking_model,
        rule_version=build.rule_version,
        solver_version=build.solver_version,
    )
