"""Tests for request-to-retrieval pipeline translation."""

from pc_build_recommender.application.pipeline import build_query_text
from pc_build_recommender.domain import (
    BuildGenerationRequest,
    WorkloadLabel,
    WorkloadPreference,
)


def test_build_query_text_includes_the_structured_performance_target() -> None:
    request = BuildGenerationRequest(
        budget_sgd=2500,
        workloads=[WorkloadPreference(name=WorkloadLabel.GAMING_1440P, weight=1.0)],
        performance_target="120 FPS at high settings",
    )

    assert "120 FPS at high settings" in build_query_text(request)
