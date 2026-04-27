"""Unit coverage for bounded recommendation-domain Prometheus metrics."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from services.api.metrics import DomainMetrics
from services.api.models import PerformanceSignal


@pytest.mark.parametrize(
    ("basis", "decision", "message"),
    [
        ("observed", "precise_model_prediction", "observed_benchmark"),
        ("predicted", "observed_benchmark", "precise_model_prediction"),
        ("relative", "observed_benchmark", "bounded fallback"),
        ("insufficient_data", "deterministic_baseline", "must not claim"),
    ],
)
def test_performance_signal_rejects_a_decision_that_conflicts_with_its_basis(
    basis: str,
    decision: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        PerformanceSignal(
            workload="local_ai",
            metric="relative_score",
            value=80.0,
            basis=basis,  # type: ignore[arg-type]
            decision=decision,  # type: ignore[arg-type]
        )


def test_performance_signal_keeps_decision_optional_for_legacy_response_payloads() -> None:
    signal = PerformanceSignal(
        workload="local_ai",
        metric="relative_score",
        value=80.0,
        basis="relative",
    )

    assert signal.decision is None


def test_domain_metrics_render_recommendation_outcomes_without_dynamic_labels() -> None:
    metrics = DomainMetrics()

    metrics.record_build_generation(
        outcome="complete",
        solver_status="OPTIMAL",
        solver_ran=True,
        build_count=3,
        validator_rejections=2,
        profile_statuses=(
            ("best_overall", "OPTIMAL"),
            ("best_value", "FEASIBLE"),
        ),
    )
    metrics.record_component_replacement(solver_status="FEASIBLE", solver_ran=True)
    metrics.record_product_search(
        result_count=4,
        ranked_candidates=4,
        retrieved_candidates=11,
        filtered_category=2,
        filtered_brand=1,
        filtered_incompatible=2,
        filtered_unknown=1,
    )
    metrics.record_product_search(
        result_count=0,
        ranked_candidates=0,
        retrieved_candidates=0,
        filtered_category=0,
        filtered_brand=0,
        filtered_incompatible=0,
        filtered_unknown=0,
    )
    metrics.record_performance_signals(
        signals=(
            ("observed", None, "observed_benchmark"),
            ("predicted", "high", "precise_model_prediction"),
            ("relative", "low", "model_not_promotion_eligible"),
        )
    )
    metrics.record_compatibility(
        status="warning",
        is_feasible=True,
        check_statuses=("pass", "warning", "unknown"),
    )
    metrics.record_interaction(event_type="build_saved")
    metrics.record_freshness(
        status="stale",
        production_ready=False,
        product_count=3000,
        listing_count=0,
        release_blocker_count=3,
        release_artifact_verification="verified",
    )
    metrics.record_admin_operations(
        entity_resolution_mapping_available=True,
        manual_review_count=7,
        unmatched_offer_count=11,
        rejected_conflict_count=2,
        model_rejected_count=3,
        missing_critical_field_value_count=14,
        pipeline_receipts_available=True,
        pipeline_failed_count=2,
        pipeline_invalid_receipt_count=1,
        pipeline_window_hours=168,
        pipeline_receipts_truncated=True,
    )

    rendered = metrics.render()

    assert (
        'pcbr_build_generation_total{outcome="complete",solver_status="OPTIMAL",'
        'solver_ran="true"} 1'
    ) in rendered
    assert 'pcbr_builds_returned_total{outcome="complete"} 3' in rendered
    assert (
        'pcbr_optimizer_profile_outcomes_total{profile="best_value",status="FEASIBLE"} 1'
        in rendered
    )
    assert "pcbr_optimizer_validator_rejections_total 2" in rendered
    assert (
        'pcbr_component_replacements_total{solver_status="FEASIBLE",solver_ran="true"} 1'
        in rendered
    )
    assert "pcbr_product_search_requests_total 2" in rendered
    assert "pcbr_product_search_results_total 4" in rendered
    assert 'pcbr_product_search_candidates_total{stage="retrieved"} 10' in rendered
    assert 'pcbr_product_search_candidates_total{stage="after_category_filter"} 8' in rendered
    assert 'pcbr_product_search_candidates_total{stage="after_brand_filter"} 7' in rendered
    assert 'pcbr_product_search_candidates_total{stage="ranked"} 4' in rendered
    assert 'pcbr_product_search_filtered_total{reason="category"} 2' in rendered
    assert 'pcbr_product_search_filtered_total{reason="brand"} 1' in rendered
    assert 'pcbr_product_search_filtered_total{reason="incompatible"} 2' in rendered
    assert 'pcbr_product_search_filtered_total{reason="unknown"} 1' in rendered
    assert "pcbr_product_search_empty_total 1" in rendered
    assert (
        'pcbr_performance_signals_total{basis="observed",confidence="not_applicable",'
        'decision="observed_benchmark"} 1' in rendered
    )
    assert (
        'pcbr_performance_signals_total{basis="predicted",confidence="high",'
        'decision="precise_model_prediction"} 1' in rendered
    )
    assert (
        'pcbr_performance_signals_total{basis="relative",confidence="low",'
        'decision="model_not_promotion_eligible"} 1' in rendered
    )
    assert 'pcbr_performance_fallbacks_total{decision="model_not_promotion_eligible"} 1' in rendered
    assert 'pcbr_compatibility_requests_total{status="warning",feasible="true"} 1' in rendered
    assert 'pcbr_compatibility_checks_total{status="unknown"} 1' in rendered
    assert 'pcbr_interaction_events_total{event_type="build_saved"} 1' in rendered
    assert (
        'pcbr_catalogue_freshness_observations_total{status="stale",production_ready="false"} 1'
    ) in rendered
    assert 'pcbr_catalogue_freshness_status{status="stale"} 1' in rendered
    assert 'pcbr_catalogue_freshness_status{status="fresh"} 0' in rendered
    assert "pcbr_catalogue_production_ready 0" in rendered
    assert "pcbr_catalogue_freshness_probe_success 1" in rendered
    assert "pcbr_catalogue_products 3000" in rendered
    assert "pcbr_catalogue_listings 0" in rendered
    assert "pcbr_catalogue_release_blockers 3" in rendered
    assert 'pcbr_serving_release_artifact_verification{status="verified"} 1' in rendered
    assert 'pcbr_entity_resolution_mapping_observation{state="available"} 1' in rendered
    assert "pcbr_entity_resolution_manual_review_queue_items 7" in rendered
    assert "pcbr_entity_resolution_unmatched_offer_items 11" in rendered
    assert "pcbr_entity_resolution_rejected_conflict_items 2" in rendered
    assert "pcbr_entity_resolution_model_rejected_items 3" in rendered
    assert "pcbr_catalogue_missing_critical_field_values 14" in rendered
    assert 'pcbr_pipeline_operations_observation{state="available"} 1' in rendered
    assert "pcbr_pipeline_operation_failures_in_window 2" in rendered
    assert "pcbr_pipeline_operation_invalid_receipts_in_window 1" in rendered
    assert "pcbr_pipeline_operation_receipt_window_hours 168" in rendered
    assert "pcbr_pipeline_operation_receipts_truncated 1" in rendered
    assert "query_id" not in rendered
    assert "model_version" not in rendered


def test_domain_metrics_freshness_probe_failure_fails_closed_and_preserves_counts() -> None:
    metrics = DomainMetrics()
    metrics.record_freshness(
        status="fresh",
        production_ready=True,
        product_count=3000,
        listing_count=10000,
        release_blocker_count=0,
        release_artifact_verification="verified",
    )

    metrics.record_freshness_probe_failure()
    rendered = metrics.render()

    assert 'pcbr_catalogue_freshness_status{status="degraded"} 1' in rendered
    assert "pcbr_catalogue_production_ready 0" in rendered
    assert "pcbr_catalogue_freshness_probe_success 0" in rendered
    assert "pcbr_catalogue_products 3000" in rendered
    assert "pcbr_catalogue_listings 10000" in rendered
    assert "pcbr_catalogue_release_blockers 1" in rendered


def test_domain_metrics_require_complete_operational_evidence_and_clear_stale_gauges() -> None:
    metrics = DomainMetrics()

    with pytest.raises(ValueError, match="available entity-resolution mapping"):
        metrics.record_admin_operations(
            entity_resolution_mapping_available=True,
            manual_review_count=1,
            unmatched_offer_count=None,
            rejected_conflict_count=0,
            model_rejected_count=0,
            missing_critical_field_value_count=0,
            pipeline_receipts_available=False,
            pipeline_failed_count=None,
            pipeline_invalid_receipt_count=None,
            pipeline_window_hours=None,
            pipeline_receipts_truncated=None,
        )

    metrics.record_admin_operations(
        entity_resolution_mapping_available=False,
        manual_review_count=None,
        unmatched_offer_count=None,
        rejected_conflict_count=None,
        model_rejected_count=None,
        missing_critical_field_value_count=None,
        pipeline_receipts_available=False,
        pipeline_failed_count=None,
        pipeline_invalid_receipt_count=None,
        pipeline_window_hours=None,
        pipeline_receipts_truncated=None,
    )

    rendered = metrics.render()

    assert 'pcbr_entity_resolution_mapping_observation{state="unavailable"} 1' in rendered
    assert 'pcbr_pipeline_operations_observation{state="unavailable"} 1' in rendered
    assert "pcbr_entity_resolution_manual_review_queue_items" not in rendered
    assert "pcbr_pipeline_operation_failures_in_window" not in rendered


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    [
        (
            "record_build_generation",
            {
                "outcome": "unexpected",
                "solver_status": "OPTIMAL",
                "solver_ran": True,
                "build_count": 1,
                "validator_rejections": 0,
                "profile_statuses": (),
            },
            "build outcome",
        ),
        (
            "record_component_replacement",
            {"solver_status": "unbounded", "solver_ran": True},
            "solver status",
        ),
        (
            "record_compatibility",
            {"status": "unbounded", "is_feasible": True, "check_statuses": ()},
            "compatibility status",
        ),
        (
            "record_performance_signals",
            {"signals": (("arbitrary-client-string", "low", None),)},
            "performance basis",
        ),
        (
            "record_interaction",
            {"event_type": "arbitrary-client-string"},
            "interaction event",
        ),
        (
            "record_freshness",
            {
                "status": "unbounded",
                "production_ready": False,
                "product_count": 0,
                "listing_count": 0,
                "release_blocker_count": 0,
                "release_artifact_verification": "verified",
            },
            "freshness status",
        ),
        (
            "record_freshness",
            {
                "status": "fresh",
                "production_ready": False,
                "product_count": 0,
                "listing_count": 0,
                "release_blocker_count": 0,
                "release_artifact_verification": "unbounded",
            },
            "release artifact verification",
        ),
    ],
)
def test_domain_metrics_reject_unbounded_labels(
    method: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    metrics = DomainMetrics()

    with pytest.raises(ValueError, match=message):
        getattr(metrics, method)(**kwargs)
