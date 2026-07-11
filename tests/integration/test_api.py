"""Contract tests for the HTTP vertical slice."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from services.api.core_service import _domain_request
from services.api.main import create_app
from services.api.models import GenerateBuildsRequest
from services.api.settings import ApiSettings

from pc_build_recommender.domain import BuildProfile as DomainBuildProfile


def _without_impression_tokens(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_impression_tokens(item)
            for key, item in value.items()
            if key != "impression_token"
        }
    if isinstance(value, list):
        return [_without_impression_tokens(item) for item in value]
    return value


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = ApiSettings(
        environment="test",
        data_version="test-data-v1",
        ranking_model_version="deterministic-test-baseline-v1",
        compatibility_rule_version="compat-test-v1",
        solver_version="solver-test-v1",
        cors_origins=["http://localhost:3000"],
        admin_token="test-admin-token-0123456789abcdef",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def build_request() -> dict[str, Any]:
    return {
        "budget_sgd": 2500,
        "workloads": [
            {"name": "local_ai", "weight": 0.6},
            {"name": "gaming_1440p", "weight": 0.4},
        ],
        "existing_products": [],
        "requirements": {
            "minimum_gpu_vram_gb": 16,
            "minimum_memory_gb": 32,
            "storage_gb": 2000,
            "wifi_required": True,
            "case_size": "mid_tower",
        },
        "preferences": {
            "noise": "low",
            "upgradeability": "high",
            "power_efficiency": "medium",
            "preferred_brands": [],
            "excluded_brands": [],
        },
    }


@pytest.mark.integration
def test_health_readiness_freshness_and_version_headers(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "caller-request-1"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-request-id"] == "caller-request-1"
    assert response.headers["x-data-version"] == "test-data-v1"
    assert response.headers["x-ranking-model"] == "deterministic-test-baseline-v1"
    assert response.headers["x-compatibility-rule-version"] == "compat-test-v1"
    assert response.headers["x-solver-version"] == "solver-test-v1"

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "checks": {
            "catalogue": "ready",
            "build_profiles": "ready",
            "compatibility_engine": "ready",
        },
        "data_version": "test-data-v1",
        "ranking_model": "deterministic-test-baseline-v1",
        "rule_version": "compat-test-v1",
        "solver_version": "solver-test-v1",
    }

    freshness = client.get("/v1/system/freshness")
    assert freshness.status_code == 200
    assert freshness.json()["status"] == "fresh"
    assert freshness.json()["product_count"] >= 20
    assert freshness.json()["data_version"] == "test-data-v1"
    assert freshness.json()["production_ready"] is False
    assert freshness.json()["release_artifact_verification"] == "development_unverified"
    assert freshness.json()["readiness_blockers"]

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "pcbr_http_requests_total" in metrics.text
    assert 'route="/health/live"' in metrics.text
    assert "pcbr_http_request_duration_seconds_bucket" in metrics.text
    assert 'pcbr_entity_resolution_mapping_observation{state="unavailable"} 1' in metrics.text
    assert 'pcbr_pipeline_operations_observation{state="unavailable"} 1' in metrics.text


@pytest.mark.integration
def test_domain_metrics_are_recorded_only_from_validated_api_responses(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    generated = client.post("/v1/builds/generate", json=build_request)
    assert generated.status_code == 200, generated.text

    search = client.post(
        "/v1/products/search",
        json={"query": "16 GB NVIDIA", "category": "gpu", "limit": 10},
    )
    assert search.status_code == 200, search.text
    search_payload = search.json()
    assert search_payload["retrieved_candidates"] >= search_payload["total"]

    compatibility = client.post(
        "/v1/compatibility/check",
        json={"components": [{"product_id": "cpu-amd-7600", "category": "cpu"}]},
    )
    assert compatibility.status_code == 200, compatibility.text
    assert compatibility.json()["status"] == "unknown"

    interaction = client.post(
        "/v1/interactions",
        json={
            "event_type": "build_saved",
            "session_id": "domain-metrics-session",
            "build_id": generated.json()["builds"][0]["build_id"],
        },
    )
    assert interaction.status_code == 202, interaction.text

    freshness = client.get("/v1/system/freshness")
    assert freshness.status_code == 200, freshness.text

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert (
        'pcbr_build_generation_total{outcome="complete",solver_status="FEASIBLE",'
        'solver_ran="false"}'
    ) in metrics.text
    assert 'pcbr_builds_returned_total{outcome="complete"}' in metrics.text
    assert "pcbr_product_search_requests_total" in metrics.text
    assert 'pcbr_product_search_candidates_total{stage="retrieved"}' in metrics.text
    assert (
        'pcbr_performance_signals_total{basis="relative",confidence="low",'
        'decision="deterministic_baseline"}'
    ) in metrics.text
    assert 'pcbr_performance_fallbacks_total{decision="deterministic_baseline"}' in metrics.text
    assert 'pcbr_compatibility_requests_total{status="unknown",feasible="false"}' in metrics.text
    assert 'pcbr_compatibility_checks_total{status="unknown"}' in metrics.text
    assert 'pcbr_interaction_events_total{event_type="build_saved"}' in metrics.text
    assert (
        'pcbr_catalogue_freshness_observations_total{status="fresh",production_ready="false"}'
    ) in metrics.text
    assert (
        'pcbr_serving_release_artifact_verification{status="development_unverified"} 1'
        in metrics.text
    )


@pytest.mark.integration
def test_generate_returns_diverse_refresh_safe_compatible_builds(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    generated = client.post("/v1/builds/generate", json=build_request)

    assert generated.status_code == 200, generated.text
    payload = generated.json()
    assert payload["status"] == "complete"
    assert 3 <= len(payload["builds"]) <= 5
    assert payload["data_version"] == "test-data-v1"
    assert payload["ranking_model"] == "deterministic-test-baseline-v1"
    assert payload["retrieval_model"] == "deterministic-token-lexical-baseline-v1"
    assert payload["performance_model"] == "deterministic-relative-performance-baseline-v1"
    assert payload["rule_version"] == "compat-test-v1"
    assert payload["solver_version"] == "solver-test-v1"
    assert payload["solver_status"] == "FEASIBLE"
    assert payload["solver_ran"] is False
    assert payload["solver_profile_statuses"] == []
    assert payload["solver_validator_rejections"] == 0

    configurations: set[tuple[str, ...]] = set()
    required_categories = {
        "cpu",
        "gpu",
        "motherboard",
        "memory",
        "storage",
        "psu",
        "cooler",
        "case",
    }
    for build in payload["builds"]:
        assert build["total_price_sgd"] <= build_request["budget_sgd"]
        assert build["compatibility_status"] in {"pass", "warning"}
        assert all(
            check["status"] not in {"fail", "unknown"} for check in build["compatibility_checks"]
        )
        categories = {component["category"] for component in build["components"]}
        assert categories == required_categories
        configuration = tuple(component["product_id"] for component in build["components"])
        configurations.add(configuration)
        assert all(component["listing_id"] for component in build["components"])
        assert all(
            signal["basis"] == "relative"
            for component in build["components"]
            for signal in component["performance_signals"]
        )
        assert all(
            signal["decision"] == "deterministic_baseline"
            for component in build["components"]
            for signal in component["performance_signals"]
        )
        assert all(
            alternative["compatibility_status"] in {"pass", "warning"}
            for component in build["components"]
            for alternative in component["alternatives"]
        )
    assert len(configurations) == len(payload["builds"])

    refreshed_request = client.get(f"/v1/requests/{payload['request_id']}/builds")
    assert refreshed_request.status_code == 200
    assert _without_impression_tokens(refreshed_request.json()) == _without_impression_tokens(
        payload
    )

    first_build = payload["builds"][0]
    refreshed_build = client.get(f"/v1/builds/{first_build['build_id']}")
    assert refreshed_build.status_code == 200
    assert _without_impression_tokens(refreshed_build.json()) == _without_impression_tokens(
        first_build
    )


@pytest.mark.integration
def test_public_build_share_is_allow_listed_and_revocable(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    generated = client.post("/v1/builds/generate", json=build_request).json()
    build = generated["builds"][0]

    created = client.post(f"/v1/builds/{build['build_id']}/shares")

    assert created.status_code == 201, created.text
    creation = created.json()
    assert creation["share_id"].startswith("share_")
    assert len(creation["revocation_token"]) >= 32
    assert creation["expires_at"] > creation["created_at"]
    assert created.headers["cache-control"] == "no-store"
    assert {
        value.strip().casefold() for value in created.headers["vary"].split(",")
    } == {"origin"}

    shared = client.get(f"/v1/build-shares/{creation['share_id']}")

    assert shared.status_code == 200, shared.text
    payload = shared.json()
    assert payload["share_id"] == creation["share_id"]
    snapshot = payload["snapshot"]
    assert {component["category"] for component in snapshot["components"]} == {
        "cpu",
        "gpu",
        "motherboard",
        "memory",
        "storage",
        "psu",
        "cooler",
        "case",
    }
    serialized = str(snapshot)
    for forbidden in (
        "build_id",
        "request_id",
        "product_id",
        "listing_id",
        "listing_url",
        "already_owned",
        "performance_signals",
        "alternatives",
    ):
        assert forbidden not in serialized

    wrong_token = client.post(
        f"/v1/build-shares/{creation['share_id']}/revoke",
        json={"revocation_token": "x" * 43},
    )
    assert wrong_token.status_code == 404

    revoked = client.post(
        f"/v1/build-shares/{creation['share_id']}/revoke",
        json={"revocation_token": creation["revocation_token"]},
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["share_id"] == creation["share_id"]
    assert client.get(f"/v1/build-shares/{creation['share_id']}").status_code == 404


@pytest.mark.integration
def test_admin_operations_is_token_protected_and_returns_only_aggregate_evidence(
    client: TestClient,
) -> None:
    disabled_settings = ApiSettings(environment="test")
    with TestClient(create_app(disabled_settings)) as disabled_client:
        disabled = disabled_client.get("/v1/admin/operations")
    assert disabled.status_code == 404
    assert disabled.json()["error"]["code"] == "admin_surface_disabled"

    unauthenticated = client.get("/v1/admin/operations")
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["error"]["code"] == "admin_authentication_required"

    operations = client.get(
        "/v1/admin/operations",
        headers={"X-PCBR-Admin-Token": "test-admin-token-0123456789abcdef"},
    )
    assert operations.status_code == 200, operations.text
    payload = operations.json()
    assert payload["mode"] == "demo"
    assert payload["data_version"] == "test-data-v1"
    assert payload["mapping_queue"] is None
    assert payload["price_freshness"] is None
    assert payload["pipeline_failure_events_available"] is False
    assert payload["notes"]
    assert "listing_url" not in str(payload)
    assert operations.headers["cache-control"] == "no-store"
    assert {
        value.strip().casefold() for value in operations.headers["vary"].split(",")
    } == {"origin", "x-pcbr-admin-token"}

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'pcbr_entity_resolution_mapping_observation{state="unavailable"} 1' in metrics.text
    assert 'pcbr_pipeline_operations_observation{state="unavailable"} 1' in metrics.text
    assert "pcbr_entity_resolution_manual_review_queue_items" not in metrics.text
    assert "pcbr_pipeline_operation_failures_in_window" not in metrics.text

    preflight = client.options(
        "/v1/admin/operations",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-PCBR-Admin-Token",
        },
    )
    assert preflight.status_code == 200
    assert "x-pcbr-admin-token" in preflight.headers["access-control-allow-headers"].casefold()


@pytest.mark.integration
def test_infeasible_response_is_explanatory_and_contains_no_builds(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    build_request["budget_sgd"] = 500

    response = client.post("/v1/builds/generate", json=build_request)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "infeasible"
    assert payload["solver_status"] == "INFEASIBLE"
    assert payload["solver_ran"] is False
    assert payload["builds"] == []
    assert payload["infeasibility"]["reasons"]
    assert "budget_too_low" in {reason["code"] for reason in payload["infeasibility"]["reasons"]}

    refreshed = client.get(f"/v1/requests/{payload['request_id']}/builds")
    assert refreshed.status_code == 200
    assert refreshed.json() == payload


@pytest.mark.integration
def test_http_requirements_and_profiles_normalize_to_domain_semantics() -> None:
    request = GenerateBuildsRequest.model_validate(
        {
            "budget_sgd": 2500,
            "workloads": [{"name": "local_ai", "weight": 1}],
            "requirements": {
                "minimum_gpu_vram_gb": 0,
                "wifi_required": None,
                "in_stock_only": False,
            },
            "max_builds": 2,
            "performance_target": "  120 FPS at 1440p high settings  ",
            "requested_profiles": ["lowest_power", "best_value"],
        }
    )

    assert request.requirements.minimum_gpu_vram_gb is None
    assert request.requirements.wifi_required is False
    domain_request = _domain_request(request)
    assert domain_request.requirements.minimum_gpu_vram_gb is None
    assert domain_request.requirements.wifi_required is False
    assert domain_request.requirements.in_stock_only is False
    assert domain_request.performance_target == "120 FPS at 1440p high settings"
    assert domain_request.requested_profiles == [
        DomainBuildProfile.LOWEST_POWER,
        DomainBuildProfile.BEST_VALUE,
    ]


@pytest.mark.integration
def test_performance_target_is_optional_and_bounded(client: TestClient) -> None:
    base_request = {
        "budget_sgd": 2500,
        "workloads": [{"name": "local_ai", "weight": 1}],
    }

    assert client.post("/v1/builds/generate", json=base_request).status_code == 200
    assert (
        client.post(
            "/v1/builds/generate",
            json={**base_request, "performance_target": " "},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/builds/generate",
            json={**base_request, "performance_target": "x" * 201},
        ).status_code
        == 422
    )


@pytest.mark.integration
def test_requested_profiles_cannot_silently_exceed_max_builds(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    build_request["max_builds"] = 1
    build_request["requested_profiles"] = ["best_overall", "best_value"]

    response = client.post("/v1/builds/generate", json=build_request)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.integration
def test_partial_response_is_explicit_when_fewer_than_three_builds_fit(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    build_request["budget_sgd"] = 1900

    response = client.post("/v1/builds/generate", json=build_request)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert 1 <= len(payload["builds"]) < 3
    assert all(build["total_price_sgd"] <= 1900 for build in payload["builds"])


@pytest.mark.integration
def test_retained_component_is_selected_and_excluded_from_total_when_owned(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    build_request["budget_sgd"] = 1500
    build_request["existing_products"] = [
        {
            "product_id": "gpu-rtx-5060ti-16",
            "category": "gpu",
            "canonical_name": "NVIDIA GeForce RTX 5060 Ti 16 GB",
            "include_in_budget": False,
        }
    ]

    response = client.post("/v1/builds/generate", json=build_request)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "complete"
    for build in payload["builds"]:
        retained_gpu = next(
            component for component in build["components"] if component["category"] == "gpu"
        )
        assert retained_gpu["product_id"] == "gpu-rtx-5060ti-16"
        assert retained_gpu["already_owned"] is True
        assert build["total_price_sgd"] <= 1500


@pytest.mark.integration
def test_validation_and_not_found_use_stable_error_envelope(client: TestClient) -> None:
    invalid = client.post(
        "/v1/builds/generate",
        headers={"X-Request-ID": "invalid-request"},
        json={"budget_sgd": -1, "workloads": []},
    )

    assert invalid.status_code == 422
    error = invalid.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"] == "invalid-request"
    assert error["details"]

    missing = client.get("/v1/builds/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "build_not_found"
    assert missing.json()["error"]["request_id"] == missing.headers["x-request-id"]


@pytest.mark.integration
def test_product_search_detail_and_evidence_contracts(client: TestClient) -> None:
    response = client.post(
        "/v1/products/search",
        json={"query": "16 GB NVIDIA", "category": "gpu", "limit": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 2
    assert payload["retrieval_model"] == "deterministic-token-baseline-v1"
    product_id = payload["products"][0]["product_id"]

    detail = client.get(f"/v1/products/{product_id}")
    prices = client.get(f"/v1/products/{product_id}/prices")
    benchmarks = client.get(f"/v1/products/{product_id}/benchmarks")
    reviews = client.get(f"/v1/products/{product_id}/reviews")

    assert detail.status_code == prices.status_code == benchmarks.status_code == 200
    assert reviews.status_code == 200
    assert detail.json()["source_url"]
    assert prices.json()["observations"][0]["stock_status"] == "in_stock"
    assert prices.json()["observations"][0]["condition"] == "new"
    assert prices.json()["observations"][0]["current_offer_eligible"] is True
    assert prices.json()["price_intelligence"]["basis"] == "descriptive_observed_history"
    assert prices.json()["price_intelligence"]["history_sufficient"] is False
    assert prices.json()["price_intelligence"]["percentile_90d"] is None
    assert prices.json()["price_intelligence"]["volatility_90d_pct"] is None
    assert prices.json()["price_intelligence"]["labels"] == ["Insufficient price history"]
    assert benchmarks.json()["benchmarks"][0]["basis"] == "predicted"
    assert reviews.json()["evidence"] == []


@pytest.mark.integration
def test_product_search_pagination_facets_coverage_and_cursor_are_deterministic(
    client: TestClient,
) -> None:
    first = client.post(
        "/v1/products/search",
        json={"query": "", "in_stock_only": False, "limit": 3, "page": 1, "page_size": 3},
    )

    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert first_payload["total"] > len(first_payload["products"])
    assert first_payload["pagination"] == {
        "page": 1,
        "page_size": 3,
        "total_pages": (first_payload["total"] + 2) // 3,
        "has_previous": False,
        "has_next": True,
        "previous_cursor": None,
        "next_cursor": first_payload["pagination"]["next_cursor"],
    }
    assert first_payload["pagination"]["next_cursor"]
    assert (
        sum(item["count"] for item in first_payload["facets"]["categories"])
        == first_payload["coverage"]["canonical_products"]
    )
    assert first_payload["coverage"]["scope_label"] == "Controlled illustrative API demo"
    assert first_payload["coverage"]["category_count"] == 8

    second_request = {
        "query": "",
        "in_stock_only": False,
        "limit": 3,
        "page": 2,
        "page_size": 3,
        "cursor": first_payload["pagination"]["next_cursor"],
    }
    second = client.post("/v1/products/search", json=second_request)
    repeated = client.post("/v1/products/search", json=second_request)

    assert second.status_code == repeated.status_code == 200
    second_payload = second.json()
    assert _without_impression_tokens(second_payload) == _without_impression_tokens(
        repeated.json()
    )
    assert second_payload["pagination"]["page"] == 2
    assert second_payload["pagination"]["has_previous"] is True
    assert second_payload["pagination"]["previous_cursor"]
    first_ids = {item["product_id"] for item in first_payload["products"]}
    second_ids = {item["product_id"] for item in second_payload["products"]}
    assert first_ids.isdisjoint(second_ids)

    malformed = client.post(
        "/v1/products/search",
        json={"query": "", "limit": 3, "page": 2, "page_size": 3, "cursor": "not-a-cursor"},
    )
    wrong_scope = client.post(
        "/v1/products/search",
        json={
            **second_request,
            "query": "different search",
        },
    )
    assert malformed.status_code == wrong_scope.status_code == 422
    assert malformed.json()["error"]["code"] == "invalid_pagination_cursor"
    assert wrong_scope.json()["error"]["code"] == "invalid_pagination_cursor"


@pytest.mark.integration
def test_compatible_search_substitutes_and_filters_failures_and_unknowns(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    generated = client.post("/v1/builds/generate", json=build_request).json()
    build_id = generated["builds"][0]["build_id"]

    hard_failure = client.post(
        "/v1/products/search",
        json={
            "query": "Oversized",
            "category": "gpu",
            "compatible_with_build_id": build_id,
        },
    )
    assert hard_failure.status_code == 200
    assert hard_failure.json()["products"] == []
    assert hard_failure.json()["filtered_incompatible"] == 1
    assert hard_failure.json()["filtered_unknown"] == 0

    required_field_unknown = client.post(
        "/v1/products/search",
        json={
            "query": "Unverified-Dimensions",
            "category": "gpu",
            "compatible_with_build_id": build_id,
        },
    )
    assert required_field_unknown.status_code == 200
    assert required_field_unknown.json()["products"] == []
    assert required_field_unknown.json()["filtered_incompatible"] == 0
    assert required_field_unknown.json()["filtered_unknown"] == 1

    compatible = client.post(
        "/v1/products/search",
        json={
            "query": "Radeon",
            "category": "gpu",
            "compatible_with_build_id": build_id,
        },
    )
    assert compatible.status_code == 200
    assert compatible.json()["products"]
    assert all(
        product["compatibility_status"] in {"pass", "warning"}
        for product in compatible.json()["products"]
    )


@pytest.mark.integration
def test_compatibility_endpoint_reports_unknown_for_incomplete_input(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/compatibility/check",
        json={"components": [{"product_id": "cpu-amd-7600", "category": "cpu"}]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "unknown"
    assert response.json()["is_feasible"] is False
    assert response.json()["rule_version"] == "compat-test-v1"


@pytest.mark.integration
def test_compatible_replacement_creates_new_refresh_safe_build(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    generated = client.post("/v1/builds/generate", json=build_request).json()
    original = generated["builds"][0]
    original_cpu = next(
        component for component in original["components"] if component["category"] == "cpu"
    )
    replacement_product_id = (
        "cpu-amd-7600" if original_cpu["product_id"] != "cpu-amd-7600" else "cpu-amd-7700"
    )

    response = client.post(
        f"/v1/builds/{original['build_id']}/replace",
        json={
            "category": "cpu",
            "replacement_product_id": replacement_product_id,
            "mode": "lock_other_components",
        },
    )

    assert response.status_code == 200, response.text
    replacement = response.json()
    assert replacement["build"]["build_id"] != original["build_id"]
    assert replacement["changed_categories"] == ["cpu"]
    assert all(
        check["status"] not in {"fail", "unknown"}
        for check in replacement["build"]["compatibility_checks"]
    )
    refreshed_new = client.get(f"/v1/builds/{replacement['build']['build_id']}")
    assert _without_impression_tokens(refreshed_new.json()) == _without_impression_tokens(
        replacement["build"]
    )
    refreshed_original = client.get(f"/v1/builds/{original['build_id']}")
    assert _without_impression_tokens(refreshed_original.json()) == _without_impression_tokens(
        original
    )


@pytest.mark.integration
def test_unlocked_reoptimization_mode_is_explicitly_rejected(
    client: TestClient, build_request: dict[str, Any]
) -> None:
    generated = client.post("/v1/builds/generate", json=build_request).json()
    original = generated["builds"][0]
    original_cpu = next(
        component for component in original["components"] if component["category"] == "cpu"
    )
    replacement_product_id = (
        "cpu-amd-7600" if original_cpu["product_id"] != "cpu-amd-7600" else "cpu-amd-7700"
    )

    response = client.post(
        f"/v1/builds/{original['build_id']}/replace",
        json={
            "category": "cpu",
            "replacement_product_id": replacement_product_id,
            "mode": "reoptimize_unlocked",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "replacement_mode_not_available"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert _without_impression_tokens(
        client.get(f"/v1/builds/{original['build_id']}").json()
    ) == _without_impression_tokens(original)


@pytest.mark.integration
def test_interaction_ingestion_is_versioned(client: TestClient) -> None:
    response = client.post(
        "/v1/interactions",
        json={
            "event_type": "build_saved",
            "session_id": "session-test",
            "build_id": "build-client-cache-id",
            "rank_position": 1,
            "model_version": "forged-client-model-v99",
            "data_version": "forged-client-data-v99",
            "rule_version": "client-rule-v7",
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert response.json()["event_id"].startswith("evt_")
    assert response.json()["data_version"] == "test-data-v1"
    assert response.json()["rule_version"] == "compat-test-v1"
    assert response.json()["trust_level"] == "legacy_untrusted"
    assert response.json()["replayed"] is False

    # Starlette's TestClient annotation exposes only the ASGI callable, while this
    # integration assertion intentionally inspects the application state fixture.
    service = cast(Any, client).app.state.application_service
    canonical_event = service._interactions[-1][1]
    assert canonical_event.model_version == "deterministic-test-baseline-v1"
    assert canonical_event.data_version == "test-data-v1"
    assert canonical_event.rule_version == "compat-test-v1"

    defaulted = client.post(
        "/v1/interactions",
        json={"event_type": "build_viewed", "session_id": "session-test", "build_id": "b1"},
    )
    assert defaulted.status_code == 202
    assert defaulted.json()["rule_version"] == "compat-test-v1"


@pytest.mark.integration
def test_signed_product_interaction_is_canonical_idempotent_and_tamper_safe(
    client: TestClient,
) -> None:
    search = client.post(
        "/v1/products/search",
        json={"query": "GPU", "page": 1, "page_size": 3, "in_stock_only": False},
    )
    assert search.status_code == 200, search.text
    item = search.json()["products"][0]
    token = item["impression_token"]
    event = {
        "event_type": "component_viewed",
        "session_id": "session-signed-product",
        "impression_token": token,
    }
    headers = {"Idempotency-Key": "product-view-retry-0001"}

    accepted = client.post("/v1/interactions", json=event, headers=headers)
    replayed = client.post("/v1/interactions", json=event, headers=headers)

    assert accepted.status_code == replayed.status_code == 202
    assert accepted.json()["trust_level"] == "verified_impression"
    assert accepted.json()["replayed"] is False
    assert replayed.json()["replayed"] is True
    assert replayed.json()["event_id"] == accepted.json()["event_id"]
    assert replayed.json()["accepted_at"] == accepted.json()["accepted_at"]

    service = cast(Any, client).app.state.application_service
    matching_events = [
        item for item in service._interactions if item[0] == accepted.json()["event_id"]
    ]
    assert len(matching_events) == 1
    canonical = next(
        item[1] for item in service._interactions if item[0] == accepted.json()["event_id"]
    )
    assert canonical.query_id == search.json()["query_id"]
    assert canonical.product_id == item["product_id"]
    assert canonical.rank_position == 1
    assert canonical.model_version == search.json()["retrieval_model"]
    assert canonical.impression_id.startswith("imp_")
    assert canonical.impression_token is None
    assert canonical.idempotency_key_sha256 != "product-view-retry-0001"

    conflict = client.post(
        "/v1/interactions",
        json={**event, "metadata": {"changed": True}},
        headers=headers,
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "interaction_idempotency_conflict"

    forged_rank = client.post(
        "/v1/interactions",
        json={**event, "rank_position": 99},
        headers={"Idempotency-Key": "product-view-retry-0002"},
    )
    assert forged_rank.status_code == 409
    assert forged_rank.json()["error"]["code"] == "impression_claim_mismatch"

    replacement = "A" if token[-1] != "A" else "B"
    tampered = client.post(
        "/v1/interactions",
        json={**event, "impression_token": token[:-1] + replacement},
        headers={"Idempotency-Key": "product-view-retry-0003"},
    )
    assert tampered.status_code == 401
    assert tampered.json()["error"]["code"] == "invalid_impression"

    metadata_claim = client.post(
        "/v1/interactions",
        json={**event, "metadata": {"rank_position": 999}},
        headers={"Idempotency-Key": "product-view-reserved-metadata"},
    )
    assert metadata_claim.status_code == 422
    assert metadata_claim.json()["error"]["code"] == "reserved_interaction_metadata"


@pytest.mark.integration
def test_signed_build_interaction_uses_server_result_claims(client: TestClient) -> None:
    generated = client.post(
        "/v1/builds/generate",
        json={
            "budget_sgd": 2500,
            "workloads": [{"name": "local_ai", "weight": 1.0}],
        },
    )
    assert generated.status_code == 200, generated.text
    build = generated.json()["builds"][0]

    accepted = client.post(
        "/v1/interactions",
        json={
            "event_type": "build_saved",
            "session_id": "session-signed-build",
            "impression_token": build["impression_token"],
        },
        headers={"Idempotency-Key": "build-save-retry-0001"},
    )

    assert accepted.status_code == 202, accepted.text
    service = cast(Any, client).app.state.application_service
    canonical = service._interactions[-1][1]
    assert canonical.query_id == generated.json()["request_id"]
    assert canonical.build_id == build["build_id"]
    assert canonical.rank_position == 1
    assert canonical.trust_level == "verified_impression"


@pytest.mark.integration
def test_interaction_rank_positions_are_one_based_and_reference_a_result(
    client: TestClient,
) -> None:
    zero_rank = client.post(
        "/v1/interactions",
        json={
            "event_type": "component_viewed",
            "session_id": "session-test",
            "product_id": "gpu-1",
            "rank_position": 0,
        },
    )
    missing_result = client.post(
        "/v1/interactions",
        json={
            "event_type": "component_viewed",
            "session_id": "session-test",
            "rank_position": 1,
        },
    )

    assert zero_rank.status_code == 422
    assert zero_rank.json()["error"]["code"] == "validation_error"
    assert missing_result.status_code == 422
    assert missing_result.json()["error"]["code"] == "validation_error"
