"""Production-facing FastAPI and application-adapter contract tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from services.api.core_service import CoreRecommendationService, _public_catalogue_readiness
from services.api.durability import SqlAlchemyDurableStore
from services.api.main import create_app
from services.api.service import InMemoryRecommendationService
from services.api.settings import ApiRuntimeSettings
from sqlalchemy import select

from pc_build_recommender.application import (
    ApplicationBuildGenerationResponse,
    ApplicationServices,
    ApplicationVersions,
)
from pc_build_recommender.catalog import (
    BuildShareRecord,
    CatalogReadinessReport,
    GeneratedBuildRecord,
    InteractionEventRecord,
    SearchQueryRecord,
    create_db_engine,
    create_session_factory,
    init_database,
)
from pc_build_recommender.domain import (
    BuildRequestSpec,
    BuildPreset,
    InteractionType,
    WorkloadLabel,
    WorkloadPreference,
)
from pc_build_recommender.domain import (
    ComponentKind as DomainCategory,
)
from pc_build_recommender.domain import (
    InteractionRecord as DomainInteractionEvent,
)
from pc_build_recommender.optimizer import OptimizationStatus
from pc_build_recommender.pipeline_operations import write_pipeline_operation_event


def _request() -> dict[str, Any]:
    return {
        "budget_sgd": 2500,
        "workloads": [{"name": "local_ai", "weight": 1}],
        "max_builds": 1,
    }


def _optimizer_response(
    status: OptimizationStatus, *, ran: bool
) -> ApplicationBuildGenerationResponse:
    return ApplicationBuildGenerationResponse(
        request_id="req_optimizer_contract",
        data_version="catalog-v1",
        ranking_model="ranker-v1",
        retrieval_model="retriever-v1",
        performance_model="performance-v1",
        rule_version="compat_v2",
        builds=[],
        infeasibility_reasons=["No complete build was published."],
        optimizer_status=status,
        optimizer_version="cp-sat-v1",
        optimizer_ran=ran,
    )


def _core_with_response(
    response: ApplicationBuildGenerationResponse,
) -> tuple[CoreRecommendationService, ApiRuntimeSettings]:
    versions = ApplicationVersions(
        data_version=response.data_version,
        ranking_model=response.ranking_model,
        rule_version=response.rule_version,
        optimizer_version=response.optimizer_version,
    )

    class Generator:
        compatibility_engine = SimpleNamespace(rule_version=response.rule_version)

        @staticmethod
        def generate(*_: Any, **__: Any) -> ApplicationBuildGenerationResponse:
            return response

    fake_services = SimpleNamespace(generate_builds=Generator(), versions=versions)
    service = object.__new__(CoreRecommendationService)
    service.services = cast(ApplicationServices, fake_services)
    service.settings = ApiRuntimeSettings(
        environment="test",
        data_version=response.data_version,
        ranking_model_version=response.ranking_model,
        compatibility_rule_version=response.rule_version,
        solver_version=response.optimizer_version,
    )
    service._mutation_lock = asyncio.Lock()
    service._durable_store = None
    service._interactions = []
    service._generated_at_by_request = {}
    return service, service.settings


def test_catalogue_readiness_summary_exposes_only_aggregate_release_evidence() -> None:
    report = CatalogReadinessReport(
        data_version="catalog-v1",
        product_count=1,
        products_by_category={"gpu": 1},
        compatibility_ready_products_by_category={"gpu": 1},
        critical_field_present_by_category={"gpu": {}},
        product_provenance_complete_count=1,
        offer_provenance_complete_count=1,
        offer_rights_explicit_count=1,
        offer_rights_production_valid_count=1,
        rights_evaluated_on=date(2026, 7, 23),
        rights_territory="SG",
        entity_resolution_evaluation=None,
        entity_resolution_model_version="entity-resolution-v1",
        entity_resolution_model_production_authorized=False,
        listing_count=1,
        listing_provenance_complete_count=1,
        offer_count=1,
        mapping_outcomes={"matched": 1},
        matched_listings_by_category={"gpu": 1},
        in_stock_listings_by_category={"gpu": 1},
    )

    summary = _public_catalogue_readiness(report, blockers=["catalogue policy is blocked"])

    assert summary is not None
    assert summary.products_by_category == {"gpu": 1}
    assert summary.matched_listings_by_category == {"gpu": 1}
    assert summary.rights_territory == "SG"
    assert summary.production_ready is False
    assert summary.production_blockers == ["catalogue policy is blocked"]


def test_processed_admin_operations_reports_aggregate_queue_freshness_and_missing_fields(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    service = object.__new__(CoreRecommendationService)
    settings = ApiRuntimeSettings(
        environment="test",
        admin_token="test-admin-token-0123456789abcdef",
        stale_after_hours=24,
        pipeline_operations_path=tmp_path,
        pipeline_operations_window_hours=24 * 7,
    )
    service.settings = settings
    write_pipeline_operation_event(
        tmp_path,
        operation_name="buildcores_catalog_ingestion",
        status="failed",
        finished_at=now - timedelta(minutes=2),
        failure_class="builtins.ValueError",
    )
    service.processed_data = SimpleNamespace(
        stats=SimpleNamespace(
            data_version="catalog-v1",
            offer_count=10,
            matched_listing_count=4,
            unmatched_offer_count=3,
            manual_review_count=1,
            rejected_conflict_count=1,
            model_rejected_count=1,
        ),
        price_snapshots=(
            SimpleNamespace(observed_at=now - timedelta(hours=2)),
            SimpleNamespace(observed_at=now - timedelta(hours=30)),
        ),
        readiness=SimpleNamespace(
            blockers=lambda: ("catalogue coverage is not production-ready",),
            critical_field_present_by_category={"gpu": {"clearance": 7}},
            products_by_category={"gpu": 10},
        ),
    )

    response = asyncio.run(service.admin_operations())

    assert response.mode == "processed_catalog"
    assert response.data_version == "catalog-v1"
    assert response.mapping_queue is not None
    assert response.mapping_queue.unmatched_count == 3
    assert response.price_freshness is not None
    assert response.price_freshness.stale_snapshot_count == 1
    assert response.release_blockers == ["catalogue coverage is not production-ready"]
    assert response.missing_critical_fields[0].category.value == "gpu"
    assert response.missing_critical_fields[0].missing_product_count == 3
    assert response.pipeline_failure_events_available is True
    assert response.pipeline_operations is not None
    assert response.pipeline_operations.failed_count == 1
    assert response.pipeline_operations.latest_failure_at is not None
    assert all("ValueError" not in note for note in response.notes)

    with TestClient(create_app(settings=settings, service=service)) as client:
        metrics = client.get("/metrics")
        operations = client.get(
            "/v1/admin/operations",
            headers={"X-PCBR-Admin-Token": "test-admin-token-0123456789abcdef"},
        )

    assert operations.status_code == 200, operations.text
    assert 'pcbr_entity_resolution_mapping_observation{state="available"} 1' in metrics.text
    assert "pcbr_entity_resolution_manual_review_queue_items 1" in metrics.text
    assert "pcbr_entity_resolution_unmatched_offer_items 3" in metrics.text
    assert "pcbr_catalogue_missing_critical_field_values 3" in metrics.text
    assert 'pcbr_pipeline_operations_observation{state="available"} 1' in metrics.text
    assert "pcbr_pipeline_operation_failures_in_window 1" in metrics.text


def test_admin_token_file_is_loaded_and_rejects_ambiguous_or_short_configuration(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "admin-token.txt"
    token_file.write_text("test-admin-token-0123456789abcdef", encoding="utf-8")

    settings = ApiRuntimeSettings(environment="test", admin_token_file=token_file)

    assert settings.admin_token is not None
    assert settings.admin_token.get_secret_value() == "test-admin-token-0123456789abcdef"
    with pytest.raises(ValueError, match="only one of admin_token or admin_token_file"):
        ApiRuntimeSettings(
            environment="test",
            admin_token="test-admin-token-0123456789abcdef",
            admin_token_file=token_file,
        )
    token_file.write_text("too-short", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 24"):
        ApiRuntimeSettings(environment="test", admin_token_file=token_file)


@pytest.mark.integration
def test_inconclusive_optimizer_is_a_structured_503() -> None:
    service, settings = _core_with_response(
        _optimizer_response(OptimizationStatus.UNKNOWN, ran=True)
    )
    with TestClient(create_app(settings=settings, service=service)) as client:
        response = client.post("/v1/builds/generate", json=_request())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "optimizer_not_conclusive"
    assert response.json()["error"]["details"]["solver_status"] == "UNKNOWN"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


@pytest.mark.integration
def test_preflight_infeasibility_preserves_optimizer_execution_metadata() -> None:
    service, settings = _core_with_response(
        _optimizer_response(OptimizationStatus.INFEASIBLE, ran=False)
    )
    with TestClient(create_app(settings=settings, service=service)) as client:
        response = client.post("/v1/builds/generate", json=_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "infeasible"
    assert payload["solver_status"] == "INFEASIBLE"
    assert payload["solver_ran"] is False
    assert payload["retrieval_model"] == "retriever-v1"
    assert payload["performance_model"] == "performance-v1"
    assert payload["solver_profile_statuses"] == []
    assert payload["solver_validator_rejections"] == 0


@pytest.mark.integration
def test_processed_readiness_requires_all_component_categories() -> None:
    response = _optimizer_response(OptimizationStatus.INFEASIBLE, ran=False)
    service, settings = _core_with_response(response)
    service.processed_data = cast(
        Any,
        SimpleNamespace(
            products=(
                SimpleNamespace(
                    category=DomainCategory.GPU,
                    updated_at=datetime.now(UTC),
                ),
            ),
            price_snapshots=(SimpleNamespace(observed_at=datetime.now(UTC)),),
            readiness=None,
            stats=SimpleNamespace(
                has_complete_priced_coverage=False,
                has_complete_in_stock_coverage=False,
            ),
        ),
    )
    with TestClient(create_app(settings=settings, service=service)) as client:
        readiness = client.get("/health/ready")

    assert readiness.status_code == 503
    assert readiness.json()["status"] == "not_ready"
    assert readiness.json()["checks"]["catalogue"] == "ready"
    assert readiness.json()["checks"]["category_coverage"] == "not_ready"


@pytest.mark.integration
def test_stale_catalogue_and_prices_fail_closed_for_readiness_and_production_use() -> None:
    response = _optimizer_response(OptimizationStatus.INFEASIBLE, ran=False)
    service, settings = _core_with_response(response)
    stale_timestamp = datetime.now(UTC) - timedelta(hours=settings.stale_after_hours + 1)
    service.processed_data = cast(
        Any,
        SimpleNamespace(
            products=tuple(
                SimpleNamespace(
                    category=category,
                    updated_at=stale_timestamp,
                    provenance=(),
                )
                for category in DomainCategory
            ),
            price_snapshots=(SimpleNamespace(observed_at=stale_timestamp),),
            listing_provenance=(),
            readiness=SimpleNamespace(blockers=lambda: ()),
            stats=SimpleNamespace(
                product_count=len(DomainCategory),
                matched_listing_count=len(DomainCategory),
                has_complete_priced_coverage=True,
                has_complete_in_stock_coverage=True,
            ),
        ),
    )

    with TestClient(create_app(settings=settings, service=service)) as client:
        readiness = client.get("/health/ready")
        freshness = client.get("/v1/system/freshness")

    assert readiness.status_code == 503
    assert readiness.json()["checks"]["catalogue_freshness"] == "not_ready"
    assert readiness.json()["checks"]["production_catalog_policy"] == "not_ready"
    assert freshness.status_code == 200
    assert freshness.json()["status"] == "stale"
    assert freshness.json()["production_ready"] is False
    assert freshness.json()["readiness_blockers"] == [
        f"Catalogue data is stale: last_catalog_update exceeds "
        f"stale_after_hours={settings.stale_after_hours}.",
        f"Price data is stale: prices_updated_at exceeds "
        f"stale_after_hours={settings.stale_after_hours}.",
    ]


@pytest.mark.integration
def test_processed_freshness_exposes_measured_readiness_and_unique_sources() -> None:
    response = _optimizer_response(OptimizationStatus.INFEASIBLE, ran=False)
    service, settings = _core_with_response(response)
    observed_at = datetime.now(UTC)
    service.processed_data = cast(
        Any,
        SimpleNamespace(
            products=(
                SimpleNamespace(
                    category=DomainCategory.GPU,
                    updated_at=observed_at,
                    provenance=(SimpleNamespace(source_name="manufacturer-a"),),
                ),
            ),
            price_snapshots=(SimpleNamespace(observed_at=observed_at),),
            listing_provenance=(
                SimpleNamespace(source_name="retailer-a"),
                SimpleNamespace(source_name="retailer-a"),
            ),
            readiness=SimpleNamespace(blockers=lambda: ("product_count=1 below minimum=750",)),
            stats=SimpleNamespace(product_count=1, matched_listing_count=1),
        ),
    )
    with TestClient(create_app(settings=settings, service=service)) as client:
        freshness = client.get("/v1/system/freshness")

    assert freshness.status_code == 200
    assert freshness.json()["source_count"] == 2
    assert freshness.json()["production_ready"] is False
    assert freshness.json()["release_artifact_verification"] == "development_unverified"
    assert freshness.json()["readiness_blockers"] == ["product_count=1 below minimum=750"]
    assert freshness.json()["catalogue_readiness"] is None


@pytest.mark.integration
def test_startup_and_metrics_scrape_refresh_freshness_without_public_freshness_call() -> None:
    class CountingFreshnessService(InMemoryRecommendationService):
        def __init__(self, settings: ApiRuntimeSettings) -> None:
            super().__init__(settings)
            self.freshness_calls = 0

        async def freshness(self) -> Any:
            self.freshness_calls += 1
            return await super().freshness()

    settings = ApiRuntimeSettings(environment="test")
    service = CountingFreshnessService(settings)
    with TestClient(create_app(settings=settings, service=service)) as client:
        assert service.freshness_calls == 1
        metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert service.freshness_calls == 2
    assert 'pcbr_catalogue_freshness_status{status="fresh"} 1' in metrics.text
    assert "pcbr_catalogue_freshness_probe_success 1" in metrics.text


@pytest.mark.integration
def test_openapi_documents_structured_errors_for_evidence_and_compatibility() -> None:
    settings = ApiRuntimeSettings(environment="test")
    with TestClient(create_app(settings)) as client:
        schema = client.get("/openapi.json").json()

    for path in (
        "/v1/products/{product_id}",
        "/v1/products/{product_id}/prices",
        "/v1/products/{product_id}/benchmarks",
        "/v1/products/{product_id}/reviews",
    ):
        error_schema = schema["paths"][path]["get"]["responses"]["404"]["content"][
            "application/json"
        ]["schema"]
        assert error_schema["$ref"].endswith("/ErrorResponse")
    validation_schema = schema["paths"]["/v1/compatibility/check"]["post"]["responses"]["422"][
        "content"
    ]["application/json"]["schema"]
    assert validation_schema["$ref"].endswith("/ErrorResponse")


@pytest.mark.integration
def test_invalid_service_response_is_withheld_by_typed_contract() -> None:
    class InvalidFreshnessService(InMemoryRecommendationService):
        async def freshness(self) -> Any:
            return {"status": "fresh"}

    settings = ApiRuntimeSettings(environment="test")
    service = InvalidFreshnessService(settings)
    with TestClient(
        create_app(settings=settings, service=service), raise_server_exceptions=False
    ) as client:
        response = client.get("/v1/system/freshness")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "response_contract_error"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_default_api_rule_version_is_compat_v2() -> None:
    assert ApiRuntimeSettings().compatibility_rule_version == "compat_v2"


@pytest.mark.integration
def test_sqlalchemy_store_survives_restart_and_commits_interaction(tmp_path: Path) -> None:
    database_path = tmp_path / "durable-results.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    first_engine = create_db_engine(database_url)
    init_database(first_engine)
    first_store = SqlAlchemyDurableStore(first_engine)
    request = BuildRequestSpec(
        budget_sgd=Decimal("2500.00"),
        workloads=[WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=1.0)],
        requested_profiles=[BuildPreset.BEST_OVERALL],
    )
    response = _optimizer_response(OptimizationStatus.INFEASIBLE, ran=False)

    first_store.save(request, response, no_cost_product_ids=frozenset())
    persisted_event = first_store.add_interaction(
        DomainInteractionEvent(
            event_id="evt_restart_contract",
            session_id="session-restart",
            event_type=InteractionType.FEEDBACK_SUBMITTED,
        )
    )
    first_engine.dispose()

    restarted_engine = create_db_engine(database_url)
    restarted_store = SqlAlchemyDurableStore(restarted_engine)
    stored = restarted_store.require_generation(response.request_id)
    with create_session_factory(restarted_engine)() as session:
        event_ids = list(session.scalars(select(InteractionEventRecord.event_id)))

    assert stored.request == request
    assert stored.response == response
    assert stored.stored_at.tzinfo is not None
    assert persisted_event.event_id in event_ids
    restarted_engine.dispose()


@pytest.mark.integration
def test_durable_build_share_is_hashed_revocable_and_survives_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "durable-build-share.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    first_engine = create_db_engine(database_url)
    init_database(first_engine)
    with create_session_factory(first_engine)() as session:
        session.add(
            SearchQueryRecord(
                query_id="query_for_share",
                raw_query=None,
                structured_constraints={},
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            GeneratedBuildRecord(
                build_id="build_for_share",
                query_id="query_for_share",
                profile="best_overall",
                total_price=Decimal("1999.00"),
                overall_score=91.0,
                workload_scores={"local_ai": 91.0},
                compatibility_status="pass",
                compatibility_checks=[],
                estimated_power_watts=600.0,
                warnings=[],
                explanation=[],
                alternatives=[],
                optimizer_status="FEASIBLE",
                rule_version="compat-test-v1",
                model_version="ranker-test-v1",
                data_version="catalog-test-v1",
                created_at=datetime.now(UTC),
            )
        )
        session.commit()

    first_store = SqlAlchemyDurableStore(first_engine)
    created = first_store.create_build_share(
        build_id="build_for_share",
        snapshot={"profile": "best_overall", "components": []},
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    with create_session_factory(first_engine)() as session:
        record = session.get(BuildShareRecord, created.share_id)
        assert record is not None
        assert record.revocation_token_sha256 != created.revocation_token
        assert len(record.revocation_token_sha256) == 64

    first_engine.dispose()
    restarted_engine = create_db_engine(database_url)
    restarted_store = SqlAlchemyDurableStore(restarted_engine)
    loaded = restarted_store.get_active_build_share(created.share_id)

    assert loaded is not None
    assert loaded.snapshot == {"profile": "best_overall", "components": []}
    assert restarted_store.revoke_build_share(created.share_id, "wrong-token") is None
    assert restarted_store.get_active_build_share(created.share_id) is not None
    assert (
        restarted_store.revoke_build_share(created.share_id, created.revocation_token) is not None
    )
    assert restarted_store.get_active_build_share(created.share_id) is None
    restarted_engine.dispose()


def test_production_processed_catalog_rejects_in_memory_storage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot use in-memory storage"):
        ApiRuntimeSettings(
            environment="production",
            service_mode="processed_catalog",
            storage_backend="memory",
            buildcores_catalog_path=tmp_path / "catalog.jsonl",
            governed_offers_path=tmp_path / "offers.jsonl",
        )


def test_production_processed_catalog_requires_serving_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires serving_manifest_path"):
        ApiRuntimeSettings(
            environment="production",
            service_mode="processed_catalog",
            storage_backend="database",
            database_url="postgresql+psycopg://pcbr:secret@postgres/pcbr",
            buildcores_catalog_path=tmp_path / "catalog.jsonl",
            governed_offers_path=tmp_path / "offers.jsonl",
        )


def test_production_processed_catalog_requires_pinned_semantic_encoder_bundle(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires semantic_encoder_bundle_path"):
        ApiRuntimeSettings(
            environment="production",
            service_mode="processed_catalog",
            storage_backend="database",
            database_url="postgresql+psycopg://pcbr:secret@postgres/pcbr",
            buildcores_catalog_path=tmp_path / "catalog.jsonl",
            governed_offers_path=tmp_path / "offers.jsonl",
            serving_manifest_path=tmp_path / "serving-manifest.json",
            serving_manifest_sha256="a" * 64,
        )


def test_production_processed_catalog_requires_pinned_review_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a pinned review_evidence_path"):
        ApiRuntimeSettings(
            environment="production",
            service_mode="processed_catalog",
            storage_backend="database",
            database_url="postgresql+psycopg://pcbr:secret@postgres/pcbr",
            buildcores_catalog_path=tmp_path / "catalog.jsonl",
            governed_offers_path=tmp_path / "offers.jsonl",
            serving_manifest_path=tmp_path / "serving-manifest.json",
            serving_manifest_sha256="a" * 64,
            semantic_encoder_bundle_path=tmp_path / "encoders" / ("b" * 64),
            semantic_encoder_bundle_sha256="b" * 64,
        )


def test_semantic_encoder_bundle_path_and_hash_must_be_paired(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be configured together"):
        ApiRuntimeSettings(
            environment="test",
            service_mode="processed_catalog",
            buildcores_catalog_path=tmp_path / "catalog.jsonl",
            governed_offers_path=tmp_path / "offers.jsonl",
            semantic_encoder_bundle_path=tmp_path / "encoder",
        )


def test_development_auto_storage_remains_in_memory_with_database_url(tmp_path: Path) -> None:
    settings = ApiRuntimeSettings(
        environment="development",
        service_mode="processed_catalog",
        storage_backend="auto",
        database_url="sqlite:///development.db",
        buildcores_catalog_path=tmp_path / "catalog.jsonl",
        governed_offers_path=tmp_path / "offers.jsonl",
        allow_development_catalog=True,
    )

    assert settings.uses_database_storage is False


def test_governed_offer_setting_accepts_legacy_env_with_canonical_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical-offers.jsonl"
    legacy = tmp_path / "legacy-offers.jsonl"
    monkeypatch.setenv("PCBR_API_GOVERNED_OFFERS_PATH", str(canonical))
    monkeypatch.setenv("PCBR_API_DYNACORE_OFFERS_PATH", str(legacy))

    settings = ApiRuntimeSettings(_env_file=None)

    assert settings.governed_offers_path == canonical
    assert settings.dynacore_offers_path == canonical


def test_governed_offer_setting_accepts_legacy_constructor_alias(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy-offers.jsonl"

    settings = ApiRuntimeSettings(_env_file=None, dynacore_offers_path=legacy)

    assert settings.governed_offers_path == legacy
