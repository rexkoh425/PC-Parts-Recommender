from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from services.api import core_service
from services.api.durability import SqlAlchemyDurableStore
from services.api.settings import ApiRuntimeSettings

from pc_build_recommender.application import ServingConfigurationError


class _DurableStore:
    def __init__(self) -> None:
        self.session_factory = object()
        self.schema_verified = False
        self.catalog_identity: (
            tuple[tuple[str, ...], tuple[str, ...], tuple[Any, ...], tuple[Any, ...]] | None
        ) = None

    def verify_schema(self) -> None:
        self.schema_verified = True

    def verify_catalog_identity(
        self,
        *,
        product_ids: Any,
        listing_ids: Any,
        canonical_products: Any = None,
        retailer_listings: Any = None,
    ) -> None:
        self.catalog_identity = (
            tuple(product_ids),
            tuple(listing_ids),
            tuple(canonical_products or ()),
            tuple(retailer_listings or ()),
        )


def _settings(tmp_path: Path) -> ApiRuntimeSettings:
    return ApiRuntimeSettings(
        environment="production",
        docs_enabled=False,
        cors_origins=["https://pcbr.example.test"],
        service_mode="processed_catalog",
        storage_backend="database",
        database_url="postgresql+psycopg://pcbr:secret@postgres/pcbr",
        buildcores_catalog_path=tmp_path / "catalog.jsonl",
        governed_offers_path=tmp_path / "offers.jsonl",
        reviewed_mapping_path=tmp_path / "reviewed-mappings.json",
        review_evidence_path=tmp_path / "review-evidence.jsonl",
        serving_manifest_path=tmp_path / "serving-manifest.json",
        serving_manifest_sha256="a" * 64,
        semantic_encoder_bundle_path=tmp_path / "encoders" / ("c" * 64),
        semantic_encoder_bundle_sha256="c" * 64,
        data_version="catalog-v1",
        ranking_model_version="ltr-v4",
    )


def _patch_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, object, _DurableStore, dict[str, object]]:
    data = SimpleNamespace(
        readiness=object(),
        products=(SimpleNamespace(product_id="product-1"),),
        listings=(SimpleNamespace(listing_id="listing-1"),),
        stats=SimpleNamespace(data_version="catalog-v1"),
    )
    reader = object()
    store = _DurableStore()
    catalog_arguments: dict[str, object] = {}

    def load_catalog(path: Path, **kwargs: object) -> SimpleNamespace:
        catalog_arguments["path"] = path
        catalog_arguments.update(kwargs)
        return data

    monkeypatch.setattr(core_service, "load_processed_catalog", load_catalog)
    monkeypatch.setattr(core_service, "validate_production_readiness", lambda _report: None)
    monkeypatch.setattr(core_service, "InMemoryCatalogReader", lambda _data: reader)
    monkeypatch.setattr(
        SqlAlchemyDurableStore,
        "from_url",
        classmethod(lambda _cls, _url: store),
    )
    return data, reader, store, catalog_arguments


def test_exact_production_composition_loads_only_the_validated_serving_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    data, reader, store, catalog_arguments = _patch_catalogue(monkeypatch)
    retriever = object()
    ranker = object()
    artifact = object()
    active_models = object()
    er_runtime = object()
    er_policy = object()
    er_production_policy = object()
    er_evaluation_path = tmp_path / "entity-resolution-evaluation.json"
    er_binding_sha256 = "b" * 64
    er_release = SimpleNamespace(
        runtime=er_runtime,
        policy=er_policy,
        identity=SimpleNamespace(binding_sha256=er_binding_sha256),
    )
    release = SimpleNamespace(
        retriever=retriever,
        ranker=ranker,
        performance_artifacts=(artifact,),
        active_models=active_models,
        semantic_encoder_ready=True,
        catalog_release=SimpleNamespace(
            entity_resolution=er_release,
            entity_resolution_evaluation_path=er_evaluation_path,
        ),
    )
    load_arguments: dict[str, object] = {}

    def load_release(path: Path, **kwargs: object) -> SimpleNamespace:
        load_arguments["path"] = path
        load_arguments.update(kwargs)
        return release

    monkeypatch.setattr(core_service, "load_production_serving_release", load_release)
    monkeypatch.setattr(
        core_service,
        "production_catalog_policy_from_entity_resolution",
        lambda policy: er_production_policy if policy is er_policy else None,
    )
    factory_arguments: dict[str, object] = {}
    application_services = SimpleNamespace(
        versions=SimpleNamespace(
            data_version="catalog-v1",
            ranking_model="ltr-v4",
            rule_version="compat-v2",
            optimizer_version="cp-sat-v1",
        )
    )

    def create_services(catalogue_reader: object, **kwargs: object) -> SimpleNamespace:
        factory_arguments["reader"] = catalogue_reader
        factory_arguments.update(kwargs)
        return application_services

    monkeypatch.setattr(core_service, "create_application_services", create_services)

    service = core_service.create_processed_catalog_service(settings)

    service_services: Any = service.services
    assert service_services is application_services
    assert store.schema_verified is True
    assert store.catalog_identity == (
        ("product-1",),
        ("listing-1",),
        data.products,
        data.listings,
    )
    assert service._release_artifact_verification == "verified"
    assert load_arguments == {
        "path": settings.serving_manifest_path,
        "catalog_path": settings.buildcores_catalog_path,
        "offers_path": settings.governed_offers_path,
        "reviewed_mappings_path": settings.reviewed_mapping_path,
        "review_evidence_path": settings.review_evidence_path,
        "session_factory": store.session_factory,
        "expected_catalog_data_version": "catalog-v1",
        "expected_ranker_version": "ltr-v4",
        "expected_manifest_sha256": "a" * 64,
        "expected_encoder_bundle_path": settings.semantic_encoder_bundle_path,
        "expected_encoder_bundle_sha256": "c" * 64,
    }
    assert catalog_arguments == {
        "path": settings.buildcores_catalog_path,
        "offer_path": settings.governed_offers_path,
        "reviewed_mapping_path": settings.reviewed_mapping_path,
        "review_evidence_path": settings.review_evidence_path,
        "entity_resolution_evaluation_path": er_evaluation_path,
        "entity_resolution_runtime": er_runtime,
        "entity_resolution_policy": er_policy,
        "entity_resolution_binding_sha256": er_binding_sha256,
        "require_production_entity_resolution": True,
        "production_policy": er_production_policy,
    }
    assert factory_arguments["reader"] is reader
    assert factory_arguments["require_promoted_models"] is True
    assert factory_arguments["retriever"] is retriever
    assert factory_arguments["ranker"] is ranker
    assert factory_arguments["performance_artifacts"] == (artifact,)
    assert factory_arguments["promoted_serving_models"] is active_models


def test_production_composition_never_reaches_factory_after_release_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    _patch_catalogue(monkeypatch)
    monkeypatch.setattr(
        core_service,
        "load_production_serving_release",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ServingConfigurationError("serving manifest content hash verification failed")
        ),
    )
    factory_called = False

    def create_services(*_args: object, **_kwargs: object) -> object:
        nonlocal factory_called
        factory_called = True
        return object()

    monkeypatch.setattr(core_service, "create_application_services", create_services)

    with pytest.raises(ServingConfigurationError, match="content hash verification failed"):
        core_service.create_processed_catalog_service(settings)

    assert factory_called is False


def test_production_composition_rejects_direct_component_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    _patch_catalogue(monkeypatch)

    with pytest.raises(RuntimeError, match="must come from the immutable serving manifest"):
        core_service.create_processed_catalog_service(settings, ranker=object())  # type: ignore[arg-type]
