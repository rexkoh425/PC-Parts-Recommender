from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from services.api import core_service
from services.api.settings import ApiRuntimeSettings

from pc_build_recommender.application import ApplicationCatalog
from pc_build_recommender.compatibility import (
    AUTHORITATIVE_COMPATIBILITY_POLICY,
    COMPATIBILITY_AUTHORITY_KEY,
    CONTROLLED_NON_PRODUCTION_POLICY,
    CompatibilityEngine,
    CompatVerdict,
)
from pc_build_recommender.domain import (
    BenchmarkResult,
    MasterProduct,
    CaseAttributes,
    ComponentKind,
    GPUAttributes,
    ProductStatus,
    RetailerOffering,
    ReviewNote,
    SourceProvenance,
    SourceType,
    StockState,
)

NOW = datetime(2026, 7, 22, tzinfo=UTC)


class CatalogReaderFixture:
    def __init__(self, products: tuple[MasterProduct, ...]) -> None:
        self.products = products

    def list_products(
        self,
        *,
        category: ComponentKind | None = None,
        brand: str | None = None,
        status: ProductStatus | None = ProductStatus.ACTIVE,
        offset: int = 0,
        limit: int = 100,
    ) -> list[MasterProduct]:
        products = [
            product
            for product in self.products
            if (category is None or product.category is category)
            and (brand is None or product.brand.casefold() == brand.casefold())
            and (status is None or product.status is status)
        ]
        return products[offset : offset + limit]

    def get_product(self, product_id: str) -> MasterProduct | None:
        return next(
            (product for product in self.products if product.product_id == product_id),
            None,
        )

    def list_listings(
        self,
        *,
        product_id: str | None = None,
        retailer: str | None = None,
        stock_status: StockState | None = None,
        limit: int = 100,
    ) -> list[RetailerOffering]:
        return []

    def list_benchmarks(
        self, product_id: str, *, workload: str | None = None
    ) -> list[BenchmarkResult]:
        return []

    def list_review_evidence(self, product_id: str) -> list[ReviewNote]:
        return []


def _provenance(product_id: str, source_type: SourceType) -> SourceProvenance:
    source_name = (
        "example_manufacturer" if source_type is SourceType.MANUFACTURER else "buildcores_open_db"
    )
    return SourceProvenance(
        provenance_id=f"src_{product_id}_{source_type.value}",
        product_id=product_id,
        source_name=source_name,
        source_url=f"https://example.test/{source_name}/{product_id}",
        source_type=source_type,
        retrieved_at=NOW,
        raw_content_hash="a" * 64,
        parser_version="authority-test-v1",
        licence_or_access_note="Test evidence only.",
        last_verified_at=NOW,
    )


def _products(source_type: SourceType) -> tuple[MasterProduct, MasterProduct]:
    gpu_id = "prod_authority_gpu"
    case_id = "prod_authority_case"
    return (
        MasterProduct(
            product_id=gpu_id,
            category=ComponentKind.GPU,
            brand="Example",
            model="GPU",
            canonical_name="Example GPU",
            category_attributes=GPUAttributes(length_mm=300, slot_width=2.5),
            provenance=[_provenance(gpu_id, source_type)],
            created_at=NOW,
            updated_at=NOW,
        ),
        MasterProduct(
            product_id=case_id,
            category=ComponentKind.CASE,
            brand="Example",
            model="Case",
            canonical_name="Example Case",
            category_attributes=CaseAttributes(
                maximum_gpu_length_mm=330,
                maximum_gpu_slot_width=3.0,
            ),
            provenance=[_provenance(case_id, source_type)],
            created_at=NOW,
            updated_at=NOW,
        ),
    )


def _pair_report(catalog: ApplicationCatalog):
    gpu = catalog.require("prod_authority_gpu").compatibility_record
    case = catalog.require("prod_authority_case").compatibility_record
    return CompatibilityEngine().check_pair("gpu", gpu, "case", case)


def test_community_only_compatibility_fields_are_unknown_in_authoritative_mode() -> None:
    catalog = ApplicationCatalog.from_repository(
        CatalogReaderFixture(_products(SourceType.IMPORT)),
    )

    report = _pair_report(catalog)

    assert catalog.compatibility_evidence_policy == AUTHORITATIVE_COMPATIBILITY_POLICY
    assert report.status is CompatVerdict.UNKNOWN
    assert {result.rule_id for result in report.results} == {"compat.evidence.authority"}
    assert all(result.status is CompatVerdict.UNKNOWN for result in report.results)
    assert all(
        "non-authoritative community data" in result.evidence["authority"]["reason"]
        for result in report.results
    )
    assert catalog.has_authoritative_compatibility_coverage is False


def test_manufacturer_authoritative_fields_receive_normal_rule_results() -> None:
    catalog = ApplicationCatalog.from_repository(
        CatalogReaderFixture(_products(SourceType.MANUFACTURER)),
        compatibility_evidence_policy=AUTHORITATIVE_COMPATIBILITY_POLICY,
    )

    report = _pair_report(catalog)

    assert report.status is CompatVerdict.PASS
    assert {result.rule_id for result in report.results} == {
        "compat.gpu_case.length",
        "compat.gpu_case.slot_width",
    }
    assert all(result.status is CompatVerdict.PASS for result in report.results)
    assert catalog.has_authoritative_compatibility_coverage is True


def test_controlled_demo_policy_is_explicitly_non_production_and_unchanged() -> None:
    catalog = ApplicationCatalog.from_repository(
        CatalogReaderFixture(_products(SourceType.IMPORT)),
        compatibility_evidence_policy=CONTROLLED_NON_PRODUCTION_POLICY,
    )

    report = _pair_report(catalog)
    authority = catalog.require("prod_authority_gpu").compatibility_record[
        COMPATIBILITY_AUTHORITY_KEY
    ]

    assert report.status is CompatVerdict.PASS
    assert authority["decision"] == "controlled_non_production"
    assert authority["production_eligible"] is False
    assert catalog.has_authoritative_compatibility_coverage is False


def test_processed_catalog_composition_enforces_authoritative_policy(
    monkeypatch,
) -> None:
    reader = CatalogReaderFixture(_products(SourceType.IMPORT))
    processed = SimpleNamespace(
        readiness=object(),
        stats=SimpleNamespace(data_version="processed-authority-test-v1"),
    )
    monkeypatch.setattr(core_service, "load_processed_catalog", lambda *args, **kwargs: processed)
    monkeypatch.setattr(core_service, "InMemoryCatalogReader", lambda data: reader)
    settings = ApiRuntimeSettings(
        environment="test",
        service_mode="processed_catalog",
        buildcores_catalog_path=Path("community-products.jsonl"),
        governed_offers_path=Path("controlled-offers.jsonl"),
        allow_development_catalog=True,
    )

    service = core_service.create_processed_catalog_service(settings)

    assert (
        service.services.catalog.compatibility_evidence_policy == AUTHORITATIVE_COMPATIBILITY_POLICY
    )
    assert service.services.catalog.has_authoritative_compatibility_coverage is False
    assert _pair_report(service.services.catalog).status is CompatVerdict.UNKNOWN
