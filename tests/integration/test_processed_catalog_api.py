from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from services.api.core_service import CoreRecommendationService
from services.api.durability import SqlAlchemyDurableStore
from services.api.main import create_app
from services.api.settings import ApiSettings
from sqlalchemy import select

from pc_build_recommender.application import create_application_services
from pc_build_recommender.catalog import (
    InMemoryCatalogReader,
    InteractionEventRecord,
    ProcessedCatalogData,
    ProcessedCatalogStats,
    SearchQueryRecord,
    create_db_engine,
    create_session_factory,
    init_database,
)
from pc_build_recommender.domain import (
    CanonicalProduct,
    ComponentCategory,
    GPUAttributes,
    ListingCondition,
    PriceSample,
    RetailerListing,
    ReviewEvidence,
    SourceProvenance,
    SourceType,
    StockStatus,
)

NOW = datetime(2026, 7, 22, tzinfo=UTC)


def _service(
    *,
    product_count: int = 1,
    condition: ListingCondition = ListingCondition.NEW,
    stock_status: StockStatus = StockStatus.UNKNOWN,
    rights_ready: bool = False,
    snapshot_prices: tuple[Decimal, ...] | None = None,
    review_evidence: tuple[ReviewEvidence, ...] = (),
    durable_store: SqlAlchemyDurableStore | None = None,
) -> CoreRecommendationService:
    if product_count < 1:
        raise ValueError("product_count must be positive")
    if snapshot_prices is not None and product_count != 1:
        raise ValueError("snapshot_prices requires a single-product fixture")
    base_product = CanonicalProduct(
        product_id="prod_real_gpu",
        category=ComponentCategory.GPU,
        brand="ASUS",
        model="Prime RTX 5060 Ti",
        manufacturer_part_number="PRIME-RTX5060TI-O16G",
        canonical_name="ASUS Prime RTX 5060 Ti 16GB",
        category_attributes=GPUAttributes(
            vram_gb=16,
            length_mm=304,
            slot_width=2.5,
            board_power_watts=180,
            power_connectors={"8_pin": 1},
        ),
        provenance=[
            SourceProvenance(
                provenance_id="src_buildcores_prod_real_gpu",
                product_id="prod_real_gpu",
                source_name="BuildCores OpenDB",
                source_url="https://github.com/buildcores/buildcores-open-db",
                source_type=SourceType.IMPORT,
                retrieved_at=NOW,
                raw_content_hash="a" * 64,
                parser_version="buildcores-open-db-v1",
                licence_or_access_note=(
                    "BuildCores OpenDB database licensed ODC-By 1.0; attribution required."
                ),
            )
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    products = tuple(
        base_product
        if index == 0
        else base_product.model_copy(
            update={
                "product_id": f"prod_real_gpu_{index + 1}",
                "model": f"Prime RTX 5060 Ti Variant {index + 1}",
                "manufacturer_part_number": f"PRIME-RTX5060TI-O16G-{index + 1}",
                "canonical_name": f"ASUS Prime RTX 5060 Ti 16GB Variant {index + 1}",
            }
        )
        for index in range(product_count)
    )
    listings = tuple(
        RetailerListing(
            listing_id=("listing_real_gpu" if index == 0 else f"listing_real_gpu_{index + 1}"),
            product_id=product.product_id,
            retailer="Controlled Retailer",
            source_listing_id=f"offer-{index + 1}",
            title=product.canonical_name,
            condition=condition,
            base_price=(
                snapshot_prices[0] if snapshot_prices is not None else Decimal("899.00") + index
            ),
            stock_status=stock_status,
            listing_url=f"https://example.test/offer/{index + 1}",
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        for index, product in enumerate(products)
    )
    snapshots = (
        tuple(
            PriceSample(
                snapshot_id=f"price_real_gpu_day_{day}",
                listing_id=listings[0].listing_id,
                observed_at=NOW - timedelta(days=day),
                base_price=price,
                stock_status=stock_status,
            )
            for day, price in enumerate(snapshot_prices)
        )
        if snapshot_prices is not None
        else tuple(
            PriceSample(
                snapshot_id=("price_real_gpu" if index == 0 else f"price_real_gpu_{index + 1}"),
                listing_id=listing.listing_id,
                observed_at=NOW,
                base_price=listing.base_price,
                stock_status=stock_status,
            )
            for index, listing in enumerate(listings)
        )
    )
    stats = ProcessedCatalogStats(
        product_count=product_count,
        offer_count=product_count,
        matched_listing_count=product_count,
        auto_matched_count=product_count,
        reviewed_matched_count=0,
        unmatched_offer_count=0,
        rejected_conflict_count=0,
        ambiguous_exact_match_count=0,
        products_by_category={"gpu": product_count},
        matched_listings_by_category={"gpu": product_count},
        in_stock_listings_by_category=(
            {"gpu": product_count} if stock_status is StockStatus.IN_STOCK else {}
        ),
        known_in_stock_listing_count=(product_count if stock_status is StockStatus.IN_STOCK else 0),
        data_version="processed-test-v1",
    )
    data = ProcessedCatalogData(
        products=products,
        listings=listings,
        price_snapshots=snapshots,
        stats=stats,
        review_evidence=review_evidence,
        match_method_by_listing={listing.listing_id: "exact_mpn_brand" for listing in listings},
        readiness=(
            cast(
                Any,
                SimpleNamespace(
                    offer_count=product_count,
                    offer_rights_production_valid_count=product_count,
                    blockers=lambda: (),
                ),
            )
            if rights_ready
            else None
        ),
    )
    reader = InMemoryCatalogReader(data)
    application = create_application_services(reader, data_version=stats.data_version)
    return CoreRecommendationService(
        ApiSettings(), application, reader, data, durable_store=durable_store
    )


def test_processed_service_search_prices_and_generation_are_truthful() -> None:
    service = _service()
    client = TestClient(create_app(settings=service.settings, service=service))

    unavailable = client.post(
        "/v1/products/search",
        json={"query": "RTX 5060 Ti", "category": "gpu", "in_stock_only": True},
    )
    assert unavailable.status_code == 200
    assert unavailable.json()["products"] == []

    search = client.post(
        "/v1/products/search",
        json={"query": "RTX 5060 Ti", "category": "gpu", "in_stock_only": False},
    )
    assert search.status_code == 200
    assert search.json()["products"][0]["lowest_price_sgd"] is None
    assert search.json()["products"][0]["stock_status"] is None
    attribution = search.json()["coverage"]["source_attributions"]
    assert attribution == [
        {
            "source_name": "BuildCores OpenDB",
            "source_url": "https://github.com/buildcores/buildcores-open-db",
            "licence_or_access_note": (
                "BuildCores OpenDB database licensed ODC-By 1.0; attribution required."
            ),
            "attribution_notice": (
                "Contains information from BuildCores OpenDB, made available under the "
                "ODC Attribution License v1.0."
            ),
            "licence_url": "https://opendatacommons.org/licenses/by/1-0/",
            "retrieved_at": "2026-07-22T00:00:00Z",
        }
    ]

    product = client.get("/v1/products/prod_real_gpu")
    assert product.status_code == 200
    assert product.json()["source_attributions"] == attribution

    all_categories = client.post(
        "/v1/products/search",
        json={"query": "RTX 5060 Ti", "in_stock_only": False},
    )
    assert all_categories.status_code == 200
    assert all_categories.json()["products"][0]["product_id"] == "prod_real_gpu"

    prices = client.get("/v1/products/prod_real_gpu/prices")
    assert prices.status_code == 200
    assert prices.json()["current_lowest_price_sgd"] is None
    assert prices.json()["observations"] == []
    assert prices.json()["price_intelligence"] is None

    generation = client.post(
        "/v1/builds/generate",
        json={
            "budget_sgd": 2500,
            "workloads": [{"name": "gaming_1440p", "weight": 1.0}],
            "requirements": {"in_stock_only": False},
            "max_builds": 1,
        },
    )
    assert generation.status_code == 200
    payload = generation.json()
    assert payload["status"] == "infeasible"
    assert payload["solver_status"] == "INFEASIBLE"
    assert payload["builds"] == []
    assert payload["infeasibility"]["reasons"]


def test_processed_service_incomplete_compatibility_is_unknown() -> None:
    service = _service()
    client = TestClient(create_app(settings=service.settings, service=service))
    response = client.post(
        "/v1/compatibility/check",
        json={"components": [{"category": "gpu", "product_id": "prod_real_gpu"}]},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "unknown"
    assert response.json()["is_feasible"] is False


@pytest.mark.parametrize(
    ("condition", "stock_status"),
    [
        (ListingCondition.USED, StockStatus.IN_STOCK),
        (ListingCondition.NEW, StockStatus.OUT_OF_STOCK),
    ],
)
def test_non_new_or_unavailable_offer_is_never_advertised_as_current_lowest(
    condition: ListingCondition,
    stock_status: StockStatus,
) -> None:
    service = _service(
        condition=condition,
        stock_status=stock_status,
        rights_ready=True,
    )
    client = TestClient(create_app(settings=service.settings, service=service))

    detail = client.get("/v1/products/prod_real_gpu")
    prices = client.get("/v1/products/prod_real_gpu/prices")

    assert detail.json()["lowest_price_sgd"] is None
    assert prices.json()["current_lowest_price_sgd"] is None
    assert prices.json()["observations"]
    assert prices.json()["observations"][0]["condition"] == condition.value
    assert prices.json()["observations"][0]["stock_status"] == stock_status.value
    assert prices.json()["observations"][0]["current_offer_eligible"] is False
    if condition is ListingCondition.USED:
        assert prices.json()["price_intelligence"] is None
    else:
        assert prices.json()["price_intelligence"]["current_delivered_price_sgd"] is None


def test_processed_price_summary_is_rights_gated_descriptive_and_newest_first() -> None:
    service = _service(
        stock_status=StockStatus.IN_STOCK,
        rights_ready=True,
        snapshot_prices=tuple(Decimal(value) for value in (50, 100, 100, 100, 100, 100, 100, 100)),
    )
    client = TestClient(create_app(settings=service.settings, service=service))

    response = client.get("/v1/products/prod_real_gpu/prices")

    assert response.status_code == 200
    payload = response.json()
    observed_at = [item["observed_at"] for item in payload["observations"]]
    assert observed_at == sorted(observed_at, reverse=True)
    summary = payload["price_intelligence"]
    assert summary["basis"] == "descriptive_observed_history"
    assert summary["current_delivered_price_sgd"] == 50
    assert summary["median_30d_sgd"] == 100
    assert summary["median_90d_sgd"] == 100
    assert summary["percentile_90d"] == pytest.approx(6.25)
    assert summary["history_sufficient"] is True
    assert summary["observations_analyzed"] == 8


def test_processed_price_observation_response_is_bounded_to_newest_year() -> None:
    service = _service(
        stock_status=StockStatus.IN_STOCK,
        rights_ready=True,
        snapshot_prices=tuple(Decimal(100 + day) for day in range(400)),
    )
    client = TestClient(create_app(settings=service.settings, service=service))

    payload = client.get("/v1/products/prod_real_gpu/prices").json()

    assert len(payload["observations"]) == 365
    assert payload["observations"][0]["observed_at"] == NOW.isoformat().replace("+00:00", "Z")
    assert payload["observations"][-1]["observed_at"] == (
        NOW - timedelta(days=364)
    ).isoformat().replace("+00:00", "Z")
    assert payload["price_intelligence"]["observations_analyzed"] == 400


def test_processed_service_serves_only_release_bound_review_evidence() -> None:
    service = _service(
        review_evidence=(
            ReviewEvidence(
                evidence_id="review-performance",
                product_id="prod_real_gpu",
                aspect="performance",
                sentiment=0.26,
                evidence_text="Permitted benchmark review evidence supports strong performance.",
                source_url="https://reviews.example.test/performance",
                published_at=NOW,
                confidence=0.91,
            ),
            ReviewEvidence(
                evidence_id="review-noise",
                product_id="prod_real_gpu",
                aspect="noise",
                sentiment=-0.25,
                evidence_text="Permitted evidence reports audible fan noise at sustained load.",
                source_url="https://reviews.example.test/noise",
                published_at=NOW,
                confidence=0.83,
            ),
            ReviewEvidence(
                evidence_id="review-neutral",
                product_id="prod_real_gpu",
                aspect="value",
                sentiment=0.24,
                evidence_text="Permitted evidence describes value as dependent on local pricing.",
                source_url="https://reviews.example.test/value",
                published_at=NOW,
                confidence=0.72,
            ),
        )
    )
    client = TestClient(create_app(settings=service.settings, service=service))

    response = client.get("/v1/products/prod_real_gpu/reviews")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_version"] == "processed-test-v1"
    assert [(item["aspect"], item["sentiment"]) for item in payload["evidence"]] == [
        ("noise", "negative"),
        ("performance", "positive"),
        ("value", "neutral"),
    ]
    assert all(
        item["source_url"].startswith("https://reviews.example.test/")
        for item in payload["evidence"]
    )


def test_processed_search_has_exact_pagination_facets_coverage_and_scoped_cursors() -> None:
    service = _service(product_count=3)
    client = TestClient(create_app(settings=service.settings, service=service))

    first = client.post(
        "/v1/products/search",
        json={
            "query": "",
            "category": "gpu",
            "in_stock_only": False,
            "limit": 1,
            "page": 1,
            "page_size": 1,
        },
    )
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert first_payload["total"] == 3
    assert first_payload["pagination"]["total_pages"] == 3
    assert first_payload["pagination"]["next_cursor"]
    assert first_payload["facets"]["categories"] == [{"value": "gpu", "count": 3}]
    assert first_payload["facets"]["brands"] == [{"value": "ASUS", "count": 3}]
    assert first_payload["coverage"] == {
        "canonical_products": 3,
        "retailer_listings": 3,
        "source_count": 2,
        "category_count": 1,
        "as_of": "2026-07-22T00:00:00Z",
        "scope_label": "Versioned processed catalogue snapshot",
        "source_attributions": [
            {
                "source_name": "BuildCores OpenDB",
                "source_url": "https://github.com/buildcores/buildcores-open-db",
                "licence_or_access_note": (
                    "BuildCores OpenDB database licensed ODC-By 1.0; attribution required."
                ),
                "attribution_notice": (
                    "Contains information from BuildCores OpenDB, made available under the "
                    "ODC Attribution License v1.0."
                ),
                "licence_url": "https://opendatacommons.org/licenses/by/1-0/",
                "retrieved_at": "2026-07-22T00:00:00Z",
            }
        ],
    }

    page_two_request = {
        "query": "",
        "category": "gpu",
        "in_stock_only": False,
        "limit": 1,
        "page": 2,
        "page_size": 1,
        "cursor": first_payload["pagination"]["next_cursor"],
    }
    page_two = client.post("/v1/products/search", json=page_two_request)
    repeat = client.post("/v1/products/search", json=page_two_request)
    assert page_two.status_code == repeat.status_code == 200
    assert page_two.json() == repeat.json()
    assert page_two.json()["pagination"]["page"] == 2
    assert (
        page_two.json()["products"][0]["product_id"] != first_payload["products"][0]["product_id"]
    )

    invalid = client.post(
        "/v1/products/search",
        json={**page_two_request, "cursor": "invalid"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_pagination_cursor"


def test_product_search_query_survives_restart_and_validates_interaction_refs(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'search-feedback.db').as_posix()}"
    first_engine = create_db_engine(database_url)
    init_database(first_engine)
    first_store = SqlAlchemyDurableStore(first_engine)
    first_service = _service(product_count=3, durable_store=first_store)

    with TestClient(create_app(settings=first_service.settings, service=first_service)) as client:
        first = client.post(
            "/v1/products/search",
            json={
                "query": "  RTX   5060 Ti ",
                "category": "gpu",
                "in_stock_only": False,
                "page": 1,
                "page_size": 1,
            },
        )
        assert first.status_code == 200, first.text
        first_payload = first.json()
        query_id = first_payload["query_id"]
        second = client.post(
            "/v1/products/search",
            json={
                "query": "rtx 5060 ti",
                "category": "gpu",
                "in_stock_only": False,
                "page": 2,
                "page_size": 1,
                "cursor": first_payload["pagination"]["next_cursor"],
            },
        )
        assert second.status_code == 200, second.text
        assert second.json()["query_id"] == query_id

    restarted_engine = create_db_engine(database_url)
    restarted_store = SqlAlchemyDurableStore(restarted_engine)
    restarted_service = _service(product_count=3, durable_store=restarted_store)
    with TestClient(
        create_app(settings=restarted_service.settings, service=restarted_service)
    ) as client:
        repeated = client.post(
            "/v1/products/search",
            json={
                "query": "RTX 5060 TI",
                "category": "gpu",
                "in_stock_only": False,
                "limit": 3,
            },
        )
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["query_id"] == query_id

        accepted = client.post(
            "/v1/interactions",
            json={
                "event_type": "search_submitted",
                "session_id": "session-search-restart",
                "query_id": query_id,
                "model_version": repeated.json()["retrieval_model"],
                "data_version": repeated.json()["data_version"],
            },
        )
        unknown = client.post(
            "/v1/interactions",
            json={
                "event_type": "search_submitted",
                "session_id": "session-unknown-search",
                "query_id": "search_unknown",
            },
        )
        assert accepted.status_code == 202, accepted.text
        assert unknown.status_code == 409, unknown.text
        assert unknown.json()["error"]["code"] == "interaction_reference_conflict"

    verification_engine = create_db_engine(database_url)
    with create_session_factory(verification_engine)() as session:
        queries = list(session.scalars(select(SearchQueryRecord)))
        interactions = list(session.scalars(select(InteractionEventRecord)))
    assert len(queries) == 1
    assert queries[0].query_id == query_id
    assert queries[0].structured_constraints["kind"] == "product_search"
    assert [event.query_id for event in interactions] == [query_id]
    verification_engine.dispose()


def test_demo_bootstrap_is_rejected_outside_development() -> None:
    with pytest.raises(RuntimeError, match="development-only"):
        create_app(
            ApiSettings(
                environment="production",
                service_mode="demo",
                docs_enabled=False,
                cors_origins=["https://pcbr.example.test"],
            )
        )
