from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pc_build_recommender.catalog import (
    CatalogRepository,
    PriceSnapshotRecord,
    create_db_engine,
    create_session_factory,
    get_database_url,
    init_database,
    session_scope,
)
from pc_build_recommender.domain import (
    BenchmarkResult,
    MasterProduct,
    ComponentKind,
    GPUAttributes,
    PriceSample,
    RetailerListing,
    SourceProvenance,
    SourceType,
    StockState,
    WorkloadLabel,
)


@pytest.fixture
def session() -> Session:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as value:
        yield value


def _gpu() -> MasterProduct:
    return MasterProduct(
        product_id="prod_gpu_1",
        category=ComponentKind.GPU,
        brand="ExampleGPU",
        model="Inference 16G",
        manufacturer_part_number="EG-16-A",
        canonical_name="ExampleGPU Inference 16G",
        category_attributes=GPUAttributes(
            vram_gb=16,
            length_mm=280,
            board_power_watts=220,
            power_connectors={"8_pin": 1},
        ),
        provenance=[
            SourceProvenance(
                provenance_id="src_gpu_1",
                source_name="Example manufacturer",
                source_url="https://manufacturer.invalid/gpu",
                source_type=SourceType.MANUFACTURER,
                raw_content_hash="sha256:abc",
                parser_version="parser_v1",
                licence_or_access_note="Test fixture",
            )
        ],
    )


def _listing() -> RetailerListing:
    observed = datetime(2026, 7, 22, tzinfo=UTC)
    return RetailerListing(
        listing_id="listing_gpu_1",
        product_id="prod_gpu_1",
        retailer="Example Retailer",
        source_listing_id="sku-1",
        title="ExampleGPU 16 GB inference graphics card",
        base_price=Decimal("899"),
        shipping_price=Decimal("8"),
        stock_status=StockState.IN_STOCK,
        listing_url="https://retailer.invalid/sku-1",
        first_seen_at=observed,
        last_seen_at=observed,
    )


def test_product_listing_and_filtered_search_round_trip(session: Session) -> None:
    repository = CatalogRepository(session)
    created = repository.add_product(_gpu())
    listing = repository.add_listing(_listing())

    assert created.product_id == "prod_gpu_1"
    assert created.provenance[0].raw_content_hash == "sha256:abc"
    assert listing.total_price == Decimal("907.00")
    assert repository.cheapest_in_stock_listing("prod_gpu_1") == listing

    results = repository.search_products(
        "16 GB inference",
        category=ComponentKind.GPU,
        in_stock_only=True,
        max_total_price=Decimal("1000"),
    )
    assert [product.product_id for product in results] == ["prod_gpu_1"]
    assert repository.search_products("inference", max_total_price=Decimal("800")) == []


def test_price_snapshots_are_idempotent_per_listing_and_time(session: Session) -> None:
    repository = CatalogRepository(session)
    repository.add_product(_gpu())
    repository.add_listing(_listing())
    observed = datetime(2026, 7, 22, 1, tzinfo=UTC)

    repository.upsert_price_snapshot(
        PriceSample(
            snapshot_id="price_first_id",
            listing_id="listing_gpu_1",
            observed_at=observed,
            base_price=899,
            stock_status=StockState.IN_STOCK,
        )
    )
    repository.upsert_price_snapshot(
        PriceSample(
            snapshot_id="price_duplicate_natural_key",
            listing_id="listing_gpu_1",
            observed_at=observed,
            base_price=879,
            stock_status=StockState.IN_STOCK,
        )
    )

    assert session.scalar(select(func.count()).select_from(PriceSnapshotRecord)) == 1
    snapshots = repository.list_price_snapshots("listing_gpu_1")
    assert snapshots[0].snapshot_id == "price_first_id"
    assert snapshots[0].base_price == Decimal("879.00")


def test_benchmark_round_trip_keeps_configuration_fields(session: Session) -> None:
    repository = CatalogRepository(session)
    repository.add_product(_gpu())
    benchmark = BenchmarkResult(
        benchmark_id="bench_gpu_1",
        product_id="prod_gpu_1",
        workload=WorkloadLabel.GAMING_1440P,
        benchmark_name="Game suite geometric mean",
        benchmark_version="2026.1",
        score=121.4,
        unit="fps",
        resolution="2560x1440",
        preset="ultra",
        operating_system="Windows 11",
        driver_version="600.1",
        source_url="https://benchmark.invalid/result",
        observed_at=datetime(2026, 7, 1, tzinfo=UTC),
    )

    repository.upsert_benchmark(benchmark)
    restored = repository.list_benchmarks("prod_gpu_1")[0]

    assert restored == benchmark
    assert restored.resolution == "2560x1440"
    assert restored.driver_version == "600.1"


def test_session_scope_commits_and_rolls_back() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as scoped:
        CatalogRepository(scoped).add_product(_gpu())

    with factory() as verification:
        assert CatalogRepository(verification).get_product("prod_gpu_1") is not None

    with pytest.raises(RuntimeError), session_scope(factory) as scoped:
        CatalogRepository(scoped).add_listing(_listing().model_copy(update={"listing_id": "x"}))
        raise RuntimeError("force rollback")

    with factory() as verification:
        assert CatalogRepository(verification).get_listing("x") is None


def test_database_url_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///environment.db")

    assert get_database_url() == "sqlite:///environment.db"
    assert get_database_url("sqlite:///explicit.db") == "sqlite:///explicit.db"

