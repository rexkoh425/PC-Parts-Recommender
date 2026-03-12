from __future__ import annotations

from sqlalchemy import func, select

from pc_build_recommender.catalog import (
    BenchmarkResultRecord,
    CanonicalProductRecord,
    CatalogRepository,
    PriceSnapshotRecord,
    RetailerListingRecord,
    create_db_engine,
    create_session_factory,
    deterministic_id,
    init_database,
    load_seed_data,
)


def _seed() -> dict[str, object]:
    product_id = deterministic_id("prod", "gpu", "Acme", "AC-16")
    listing_id = deterministic_id("listing", "SG Parts", "offer-1")
    return {
        "products": [
            {
                "category": "gpu",
                "brand": "Acme",
                "model": "Compute 16",
                "manufacturer_part_number": "AC-16",
                "category_attributes": {"vram_gb": 16, "length_mm": 270},
                "provenance": [
                    {
                        "source_name": "Acme",
                        "source_url": "https://manufacturer.invalid/ac-16",
                        "source_type": "manufacturer",
                        "raw_content_hash": "hash-product",
                        "parser_version": "v1",
                        "licence_or_access_note": "Public specification",
                    }
                ],
            }
        ],
        "listings": [
            {
                "product_id": product_id,
                "retailer": "SG Parts",
                "source_listing_id": "offer-1",
                "title": "Acme Compute 16 GB",
                "base_price": 799,
                "stock_status": "in_stock",
            }
        ],
        "price_snapshots": [
            {
                "listing_id": listing_id,
                "observed_at": "2026-07-22T00:00:00+08:00",
                "base_price": 799,
                "stock_status": "in_stock",
            }
        ],
        "benchmarks": [
            {
                "product_id": product_id,
                "workload": "local_ai",
                "benchmark_name": "tokens per second",
                "benchmark_version": "1",
                "score": 42.5,
                "unit": "tokens/s",
                "source_url": "https://benchmark.invalid/ac-16",
                "observed_at": "2026-07-01T00:00:00+00:00",
            }
        ],
    }


def test_seed_loader_is_deterministic_and_idempotent() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        first = load_seed_data(session, _seed())
        second = load_seed_data(session, _seed())
        session.commit()

        assert first == second
        assert first.products == 1
        assert first.total_records == 5
        assert session.scalar(select(func.count()).select_from(CanonicalProductRecord)) == 1
        assert session.scalar(select(func.count()).select_from(RetailerListingRecord)) == 1
        assert session.scalar(select(func.count()).select_from(PriceSnapshotRecord)) == 1
        assert session.scalar(select(func.count()).select_from(BenchmarkResultRecord)) == 1

        products = CatalogRepository(session).list_products()
        assert products[0].product_id == deterministic_id("prod", "gpu", "Acme", "AC-16")
        assert products[0].created_at.isoformat().startswith("1970-01-01")


def test_deterministic_ids_normalise_case_and_spacing() -> None:
    assert deterministic_id("prod", " GPU ", "Acme", "AC-16") == deterministic_id(
        "prod", "gpu", "acme", "ac-16"
    )

