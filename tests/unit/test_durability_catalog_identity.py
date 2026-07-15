from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from services.api.durability import DurableStorageError, SqlAlchemyDurableStore
from sqlalchemy import update

from pc_build_recommender.catalog import (
    CanonicalProductRecord,
    CatalogRepository,
    RetailerListingRecord,
    create_db_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from pc_build_recommender.domain import (
    MasterProduct,
    ComponentKind,
    GPUAttributes,
    RetailerListing,
    StockStatus,
)
from pc_build_recommender.retrieval.embedding_index import (
    TEXT_BUILDER_VERSION,
    build_product_embedding_text,
)

NOW = datetime(2026, 7, 29, tzinfo=UTC)


def _product(product_id: str = "product-1") -> MasterProduct:
    return MasterProduct(
        product_id=product_id,
        category=ComponentKind.GPU,
        brand="Example",
        model=f"Model {product_id}",
        manufacturer_part_number=f"MPN-{product_id}",
        canonical_name=f"Example Model {product_id}",
        category_attributes=GPUAttributes(
            vram_gb=16,
            length_mm=280,
            board_power_watts=220,
            power_connectors={"pcie_8_pin": 1},
        ),
        created_at=NOW,
        updated_at=NOW,
    )


def _listing(
    *,
    listing_id: str = "listing-1",
    product_id: str = "product-1",
) -> RetailerListing:
    return RetailerListing(
        listing_id=listing_id,
        product_id=product_id,
        retailer="Example Retailer",
        source_listing_id=f"source-{listing_id}",
        title=f"Example listing {listing_id}",
        base_price=Decimal("899.00"),
        shipping_price=Decimal("8.00"),
        stock_status=StockStatus.IN_STOCK,
        listing_url=f"https://retailer.invalid/{listing_id}",
        first_seen_at=NOW,
        last_seen_at=NOW,
    )


def _release_search_identity(product: MasterProduct) -> tuple[str, str]:
    document = build_product_embedding_text(product.model_dump(mode="json"))
    content = f"{TEXT_BUILDER_VERSION}\0{product.product_id}\0{document}".encode()
    return document, hashlib.sha256(content).hexdigest()


@pytest.fixture
def durable_catalog() -> tuple[
    SqlAlchemyDurableStore,
    MasterProduct,
    RetailerListing,
]:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    factory = create_session_factory(engine)
    product = _product()
    listing = _listing()
    with session_scope(factory) as session:
        repository = CatalogRepository(session)
        repository.add_product(product)
        repository.add_listing(listing)
        document, content_hash = _release_search_identity(product)
        session.execute(
            update(CanonicalProductRecord)
            .where(CanonicalProductRecord.product_id == product.product_id)
            .values(
                search_document=document,
                search_document_hash=content_hash,
                updated_at=product.updated_at,
            )
        )
    return SqlAlchemyDurableStore(engine, factory), product, listing


def _verify(
    store: SqlAlchemyDurableStore,
    products: tuple[MasterProduct, ...],
    listings: tuple[RetailerListing, ...],
) -> None:
    store.verify_catalog_identity(
        product_ids=(product.product_id for product in products),
        listing_ids=(listing.listing_id for listing in listings),
        canonical_products=products,
        retailer_listings=listings,
    )


def test_strict_catalog_identity_accepts_exact_release_rows(
    durable_catalog: tuple[
        SqlAlchemyDurableStore,
        MasterProduct,
        RetailerListing,
    ],
) -> None:
    store, product, listing = durable_catalog

    _verify(store, (product,), (listing,))


def test_id_only_catalog_identity_keeps_development_import_check_non_strict(
    durable_catalog: tuple[
        SqlAlchemyDurableStore,
        MasterProduct,
        RetailerListing,
    ],
) -> None:
    store, product, listing = durable_catalog
    with session_scope(store.session_factory) as session:
        repository = CatalogRepository(session)
        repository.add_product(_product("product-extra"))
        repository.add_listing(
            _listing(listing_id="listing-extra", product_id=product.product_id)
        )
        session.execute(
            update(CanonicalProductRecord)
            .where(CanonicalProductRecord.product_id == product.product_id)
            .values(search_document="development import has not loaded vectors")
        )

    store.verify_catalog_identity(
        product_ids=(product.product_id,),
        listing_ids=(listing.listing_id,),
    )


def test_stale_row_preflight_allows_missing_rows_but_rejects_unexpected_rows(
    durable_catalog: tuple[
        SqlAlchemyDurableStore,
        MasterProduct,
        RetailerListing,
    ],
) -> None:
    store, product, listing = durable_catalog

    store.verify_no_unexpected_catalog_ids(
        product_ids=(product.product_id, "product-not-imported-yet"),
        listing_ids=(listing.listing_id, "listing-not-imported-yet"),
    )

    with pytest.raises(
        DurableStorageError,
        match="explicit audited stale-row reconciliation.*stale canonical products",
    ):
        store.verify_no_unexpected_catalog_ids(
            product_ids=("different-release-product",),
            listing_ids=("different-release-listing",),
        )


@pytest.mark.parametrize("extra_kind", ["product", "listing"])
def test_strict_catalog_identity_rejects_extra_rows(
    durable_catalog: tuple[
        SqlAlchemyDurableStore,
        MasterProduct,
        RetailerListing,
    ],
    extra_kind: str,
) -> None:
    store, product, listing = durable_catalog
    with session_scope(store.session_factory) as session:
        repository = CatalogRepository(session)
        if extra_kind == "product":
            repository.add_product(_product("product-extra"))
        else:
            repository.add_listing(
                _listing(listing_id="listing-extra", product_id=product.product_id)
            )

    with pytest.raises(DurableStorageError, match=f"unexpected .*{extra_kind}"):
        _verify(store, (product,), (listing,))


def test_strict_catalog_identity_rejects_modified_canonical_row(
    durable_catalog: tuple[
        SqlAlchemyDurableStore,
        MasterProduct,
        RetailerListing,
    ],
) -> None:
    store, product, listing = durable_catalog
    with session_scope(store.session_factory) as session:
        session.execute(
            update(CanonicalProductRecord)
            .where(CanonicalProductRecord.product_id == product.product_id)
            .values(source_confidence=0.5)
        )

    with pytest.raises(DurableStorageError, match="modified canonical product rows"):
        _verify(store, (product,), (listing,))


def test_strict_catalog_identity_rejects_modified_listing_row(
    durable_catalog: tuple[
        SqlAlchemyDurableStore,
        MasterProduct,
        RetailerListing,
    ],
) -> None:
    store, product, listing = durable_catalog
    with session_scope(store.session_factory) as session:
        session.execute(
            update(RetailerListingRecord)
            .where(RetailerListingRecord.listing_id == listing.listing_id)
            .values(base_price=Decimal("799.00"))
        )

    with pytest.raises(DurableStorageError, match="modified retailer listing rows"):
        _verify(store, (product,), (listing,))


@pytest.mark.parametrize(
    "mutation",
    [
        {"search_document": "tampered search corpus"},
        {"search_document_hash": "0" * 64},
    ],
    ids=["document", "content-hash"],
)
def test_strict_catalog_identity_rejects_modified_search_document_or_hash(
    durable_catalog: tuple[
        SqlAlchemyDurableStore,
        MasterProduct,
        RetailerListing,
    ],
    mutation: dict[str, str],
) -> None:
    store, product, listing = durable_catalog
    with session_scope(store.session_factory) as session:
        session.execute(
            update(CanonicalProductRecord)
            .where(CanonicalProductRecord.product_id == product.product_id)
            .values(**mutation)
        )

    with pytest.raises(DurableStorageError, match="modified search documents"):
        _verify(store, (product,), (listing,))
