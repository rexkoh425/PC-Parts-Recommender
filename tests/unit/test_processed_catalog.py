from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.review_catalog_mappings import main as review_mappings_main
from services.api.main import create_app
from services.api.settings import ApiSettings

from pc_build_recommender.catalog import (
    REVIEW_EVIDENCE_SCHEMA_VERSION,
    CatalogRepository,
    InMemoryCatalogReader,
    MappingOutcome,
    ProductionCatalogPolicy,
    ProductionCatalogReadinessError,
    ReviewStatus,
    create_db_engine,
    create_session_factory,
    init_database,
    iter_jsonl_objects,
    load_mapping_reviews,
    load_processed_catalog,
    seed_processed_catalog,
    session_scope,
    stream_processed_catalog,
    upsert_mapping_review,
    validate_production_readiness,
    validate_review_target,
)
from pc_build_recommender.domain import ComponentCategory, StockStatus

NOW = datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _product(
    *,
    product_id: str = "prod_asus_5060ti",
    mpn: str = "PRIME-RTX5060TI-O16G",
    vram_gb: int = 16,
) -> dict[str, object]:
    return {
        "schema_version": "pc-build-recommender.normalised-record.v1",
        "record_type": "canonical_product",
        "training_eligible": True,
        "published_claims_eligible": True,
        "data": {
            "product_id": product_id,
            "category": "gpu",
            "brand": "ASUS",
            "model": "Prime RTX 5060 Ti",
            "manufacturer_part_number": mpn,
            "canonical_name": f"ASUS Prime RTX 5060 Ti {vram_gb}GB",
            "status": "active",
            "common_attributes": {"colour": "Black"},
            "category_attributes": {
                "vram_gb": vram_gb,
                "length_mm": 304,
                "slot_width": 2.5,
                "board_power_watts": 180,
                "power_connectors": {"8_pin": 1},
            },
            "source_confidence": 0.95,
            "provenance": [
                {
                    "provenance_id": f"src_{product_id}",
                    "product_id": product_id,
                    "source_name": "buildcores_open_db",
                    "source_url": "https://example.test/product",
                    "source_type": "import",
                    "retrieved_at": NOW,
                    "raw_content_hash": "a" * 64,
                    "parser_version": "test-v1",
                    "licence_or_access_note": "ODC-By test fixture",
                    "extraction_confidence": 0.95,
                }
            ],
            "created_at": NOW,
            "updated_at": NOW,
        },
    }


def _offer(
    *,
    listing_id: str = "listing_asus_5060ti",
    title: str = "ASUS Prime RTX5060Ti O16G",
) -> dict[str, object]:
    return {
        "schema_version": "pc-build-recommender.normalised-record.v1",
        "record_type": "retailer_listing",
        "training_eligible": False,
        "published_claims_eligible": False,
        "development_only": False,
        "data_use_rights": _rights(grant_required_uses=True),
        "raw_record_sha256": "b" * 64,
        "provenance": {
            "source_name": "controlled-retailer-fixture",
            "source_url": "controlled-import://fixture/2026-07-22",
            "source_type": "retailer",
            "retrieved_at": NOW,
            "parser_version": "fixture-v1",
            "licence_or_access_note": "Development-only controlled test fixture.",
            "extraction_confidence": 0.95,
        },
        "data": {
            "listing": {
                "listing_id": listing_id,
                "product_id": "unmatched_fixture",
                "retailer": "Controlled Retailer",
                "source_listing_id": listing_id,
                "title": title,
                "condition": "new",
                "currency": "SGD",
                "base_price": "899.00",
                "shipping_price": "0.00",
                "stock_status": "unknown",
                "listing_url": "https://example.test/retailer",
                "first_seen_at": NOW,
                "last_seen_at": NOW,
            },
            "price_snapshot": {
                "snapshot_id": f"price_{listing_id}",
                "listing_id": listing_id,
                "observed_at": NOW,
                "base_price": "899.00",
                "shipping_price": "0.00",
                "stock_status": "unknown",
            },
        },
        "normalisation_metadata": {"category": "gpu"},
    }


def _research_offer() -> dict[str, object]:
    offer = _offer()
    offer["development_only"] = True
    offer["data_use_rights"] = _rights(grant_required_uses=False)
    return offer


def _dynacore_offer() -> dict[str, object]:
    offer = _research_offer()
    offer.pop("development_only")
    offer.pop("data_use_rights")
    provenance = offer["provenance"]
    assert isinstance(provenance, dict)
    provenance["source_name"] = "dynacore_controlled_pdf"
    metadata = offer["normalisation_metadata"]
    assert isinstance(metadata, dict)
    metadata["development_only"] = True
    return offer


def _rights(*, grant_required_uses: bool) -> dict[str, object]:
    return {
        "contract_reference": "fixture-feed-v1",
        "contract_version_url": "contract://fixture-feed/v1",
        "consent_effective_on": "2026-01-01",
        "consent_expires_on": None,
        "retention_days": 365,
        "deletion_required_on_termination": True,
        "deletion_sla_days": 30,
        "territories": ["SG"],
        "may_display": grant_required_uses,
        "may_cache": grant_required_uses,
        "may_store_history": grant_required_uses,
        "may_redistribute": False,
        "may_embed": False,
        "may_train": False,
        "may_derive": grant_required_uses,
    }


def _review_evidence(*, evidence_id: str = "review_fixture_noise") -> dict[str, object]:
    return {
        "schema_version": REVIEW_EVIDENCE_SCHEMA_VERSION,
        "record_type": "review_evidence",
        "data": {
            "evidence_id": evidence_id,
            "product_id": "prod_asus_5060ti",
            "aspect": "noise",
            "sentiment": -0.7,
            "evidence_text": "Permitted fixture evidence reports audible fan noise under load.",
            "source_url": "https://reviews.example.test/asus-5060ti",
            "published_at": NOW,
            "confidence": 0.9,
        },
        "data_use_rights": _rights(grant_required_uses=True),
        "provenance": {
            "source_name": "fixture_permitted_reviews",
            "source_url": "https://reviews.example.test/asus-5060ti",
            "retrieved_at": NOW,
            "raw_content_hash": "c" * 64,
            "parser_version": "fixture-review-v1",
            "licence_or_access_note": "Fixture contract permits cited Singapore display.",
        },
    }


def _rights_only_production_policy() -> ProductionCatalogPolicy:
    return ProductionCatalogPolicy(
        minimum_products=0,
        minimum_products_per_category=0,
        minimum_mapping_rate=0,
        minimum_critical_field_rate=0,
        require_complete_priced_coverage=False,
        require_complete_in_stock_coverage=False,
        require_complete_product_provenance=False,
        require_complete_offer_provenance=False,
        require_explicit_offer_rights=True,
        require_production_offer_rights=True,
        require_complete_listing_provenance=False,
        minimum_er_precision=0,
        minimum_er_labelled_pairs=1,
        require_promoted_entity_resolution_model=False,
    )


def _er_evaluation(
    path: Path,
    *,
    synthetic: bool = False,
    precision: float = 0.995,
    labelled_pair_count: int = 1000,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "pc-build-recommender.er-production-evaluation.v2",
                "evaluation_id": "er-eval-fixture-v1",
                "dataset_version": "human-labels-fixture-v1",
                "model_version": "er-model-fixture-v1",
                "label_source": "human_reviewed",
                "synthetic": synthetic,
                "precision": precision,
                "labelled_pair_count": labelled_pair_count,
                "evaluated_at": NOW,
                "artifact_sha256": "a" * 64,
                "review_queue_sha256": "b" * 64,
                "frozen_test_groups_sha256": "c" * 64,
                "auto_match_threshold": 0.98,
                "precision_numerator": round(precision * labelled_pair_count),
                "precision_denominator": labelled_pair_count,
                "precision_ci_lower": precision,
                "precision_ci_upper": precision,
                "recall": 0.95,
                "f1": 0.97,
                "reportable": True,
                "deployment_eligible": True,
            }
        ),
        encoding="utf-8",
    )


def test_exact_mpn_brand_mapping_preserves_unknown_stock(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])

    data = load_processed_catalog(products, offers)

    assert data.stats.product_count == 1
    assert data.stats.offer_count == 1
    assert data.stats.auto_matched_count == 1
    assert data.stats.matched_listings_by_category == {"gpu": 1}
    assert data.stats.known_in_stock_listing_count == 0
    assert not data.stats.has_complete_priced_coverage
    assert data.listings[0].product_id == "prod_asus_5060ti"
    assert data.listings[0].stock_status is StockStatus.UNKNOWN
    assert data.match_method_by_listing[data.listings[0].listing_id] == "exact_mpn_brand"

    reader = InMemoryCatalogReader(data)
    assert reader.get_product("prod_asus_5060ti") is not None
    assert reader.list_products(category=ComponentCategory.GPU)[0].product_id == (
        "prod_asus_5060ti"
    )


def test_streaming_and_in_memory_loaders_share_exact_mapping_decisions(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])

    in_memory = load_processed_catalog(products, offers)
    streamed = stream_processed_catalog(products, offers)

    assert streamed.stats.to_dict() == in_memory.stats.to_dict()
    assert [item.to_dict() for item in streamed.mapping_decisions] == [
        item.to_dict() for item in in_memory.mapping_decisions
    ]


def test_review_evidence_is_version_bound_and_persisted_by_both_catalog_paths(
    tmp_path: Path,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])
    _write_jsonl(reviews, [_review_evidence()])

    in_memory = load_processed_catalog(products, offers, review_evidence_path=reviews)
    streamed = stream_processed_catalog(products, offers, review_evidence_path=reviews)

    assert [item.evidence_id for item in in_memory.review_evidence] == ["review_fixture_noise"]
    assert streamed.review_evidence_count == 1
    assert streamed.stats.data_version == in_memory.stats.data_version

    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        persisted = stream_processed_catalog(
            products,
            offers,
            session=session,
            review_evidence_path=reviews,
        )
        assert persisted.review_evidence_count == 1
    with session_scope(factory) as session:
        repository = CatalogRepository(session)
        saved = repository.list_review_evidence("prod_asus_5060ti")
        assert [item.evidence_id for item in saved] == ["review_fixture_noise"]
    engine.dispose()


def test_numeric_variant_conflict_is_not_mapped(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer(title="ASUS Prime RTX5060Ti O16G 8GB")])

    data = load_processed_catalog(products, offers)

    assert data.listings == ()
    assert data.stats.rejected_conflict_count == 1


def test_combined_colour_row_is_not_a_unique_sku_match(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer(title="ASUS Prime RTX5060Ti O16G Black/White")])

    data = load_processed_catalog(products, offers)

    assert data.listings == ()
    assert data.stats.rejected_conflict_count == 1


def test_reviewed_mapping_requires_evidence_and_still_obeys_hard_conflicts(
    tmp_path: Path,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    mappings = tmp_path / "reviewed.json"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer(title="ASUS Prime current generation graphics card")])
    mappings.write_text(
        json.dumps(
            {
                "schema_version": "pc-build-recommender.reviewed-mappings.v1",
                "mappings": [
                    {
                        "listing_id": "listing_asus_5060ti",
                        "product_id": "prod_asus_5060ti",
                        "review_status": "approved",
                        "reviewed_by": "fixture-reviewer",
                        "evidence": "Manufacturer part number checked against source document.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    data = load_processed_catalog(products, offers, reviewed_mapping_path=mappings)
    assert data.stats.reviewed_matched_count == 1
    assert data.match_method_by_listing["listing_asus_5060ti"] == "reviewed"

    _write_jsonl(offers, [_offer(title="ASUS Prime RTX5060Ti 8GB")])
    with pytest.raises(ValueError, match="hard variant conflict"):
        load_processed_catalog(products, offers, reviewed_mapping_path=mappings)


def test_research_rows_cannot_become_training_eligible(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    unsafe = _research_offer()
    unsafe["training_eligible"] = True
    _write_jsonl(offers, [unsafe])

    with pytest.raises(ValueError, match="training/claims ineligible"):
        load_processed_catalog(products, offers)


def test_research_rows_cannot_claim_any_data_use_right(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    unsafe = _research_offer()
    rights = _rights(grant_required_uses=False)
    rights["may_train"] = True
    unsafe["data_use_rights"] = rights
    _write_jsonl(offers, [unsafe])

    with pytest.raises(ValueError, match="barred for every data use"):
        load_processed_catalog(products, offers)


def test_authorized_generic_offer_is_accepted_by_sg_production_rights_gate(
    tmp_path: Path,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    evaluation = tmp_path / "er-evaluation.json"
    _write_jsonl(products, [_product()])
    offer = _offer()
    offer["development_only"] = False
    offer["data_use_rights"] = _rights(grant_required_uses=True)
    _write_jsonl(offers, [offer])
    _er_evaluation(evaluation)

    data = load_processed_catalog(
        products,
        offer_path=offers,
        entity_resolution_evaluation_path=evaluation,
    )

    assert data.stats.auto_matched_count == 1
    assert data.readiness is not None
    assert data.readiness.offer_rights_explicit_count == 1
    assert data.readiness.offer_rights_production_valid_count == 1
    validate_production_readiness(data.readiness, _rights_only_production_policy())
    streamed = stream_processed_catalog(
        products,
        offer_path=offers,
        entity_resolution_evaluation_path=evaluation,
        require_production_ready=True,
        production_policy=_rights_only_production_policy(),
    )
    assert streamed.readiness.offer_rights_production_valid_count == 1


def test_research_web_offer_is_quarantined_before_serving_catalogue_load(
    tmp_path: Path,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "research-web-offers.jsonl"
    evaluation = tmp_path / "er-evaluation.json"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_research_offer()])
    _er_evaluation(evaluation)

    with pytest.raises(ValueError, match="quarantined.*serving catalogue"):
        load_processed_catalog(
            products,
            offers,
            entity_resolution_evaluation_path=evaluation,
        )
    with pytest.raises(ValueError, match="quarantined.*serving catalogue"):
        stream_processed_catalog(
            products,
            offers,
            entity_resolution_evaluation_path=evaluation,
        )


@pytest.mark.parametrize(
    ("retrieved_at", "expected"),
    [
        ("2026-07-22T00:00:00", "timezone-aware"),
        (
            (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "cannot be in the future",
        ),
        (
            (datetime.now(UTC) - timedelta(days=366)).isoformat(),
            "stale.*retention_days=365",
        ),
    ],
)
def test_generic_offer_freshness_fails_closed(
    tmp_path: Path,
    retrieved_at: str,
    expected: str,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    provenance = offer["provenance"]
    rights = offer["data_use_rights"]
    assert isinstance(provenance, dict)
    assert isinstance(rights, dict)
    provenance["retrieved_at"] = retrieved_at
    if expected.startswith("stale"):
        rights["consent_effective_on"] = "2020-01-01"
    _write_jsonl(offers, [offer])

    with pytest.raises(ValueError, match=expected):
        load_processed_catalog(products, offers)


@pytest.mark.parametrize(
    ("timestamp_location", "expected_label"),
    [
        ("first_seen_at", "listing.first_seen_at"),
        ("last_seen_at", "listing.last_seen_at"),
        ("observed_at", "price_snapshot.observed_at"),
    ],
)
def test_generic_offer_requires_strict_aware_observation_timestamps(
    tmp_path: Path,
    timestamp_location: str,
    expected_label: str,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    data = offer["data"]
    assert isinstance(data, dict)
    listing = data["listing"]
    snapshot = data["price_snapshot"]
    assert isinstance(listing, dict)
    assert isinstance(snapshot, dict)
    target = snapshot if timestamp_location == "observed_at" else listing
    target[timestamp_location] = "2026-07-22 00:00:00+00:00"
    _write_jsonl(offers, [offer])

    with pytest.raises(ValueError, match=rf"{expected_label}.*strict aware ISO-8601"):
        load_processed_catalog(products, offers)


@pytest.mark.parametrize(
    ("timestamp_location", "expected_label"),
    [
        ("first_seen_at", "listing.first_seen_at"),
        ("last_seen_at", "listing.last_seen_at"),
        ("observed_at", "price_snapshot.observed_at"),
    ],
)
def test_generic_offer_rejects_future_observation_timestamps(
    tmp_path: Path,
    timestamp_location: str,
    expected_label: str,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    data = offer["data"]
    assert isinstance(data, dict)
    listing = data["listing"]
    snapshot = data["price_snapshot"]
    assert isinstance(listing, dict)
    assert isinstance(snapshot, dict)
    target = snapshot if timestamp_location == "observed_at" else listing
    target[timestamp_location] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    _write_jsonl(offers, [offer])

    with pytest.raises(ValueError, match=rf"{expected_label} cannot be in the future"):
        load_processed_catalog(products, offers)


@pytest.mark.parametrize(
    ("timestamp_location", "expected_label"),
    [
        ("retrieved_at", "provenance.retrieved_at"),
        ("first_seen_at", "listing.first_seen_at"),
        ("last_seen_at", "listing.last_seen_at"),
        ("observed_at", "price_snapshot.observed_at"),
    ],
)
def test_generic_offer_rejects_pre_consent_timestamps(
    tmp_path: Path,
    timestamp_location: str,
    expected_label: str,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    rights = offer["data_use_rights"]
    data = offer["data"]
    provenance = offer["provenance"]
    assert isinstance(rights, dict)
    assert isinstance(data, dict)
    assert isinstance(provenance, dict)
    listing = data["listing"]
    snapshot = data["price_snapshot"]
    assert isinstance(listing, dict)
    assert isinstance(snapshot, dict)
    consent_effective_on = (datetime.now(UTC) - timedelta(days=1)).date()
    rights["consent_effective_on"] = consent_effective_on.isoformat()
    pre_consent = datetime.combine(
        consent_effective_on - timedelta(days=1),
        datetime.max.time(),
        tzinfo=UTC,
    ).isoformat()
    if timestamp_location == "retrieved_at":
        provenance[timestamp_location] = pre_consent
    elif timestamp_location == "observed_at":
        snapshot[timestamp_location] = pre_consent
    else:
        listing[timestamp_location] = pre_consent
    _write_jsonl(offers, [offer])

    with pytest.raises(
        ValueError,
        match=rf"{expected_label} predates consent_effective_on={consent_effective_on}",
    ):
        load_processed_catalog(products, offers)


def test_generic_offer_accepts_timestamps_at_consent_boundary(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    rights = offer["data_use_rights"]
    data = offer["data"]
    provenance = offer["provenance"]
    assert isinstance(rights, dict)
    assert isinstance(data, dict)
    assert isinstance(provenance, dict)
    listing = data["listing"]
    snapshot = data["price_snapshot"]
    assert isinstance(listing, dict)
    assert isinstance(snapshot, dict)
    consent_effective_on = (datetime.now(UTC) - timedelta(days=1)).date()
    consent_boundary = datetime.combine(
        consent_effective_on,
        datetime.min.time(),
        tzinfo=UTC,
    ).isoformat()
    rights["consent_effective_on"] = consent_effective_on.isoformat()
    provenance["retrieved_at"] = consent_boundary
    listing["first_seen_at"] = consent_boundary
    listing["last_seen_at"] = consent_boundary
    snapshot["observed_at"] = consent_boundary
    _write_jsonl(offers, [offer])

    loaded = load_processed_catalog(products, offers)

    assert loaded.readiness.offer_count == 1


def test_generic_offer_requires_monotonic_listing_timestamps(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    data = offer["data"]
    assert isinstance(data, dict)
    listing = data["listing"]
    snapshot = data["price_snapshot"]
    assert isinstance(listing, dict)
    assert isinstance(snapshot, dict)
    last_seen_at = datetime.now(UTC) - timedelta(hours=2)
    listing["first_seen_at"] = (last_seen_at + timedelta(hours=1)).isoformat()
    listing["last_seen_at"] = last_seen_at.isoformat()
    snapshot["observed_at"] = last_seen_at.isoformat()
    _write_jsonl(offers, [offer])

    with pytest.raises(ValueError, match="first_seen_at cannot be later than.*last_seen_at"):
        load_processed_catalog(products, offers)


def test_generic_offer_snapshot_must_match_listing_observation(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    data = offer["data"]
    assert isinstance(data, dict)
    listing = data["listing"]
    snapshot = data["price_snapshot"]
    assert isinstance(listing, dict)
    assert isinstance(snapshot, dict)
    listing["first_seen_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    snapshot["observed_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    _write_jsonl(offers, [offer])

    with pytest.raises(ValueError, match="observed_at must equal listing.last_seen_at"):
        load_processed_catalog(products, offers)


def test_generic_offer_provenance_cannot_predate_observation(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    observed_at = datetime.now(UTC) - timedelta(hours=1)
    data = offer["data"]
    provenance = offer["provenance"]
    assert isinstance(data, dict)
    assert isinstance(provenance, dict)
    listing = data["listing"]
    snapshot = data["price_snapshot"]
    assert isinstance(listing, dict)
    assert isinstance(snapshot, dict)
    listing["first_seen_at"] = observed_at.isoformat()
    listing["last_seen_at"] = observed_at.isoformat()
    snapshot["observed_at"] = observed_at.isoformat()
    provenance["retrieved_at"] = (observed_at - timedelta(minutes=1)).isoformat()
    _write_jsonl(offers, [offer])

    with pytest.raises(ValueError, match="retrieved_at cannot be earlier than.*observed_at"):
        load_processed_catalog(products, offers)


def test_generic_offer_retention_applies_to_price_observation(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    stale_observation = (datetime.now(UTC) - timedelta(days=366)).isoformat()
    data = offer["data"]
    rights = offer["data_use_rights"]
    assert isinstance(data, dict)
    assert isinstance(rights, dict)
    rights["consent_effective_on"] = "2020-01-01"
    listing = data["listing"]
    snapshot = data["price_snapshot"]
    assert isinstance(listing, dict)
    assert isinstance(snapshot, dict)
    listing["first_seen_at"] = stale_observation
    listing["last_seen_at"] = stale_observation
    snapshot["observed_at"] = stale_observation
    _write_jsonl(offers, [offer])

    with pytest.raises(
        ValueError,
        match="price_snapshot.observed_at exceeds retention_days=365",
    ):
        load_processed_catalog(products, offers)


@pytest.mark.parametrize("rights_failure", ["missing", "partial", "expired", "wrong_territory"])
def test_generic_offer_rights_fail_closed(
    tmp_path: Path,
    rights_failure: str,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "web-offers.jsonl"
    _write_jsonl(products, [_product()])
    offer = _offer()
    offer["development_only"] = False
    if rights_failure == "missing":
        offer.pop("data_use_rights")
    else:
        rights = _rights(grant_required_uses=True)
        if rights_failure == "partial":
            rights.pop("may_cache")
        elif rights_failure == "expired":
            rights["consent_effective_on"] = "2019-01-01"
            rights["consent_expires_on"] = "2020-01-01"
        else:
            rights["territories"] = ["US"]
        offer["data_use_rights"] = rights
    _write_jsonl(offers, [offer])

    with pytest.raises(ValueError, match="rights"):
        load_processed_catalog(products, offers)


def test_dynacore_legacy_quarantine_remains_barred(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "dynacore.jsonl"
    _write_jsonl(products, [_product()])
    legacy_offer = _dynacore_offer()
    provenance = legacy_offer["provenance"]
    assert isinstance(provenance, dict)
    provenance["retrieved_at"] = "2020-01-01T00:00:00+00:00"
    _write_jsonl(offers, [legacy_offer])

    legacy = load_processed_catalog(products, dynacore_path=offers)
    assert legacy.stats.auto_matched_count == 1
    assert legacy.readiness is not None
    assert legacy.readiness.offer_rights_explicit_count == 0
    assert legacy.readiness.offer_rights_production_valid_count == 0

    unsafe = _dynacore_offer()
    unsafe["data_use_rights"] = _rights(grant_required_uses=True)
    _write_jsonl(offers, [unsafe])
    with pytest.raises(ValueError, match="controlled Dynacore.*barred"):
        load_processed_catalog(products, dynacore_path=offers)


def test_all_false_rights_are_quarantined_from_serving_catalogue(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    evaluation = tmp_path / "er-evaluation.json"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_research_offer()])
    _er_evaluation(evaluation)

    with pytest.raises(ValueError, match="quarantined.*serving catalogue"):
        load_processed_catalog(
            products,
            offers,
            entity_resolution_evaluation_path=evaluation,
        )


@pytest.mark.parametrize(
    ("synthetic", "precision", "labelled_pair_count", "expected"),
    [
        (True, 0.999, 1000, "evaluation is synthetic"),
        (False, 0.98, 1000, "precision=0.9800 below minimum=0.9900"),
        (False, 0.999, 999, "labelled_pair_count=999 below minimum=1000"),
    ],
)
def test_er_evaluation_gate_rejects_nonqualifying_evidence(
    tmp_path: Path,
    synthetic: bool,
    precision: float,
    labelled_pair_count: int,
    expected: str,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    evaluation = tmp_path / "er-evaluation.json"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])
    _er_evaluation(
        evaluation,
        synthetic=synthetic,
        precision=precision,
        labelled_pair_count=labelled_pair_count,
    )

    data = load_processed_catalog(
        products,
        offers,
        entity_resolution_evaluation_path=evaluation,
    )

    assert data.readiness is not None
    assert any(expected in blocker for blocker in data.readiness.blockers())


def test_missing_er_evaluation_is_a_production_blocker(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])

    data = load_processed_catalog(products, offers)

    assert data.readiness is not None
    assert any(
        "no versioned human-labelled entity-resolution evaluation" in blocker
        for blocker in data.readiness.blockers()
    )

    with pytest.raises(FileNotFoundError):
        load_processed_catalog(
            products,
            offers,
            entity_resolution_evaluation_path=tmp_path / "missing-er-evaluation.json",
        )


def test_processed_production_startup_rejects_barred_rights_and_dev_is_explicit(
    tmp_path: Path,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    evaluation = tmp_path / "er-evaluation.json"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_dynacore_offer()])
    _er_evaluation(evaluation)
    settings_values = {
        "service_mode": "processed_catalog",
        "database_url": "postgresql+psycopg://test:test@localhost/test",
        "buildcores_catalog_path": products,
        "governed_offers_path": offers,
        "entity_resolution_evaluation_path": evaluation,
        "serving_manifest_path": tmp_path / "serving-manifest.json",
        "serving_manifest_sha256": "a" * 64,
    }

    with pytest.raises(
        ValidationError,
        match="entity-resolution evaluation must come from the immutable serving manifest",
    ):
        create_app(ApiSettings(environment="production", **settings_values))

    with pytest.raises(ValidationError, match="forbidden outside development/test"):
        ApiSettings(
            environment="production",
            allow_development_catalog=True,
            **settings_values,
        )

    development = create_app(
        ApiSettings(
            environment="test",
            allow_development_catalog=True,
            **settings_values,
        )
    )
    assert development.state.settings.allow_development_catalog is True


def test_processed_seed_is_idempotent(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])
    data = load_processed_catalog(products, offers)
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    factory = create_session_factory(engine)

    for _ in range(2):
        with session_scope(factory) as session:
            seed_processed_catalog(session, data)

    with session_scope(factory) as session:
        repository = CatalogRepository(session)
        assert len(repository.list_products()) == 1
        assert len(repository.list_listings(product_id="prod_asus_5060ti")) == 1
        assert len(repository.list_price_snapshots("listing_asus_5060ti")) == 1
        provenance = repository.list_provenance(listing_id="listing_asus_5060ti")
        assert len(provenance) == 1
        assert provenance[0].source_name == "controlled-retailer-fixture"


def test_rejected_review_suppresses_an_automatic_match(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    mappings = tmp_path / "reviewed.json"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])
    upsert_mapping_review(
        mappings,
        listing_id="listing_asus_5060ti",
        status=ReviewStatus.REJECTED,
        reviewed_by="fixture-reviewer",
        evidence="Retailer confirmed this is a bundled row rather than a unique SKU.",
    )

    data = load_processed_catalog(products, offers, reviewed_mapping_path=mappings)

    assert data.listings == ()
    assert data.stats.auto_matched_count == 0
    assert data.stats.review_rejected_count == 1
    assert data.mapping_decisions[0].outcome is MappingOutcome.REVIEW_REJECTED
    assert load_mapping_reviews(mappings)["listing_asus_5060ti"].product_id is None


def test_readiness_reports_exact_fields_and_fails_closed(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])

    data = load_processed_catalog(products, offers)
    assert data.readiness is not None
    readiness = data.readiness

    assert readiness.product_provenance_complete_count == 1
    assert readiness.offer_provenance_complete_count == 1
    assert readiness.offer_rights_explicit_count == 1
    assert readiness.offer_rights_missing_count == 0
    assert readiness.offer_rights_production_valid_count == 1
    assert readiness.listing_provenance_complete_count == 1
    assert readiness.compatibility_ready_products_by_category["gpu"] == 1
    assert readiness.compatibility_ready_products_by_category["case"] == 0
    assert readiness.critical_field_rate("gpu", "vram_gb") == 1.0
    assert readiness.mapping_rate == 1.0
    assert not readiness.has_complete_in_stock_coverage
    with pytest.raises(ProductionCatalogReadinessError, match="not production-ready"):
        validate_production_readiness(readiness)


def test_jsonl_reader_rejects_an_oversized_record(tmp_path: Path) -> None:
    source = tmp_path / "oversized.jsonl"
    _write_jsonl(source, [{"value": "x" * 128}])

    with pytest.raises(ValueError, match="exceeds 32 bytes"):
        list(iter_jsonl_objects(source, max_line_bytes=32))


def test_streaming_import_is_idempotent_and_reports_same_mapping(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    factory = create_session_factory(engine)

    for _ in range(2):
        with session_scope(factory) as session:
            result = stream_processed_catalog(
                products,
                offers,
                session=session,
                batch_size=1,
            )
            assert result.stats.auto_matched_count == 1
            assert result.readiness.listing_provenance_complete_count == 1
            assert result.product_ids == ("prod_asus_5060ti",)
            assert result.listing_ids == ("listing_asus_5060ti",)

    with session_scope(factory) as session:
        repository = CatalogRepository(session)
        assert len(repository.list_products()) == 1
        assert len(repository.list_listings(product_id="prod_asus_5060ti")) == 1
        assert len(repository.list_provenance(listing_id="listing_asus_5060ti")) == 1


def test_streaming_production_gate_rolls_back_import(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer()])
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    factory = create_session_factory(engine)

    with (
        pytest.raises(ProductionCatalogReadinessError),
        session_scope(factory) as session,
    ):
        stream_processed_catalog(
            products,
            offers,
            session=session,
            batch_size=1,
            require_production_ready=True,
        )

    with session_scope(factory) as session:
        assert CatalogRepository(session).list_products() == []


def test_review_target_validation_rejects_hard_variant_conflict(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer(title="ASUS Prime RTX5060Ti O16G 8GB")])

    with pytest.raises(ValueError, match="hard variant conflict"):
        validate_review_target(
            offers,
            listing_id="listing_asus_5060ti",
            buildcores_path=products,
            product_id="prod_asus_5060ti",
        )


def test_review_cli_generates_a_persisted_unresolved_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    products = tmp_path / "products.jsonl"
    offers = tmp_path / "offers.jsonl"
    queue = tmp_path / "review-queue.json"
    _write_jsonl(products, [_product()])
    _write_jsonl(offers, [_offer(title="ASUS current generation graphics card")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_catalog_mappings.py",
            "queue",
            "--buildcores",
            str(products),
            "--offers",
            str(offers),
            "--output",
            str(queue),
        ],
    )

    assert review_mappings_main() == 0
    payload = json.loads(queue.read_text(encoding="utf-8"))
    assert payload["unresolved_count"] == 1
    assert payload["decisions"][0]["outcome"] == "unmatched"
