from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import pc_build_recommender.catalog.streaming as streaming_module
from pc_build_recommender.catalog import (
    CANONICAL_IDENTITY_PREFLIGHT_SCHEMA_VERSION,
    CANONICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION,
    CanonicalIdentityResolutionError,
    audit_canonical_product_identities,
    canonical_identity_conflict_set_sha256,
    canonical_identity_resolution_findings,
    load_and_apply_canonical_identity_resolution,
    stream_processed_catalog,
)
from pc_build_recommender.domain import CanonicalProduct

NOW = datetime(2026, 8, 15, 8, 0, tzinfo=UTC)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _product(product_id: str, mpn: str | None) -> CanonicalProduct:
    return CanonicalProduct.model_validate(
        {
            "product_id": product_id,
            "category": "gpu",
            "brand": "Example",
            "model": product_id,
            "manufacturer_part_number": mpn,
            "canonical_name": f"Example {product_id}",
            "status": "active",
            "common_attributes": {"colour": "Black"},
            "category_attributes": {
                "vram_gb": 16,
                "length_mm": 300,
                "slot_width": 2.5,
                "board_power_watts": 200,
                "power_connectors": {"8_pin": 1},
            },
            "source_confidence": 1.0,
            "provenance": [
                {
                    "provenance_id": f"src_{product_id}",
                    "product_id": product_id,
                    "source_name": "identity-test",
                    "source_url": f"https://example.test/{product_id}",
                    "source_type": "manufacturer",
                    "retrieved_at": NOW.isoformat(),
                    "raw_content_hash": "a" * 64,
                    "parser_version": "test-v1",
                    "licence_or_access_note": "Test fixture",
                    "extraction_confidence": 1.0,
                }
            ],
            "created_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
        }
    )


def _write_catalog(path: Path, products: tuple[CanonicalProduct, ...]) -> None:
    rows = [
        {
            "schema_version": "pc-build-recommender.normalised-record.v1",
            "record_type": "canonical_product",
            "data": product.model_dump(mode="json"),
        }
        for product in products
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _evidence(suffix: str) -> list[dict[str, object]]:
    return [
        {
            "evidence_id": f"evidence-{suffix}",
            "source_url": f"https://example.test/evidence/{suffix}",
            "content_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
            "note": f"Manufacturer specification checked by {suffix}",
        }
    ]


def _decision(finding_type: str, product_ids: list[str]) -> dict[str, object]:
    if finding_type == "missing_mpn":
        product_id = product_ids[0]
        return {
            "assignments": [
                {
                    "source_product_id": product_id,
                    "canonical_product_id": product_id,
                    "manufacturer_part_number": "VERIFIED-MISSING-MPN",
                }
            ]
        }
    retained = product_ids[0]
    return {
        "assignments": [
            {
                "source_product_id": product_id,
                "canonical_product_id": retained,
                "manufacturer_part_number": "DUPLICATE-MPN",
            }
            for product_id in product_ids
        ]
    }


def _artifact_payload(
    source_path: Path,
    products: tuple[CanonicalProduct, ...],
) -> dict[str, Any]:
    report = audit_canonical_product_identities(products)
    resolutions: list[dict[str, object]] = []
    for finding_index, finding in enumerate(canonical_identity_resolution_findings(report)):
        decision = _decision(
            finding.finding_type,
            [product.product_id for product in finding.products],
        )
        resolutions.append(
            {
                "finding_id": finding.finding_id,
                "reviews": [
                    {
                        "review_id": f"review-{finding_index}-a",
                        "reviewer_id": "reviewer-a",
                        "reviewed_at": "2026-08-15T08:10:00+00:00",
                        "rationale": "Verified against the cited manufacturer specification.",
                        "evidence": _evidence(f"{finding_index}-review-a"),
                        "decision": deepcopy(decision),
                    },
                    {
                        "review_id": f"review-{finding_index}-b",
                        "reviewer_id": "reviewer-b",
                        "reviewed_at": "2026-08-15T08:20:00+00:00",
                        "rationale": "Independently verified the model and part number.",
                        "evidence": _evidence(f"{finding_index}-review-b"),
                        "decision": deepcopy(decision),
                    },
                ],
                "adjudication": {
                    "adjudication_id": f"adjudication-{finding_index}",
                    "adjudicator_id": "adjudicator-c",
                    "adjudicated_at": "2026-08-15T08:30:00+00:00",
                    "rationale": "The independent evidence supports this final identity.",
                    "evidence": _evidence(f"{finding_index}-adjudication"),
                    "decision": deepcopy(decision),
                },
            }
        )
    payload: dict[str, Any] = {
        "schema_version": CANONICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION,
        "resolution_id": "identity-resolution-test-v1",
        "created_at": "2026-08-15T08:40:00+00:00",
        "source_catalog": {
            "sha256": _sha256(source_path),
            "size_bytes": source_path.stat().st_size,
            "preflight_schema_version": CANONICAL_IDENTITY_PREFLIGHT_SCHEMA_VERSION,
            "preflight_content_sha256": report.to_dict()["content_sha256"],
            "conflict_set_sha256": canonical_identity_conflict_set_sha256(report),
        },
        "resolutions": resolutions,
    }
    payload["content_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _write_artifact(path: Path, payload: dict[str, Any]) -> str:
    path.write_bytes(_canonical_bytes(payload))
    return _sha256(path)


@pytest.fixture
def identity_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, tuple[CanonicalProduct, ...], dict[str, Any], str]:
    products = (
        _product("prod-a", "DUPLICATE-MPN"),
        _product("prod-b", "DUPLICATE-MPN"),
        _product("prod-c", None),
        _product("prod-d", "UNIQUE-MPN"),
    )
    source_path = tmp_path / "products.jsonl"
    artifact_path = tmp_path / "identity-resolution.json"
    _write_catalog(source_path, products)
    payload = _artifact_payload(source_path, products)
    artifact_sha256 = _write_artifact(artifact_path, payload)
    return source_path, artifact_path, products, payload, artifact_sha256


def test_applies_complete_reviewed_resolution_without_mutating_source(
    identity_fixture: tuple[Path, Path, tuple[CanonicalProduct, ...], dict[str, Any], str],
) -> None:
    source_path, artifact_path, products, _, artifact_sha256 = identity_fixture

    application = load_and_apply_canonical_identity_resolution(
        artifact_path,
        expected_artifact_sha256=artifact_sha256,
        source_catalog_path=source_path,
        products=products,
    )

    assert [product.product_id for product in application.products] == [
        "prod-a",
        "prod-c",
        "prod-d",
    ]
    assert products[1].product_id == "prod-b"
    assert products[2].manufacturer_part_number is None
    assert application.products[1].manufacturer_part_number == "VERIFIED-MISSING-MPN"
    assert application.summary.aliases == {"prod-b": "prod-a"}
    assert application.summary.mpn_overrides == {"prod-c": "VERIFIED-MISSING-MPN"}
    assert application.summary.resolved_finding_count == 2
    assert application.summary.source_preflight.production_ready is False
    assert application.summary.effective_preflight.production_ready is True


def test_rejects_incomplete_exact_conflict_set(
    identity_fixture: tuple[Path, Path, tuple[CanonicalProduct, ...], dict[str, Any], str],
) -> None:
    source_path, artifact_path, products, payload, _ = identity_fixture
    payload["resolutions"].pop()
    payload["content_sha256"] = hashlib.sha256(
        _canonical_bytes({key: value for key, value in payload.items() if key != "content_sha256"})
    ).hexdigest()
    artifact_sha256 = _write_artifact(artifact_path, payload)

    with pytest.raises(CanonicalIdentityResolutionError, match="incomplete"):
        load_and_apply_canonical_identity_resolution(
            artifact_path,
            expected_artifact_sha256=artifact_sha256,
            source_catalog_path=source_path,
            products=products,
        )


def test_rejects_stale_catalogue_and_tampered_artifact(
    identity_fixture: tuple[Path, Path, tuple[CanonicalProduct, ...], dict[str, Any], str],
) -> None:
    source_path, artifact_path, products, payload, artifact_sha256 = identity_fixture
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(CanonicalIdentityResolutionError, match="stale"):
        load_and_apply_canonical_identity_resolution(
            artifact_path,
            expected_artifact_sha256=artifact_sha256,
            source_catalog_path=source_path,
            products=products,
        )

    _write_catalog(source_path, products)
    payload["resolution_id"] = "tampered"
    tampered_file_sha256 = _write_artifact(artifact_path, payload)
    with pytest.raises(CanonicalIdentityResolutionError, match="content SHA-256"):
        load_and_apply_canonical_identity_resolution(
            artifact_path,
            expected_artifact_sha256=tampered_file_sha256,
            source_catalog_path=source_path,
            products=products,
        )


def test_rejects_non_independent_reviews(
    identity_fixture: tuple[Path, Path, tuple[CanonicalProduct, ...], dict[str, Any], str],
) -> None:
    source_path, artifact_path, products, payload, _ = identity_fixture
    payload["resolutions"][0]["reviews"][1]["reviewer_id"] = "reviewer-a"
    payload["content_sha256"] = hashlib.sha256(
        _canonical_bytes({key: value for key, value in payload.items() if key != "content_sha256"})
    ).hexdigest()
    artifact_sha256 = _write_artifact(artifact_path, payload)

    with pytest.raises(CanonicalIdentityResolutionError, match="distinct review"):
        load_and_apply_canonical_identity_resolution(
            artifact_path,
            expected_artifact_sha256=artifact_sha256,
            source_catalog_path=source_path,
            products=products,
        )


def test_streaming_applies_only_in_production_preflight(
    identity_fixture: tuple[Path, Path, tuple[CanonicalProduct, ...], dict[str, Any], str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, artifact_path, _, _, artifact_sha256 = identity_fixture
    offers_path = tmp_path / "offers.jsonl"
    offers_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="development inspection retains every source row"):
        stream_processed_catalog(
            source_path,
            offers_path,
            canonical_identity_resolution_path=artifact_path,
            canonical_identity_resolution_sha256=artifact_sha256,
        )

    monkeypatch.setattr(streaming_module, "validate_production_readiness", lambda *_: None)
    result = stream_processed_catalog(
        source_path,
        offers_path,
        canonical_identity_resolution_path=artifact_path,
        canonical_identity_resolution_sha256=artifact_sha256,
        require_production_ready=True,
    )

    assert result.product_ids == ("prod-a", "prod-c", "prod-d")
    assert result.readiness.canonical_identity_preflight is not None
    assert result.readiness.canonical_identity_preflight.production_ready is True
    assert result.canonical_identity_resolution is not None
    assert result.to_dict()["canonical_identity_resolution"]["aliases"] == {"prod-b": "prod-a"}
