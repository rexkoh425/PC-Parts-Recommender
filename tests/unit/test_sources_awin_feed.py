from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pipelines.parsing.streaming_writer import StreamingPublicationError
from pipelines.sources.awin_feed import (
    AwinFeedError,
    AwinFeedLimitError,
    AwinLocalFeedAdapter,
)
from pipelines.sources.signed_policy import (
    DETACHED_SIGNATURE_SCHEMA_VERSION,
    SIGNED_POLICY_SCHEMA_VERSION,
    TRUST_ROOT_SCHEMA_VERSION,
    SignedPolicyError,
    VerifiedSignedPolicy,
    verify_signed_policy,
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _policy_payload(
    *,
    compression: str = "none",
    limits: dict[str, int | float] | None = None,
    production: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": "pc-build-recommender.awin-feed-policy.v1",
        "advertiser_id": "12345",
        "feed_id": "67890",
        "retailer": "Fixture Hardware",
        "licence_or_access_note": "Synthetic test agreement; not a real data grant.",
        "rights": {
            "contract_reference": "fixture-awin-agreement-v1",
            "contract_version_url": "contract://fixture/awin-agreement-v1",
            "consent_effective_on": "2026-01-01",
            "consent_expires_on": "2026-12-31",
            "retention_days": 30,
            "deletion_required_on_termination": True,
            "deletion_sla_days": 7,
            "territories": ["SG"],
            "grants": ["display", "cache", "store_history", "derive"],
        },
        "allowed_currencies": ["SGD"],
        "allowed_link_hosts": ["shop.example.test"],
        "category_mappings": {"id:gpu": "gpu"},
        "feed": {"format": "csv", "compression": compression, "delimiter": ","},
        "default_condition": "new",
        "production_catalog_eligible": production,
        "training_eligible": False,
        "published_claims_eligible": False,
        "allow_non_new": False,
        **({"limits": limits} if limits is not None else {}),
    }


def _signed_policy(
    tmp_path: Path,
    payload: dict[str, Any],
    *,
    stem: str = "awin",
) -> tuple[VerifiedSignedPolicy, Path, Path, Path, str]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_root_path = tmp_path / f"{stem}-trust.json"
    trust_root_path.write_bytes(
        _json_bytes(
            {
                "schema_version": TRUST_ROOT_SCHEMA_VERSION,
                "keys": [
                    {
                        "key_id": "fixture-governance-key",
                        "algorithm": "Ed25519",
                        "public_key": base64.b64encode(public_key).decode("ascii"),
                        "status": "active",
                        "valid_from": "2025-01-01T00:00:00Z",
                        "valid_until": "2028-01-01T00:00:00Z",
                    }
                ],
            }
        )
    )
    policy_path = tmp_path / f"{stem}-policy.json"
    policy_bytes = _json_bytes(
        {
            "schema_version": SIGNED_POLICY_SCHEMA_VERSION,
            "policy_id": f"fixture-{stem}",
            "issued_at": "2026-01-01T00:00:00Z",
            "expires_at": "2026-12-31T00:00:00Z",
            "payload": payload,
        }
    )
    policy_path.write_bytes(policy_bytes)
    signature_path = tmp_path / f"{stem}-policy.sig.json"
    signature_path.write_bytes(
        _json_bytes(
            {
                "schema_version": DETACHED_SIGNATURE_SCHEMA_VERSION,
                "key_id": "fixture-governance-key",
                "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
                "signature": base64.b64encode(private_key.sign(policy_bytes)).decode("ascii"),
            }
        )
    )
    trust_sha = hashlib.sha256(trust_root_path.read_bytes()).hexdigest()
    verified = verify_signed_policy(
        policy_path=policy_path,
        signature_path=signature_path,
        trust_root_path=trust_root_path,
        expected_trust_root_sha256=trust_sha,
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )
    return verified, policy_path, signature_path, trust_root_path, trust_sha


_FIELDS = [
    "merchant_id",
    "merchant_name",
    "merchant_product_id",
    "product_name",
    "search_price",
    "currency",
    "delivery_cost",
    "aw_deep_link",
    "category_id",
    "stock_status",
    "in_stock",
    "is_for_sale",
    "condition",
    "brand_name",
    "mpn",
    "product_GTIN",
]


def _row(index: int = 1, **updates: str) -> dict[str, str]:
    result = {
        "merchant_id": "12345",
        "merchant_name": "Fixture Hardware",
        "merchant_product_id": f"gpu-{index}",
        "product_name": f"Fixture GPU {index}",
        "search_price": "599.90",
        "currency": "SGD",
        "delivery_cost": "0",
        "aw_deep_link": f"https://shop.example.test/products/gpu-{index}?utm_source=awin",
        "category_id": "gpu",
        "stock_status": "in_stock",
        "in_stock": "1",
        "is_for_sale": "1",
        "condition": "new",
        "brand_name": "Fixture",
        "mpn": f"GPU-{index}",
        "product_GTIN": f"88888888{index:05d}",
    }
    result.update(updates)
    return result


def _write_feed(path: Path, rows: list[dict[str, str]], *, gzip_encoded: bool = False) -> None:
    if gzip_encoded:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


@pytest.mark.parametrize("compression", ["none", "gzip"])
def test_signed_awin_feed_streams_to_secret_safe_idempotent_artifact(
    tmp_path: Path,
    compression: str,
) -> None:
    verified, *_ = _signed_policy(
        tmp_path,
        _policy_payload(compression=compression),
        stem=f"valid-{compression}",
    )
    # A local operator filename may be arbitrary; it must never enter provenance.
    suffix = "csv.gz" if compression == "gzip" else "csv"
    feed_path = tmp_path / f"apikey_SUPERSECRET_feed.{suffix}"
    _write_feed(feed_path, [_row()], gzip_encoded=compression == "gzip")
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)

    snapshot = adapter.fetch(feed_path=feed_path)
    first = adapter.materialize(snapshot, processed_root=tmp_path / "processed")
    second = adapter.materialize(snapshot, processed_root=tmp_path / "processed")

    assert first.accepted_count == 1
    assert first.rejected_count == 0
    assert second.reused is True
    record = json.loads(first.records_jsonl.read_text(encoding="utf-8"))
    assert record["normalisation_metadata"]["category"] == "gpu"
    assert record["normalisation_metadata"]["canonical_mapping_status"] == "unmatched"
    assert record["data"]["listing"]["product_id"].startswith("unmatched_product_")
    assert record["data"]["listing"]["stock_status"] == "in_stock"
    assert record["data_use_rights"]["may_train"] is False
    assert record["provenance"]["source_url"] == "awin://advertisers/12345/feeds/67890"
    assert record["rights_authority"]["policy_id"] == f"fixture-valid-{compression}"
    raw_metadata = json.loads(snapshot.raw.metadata_path.read_text(encoding="utf-8"))
    assert raw_metadata["source_url"] == "awin://advertisers/12345/feeds/67890"

    persisted_control = b"\n".join(
        path.read_bytes()
        for path in (
            snapshot.raw.metadata_path,
            snapshot.authorization_receipt_path,
            first.records_jsonl,
            first.manifest_json,
            first.quality_json,
        )
    ).lower()
    assert b"supersecret" not in persisted_control
    assert b"/apikey/" not in persisted_control
    raw_bytes = snapshot.raw.path.read_bytes().lower()
    assert b"/apikey/" not in raw_bytes
    assert b"apikey=" not in raw_bytes


def test_streaming_awin_quality_gate_rejects_a_material_listing_count_drop(
    tmp_path: Path,
) -> None:
    verified, *_ = _signed_policy(tmp_path, _policy_payload(), stem="quality-regression")
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)
    processed_root = tmp_path / "processed"
    baseline_feed = tmp_path / "baseline.csv"
    _write_feed(baseline_feed, [_row(index) for index in range(1, 11)])

    baseline = adapter.materialize(
        adapter.fetch(feed_path=baseline_feed),
        processed_root=processed_root,
    )
    baseline_quality = json.loads(baseline.quality_json.read_text(encoding="utf-8"))
    assert baseline_quality["status"] == "pass"

    reduced_feed = tmp_path / "reduced.csv"
    _write_feed(reduced_feed, [_row(index) for index in range(1, 7)])
    reduced = adapter.materialize(
        adapter.fetch(feed_path=reduced_feed),
        processed_root=processed_root,
    )
    reduced_quality = json.loads(reduced.quality_json.read_text(encoding="utf-8"))
    accepted_check = next(
        check
        for check in reduced_quality["checks"]
        if check["name"] == "accepted_count_regression"
    )
    listing_check = next(
        check
        for check in reduced_quality["checks"]
        if check["name"] == "retailer_listing_count_regression"
    )

    assert accepted_check["count"] == 1
    assert listing_check["count"] == 1
    assert reduced_quality["status"] == "fail"


def test_policy_tamper_fails_before_raw_snapshot_directory_exists(tmp_path: Path) -> None:
    _, policy_path, signature_path, trust_path, trust_sha = _signed_policy(
        tmp_path,
        _policy_payload(),
        stem="tampered",
    )
    policy_path.write_bytes(policy_path.read_bytes() + b" ")

    with pytest.raises(SignedPolicyError, match="different policy bytes"):
        verify_signed_policy(
            policy_path=policy_path,
            signature_path=signature_path,
            trust_root_path=trust_path,
            expected_trust_root_sha256=trust_sha,
            now=datetime(2026, 7, 23, tzinfo=UTC),
        )
    assert not (tmp_path / "raw").exists()


def test_unsigned_claim_eligibility_needs_distinct_contractual_grant(tmp_path: Path) -> None:
    payload = _policy_payload()
    payload["published_claims_eligible"] = True
    verified, *_ = _signed_policy(tmp_path, payload, stem="claims-no-grant")

    with pytest.raises(PermissionError, match="contractual grant reference"):
        AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)


def test_signed_claim_grant_reference_is_bound_into_each_record(tmp_path: Path) -> None:
    payload = _policy_payload()
    payload["published_claims_eligible"] = True
    payload["published_claims_grant_reference"] = "contract://fixture/published-claims-v1"
    verified, *_ = _signed_policy(tmp_path, payload, stem="claims-with-grant")
    feed_path = tmp_path / "feed.csv"
    _write_feed(feed_path, [_row()])
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)

    artifacts = adapter.materialize(
        adapter.fetch(feed_path=feed_path),
        processed_root=tmp_path / "processed",
    )

    record = json.loads(artifacts.records_jsonl.read_text(encoding="utf-8"))
    assert record["published_claims_eligible"] is True
    assert (
        record["rights_authority"]["published_claims_grant_reference"]
        == "contract://fixture/published-claims-v1"
    )


@pytest.mark.parametrize(
    "credential_url",
    [
        "https://shop.example.test/apikey/SUPERSECRET/product",
        "https://shop.example.test/%2Fapikey%2FSUPERSECRET/product",
    ],
)
def test_credential_bearing_row_aborts_without_processed_publication(
    tmp_path: Path,
    credential_url: str,
) -> None:
    verified, *_ = _signed_policy(tmp_path, _policy_payload(), stem="credential-row")
    feed_path = tmp_path / "feed.csv"
    _write_feed(
        feed_path,
        [_row(aw_deep_link=credential_url)],
    )
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)

    with pytest.raises(AwinFeedError, match="credential-bearing"):
        adapter.fetch(feed_path=feed_path)
    assert not (tmp_path / "raw").exists()
    assert not list((tmp_path / "processed").glob("awin_*/*/manifest.json"))


def test_gzip_expansion_is_bounded_before_raw_snapshot_publication(tmp_path: Path) -> None:
    verified, *_ = _signed_policy(
        tmp_path,
        _policy_payload(
            compression="gzip",
            limits={"maximum_decompressed_bytes": 128},
        ),
        stem="gzip-limit",
    )
    feed_path = tmp_path / "feed.csv.gz"
    _write_feed(feed_path, [_row()], gzip_encoded=True)
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)

    with pytest.raises(AwinFeedLimitError, match="decompressed bytes"):
        adapter.fetch(feed_path=feed_path)

    assert not (tmp_path / "raw").exists()


def test_strict_offer_fields_are_quarantined_with_bounded_reasons(tmp_path: Path) -> None:
    verified, *_ = _signed_policy(tmp_path, _policy_payload(), stem="invalid-rows")
    rows = [
        _row(1),
        _row(2, merchant_id="999"),
        _row(3, currency="USD"),
        _row(4, delivery_cost=""),
        _row(5, category_id="case"),
        _row(6, stock_status="in_stock", in_stock="0"),
        _row(7, search_price="1,23.00"),
        _row(1),
    ]
    feed_path = tmp_path / "invalid.csv"
    _write_feed(feed_path, rows)
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)
    artifacts = adapter.materialize(
        adapter.fetch(feed_path=feed_path), processed_root=tmp_path / "processed"
    )

    assert artifacts.accepted_count == 1
    assert artifacts.rejected_count == 7
    reasons = {
        json.loads(line)["reason"]
        for line in artifacts.rejections_jsonl.read_text(encoding="utf-8").splitlines()
    }
    assert reasons == {
        "merchant_id_mismatch",
        "currency_not_allowed",
        "shipping_price_missing",
        "category_unmapped",
        "stock_signal_conflict",
        "invalid_money",
        "duplicate_source_listing_id",
    }


def test_large_feed_uses_disk_backed_uniqueness_and_bounded_python_memory(
    tmp_path: Path,
) -> None:
    row_count = 10_000
    verified, *_ = _signed_policy(
        tmp_path,
        _policy_payload(
            limits={
                "maximum_records": row_count,
                "maximum_rejections": 10,
                "maximum_output_bytes": 64 * 1024 * 1024,
            }
        ),
        stem="bounded",
    )
    feed_path = tmp_path / "large.csv"
    with feed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(_row(index))
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)
    snapshot = adapter.fetch(feed_path=feed_path)

    tracemalloc.start()
    artifacts = adapter.materialize(snapshot, processed_root=tmp_path / "processed")
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert artifacts.accepted_count == row_count
    assert artifacts.rejected_count == 0
    assert peak < 64 * 1024 * 1024
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["metadata"]["statistics"]["unique_source_listing_ids"] == row_count
    assert not list((tmp_path / "processed").glob(".awin-dedupe.*"))


def test_existing_artifact_tamper_is_rejected(tmp_path: Path) -> None:
    verified, *_ = _signed_policy(tmp_path, _policy_payload(), stem="artifact-tamper")
    feed_path = tmp_path / "feed.csv"
    _write_feed(feed_path, [_row()])
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)
    snapshot = adapter.fetch(feed_path=feed_path)
    artifacts = adapter.materialize(snapshot, processed_root=tmp_path / "processed")
    artifacts.records_jsonl.write_text("{}\n", encoding="utf-8")

    with pytest.raises(StreamingPublicationError, match="hash mismatch"):
        adapter.materialize(snapshot, processed_root=tmp_path / "processed")
