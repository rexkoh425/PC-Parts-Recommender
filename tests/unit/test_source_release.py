from __future__ import annotations

import base64
import csv
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pipelines.source_release import (
    AuthorizedBatchAuthorityArtifacts,
    AuthorizedBatchReleaseArtifacts,
    AuthorizedBatchReleaseError,
    VerifiedAuthorizedBatchRelease,
    publish_awin_production_batch_release,
    verify_awin_production_batch_release,
)
from pipelines.sources.awin_feed import AwinLocalFeedAdapter
from pipelines.sources.signed_policy import (
    DETACHED_SIGNATURE_SCHEMA_VERSION,
    SIGNED_POLICY_SCHEMA_VERSION,
    TRUST_ROOT_SCHEMA_VERSION,
    VerifiedSignedPolicy,
    verify_signed_policy,
)
from scripts.verify_authorized_source_release import main as verify_release_main
from services.api import serving_release

from pc_build_recommender.application import ServingConfigurationError

_VERIFY_AT = datetime.now(UTC)
_REGISTRY_VERIFIED_ON = (_VERIFY_AT - timedelta(days=1)).date().isoformat()
_STALE_REGISTRY_VERIFIED_ON = (_VERIFY_AT - timedelta(days=31)).date().isoformat()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"


def _serving_reference(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _content_digest(path: Path) -> dict[str, object]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _policy_payload(
    *,
    retention_days: int = 36_500,
    consent_expires_on: str = "2098-12-31",
) -> dict[str, Any]:
    return {
        "schema_version": "pc-build-recommender.awin-feed-policy.v1",
        "advertiser_id": "12345",
        "feed_id": "67890",
        "retailer": "Synthetic Hardware",
        "licence_or_access_note": "Synthetic signed test grant; not real retailer authority.",
        "rights": {
            "contract_reference": "contract://synthetic/awin-v1",
            "contract_version_url": "contract://synthetic/awin-v1",
            "consent_effective_on": "2025-01-01",
            "consent_expires_on": consent_expires_on,
            "retention_days": retention_days,
            "deletion_required_on_termination": True,
            "deletion_sla_days": 7,
            "territories": ["SG"],
            "grants": ["display", "cache", "store_history", "derive"],
        },
        "allowed_currencies": ["SGD"],
        "allowed_link_hosts": ["shop.example.test"],
        "category_mappings": {"id:gpu": "gpu"},
        "feed": {"format": "csv", "compression": "none", "delimiter": ","},
        "default_condition": "new",
        "production_catalog_eligible": True,
        "training_eligible": False,
        "published_claims_eligible": False,
        "allow_non_new": False,
    }


def _signed_authority(
    tmp_path: Path,
    *,
    retention_days: int = 36_500,
    consent_expires_on: str = "2098-12-31",
) -> tuple[AuthorizedBatchAuthorityArtifacts, str, VerifiedSignedPolicy]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_root = tmp_path / "trust-root.json"
    trust_root.write_bytes(
        _json_bytes(
            {
                "schema_version": TRUST_ROOT_SCHEMA_VERSION,
                "keys": [
                    {
                        "key_id": "synthetic-governance-key",
                        "algorithm": "Ed25519",
                        "public_key": base64.b64encode(public_key).decode("ascii"),
                        "status": "active",
                        "valid_from": "2025-01-01T00:00:00Z",
                        "valid_until": "2099-01-01T00:00:00Z",
                    }
                ],
            }
        )
    )
    policy = tmp_path / "policy.json"
    policy_bytes = _json_bytes(
        {
            "schema_version": SIGNED_POLICY_SCHEMA_VERSION,
            "policy_id": "synthetic-awin-production-v1",
            "issued_at": "2025-01-01T00:00:00Z",
            "expires_at": "2098-12-31T00:00:00Z",
            "payload": _policy_payload(
                retention_days=retention_days,
                consent_expires_on=consent_expires_on,
            ),
        }
    )
    policy.write_bytes(policy_bytes)
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    signature = tmp_path / "policy.sig.json"
    signature.write_bytes(
        _json_bytes(
            {
                "schema_version": DETACHED_SIGNATURE_SCHEMA_VERSION,
                "key_id": "synthetic-governance-key",
                "policy_sha256": policy_sha256,
                "signature": base64.b64encode(private_key.sign(policy_bytes)).decode("ascii"),
            }
        )
    )
    source_registry = tmp_path / "source-registry.yaml"
    source_registry.write_text(
        f"""schema_version: pc-build-recommender.source-registry.v1
verified_on: {_REGISTRY_VERIFIED_ON}
sources:
  awin_12345_67890:
    kind: authorized_retailer_product_feed
    template: awin_authorized_local_feed
    status: active
    source_url: awin://advertisers/12345/feeds/67890
    advertiser_id: "12345"
    feed_id: "67890"
    retailer: Synthetic Hardware
    parser_version: awin-local-csv-stream-v1
    policy_id: synthetic-awin-production-v1
    policy_sha256: {policy_sha256}
    admitted_on: {_REGISTRY_VERIFIED_ON}
    admission_expires_on: 2098-12-30
    revoked_on: null
    revocation_reason: null
    production_catalog_eligible: true
    access_note: Synthetic explicit per-feed test admission.
source_templates:
  awin_authorized_local_feed:
    kind: authorized_retailer_product_feed
    parser_version: awin-local-csv-stream-v1
    access: controlled_local_import_only
    scheduled_fetch: false
    source_url: awin://advertisers/supplied-per-policy/feeds/supplied-per-policy
    version_policy: content_sha256_and_signed_policy
    licence: supplied-per-policy
    attribution_required: supplied-per-policy
    production_catalog_eligible: false
    training_eligible: false
    published_claims_eligible: false
    redistribution_eligible: false
    requirements:
      - already_downloaded_authorized_local_csv_or_gzip
      - detached_ed25519_policy_signature
      - independently_pinned_trust_root_sha256
      - exact_advertiser_feed_merchant_host_and_category_mapping
      - exact_sgd_currency_and_explicit_shipping
      - bounded_input_decompression_records_rejections_and_output
      - separate_signed_contract_reference_for_published_claims
    data_use_rights:
      rights_status: requires_per_feed_signed_contract
      contract_reference: supplied-per-policy
      contract_version_url: supplied-per-policy
      consent_effective_on: supplied-per-policy
      consent_expires_on: supplied-per-policy
      retention_days: supplied-per-policy
      deletion_required_on_termination: supplied-per-policy
      deletion_sla_days: supplied-per-policy
      territories: supplied-per-policy
      may_display: false
      may_cache: false
      may_store_history: false
      may_redistribute: false
      may_embed: false
      may_train: false
      may_derive: false
    access_note: Synthetic fail-closed test template.
auxiliary_sources: {{}}
blocked_or_restricted_sources: {{}}
""",
        encoding="utf-8",
    )
    trust_sha = hashlib.sha256(trust_root.read_bytes()).hexdigest()
    verified = verify_signed_policy(
        policy_path=policy,
        signature_path=signature,
        trust_root_path=trust_root,
        expected_trust_root_sha256=trust_sha,
        now=_VERIFY_AT,
    )
    return (
        AuthorizedBatchAuthorityArtifacts(
            policy=policy,
            policy_signature=signature,
            trust_root=trust_root,
            source_registry=source_registry,
        ),
        trust_sha,
        verified,
    )


def _write_feed(path: Path, *, count: int) -> None:
    fields = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(1, count + 1):
            writer.writerow(
                {
                    "merchant_id": "12345",
                    "merchant_name": "Synthetic Hardware",
                    "merchant_product_id": f"gpu-{index}",
                    "product_name": f"Synthetic GPU {index}",
                    "search_price": "599.90",
                    "currency": "SGD",
                    "delivery_cost": "0",
                    "aw_deep_link": f"https://shop.example.test/products/gpu-{index}",
                    "category_id": "gpu",
                    "stock_status": "in_stock",
                    "in_stock": "1",
                    "is_for_sale": "1",
                    "condition": "new",
                }
            )


def _materialize(
    tmp_path: Path,
    authority: AuthorizedBatchAuthorityArtifacts,
    trust_sha: str,
    *,
    count: int = 1,
    feed_name: str = "feed.csv",
) -> tuple[AuthorizedBatchReleaseArtifacts, AwinLocalFeedAdapter]:
    verified = verify_signed_policy(
        policy_path=authority.policy,
        signature_path=authority.policy_signature,
        trust_root_path=authority.trust_root,
        expected_trust_root_sha256=trust_sha,
        now=_VERIFY_AT,
    )
    adapter = AwinLocalFeedAdapter(raw_root=tmp_path / "raw", verified_policy=verified)
    feed = tmp_path / feed_name
    _write_feed(feed, count=count)
    snapshot = adapter.fetch(feed_path=feed)
    processed = adapter.materialize(snapshot, processed_root=tmp_path / "processed")
    return (
        AuthorizedBatchReleaseArtifacts(
            raw_snapshot=snapshot.raw.path,
            raw_metadata=snapshot.raw.metadata_path,
            authorization_receipt=snapshot.authorization_receipt_path,
            records=processed.records_jsonl,
            rejections=processed.rejections_jsonl,
            processed_manifest=processed.manifest_json,
            quality_report=processed.quality_json,
        ),
        adapter,
    )


def _publish(
    tmp_path: Path,
    artifacts: AuthorizedBatchReleaseArtifacts,
    authority: AuthorizedBatchAuthorityArtifacts,
    trust_sha: str,
    *,
    now: datetime | None = None,
) -> VerifiedAuthorizedBatchRelease:
    return publish_awin_production_batch_release(
        release_root=tmp_path / "releases",
        artifacts=artifacts,
        authority=authority,
        expected_trust_root_sha256=trust_sha,
        now=now or datetime.now(UTC),
    )


def _rewrite_processed_file_reference(
    artifacts: AuthorizedBatchReleaseArtifacts,
    *,
    filename: str,
    path: Path,
) -> None:
    manifest = json.loads(artifacts.processed_manifest.read_text(encoding="utf-8"))
    manifest["files"][filename] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }
    manifest.pop("content_sha256")
    semantic = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    manifest["content_sha256"] = hashlib.sha256(semantic).hexdigest()
    artifacts.processed_manifest.write_bytes(_json_bytes(manifest))


def _rewrite_quality(
    artifacts: AuthorizedBatchReleaseArtifacts,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = json.loads(artifacts.quality_report.read_text(encoding="utf-8"))
    mutate(payload)
    artifacts.quality_report.write_bytes(_json_bytes(payload))
    _rewrite_processed_file_reference(
        artifacts,
        filename="data-quality.json",
        path=artifacts.quality_report,
    )


def _rewrite_first_record(
    artifacts: AuthorizedBatchReleaseArtifacts,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    lines = artifacts.records.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    mutate(record)
    lines[0] = json.dumps(record, separators=(",", ":"), sort_keys=True)
    artifacts.records.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rewrite_processed_file_reference(
        artifacts,
        filename="records.jsonl",
        path=artifacts.records,
    )


def test_release_binds_and_reverifies_complete_signed_ingestion_chain(tmp_path: Path) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)

    release = _publish(tmp_path, artifacts, authority, trust_sha)
    repeated = _publish(tmp_path, artifacts, authority, trust_sha)
    independently_verified = verify_awin_production_batch_release(
        manifest_path=release.manifest_path,
        expected_manifest_sha256=release.manifest_sha256,
        expected_trust_root_sha256=trust_sha,
        current_source_registry=authority.source_registry,
        raw_snapshot=artifacts.raw_snapshot,
        records=artifacts.records,
        rejections=artifacts.rejections,
        now=datetime.now(UTC),
    )

    assert release.accepted_count == 1
    assert release.rejected_count == 0
    assert repeated.reused is True
    assert independently_verified.manifest_sha256 == release.manifest_sha256
    assert independently_verified.authority_expires_at == release.authority_expires_at
    manifest = json.loads(release.manifest_path.read_text(encoding="utf-8"))
    assert manifest["authority"]["authority_type"] == "ed25519_signed_policy"
    assert manifest["authority"]["authority_expires_at"] == (
        release.authority_expires_at.isoformat()
    )
    assert manifest["processed_batch"]["quality_status"] == "pass"
    assert manifest["external_files"]["raw_snapshot"]["sha256"] == (release.raw_snapshot_sha256)
    assert set(manifest["bundle_files"]) == {
        "authorization-receipt.json",
        "data-quality.json",
        "policy-trust-root.json",
        "processed-manifest.json",
        "raw-snapshot.metadata.json",
        "source-policy.json",
        "source-policy.sig.json",
        "source-registry.yaml",
    }


def test_serving_admission_reverifies_real_source_chain_and_exact_offer_bytes(
    tmp_path: Path,
) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    release = _publish(tmp_path, artifacts, authority, trust_sha)
    config = {
        "manifest": _serving_reference(tmp_path, release.manifest_path),
        "raw_snapshot": _serving_reference(tmp_path, artifacts.raw_snapshot),
        "rejections": _serving_reference(tmp_path, artifacts.rejections),
        "current_source_registry": _content_digest(authority.source_registry),
        "expected_trust_root_sha256": trust_sha,
    }

    admitted = serving_release._verified_authorized_source_release(
        config,
        root=tmp_path.resolve(),
        offers_path=artifacts.records.resolve(),
        current_source_registry_path=authority.source_registry.resolve(),
        expected_source_trust_root_sha256=trust_sha,
    )

    assert admitted.manifest_sha256 == release.manifest_sha256
    artifacts.records.write_bytes(artifacts.records.read_bytes() + b"tampered\n")
    with pytest.raises(
        ServingConfigurationError,
        match="authorized source release failed validation",
    ):
        serving_release._verified_authorized_source_release(
            config,
            root=tmp_path.resolve(),
            offers_path=artifacts.records.resolve(),
            current_source_registry_path=authority.source_registry.resolve(),
            expected_source_trust_root_sha256=trust_sha,
        )


def test_serving_admission_rejects_bundled_registry_as_current_authority(
    tmp_path: Path,
) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    release = _publish(tmp_path, artifacts, authority, trust_sha)
    bundled_registry = release.manifest_path.parent / "source-registry.yaml"
    config = {
        "manifest": _serving_reference(tmp_path, release.manifest_path),
        "raw_snapshot": _serving_reference(tmp_path, artifacts.raw_snapshot),
        "rejections": _serving_reference(tmp_path, artifacts.rejections),
        "current_source_registry": _content_digest(bundled_registry),
        "expected_trust_root_sha256": trust_sha,
    }

    with pytest.raises(ServingConfigurationError, match="must be independent"):
        serving_release._verified_authorized_source_release(
            config,
            root=tmp_path.resolve(),
            offers_path=artifacts.records.resolve(),
            current_source_registry_path=bundled_registry.resolve(),
            expected_source_trust_root_sha256=trust_sha,
        )


def test_independent_verifier_cli_reports_only_content_identities(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    release = _publish(tmp_path, artifacts, authority, trust_sha)
    arguments = [
        "--manifest",
        str(release.manifest_path),
        "--manifest-sha256",
        release.manifest_sha256,
        "--trust-root-sha256",
        trust_sha,
        "--source-registry",
        str(authority.source_registry),
        "--raw-snapshot",
        str(artifacts.raw_snapshot),
        "--records",
        str(artifacts.records),
        "--rejections",
        str(artifacts.rejections),
    ]

    assert verify_release_main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["manifest_sha256"] == release.manifest_sha256
    assert report["accepted_count"] == 1
    assert "manifest_path" not in report

    artifacts.raw_snapshot.write_bytes(artifacts.raw_snapshot.read_bytes() + b"tamper")
    assert verify_release_main(arguments) == 2
    error_text = capsys.readouterr().err
    assert json.loads(error_text)["error"] == (
        "authorized source release verification failed (AuthorizedBatchReleaseError)"
    )
    assert str(artifacts.raw_snapshot) not in error_text


def test_release_rejects_self_asserted_unsigned_policy_for_production(tmp_path: Path) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    unsigned = tmp_path / "self-asserted-retailer-policy.json"
    unsigned.write_bytes(_json_bytes(_policy_payload()))
    unsigned_authority = AuthorizedBatchAuthorityArtifacts(
        policy=unsigned,
        policy_signature=authority.policy_signature,
        trust_root=authority.trust_root,
        source_registry=authority.source_registry,
    )

    with pytest.raises(AuthorizedBatchReleaseError, match="valid, active signed policy"):
        _publish(tmp_path, artifacts, unsigned_authority, trust_sha)
    assert not (tmp_path / "releases").exists()


def test_release_fails_when_current_registry_or_raw_bytes_drift(tmp_path: Path) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    release = _publish(tmp_path, artifacts, authority, trust_sha)

    authority.source_registry.write_text(
        authority.source_registry.read_text(encoding="utf-8") + "# reviewed change\n",
        encoding="utf-8",
    )
    with pytest.raises(AuthorizedBatchReleaseError, match="not the current registry"):
        verify_awin_production_batch_release(
            manifest_path=release.manifest_path,
            expected_manifest_sha256=release.manifest_sha256,
            expected_trust_root_sha256=trust_sha,
            current_source_registry=authority.source_registry,
            raw_snapshot=artifacts.raw_snapshot,
            records=artifacts.records,
            rejections=artifacts.rejections,
            now=datetime.now(UTC),
        )

    bundled_registry = release.manifest_path.parent / "source-registry.yaml"
    authority.source_registry.write_bytes(bundled_registry.read_bytes())
    artifacts.raw_snapshot.write_bytes(artifacts.raw_snapshot.read_bytes() + b"tamper")
    with pytest.raises(AuthorizedBatchReleaseError, match="does not bind the raw bytes"):
        verify_awin_production_batch_release(
            manifest_path=release.manifest_path,
            expected_manifest_sha256=release.manifest_sha256,
            expected_trust_root_sha256=trust_sha,
            current_source_registry=authority.source_registry,
            raw_snapshot=artifacts.raw_snapshot,
            records=artifacts.records,
            rejections=artifacts.rejections,
            now=datetime.now(UTC),
        )


def test_release_rejects_quality_failure_and_record_authority_rewrite(tmp_path: Path) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    baseline, _adapter = _materialize(
        tmp_path,
        authority,
        trust_sha,
        count=10,
        feed_name="baseline.csv",
    )
    assert json.loads(baseline.quality_report.read_text(encoding="utf-8"))["status"] == "pass"
    reduced, _adapter = _materialize(
        tmp_path,
        authority,
        trust_sha,
        count=6,
        feed_name="reduced.csv",
    )
    assert json.loads(reduced.quality_report.read_text(encoding="utf-8"))["status"] == "fail"
    with pytest.raises(AuthorizedBatchReleaseError, match="not a passing batch report"):
        _publish(tmp_path, reduced, authority, trust_sha)

    record = json.loads(baseline.records.read_text(encoding="utf-8").splitlines()[0])
    record["rights_authority"]["policy_id"] = "self-asserted-rewrite"
    remaining = baseline.records.read_text(encoding="utf-8").splitlines()[1:]
    lines = [json.dumps(record, separators=(",", ":"), sort_keys=True), *remaining]
    baseline.records.write_text("\n".join(lines) + "\n", encoding="utf-8")
    processed_manifest = json.loads(baseline.processed_manifest.read_text(encoding="utf-8"))
    processed_manifest["files"]["records.jsonl"] = {
        "sha256": hashlib.sha256(baseline.records.read_bytes()).hexdigest(),
        "byte_count": baseline.records.stat().st_size,
    }
    processed_manifest.pop("content_sha256")
    semantic = json.dumps(
        processed_manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    processed_manifest["content_sha256"] = hashlib.sha256(semantic).hexdigest()
    baseline.processed_manifest.write_bytes(_json_bytes(processed_manifest))

    with pytest.raises(AuthorizedBatchReleaseError, match="not bound to signed authority"):
        _publish(tmp_path, baseline, authority, trust_sha)


def test_release_replays_raw_bytes_before_accepting_semantically_valid_record_rewrite(
    tmp_path: Path,
) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    _rewrite_first_record(
        artifacts,
        lambda record: record["data"]["listing"].__setitem__(
            "title", "Valid schema, but not present in the signed raw feed"
        ),
    )

    with pytest.raises(AuthorizedBatchReleaseError, match="do not reproduce from raw snapshot"):
        _publish(tmp_path, artifacts, authority, trust_sha)


def test_release_replays_raw_derived_quality_statistics(tmp_path: Path) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    _rewrite_quality(
        artifacts,
        lambda payload: payload["source_statistics"].__setitem__(
            "decompressed_bytes",
            payload["source_statistics"]["decompressed_bytes"] + 1,
        ),
    )

    with pytest.raises(AuthorizedBatchReleaseError, match="does not reproduce from raw snapshot"):
        _publish(tmp_path, artifacts, authority, trust_sha)


def test_release_reverification_enforces_signed_retention_deadline(tmp_path: Path) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path, retention_days=1)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    metadata = json.loads(artifacts.raw_metadata.read_text(encoding="utf-8"))
    retrieved_at = datetime.fromisoformat(metadata["retrieved_at"])
    release = _publish(
        tmp_path,
        artifacts,
        authority,
        trust_sha,
        now=retrieved_at + timedelta(hours=1),
    )
    retention_deadline = retrieved_at + timedelta(days=1)

    assert release.authority_expires_at == retention_deadline
    with pytest.raises(AuthorizedBatchReleaseError, match="signed retention period"):
        verify_awin_production_batch_release(
            manifest_path=release.manifest_path,
            expected_manifest_sha256=release.manifest_sha256,
            expected_trust_root_sha256=trust_sha,
            current_source_registry=authority.source_registry,
            raw_snapshot=artifacts.raw_snapshot,
            records=artifacts.records,
            rejections=artifacts.rejections,
            now=retention_deadline,
        )


def test_release_reverification_enforces_consent_and_policy_expiry(tmp_path: Path) -> None:
    authority, trust_sha, _verified = _signed_authority(
        tmp_path,
        consent_expires_on="2027-01-01",
    )
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    release = _publish(tmp_path, artifacts, authority, trust_sha)
    consent_deadline = datetime(2027, 1, 2, tzinfo=UTC)

    assert release.authority_expires_at == consent_deadline
    with pytest.raises(AuthorizedBatchReleaseError, match="valid, active signed policy"):
        verify_awin_production_batch_release(
            manifest_path=release.manifest_path,
            expected_manifest_sha256=release.manifest_sha256,
            expected_trust_root_sha256=trust_sha,
            current_source_registry=authority.source_registry,
            raw_snapshot=artifacts.raw_snapshot,
            records=artifacts.records,
            rejections=artifacts.rejections,
            now=consent_deadline,
        )
    with pytest.raises(AuthorizedBatchReleaseError, match="valid, active signed policy"):
        verify_awin_production_batch_release(
            manifest_path=release.manifest_path,
            expected_manifest_sha256=release.manifest_sha256,
            expected_trust_root_sha256=trust_sha,
            current_source_registry=authority.source_registry,
            raw_snapshot=artifacts.raw_snapshot,
            records=artifacts.records,
            rejections=artifacts.rejections,
            now=datetime(2098, 12, 31, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing_verified_on", "source registry root fields"),
        ("missing_required_control", "source requirements are incomplete"),
    ],
)
def test_release_rejects_incomplete_current_registry(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    registry = authority.source_registry.read_text(encoding="utf-8")
    if tamper == "missing_verified_on":
        registry = registry.replace(f"verified_on: {_REGISTRY_VERIFIED_ON}\n", "")
    else:
        registry = registry.replace(
            "      - exact_sgd_currency_and_explicit_shipping\n",
            "",
        )
    authority.source_registry.write_text(registry, encoding="utf-8")

    with pytest.raises(AuthorizedBatchReleaseError, match=message):
        _publish(tmp_path, artifacts, authority, trust_sha)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("stale", "older than 30 days"),
        ("revoked", "revoked or inactive"),
        ("policy_drift", "does not bind the exact feed and policy"),
        ("missing_feed", "no explicit source admission"),
    ],
)
def test_release_requires_fresh_explicit_active_feed_admission(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)
    registry = authority.source_registry.read_text(encoding="utf-8")
    if tamper == "stale":
        registry = registry.replace(
            f"verified_on: {_REGISTRY_VERIFIED_ON}",
            f"verified_on: {_STALE_REGISTRY_VERIFIED_ON}",
        )
    elif tamper == "revoked":
        registry = registry.replace("    status: active", "    status: revoked")
    elif tamper == "policy_drift":
        registry = "\n".join(
            "    policy_sha256: " + "0" * 64
            if line.startswith("    policy_sha256:")
            else line
            for line in registry.splitlines()
        ) + "\n"
    else:
        sources_start = registry.index("sources:\n")
        templates_start = registry.index("source_templates:\n")
        registry = registry[:sources_start] + "sources: {}\n" + registry[templates_start:]
    authority.source_registry.write_text(registry, encoding="utf-8")

    with pytest.raises(AuthorizedBatchReleaseError, match=message):
        _publish(tmp_path, artifacts, authority, trust_sha)


@pytest.mark.parametrize("tamper", ["duplicate_check", "boolean_count"])
def test_release_rejects_ambiguous_quality_checks(tmp_path: Path, tamper: str) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)

    def mutate(payload: dict[str, Any]) -> None:
        checks = payload["checks"]
        if tamper == "duplicate_check":
            checks.append(dict(checks[0]))
        else:
            next(check for check in checks if check["name"] == "signed_policy_verified")[
                "count"
            ] = False

    _rewrite_quality(artifacts, mutate)
    expected = "duplicated" if tamper == "duplicate_check" else "non-negative integer"
    with pytest.raises(AuthorizedBatchReleaseError, match=expected):
        _publish(tmp_path, artifacts, authority, trust_sha)


@pytest.mark.parametrize(
    "tamper",
    [
        "extra_envelope_field",
        "missing_normalisation_field",
        "extra_listing_field",
        "inconsistent_price_snapshot",
    ],
)
def test_release_rejects_noncanonical_accepted_record_schema(
    tmp_path: Path,
    tamper: str,
) -> None:
    authority, trust_sha, _verified = _signed_authority(tmp_path)
    artifacts, _adapter = _materialize(tmp_path, authority, trust_sha)

    def mutate(record: dict[str, Any]) -> None:
        if tamper == "extra_envelope_field":
            record["self_asserted_authority"] = True
        elif tamper == "missing_normalisation_field":
            record["normalisation_metadata"].pop("feed_id")
        elif tamper == "extra_listing_field":
            record["data"]["listing"]["unreviewed_price"] = "1.00"
        else:
            record["data"]["price_snapshot"]["listing_id"] = "listing_rewritten"

    _rewrite_first_record(artifacts, mutate)
    with pytest.raises(AuthorizedBatchReleaseError):
        _publish(tmp_path, artifacts, authority, trust_sha)
