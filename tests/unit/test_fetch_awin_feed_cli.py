from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pipelines.sources.signed_policy import (
    DETACHED_SIGNATURE_SCHEMA_VERSION,
    SIGNED_POLICY_SCHEMA_VERSION,
    TRUST_ROOT_SCHEMA_VERSION,
)
from scripts.fetch_open_data import main as fetch_open_data_main


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_signed_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_root = tmp_path / "LOCAL-TRUST-ROOT-SHOULD-NOT-LEAK.json"
    trust_root.write_bytes(
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
                        "valid_until": "2099-01-01T00:00:00Z",
                    }
                ],
            }
        )
    )
    payload: dict[str, Any] = {
        "schema_version": "pc-build-recommender.awin-feed-policy.v1",
        "advertiser_id": "12345",
        "feed_id": "67890",
        "retailer": "Fixture Hardware",
        "licence_or_access_note": "Synthetic test agreement; not a real data grant.",
        "rights": {
            "contract_reference": "fixture-awin-agreement-v1",
            "contract_version_url": "contract://fixture/awin-agreement-v1",
            "consent_effective_on": "2025-01-01",
            "consent_expires_on": "2098-12-31",
            "retention_days": 30,
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
    policy = tmp_path / "LOCAL-POLICY-SHOULD-NOT-LEAK.json"
    policy_bytes = _json_bytes(
        {
            "schema_version": SIGNED_POLICY_SCHEMA_VERSION,
            "policy_id": "fixture-awin-cli-policy",
            "issued_at": "2025-01-01T00:00:00Z",
            "expires_at": "2098-12-31T00:00:00Z",
            "payload": payload,
        }
    )
    policy.write_bytes(policy_bytes)
    signature = tmp_path / "LOCAL-SIGNATURE-SHOULD-NOT-LEAK.json"
    signature.write_bytes(
        _json_bytes(
            {
                "schema_version": DETACHED_SIGNATURE_SCHEMA_VERSION,
                "key_id": "fixture-governance-key",
                "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
                "signature": base64.b64encode(private_key.sign(policy_bytes)).decode("ascii"),
            }
        )
    )
    trust_root_sha256 = hashlib.sha256(trust_root.read_bytes()).hexdigest()
    return policy, signature, trust_root, trust_root_sha256


def _write_feed(path: Path) -> None:
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
        writer.writerow(
            {
                "merchant_id": "12345",
                "merchant_name": "Fixture Hardware",
                "merchant_product_id": "fixture-gpu-1",
                "product_name": "Fixture GPU",
                "search_price": "599.90",
                "currency": "SGD",
                "delivery_cost": "0",
                "aw_deep_link": "https://shop.example.test/products/fixture-gpu-1",
                "category_id": "gpu",
                "stock_status": "in_stock",
                "in_stock": "1",
                "is_for_sale": "1",
                "condition": "new",
            }
        )


def _arguments(
    *,
    feed: Path,
    policy: Path,
    signature: Path,
    trust_root: Path,
    trust_root_sha256: str,
    raw_root: Path,
    processed_root: Path,
) -> list[str]:
    return [
        "--source",
        "awin_feed",
        "--awin-feed",
        str(feed),
        "--awin-policy-json",
        str(policy),
        "--awin-policy-signature",
        str(signature),
        "--awin-trust-root",
        str(trust_root),
        "--awin-trust-root-sha256",
        trust_root_sha256,
        "--raw-root",
        str(raw_root),
        "--processed-root",
        str(processed_root),
    ]


def test_awin_cli_verifies_and_materialises_without_disclosing_input_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy, signature, trust_root, trust_sha = _write_signed_fixture(tmp_path)
    feed = tmp_path / "LOCAL-FEED-SHOULD-NOT-LEAK.csv"
    _write_feed(feed)

    exit_code = fetch_open_data_main(
        _arguments(
            feed=feed,
            policy=policy,
            signature=signature,
            trust_root=trust_root,
            trust_root_sha256=trust_sha,
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
        )
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    summary = report["sources"][0]
    assert report["status"] == "ok"
    assert summary["source_name"] == "awin_12345_67890"
    assert summary["source_uri"] == "awin://advertisers/12345/feeds/67890"
    assert summary["accepted_count"] == 1
    assert summary["recorded_rejection_count"] == 0
    assert summary["data_quality_status"] == "pass"
    assert len(summary["raw_snapshot"]["sha256"]) == 64
    assert len(summary["policy_authority"]["policy_sha256"]) == 64
    assert len(summary["manifest"]["sha256"]) == 64
    assert Path(summary["manifest"]["artifact_path"]).is_file()
    production_release = summary["production_release"]
    assert len(production_release["manifest_sha256"]) == 64
    assert len(production_release["content_sha256"]) == 64
    assert len(production_release["source_registry_sha256"]) == 64
    assert production_release["reused"] is False
    assert Path(production_release["artifact_path"]).is_file()
    for input_path in (feed, policy, signature, trust_root):
        assert str(input_path) not in output
        assert input_path.name not in output


def test_awin_cli_tamper_failure_does_not_disclose_policy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy, signature, trust_root, trust_sha = _write_signed_fixture(tmp_path)
    feed = tmp_path / "LOCAL-FEED-SHOULD-NOT-LEAK.csv"
    _write_feed(feed)
    policy.write_bytes(policy.read_bytes() + b" ")

    exit_code = fetch_open_data_main(
        _arguments(
            feed=feed,
            policy=policy,
            signature=signature,
            trust_root=trust_root,
            trust_root_sha256=trust_sha,
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
        )
    )

    assert exit_code == 2
    error_text = capsys.readouterr().err
    error = json.loads(error_text)
    assert error == {
        "status": "error",
        "error": "Awin signed policy verification failed (SignedPolicyError)",
    }
    assert str(policy) not in error_text
    assert policy.name not in error_text
    assert not (tmp_path / "raw").exists()


def test_awin_cli_feed_failure_does_not_disclose_feed_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy, signature, trust_root, trust_sha = _write_signed_fixture(tmp_path)
    missing_feed = tmp_path / "LOCAL-FEED-SHOULD-NOT-LEAK.csv"

    exit_code = fetch_open_data_main(
        _arguments(
            feed=missing_feed,
            policy=policy,
            signature=signature,
            trust_root=trust_root,
            trust_root_sha256=trust_sha,
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
        )
    )

    assert exit_code == 2
    error_text = capsys.readouterr().err
    error = json.loads(error_text)
    assert error == {
        "status": "error",
        "error": "Awin feed ingestion failed (AwinFeedError)",
    }
    assert str(missing_feed) not in error_text
    assert missing_feed.name not in error_text


def test_awin_cli_requires_every_local_authority_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = fetch_open_data_main(["--source", "awin_feed"])

    assert exit_code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert error["error"].startswith("Awin feed import is missing arguments:")
    assert "--awin-feed" in error["error"]
    assert "--awin-trust-root-sha256" in error["error"]
