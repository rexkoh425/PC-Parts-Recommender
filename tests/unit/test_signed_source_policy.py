from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pipelines.sources.signed_policy import (
    DETACHED_SIGNATURE_SCHEMA_VERSION,
    MAX_POLICY_BYTES,
    MAX_SIGNATURE_DOCUMENT_BYTES,
    SIGNED_POLICY_SCHEMA_VERSION,
    TRUST_ROOT_SCHEMA_VERSION,
    SignedPolicyError,
    VerifiedSignedPolicy,
    verify_signed_policy,
)

NOW = datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


@dataclass
class _Bundle:
    policy_path: Path
    signature_path: Path
    trust_root_path: Path
    private_key: Ed25519PrivateKey
    key_id: str
    policy: dict[str, Any]
    trust_root: dict[str, Any]
    trust_root_sha256: str

    def write_policy(
        self,
        policy: dict[str, Any] | None = None,
        *,
        raw: bytes | None = None,
        signature_overrides: dict[str, Any] | None = None,
        signing_key: Ed25519PrivateKey | None = None,
    ) -> bytes:
        policy_bytes = raw if raw is not None else _json_bytes(policy or self.policy)
        self.policy_path.write_bytes(policy_bytes)
        signature = (signing_key or self.private_key).sign(policy_bytes)
        document: dict[str, Any] = {
            "schema_version": DETACHED_SIGNATURE_SCHEMA_VERSION,
            "key_id": self.key_id,
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "signature": base64.b64encode(signature).decode(),
        }
        if signature_overrides:
            document.update(signature_overrides)
        self.signature_path.write_bytes(_json_bytes(document))
        return policy_bytes

    def write_trust_root(self, root: dict[str, Any] | None = None) -> str:
        self.trust_root_path.write_bytes(_json_bytes(root or self.trust_root))
        self.trust_root_sha256 = hashlib.sha256(self.trust_root_path.read_bytes()).hexdigest()
        return self.trust_root_sha256

    def verify(self, **overrides: Any) -> VerifiedSignedPolicy:
        arguments: dict[str, Any] = {
            "policy_path": self.policy_path,
            "signature_path": self.signature_path,
            "trust_root_path": self.trust_root_path,
            "expected_trust_root_sha256": self.trust_root_sha256,
            "now": NOW,
        }
        arguments.update(overrides)
        return verify_signed_policy(**arguments)


@pytest.fixture
def bundle(tmp_path: Path) -> _Bundle:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_id = "governance-2026-01"
    policy: dict[str, Any] = {
        "schema_version": SIGNED_POLICY_SCHEMA_VERSION,
        "policy_id": "awin:advertiser-42:feed-7:v1",
        "issued_at": "2026-07-01T00:00:00Z",
        "expires_at": "2026-08-01T00:00:00Z",
        "payload": {
            "advertiser_id": "42",
            "grants": ["cache", "derive"],
            "limits": {"maximum_records": 100_000},
        },
    }
    trust_root: dict[str, Any] = {
        "schema_version": TRUST_ROOT_SCHEMA_VERSION,
        "keys": [
            {
                "key_id": key_id,
                "algorithm": "Ed25519",
                "public_key": base64.b64encode(public_key).decode(),
                "status": "active",
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2027-01-01T00:00:00Z",
            }
        ],
    }
    result = _Bundle(
        policy_path=tmp_path / "policy.json",
        signature_path=tmp_path / "policy.sig.json",
        trust_root_path=tmp_path / "trust-root.json",
        private_key=private_key,
        key_id=key_id,
        policy=policy,
        trust_root=trust_root,
        trust_root_sha256="",
    )
    result.write_policy()
    result.write_trust_root()
    return result


def test_verifies_exact_policy_bytes_and_returns_content_identities(bundle: _Bundle) -> None:
    policy_bytes = bundle.policy_path.read_bytes()
    signature_bytes = bundle.signature_path.read_bytes()
    trust_bytes = bundle.trust_root_path.read_bytes()

    verified = bundle.verify()

    assert verified.policy_id == "awin:advertiser-42:feed-7:v1"
    assert verified.key_id == bundle.key_id
    assert verified.issued_at == datetime(2026, 7, 1, tzinfo=UTC)
    assert verified.expires_at == datetime(2026, 8, 1, tzinfo=UTC)
    assert verified.policy_sha256 == hashlib.sha256(policy_bytes).hexdigest()
    assert verified.signature_sha256 == hashlib.sha256(signature_bytes).hexdigest()
    assert verified.trust_root_sha256 == hashlib.sha256(trust_bytes).hexdigest()
    assert verified.payload["advertiser_id"] == "42"
    assert verified.payload["grants"] == ("cache", "derive")
    with pytest.raises(TypeError):
        verified.payload["advertiser_id"] = "attacker"  # type: ignore[index]


def test_trust_root_must_match_the_external_exact_byte_pin(bundle: _Bundle) -> None:
    with pytest.raises(SignedPolicyError, match="operator-provided pin"):
        bundle.verify(expected_trust_root_sha256="0" * 64)
    with pytest.raises(SignedPolicyError, match="lowercase SHA-256"):
        bundle.verify(expected_trust_root_sha256="A" * 64)


def test_policy_hash_binding_detects_byte_tampering_before_signature_use(bundle: _Bundle) -> None:
    bundle.policy_path.write_bytes(bundle.policy_path.read_bytes() + b" ")

    with pytest.raises(SignedPolicyError, match="bound to different policy bytes"):
        bundle.verify()


def test_valid_hash_with_signature_from_another_key_is_rejected(bundle: _Bundle) -> None:
    bundle.write_policy(signing_key=Ed25519PrivateKey.generate())

    with pytest.raises(SignedPolicyError, match="signature verification failed"):
        bundle.verify()


@pytest.mark.parametrize(
    ("status", "message"),
    [("revoked", "not active"), ("retired", "not active"), ("unknown", "unsupported")],
)
def test_non_active_or_unknown_key_status_fails_closed(
    bundle: _Bundle, status: str, message: str
) -> None:
    bundle.trust_root["keys"][0]["status"] = status
    bundle.write_trust_root()

    with pytest.raises(SignedPolicyError, match=message):
        bundle.verify()


@pytest.mark.parametrize(
    ("valid_from", "valid_until"),
    [
        ("2026-07-24T00:00:00Z", "2027-01-01T00:00:00Z"),
        ("2026-01-01T00:00:00Z", "2026-07-23T12:00:00Z"),
    ],
)
def test_selected_key_must_be_currently_valid(
    bundle: _Bundle, valid_from: str, valid_until: str
) -> None:
    bundle.trust_root["keys"][0]["valid_from"] = valid_from
    bundle.trust_root["keys"][0]["valid_until"] = valid_until
    bundle.write_trust_root()

    with pytest.raises(SignedPolicyError, match="outside its validity interval"):
        bundle.verify()


@pytest.mark.parametrize(
    ("issued_at", "expires_at"),
    [
        ("2026-07-24T00:00:00Z", "2026-08-01T00:00:00Z"),
        ("2026-07-01T00:00:00Z", "2026-07-23T12:00:00Z"),
    ],
)
def test_policy_must_be_currently_valid(bundle: _Bundle, issued_at: str, expires_at: str) -> None:
    bundle.policy["issued_at"] = issued_at
    bundle.policy["expires_at"] = expires_at
    bundle.write_policy()

    with pytest.raises(SignedPolicyError, match="policy is not currently valid"):
        bundle.verify()


def test_policy_must_have_been_issued_during_key_validity(bundle: _Bundle) -> None:
    bundle.trust_root["keys"][0]["valid_from"] = "2026-07-15T00:00:00Z"
    bundle.write_trust_root()

    with pytest.raises(SignedPolicyError, match="issuance is outside"):
        bundle.verify()


@pytest.mark.parametrize("document", ["policy", "signature", "trust_root"])
def test_unknown_control_document_fields_are_rejected(bundle: _Bundle, document: str) -> None:
    if document == "policy":
        bundle.policy["unexpected"] = True
        bundle.write_policy()
    elif document == "signature":
        bundle.write_policy(signature_overrides={"unexpected": True})
    else:
        bundle.trust_root["unexpected"] = True
        bundle.write_trust_root()

    with pytest.raises(SignedPolicyError, match=r"extra=\['unexpected'\]"):
        bundle.verify()


def test_unknown_trust_key_fields_are_rejected(bundle: _Bundle) -> None:
    bundle.trust_root["keys"][0]["purpose"] = "policy-signing"
    bundle.write_trust_root()

    with pytest.raises(SignedPolicyError, match=r"extra=\['purpose'\]"):
        bundle.verify()


@pytest.mark.parametrize(
    "raw_policy",
    [
        b'{"schema_version":"pc-build-recommender.signed-policy.v1",'
        b'"policy_id":"first","policy_id":"second",'
        b'"issued_at":"2026-07-01T00:00:00Z",'
        b'"expires_at":"2026-08-01T00:00:00Z","payload":{}}',
        b'{"schema_version":"pc-build-recommender.signed-policy.v1",'
        b'"policy_id":"policy-1","issued_at":"2026-07-01T00:00:00Z",'
        b'"expires_at":"2026-08-01T00:00:00Z","payload":{"limit":NaN}}',
        b'{"schema_version":"pc-build-recommender.signed-policy.v1",'
        b'"policy_id":"policy-1","issued_at":"2026-07-01T00:00:00Z",'
        b'"expires_at":"2026-08-01T00:00:00Z","payload":{"limit":1e999}}',
    ],
)
def test_signed_duplicate_keys_and_nonfinite_numbers_are_rejected(
    bundle: _Bundle, raw_policy: bytes
) -> None:
    bundle.write_policy(raw=raw_policy)

    with pytest.raises(SignedPolicyError, match="duplicate|non-finite"):
        bundle.verify()


def test_detached_signature_must_be_canonical_base64_of_exactly_64_bytes(
    bundle: _Bundle,
) -> None:
    bundle.write_policy(signature_overrides={"signature": base64.b64encode(b"x" * 63).decode()})
    with pytest.raises(SignedPolicyError, match="exactly 64 bytes"):
        bundle.verify()

    bundle.write_policy(signature_overrides={"signature": "!" * 88})
    with pytest.raises(SignedPolicyError, match="canonical base64"):
        bundle.verify()


def test_signature_key_must_exist_in_pinned_trust_root(bundle: _Bundle) -> None:
    bundle.write_policy(signature_overrides={"key_id": "absent-key"})

    with pytest.raises(SignedPolicyError, match="absent from the pinned trust root"):
        bundle.verify()


def test_duplicate_key_ids_and_public_key_aliases_are_rejected(bundle: _Bundle) -> None:
    duplicate = dict(bundle.trust_root["keys"][0])
    bundle.trust_root["keys"].append(duplicate)
    bundle.write_trust_root()
    with pytest.raises(SignedPolicyError, match="duplicate trust-root key ID"):
        bundle.verify()

    bundle.trust_root["keys"][1]["key_id"] = "alias-key"
    bundle.write_trust_root()
    with pytest.raises(SignedPolicyError, match="duplicate trust-root public keys"):
        bundle.verify()


def test_naive_verification_time_is_rejected(bundle: _Bundle) -> None:
    with pytest.raises(SignedPolicyError, match="timezone-aware"):
        bundle.verify(now=datetime(2026, 7, 23, 12, 0, 0))


@pytest.mark.parametrize("artifact", ["policy_path", "signature_path", "trust_root_path"])
def test_symlink_artifacts_are_rejected(bundle: _Bundle, artifact: str, tmp_path: Path) -> None:
    original = getattr(bundle, artifact)
    linked = tmp_path / f"linked-{original.name}"
    try:
        linked.symlink_to(original)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable on this host: {error}")

    with pytest.raises(SignedPolicyError, match="symlinks or junctions"):
        bundle.verify(**{artifact: linked})


def test_junction_artifacts_are_rejected_before_open(
    bundle: _Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = bundle.policy_path.absolute()
    original = getattr(os.path, "isjunction", None)

    def simulated_isjunction(path: str | os.PathLike[str]) -> bool:
        if Path(path).absolute() == target:
            return True
        return bool(original is not None and original(path))

    monkeypatch.setattr(os.path, "isjunction", simulated_isjunction, raising=False)
    with pytest.raises(SignedPolicyError, match="symlinks or junctions"):
        bundle.verify()


def test_non_regular_and_oversized_artifacts_are_rejected(bundle: _Bundle, tmp_path: Path) -> None:
    with pytest.raises(SignedPolicyError, match="regular file"):
        bundle.verify(policy_path=tmp_path)

    oversized = tmp_path / "oversized-policy.json"
    oversized.write_bytes(b" " * (MAX_POLICY_BYTES + 1))
    with pytest.raises(SignedPolicyError, match="exceeds"):
        bundle.verify(policy_path=oversized)

    oversized_signature = tmp_path / "oversized-signature.json"
    oversized_signature.write_bytes(b" " * (MAX_SIGNATURE_DOCUMENT_BYTES + 1))
    with pytest.raises(SignedPolicyError, match="exceeds"):
        bundle.verify(signature_path=oversized_signature)
