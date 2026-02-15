"""Verify content-addressed, Ed25519-signed source-policy envelopes.

This module deliberately does not know the schema of a source-specific
``payload``.  It authenticates a small, exact policy envelope and returns the
nested payload to the source adapter, which must then validate its own exact
schema before taking any action.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

SIGNED_POLICY_SCHEMA_VERSION: Final = "pc-build-recommender.signed-policy.v1"
DETACHED_SIGNATURE_SCHEMA_VERSION: Final = "pc-build-recommender.detached-policy-signature.v1"
TRUST_ROOT_SCHEMA_VERSION: Final = "pc-build-recommender.policy-trust-root.v1"

MAX_POLICY_BYTES: Final = 1024 * 1024
MAX_SIGNATURE_DOCUMENT_BYTES: Final = 16 * 1024
MAX_TRUST_ROOT_BYTES: Final = 1024 * 1024
MAX_TRUST_ROOT_KEYS: Final = 128

_POLICY_FIELDS: Final = frozenset(
    {"schema_version", "policy_id", "issued_at", "expires_at", "payload"}
)
_SIGNATURE_FIELDS: Final = frozenset({"schema_version", "key_id", "policy_sha256", "signature"})
_TRUST_ROOT_FIELDS: Final = frozenset({"schema_version", "keys"})
_TRUST_KEY_FIELDS: Final = frozenset(
    {"key_id", "algorithm", "public_key", "status", "valid_from", "valid_until"}
)
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_UTC_TIMESTAMP_PATTERN: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_ACTIVE_KEY_STATUS: Final = "active"
_KNOWN_KEY_STATUSES: Final = frozenset({_ACTIVE_KEY_STATUS, "retired", "revoked"})


class SignedPolicyError(ValueError):
    """Raised when a policy cannot be authenticated safely."""


@dataclass(frozen=True, slots=True)
class VerifiedSignedPolicy:
    """An authenticated policy envelope and its exact artifact identities.

    ``signature_sha256`` identifies the complete detached-signature document,
    not merely the decoded 64-byte Ed25519 value.
    """

    payload: Mapping[str, Any]
    policy_id: str
    issued_at: datetime
    expires_at: datetime
    policy_sha256: str
    signature_sha256: str
    trust_root_sha256: str
    key_id: str


@dataclass(frozen=True, slots=True)
class _TrustKey:
    key_id: str
    public_key: Ed25519PublicKey
    public_key_bytes: bytes
    status: str
    valid_from: datetime
    valid_until: datetime


def _is_linklike(path: Path, result: os.stat_result | None = None) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if is_junction is not None and is_junction(path):
            return True
    except OSError as error:
        raise SignedPolicyError(f"cannot inspect policy artifact path {path}: {error}") from error

    if result is None:
        try:
            result = os.lstat(path)
        except OSError as error:
            raise SignedPolicyError(
                f"cannot inspect policy artifact path {path}: {error}"
            ) from error
    if stat.S_ISLNK(result.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(result, "st_file_attributes", 0)
    return bool(reparse_flag and file_attributes & reparse_flag)


def _path_components(path: Path) -> tuple[Path, ...]:
    absolute = Path(os.path.abspath(path))
    components: list[Path] = []
    current = absolute
    while True:
        components.append(current)
        if current.parent == current:
            break
        current = current.parent
    components.reverse()
    return tuple(components)


def _assert_no_linklike_components(path: Path) -> os.stat_result:
    components = _path_components(path)
    target_result: os.stat_result | None = None
    for index, component in enumerate(components):
        try:
            result = os.lstat(component)
        except OSError as error:
            raise SignedPolicyError(f"policy artifact path is unavailable: {component}") from error
        if _is_linklike(component, result):
            raise SignedPolicyError(
                f"policy artifact paths must not contain symlinks or junctions: {component}"
            )
        if index < len(components) - 1 and not stat.S_ISDIR(result.st_mode):
            raise SignedPolicyError(f"policy artifact parent is not a directory: {component}")
        target_result = result
    assert target_result is not None
    return target_result


def _stat_identity(result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        result.st_dev,
        result.st_ino,
        stat.S_IFMT(result.st_mode),
        result.st_size,
        result.st_mtime_ns,
    )


def _read_stable_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    """Read one direct regular file while detecting replacement or mutation."""

    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    path = Path(os.path.abspath(path))
    before_path = _assert_no_linklike_components(path)
    if not stat.S_ISREG(before_path.st_mode):
        raise SignedPolicyError(f"{label} must be a regular file: {path}")
    if before_path.st_size > maximum_bytes:
        raise SignedPolicyError(f"{label} exceeds the {maximum_bytes}-byte limit")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SignedPolicyError(f"cannot open {label}: {path}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise SignedPolicyError(f"{label} must be a regular file: {path}")
        if _stat_identity(opened) != _stat_identity(before_path):
            raise SignedPolicyError(f"{label} changed while it was being opened")
        if opened.st_size > maximum_bytes:
            raise SignedPolicyError(f"{label} exceeds the {maximum_bytes}-byte limit")

        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - byte_count))
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise SignedPolicyError(f"{label} exceeds the {maximum_bytes}-byte limit")
        after_handle = os.fstat(descriptor)
    except OSError as error:
        raise SignedPolicyError(f"cannot read {label}: {path}") from error
    finally:
        os.close(descriptor)

    after_path = _assert_no_linklike_components(path)
    expected_identity = _stat_identity(before_path)
    if (
        _stat_identity(opened) != expected_identity
        or _stat_identity(after_handle) != expected_identity
        or _stat_identity(after_path) != expected_identity
    ):
        raise SignedPolicyError(f"{label} changed while it was being read")
    payload = b"".join(chunks)
    if len(payload) != before_path.st_size:
        raise SignedPolicyError(f"{label} size changed while it was being read")
    return payload


def _reject_nonfinite(value: str) -> None:
    raise SignedPolicyError(f"non-finite JSON number is forbidden: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SignedPolicyError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SignedPolicyError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_float,
        )
    except SignedPolicyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise SignedPolicyError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise SignedPolicyError(f"{label} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any], *, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SignedPolicyError(
            f"{label} fields do not match the schema; missing={missing}, extra={extra}"
        )
    return value


def _required_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise SignedPolicyError(f"{label} is not a valid identifier")
    return value


def _required_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise SignedPolicyError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _parse_utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise SignedPolicyError(f"{label} must be an RFC 3339 UTC timestamp in whole seconds")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise SignedPolicyError(f"{label} is not a valid UTC timestamp") from error


def _verification_time(now: datetime | None) -> datetime:
    result = datetime.now(UTC) if now is None else now
    if result.tzinfo is None or result.utcoffset() is None:
        raise SignedPolicyError("verification time must be timezone-aware")
    return result.astimezone(UTC)


def _canonical_base64(value: object, *, expected_bytes: int, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise SignedPolicyError(f"{label} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SignedPolicyError(f"{label} is not canonical base64") from error
    if len(decoded) != expected_bytes:
        raise SignedPolicyError(f"{label} must decode to exactly {expected_bytes} bytes")
    canonical = base64.b64encode(decoded).decode("ascii")
    if not hmac.compare_digest(value, canonical):
        raise SignedPolicyError(f"{label} is not canonical base64")
    return decoded


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _load_trust_keys(raw: bytes) -> dict[str, _TrustKey]:
    root = _exact_fields(
        _decode_json_object(raw, label="trust root"),
        expected=_TRUST_ROOT_FIELDS,
        label="trust root",
    )
    if root["schema_version"] != TRUST_ROOT_SCHEMA_VERSION:
        raise SignedPolicyError("unsupported trust-root schema version")
    raw_keys = root["keys"]
    if not isinstance(raw_keys, list) or not 1 <= len(raw_keys) <= MAX_TRUST_ROOT_KEYS:
        raise SignedPolicyError(
            f"trust root keys must contain from 1 to {MAX_TRUST_ROOT_KEYS} entries"
        )

    keys: dict[str, _TrustKey] = {}
    public_keys: set[bytes] = set()
    for index, raw_key in enumerate(raw_keys):
        label = f"trust root keys[{index}]"
        if not isinstance(raw_key, dict):
            raise SignedPolicyError(f"{label} must be an object")
        key = _exact_fields(raw_key, expected=_TRUST_KEY_FIELDS, label=label)
        key_id = _required_identifier(key["key_id"], label=f"{label}.key_id")
        if key_id in keys:
            raise SignedPolicyError(f"duplicate trust-root key ID is forbidden: {key_id}")
        if key["algorithm"] != "Ed25519":
            raise SignedPolicyError(f"{label}.algorithm must be Ed25519")
        status = key["status"]
        if not isinstance(status, str) or status not in _KNOWN_KEY_STATUSES:
            raise SignedPolicyError(f"{label}.status is unsupported")
        public_key_bytes = _canonical_base64(
            key["public_key"], expected_bytes=32, label=f"{label}.public_key"
        )
        if public_key_bytes in public_keys:
            raise SignedPolicyError("duplicate trust-root public keys are forbidden")
        valid_from = _parse_utc_timestamp(key["valid_from"], label=f"{label}.valid_from")
        valid_until = _parse_utc_timestamp(key["valid_until"], label=f"{label}.valid_until")
        if valid_from >= valid_until:
            raise SignedPolicyError(f"{label} validity interval is empty or reversed")
        try:
            parsed_public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        except ValueError as error:
            raise SignedPolicyError(f"{label}.public_key is not a valid Ed25519 key") from error
        keys[key_id] = _TrustKey(
            key_id=key_id,
            public_key=parsed_public_key,
            public_key_bytes=public_key_bytes,
            status=status,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        public_keys.add(public_key_bytes)
    return keys


def verify_signed_policy(
    *,
    policy_path: str | Path,
    signature_path: str | Path,
    trust_root_path: str | Path,
    expected_trust_root_sha256: str,
    now: datetime | None = None,
) -> VerifiedSignedPolicy:
    """Authenticate one exact policy using an externally pinned trust root.

    The SHA-256 pin is checked against the exact trust-root bytes before those
    bytes are parsed or used.  The detached Ed25519 signature is verified over
    the exact policy bytes, including their whitespace and final newline.
    """

    expected_trust_digest = _required_sha256(
        expected_trust_root_sha256,
        label="expected_trust_root_sha256",
    )
    verification_time = _verification_time(now)
    trust_bytes = _read_stable_regular_file(
        Path(trust_root_path), maximum_bytes=MAX_TRUST_ROOT_BYTES, label="trust root"
    )
    trust_digest = hashlib.sha256(trust_bytes).hexdigest()
    if not hmac.compare_digest(trust_digest, expected_trust_digest):
        raise SignedPolicyError("trust-root SHA-256 does not match the operator-provided pin")
    trust_keys = _load_trust_keys(trust_bytes)

    policy_bytes = _read_stable_regular_file(
        Path(policy_path), maximum_bytes=MAX_POLICY_BYTES, label="signed policy"
    )
    signature_bytes = _read_stable_regular_file(
        Path(signature_path),
        maximum_bytes=MAX_SIGNATURE_DOCUMENT_BYTES,
        label="detached signature",
    )
    policy_digest = hashlib.sha256(policy_bytes).hexdigest()
    signature_document_digest = hashlib.sha256(signature_bytes).hexdigest()

    signature_document = _exact_fields(
        _decode_json_object(signature_bytes, label="detached signature"),
        expected=_SIGNATURE_FIELDS,
        label="detached signature",
    )
    if signature_document["schema_version"] != DETACHED_SIGNATURE_SCHEMA_VERSION:
        raise SignedPolicyError("unsupported detached-signature schema version")
    key_id = _required_identifier(signature_document["key_id"], label="signature.key_id")
    bound_policy_digest = _required_sha256(
        signature_document["policy_sha256"], label="signature.policy_sha256"
    )
    if not hmac.compare_digest(policy_digest, bound_policy_digest):
        raise SignedPolicyError("detached signature is bound to different policy bytes")
    detached_signature = _canonical_base64(
        signature_document["signature"],
        expected_bytes=64,
        label="signature.signature",
    )

    trust_key = trust_keys.get(key_id)
    if trust_key is None:
        raise SignedPolicyError(f"signature key ID is absent from the pinned trust root: {key_id}")
    if trust_key.status != _ACTIVE_KEY_STATUS:
        raise SignedPolicyError(f"signature key is not active: status={trust_key.status}")
    if not trust_key.valid_from <= verification_time < trust_key.valid_until:
        raise SignedPolicyError("signature key is outside its validity interval")
    try:
        trust_key.public_key.verify(detached_signature, policy_bytes)
    except InvalidSignature as error:
        raise SignedPolicyError("Ed25519 policy signature verification failed") from error

    policy = _exact_fields(
        _decode_json_object(policy_bytes, label="signed policy"),
        expected=_POLICY_FIELDS,
        label="signed policy",
    )
    if policy["schema_version"] != SIGNED_POLICY_SCHEMA_VERSION:
        raise SignedPolicyError("unsupported signed-policy schema version")
    policy_id = _required_identifier(policy["policy_id"], label="policy.policy_id")
    issued_at = _parse_utc_timestamp(policy["issued_at"], label="policy.issued_at")
    expires_at = _parse_utc_timestamp(policy["expires_at"], label="policy.expires_at")
    if issued_at >= expires_at:
        raise SignedPolicyError("policy validity interval is empty or reversed")
    if not issued_at <= verification_time < expires_at:
        raise SignedPolicyError("signed policy is not currently valid")
    if not trust_key.valid_from <= issued_at < trust_key.valid_until:
        raise SignedPolicyError("policy issuance is outside the signing key validity interval")
    raw_payload = policy["payload"]
    if not isinstance(raw_payload, dict):
        raise SignedPolicyError("policy.payload must be a JSON object")
    payload = _deep_freeze(raw_payload)
    assert isinstance(payload, Mapping)

    return VerifiedSignedPolicy(
        payload=payload,
        policy_id=policy_id,
        issued_at=issued_at,
        expires_at=expires_at,
        policy_sha256=policy_digest,
        signature_sha256=signature_document_digest,
        trust_root_sha256=trust_digest,
        key_id=key_id,
    )


__all__ = [
    "DETACHED_SIGNATURE_SCHEMA_VERSION",
    "MAX_POLICY_BYTES",
    "MAX_SIGNATURE_DOCUMENT_BYTES",
    "MAX_TRUST_ROOT_BYTES",
    "SIGNED_POLICY_SCHEMA_VERSION",
    "TRUST_ROOT_SCHEMA_VERSION",
    "SignedPolicyError",
    "VerifiedSignedPolicy",
    "verify_signed_policy",
]
