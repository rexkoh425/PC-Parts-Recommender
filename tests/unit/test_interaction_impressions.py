from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from services.api.impressions import ImpressionSigner, InvalidImpressionToken
from services.api.settings import ApiSettings

_SECRET = "impression-test-secret-0123456789-abcdef"
_NOW = datetime(2026, 8, 15, 10, 0, tzinfo=UTC)


def _signer(*, environment: str = "test") -> ImpressionSigner:
    return ImpressionSigner(_SECRET, ttl_seconds=120, environment=environment)


def test_impression_round_trip_is_opaque_and_server_authoritative() -> None:
    signer = _signer()
    actor = signer.issue_actor(now=_NOW)
    token = signer.issue(
        actor_id=actor.actor_id,
        query_id="search_sensitive-query-text-is-not-here",
        kind="product_search_result",
        rank_position=23,
        product_id="gpu-real-23",
        model_version="rrf-v3",
        data_version="catalog-v8",
        rule_version="compat-v7",
        now=_NOW,
    )

    assert token.startswith("imp_v1.")
    assert "gpu-real-23" not in token
    assert "sensitive-query" not in token
    claims = signer.verify(
        token, actor_token=actor.token, now=_NOW + timedelta(seconds=30)
    )
    assert claims.query_id == "search_sensitive-query-text-is-not-here"
    assert claims.product_id == "gpu-real-23"
    assert claims.build_id is None
    assert claims.rank_position == 23
    assert claims.model_version == "rrf-v3"
    assert claims.environment == "test"
    assert claims.expires_at == _NOW + timedelta(seconds=120)


def test_impression_rejects_tampering_expiry_wrong_key_and_wrong_environment() -> None:
    signer = _signer()
    actor = signer.issue_actor(now=_NOW)
    token = signer.issue(
        actor_id=actor.actor_id,
        query_id="req-1",
        kind="build_result",
        rank_position=1,
        build_id="build-1",
        model_version="ltr-v4",
        data_version="catalog-v8",
        rule_version="compat-v7",
        now=_NOW,
    )
    replacement = "A" if token[-1] != "A" else "B"

    with pytest.raises(InvalidImpressionToken):
        signer.verify(token[:-1] + replacement, actor_token=actor.token, now=_NOW)
    with pytest.raises(InvalidImpressionToken):
        signer.verify(token, actor_token=actor.token, now=_NOW + timedelta(seconds=120))
    with pytest.raises(InvalidImpressionToken):
        ImpressionSigner(
            "different-impression-secret-0123456789-abcdef",
            ttl_seconds=120,
            environment="test",
        ).verify(token, actor_token=actor.token, now=_NOW)
    with pytest.raises(InvalidImpressionToken):
        _signer(environment="production").verify(token, actor_token=actor.token, now=_NOW)


@pytest.mark.parametrize("token", ["", "not-an-impression", "imp_v1.\u2603", "x" * 4097])
def test_malformed_impressions_fail_with_one_public_error(token: str) -> None:
    with pytest.raises(InvalidImpressionToken, match="invalid impression token"):
        _signer().verify(token, now=_NOW)


def test_idempotency_digest_is_keyed_stable_and_session_scoped() -> None:
    signer = _signer()

    first = signer.idempotency_key_sha256(session_id="session-a", key="retry-key-123")
    repeated = signer.idempotency_key_sha256(session_id="session-a", key="retry-key-123")
    other_session = signer.idempotency_key_sha256(session_id="session-b", key="retry-key-123")

    assert first == repeated
    assert first != other_session
    assert len(first) == 64
    assert "retry-key-123" not in first


def test_signing_key_file_is_loaded_without_serializing_the_secret(tmp_path: Path) -> None:
    key_file = tmp_path / "impression-key.txt"
    raw_secret = "file-backed-impression-secret-0123456789abcdef"
    key_file.write_text(raw_secret, encoding="utf-8")

    settings = ApiSettings(environment="test", impression_signing_key_file=key_file)

    assert settings.impression_signing_key is not None
    assert settings.impression_signing_key.get_secret_value() == raw_secret
    assert raw_secret not in repr(settings)
    assert raw_secret not in settings.model_dump_json()


@pytest.mark.parametrize(
    "content",
    ["too-short", "first-valid-secret-0123456789abcdef\nsecond-secret-value"],
)
def test_invalid_signing_key_files_fail_closed(tmp_path: Path, content: str) -> None:
    key_file = tmp_path / "impression-key.txt"
    key_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationError, match="impression_signing_key_file"):
        ApiSettings(environment="test", impression_signing_key_file=key_file)


def test_signing_key_cannot_reuse_admin_secret() -> None:
    shared_secret = "shared-administrator-secret-0123456789abcdef"

    with pytest.raises(ValidationError, match="must not reuse"):
        ApiSettings(
            environment="test",
            impression_signing_key=shared_secret,
            admin_token=shared_secret,
        )


def test_nondevelopment_signer_fails_closed_without_a_configured_key() -> None:
    settings = ApiSettings(
        environment="production",
        docs_enabled=False,
        cors_origins=["https://pcbr.example.test"],
    )

    with pytest.raises(RuntimeError, match="required outside development"):
        ImpressionSigner.from_settings(settings)
