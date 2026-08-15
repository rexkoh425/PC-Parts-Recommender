"""Opaque, authenticated recommendation-impression identities.

The browser receives a token, never the signing key or a caller-editable claim set.  Tokens are
encrypted and authenticated with Fernet, then decoded only at the interaction boundary.  This
keeps query/result/rank/version attribution under server control before an event can become a
trusted learning-to-rank signal.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken
from starlette.requests import Request
from starlette.responses import Response

from services.api.settings import ApiSettings

_TOKEN_PREFIX = "imp_v1."
ACTOR_COOKIE_NAME = "pcbr_actor"
_ACTOR_TOKEN_PREFIX = "actor_v1."
_TOKEN_VERSION = 1
_ACTOR_TOKEN_VERSION = 1
_KEY_DOMAIN = b"pcbr-impression-fernet-v1\x00"
_ACTOR_KEY_DOMAIN = b"pcbr-impression-actor-fernet-v1\x00"
_IDEMPOTENCY_DOMAIN = b"pcbr-interaction-idempotency-v1\x00"
_ACTOR_TTL_SECONDS = 365 * 24 * 60 * 60
_PAYLOAD_KEYS = frozenset(
    {
        "aid",
        "aud",
        "bid",
        "dv",
        "env",
        "exp",
        "iat",
        "iid",
        "kind",
        "mv",
        "pid",
        "qid",
        "rank",
        "rv",
        "v",
    }
)
_ACTOR_PAYLOAD_KEYS = frozenset({"aid", "aud", "env", "exp", "iat", "v"})

ImpressionKind = Literal["product_search_result", "build_result", "build_component_result"]
_TOKEN_AUDIENCE = "pcbr-interactions-v1"
_ACTOR_TOKEN_AUDIENCE = "pcbr-impression-actor-v1"


class InvalidImpressionToken(ValueError):
    """Raised when an impression token is malformed, expired, or unauthenticated."""


@dataclass(frozen=True, slots=True)
class ImpressionClaims:
    """Server-authoritative attribution recovered from one opaque token."""

    impression_id: str
    actor_id: str
    query_id: str
    kind: ImpressionKind
    rank_position: int
    model_version: str
    data_version: str
    rule_version: str
    environment: str
    issued_at: datetime
    expires_at: datetime
    product_id: str | None = None
    build_id: str | None = None


@dataclass(frozen=True, slots=True)
class ImpressionActor:
    """A server-authenticated anonymous actor used to bind displayed results."""

    actor_id: str
    token: str
    issued_at: datetime
    expires_at: datetime


class ImpressionSigner:
    """Issue and verify bounded-lived opaque recommendation impressions."""

    def __init__(self, secret: str | bytes, *, ttl_seconds: int, environment: str) -> None:
        secret_bytes = secret.encode("utf-8") if isinstance(secret, str) else secret
        if len(secret_bytes) < 32:
            raise ValueError("impression signing secret must contain at least 32 bytes")
        if ttl_seconds < 60:
            raise ValueError("impression token TTL must be at least 60 seconds")
        key_material = hashlib.sha256(_KEY_DOMAIN + secret_bytes).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(key_material))
        actor_key_material = hashlib.sha256(_ACTOR_KEY_DOMAIN + secret_bytes).digest()
        self._actor_fernet = Fernet(base64.urlsafe_b64encode(actor_key_material))
        self._idempotency_key = hashlib.sha256(_IDEMPOTENCY_DOMAIN + secret_bytes).digest()
        self._ttl_seconds = ttl_seconds
        self._environment = environment.casefold().strip()
        if not self._environment:
            raise ValueError("impression environment must not be empty")

    @classmethod
    def from_settings(cls, settings: ApiSettings) -> ImpressionSigner:
        configured = settings.impression_signing_key
        secret: str | bytes
        if configured is None:
            if not settings.is_development_environment:
                raise RuntimeError("an impression signing key is required outside development")
            # Development compatibility is process-local by design.  A restart invalidates tokens.
            secret = secrets.token_bytes(48)
        else:
            secret = configured.get_secret_value()
        return cls(
            secret,
            ttl_seconds=settings.impression_ttl_minutes * 60,
            environment=settings.environment,
        )

    def issue(
        self,
        *,
        actor_id: str,
        query_id: str,
        kind: ImpressionKind,
        rank_position: int,
        model_version: str,
        data_version: str,
        rule_version: str,
        product_id: str | None = None,
        build_id: str | None = None,
        now: datetime | None = None,
    ) -> str:
        issued_at = _as_utc(now or datetime.now(UTC))
        expires_at = issued_at + timedelta(seconds=self._ttl_seconds)
        _validate_subject(kind=kind, product_id=product_id, build_id=build_id)
        _validate_actor_id(actor_id)
        if rank_position < 1:
            raise ValueError("impression rank_position must be one-based")
        payload = {
            "aid": actor_id,
            "aud": _TOKEN_AUDIENCE,
            "bid": build_id,
            "dv": data_version,
            "env": self._environment,
            "exp": int(expires_at.timestamp()),
            "iat": int(issued_at.timestamp()),
            "iid": f"imp_{uuid4().hex}",
            "kind": kind,
            "mv": model_version,
            "pid": product_id,
            "qid": query_id,
            "rank": rank_position,
            "rv": rule_version,
            "v": _TOKEN_VERSION,
        }
        plaintext = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        token = self._fernet.encrypt_at_time(plaintext, int(issued_at.timestamp()))
        return _TOKEN_PREFIX + token.decode("ascii")

    def verify(
        self,
        token: str,
        *,
        actor_token: str | None = None,
        now: datetime | None = None,
    ) -> ImpressionClaims:
        if len(token) > 4096 or not token.startswith(_TOKEN_PREFIX):
            raise InvalidImpressionToken("invalid impression token")
        current = _as_utc(now or datetime.now(UTC))
        try:
            encoded = token.removeprefix(_TOKEN_PREFIX).encode("ascii", errors="strict")
            plaintext = self._fernet.decrypt_at_time(
                encoded,
                ttl=self._ttl_seconds,
                current_time=int(current.timestamp()),
            )
            payload = json.loads(plaintext)
        except (
            InvalidToken,
            UnicodeDecodeError,
            UnicodeEncodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise InvalidImpressionToken("invalid impression token") from error
        try:
            claims = _claims_from_payload(
                payload,
                now=current,
                ttl_seconds=self._ttl_seconds,
                environment=self._environment,
            )
            actor = self.verify_actor(actor_token or "", now=current)
            if not hmac.compare_digest(claims.actor_id, actor.actor_id):
                raise ValueError("impression actor does not match")
            return claims
        except (TypeError, ValueError, KeyError) as error:
            raise InvalidImpressionToken("invalid impression token") from error

    def issue_actor(self, *, now: datetime | None = None) -> ImpressionActor:
        """Create an opaque long-lived anonymous actor capability."""

        issued_at = _as_utc(now or datetime.now(UTC))
        expires_at = issued_at + timedelta(seconds=_ACTOR_TTL_SECONDS)
        actor_id = f"actor_{uuid4().hex}"
        payload = {
            "aid": actor_id,
            "aud": _ACTOR_TOKEN_AUDIENCE,
            "env": self._environment,
            "exp": int(expires_at.timestamp()),
            "iat": int(issued_at.timestamp()),
            "v": _ACTOR_TOKEN_VERSION,
        }
        plaintext = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encoded = self._actor_fernet.encrypt_at_time(plaintext, int(issued_at.timestamp()))
        return ImpressionActor(
            actor_id=actor_id,
            token=_ACTOR_TOKEN_PREFIX + encoded.decode("ascii"),
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def verify_actor(
        self, token: str, *, now: datetime | None = None
    ) -> ImpressionActor:
        """Authenticate an anonymous actor cookie without trusting caller identity text."""

        if len(token) > 4096 or not token.startswith(_ACTOR_TOKEN_PREFIX):
            raise InvalidImpressionToken("invalid impression actor")
        current = _as_utc(now or datetime.now(UTC))
        try:
            encoded = token.removeprefix(_ACTOR_TOKEN_PREFIX).encode(
                "ascii", errors="strict"
            )
            plaintext = self._actor_fernet.decrypt_at_time(
                encoded,
                ttl=_ACTOR_TTL_SECONDS,
                current_time=int(current.timestamp()),
            )
            raw_payload = json.loads(plaintext)
            if not isinstance(raw_payload, dict) or set(raw_payload) != _ACTOR_PAYLOAD_KEYS:
                raise ValueError("unsupported actor payload")
            payload: dict[str, object] = raw_payload
            if (
                payload["v"] != _ACTOR_TOKEN_VERSION
                or payload["aud"] != _ACTOR_TOKEN_AUDIENCE
                or payload["env"] != self._environment
            ):
                raise ValueError("actor audience or environment does not match")
            issued = payload["iat"]
            expires = payload["exp"]
            if (
                type(issued) is not int
                or type(expires) is not int
                or expires != issued + _ACTOR_TTL_SECONDS
            ):
                raise ValueError("invalid actor timing")
            issued_at = datetime.fromtimestamp(issued, tz=UTC)
            expires_at = datetime.fromtimestamp(expires, tz=UTC)
            if issued_at > current + timedelta(seconds=60) or expires_at <= current:
                raise ValueError("actor is outside its valid time window")
            actor_id = _required_string(payload, "aid")
            _validate_actor_id(actor_id)
        except (
            InvalidToken,
            UnicodeDecodeError,
            UnicodeEncodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise InvalidImpressionToken("invalid impression actor") from error
        return ImpressionActor(
            actor_id=actor_id,
            token=token,
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def resolve_actor(
        self, token: str | None, *, now: datetime | None = None
    ) -> tuple[ImpressionActor, bool]:
        """Return a verified actor, rotating malformed, missing, or expired cookies."""

        if token:
            try:
                return self.verify_actor(token, now=now), False
            except InvalidImpressionToken:
                pass
        return self.issue_actor(now=now), True

    def idempotency_key_sha256(self, *, session_id: str, key: str) -> str:
        """Return a server-keyed digest; the raw retry key is never persisted."""

        canonical = f"{session_id}\x00{key}".encode()
        return hmac.new(self._idempotency_key, canonical, hashlib.sha256).hexdigest()


def prepare_impression_response(
    request: Request,
    response: Response,
    *,
    signer: ImpressionSigner,
    secure_cookie: bool,
) -> str:
    """Bind token-bearing output to one authenticated actor cookie and disable caching."""

    actor, rotated = signer.resolve_actor(request.cookies.get(ACTOR_COOKIE_NAME))
    if rotated:
        response.set_cookie(
            key=ACTOR_COOKIE_NAME,
            value=actor.token,
            max_age=max(1, int((actor.expires_at - actor.issued_at).total_seconds())),
            expires=actor.expires_at,
            path="/",
            secure=secure_cookie,
            httponly=True,
            samesite="none" if secure_cookie else "lax",
        )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Cookie"
    return actor.actor_id


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("impression timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _validate_subject(
    *, kind: ImpressionKind, product_id: str | None, build_id: str | None
) -> None:
    valid = (
        (kind == "product_search_result" and product_id is not None and build_id is None)
        or (kind == "build_result" and build_id is not None and product_id is None)
        or (kind == "build_component_result" and build_id is not None and product_id is not None)
    )
    if not valid:
        raise ValueError("impression subject does not match its result kind")


def _validate_actor_id(actor_id: str) -> None:
    if (
        not actor_id.startswith("actor_")
        or len(actor_id) != 38
        or any(character not in "0123456789abcdef" for character in actor_id[6:])
    ):
        raise ValueError("invalid impression actor identity")


def _claims_from_payload(
    raw_payload: object, *, now: datetime, ttl_seconds: int, environment: str
) -> ImpressionClaims:
    if not isinstance(raw_payload, dict) or set(raw_payload) != _PAYLOAD_KEYS:
        raise ValueError("unsupported impression payload")
    payload: dict[str, object] = raw_payload
    if payload["v"] != _TOKEN_VERSION:
        raise ValueError("unsupported impression version")
    if payload["aud"] != _TOKEN_AUDIENCE or payload["env"] != environment:
        raise ValueError("impression audience or environment does not match")
    kind = payload["kind"]
    if kind not in {"product_search_result", "build_result", "build_component_result"}:
        raise ValueError("unsupported impression kind")
    impression_kind = cast(ImpressionKind, kind)
    rank = payload["rank"]
    issued = payload["iat"]
    expires = payload["exp"]
    if (
        type(rank) is not int
        or rank < 1
        or type(issued) is not int
        or type(expires) is not int
        or expires != issued + ttl_seconds
    ):
        raise ValueError("invalid impression timing or rank")
    issued_at = datetime.fromtimestamp(issued, tz=UTC)
    expires_at = datetime.fromtimestamp(expires, tz=UTC)
    if issued_at > now + timedelta(seconds=60) or expires_at <= now:
        raise ValueError("impression is outside its valid time window")
    product_id = _optional_string(payload, "pid")
    build_id = _optional_string(payload, "bid")
    actor_id = _required_string(payload, "aid")
    _validate_actor_id(actor_id)
    _validate_subject(kind=impression_kind, product_id=product_id, build_id=build_id)
    return ImpressionClaims(
        impression_id=_required_string(payload, "iid"),
        actor_id=actor_id,
        query_id=_required_string(payload, "qid"),
        kind=impression_kind,
        rank_position=rank,
        model_version=_required_string(payload, "mv"),
        data_version=_required_string(payload, "dv"),
        rule_version=_required_string(payload, "rv"),
        environment=_required_string(payload, "env"),
        issued_at=issued_at,
        expires_at=expires_at,
        product_id=product_id,
        build_id=build_id,
    )
