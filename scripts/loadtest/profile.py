"""Strict, replayable load-profile parsing for the Locust harness."""

from __future__ import annotations

import hashlib
import ipaddress
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

PROFILE_SCHEMA_VERSION: Final = "pc-build-recommender.locust-load-profile.v1"
MAXIMUM_PROFILE_BYTES: Final = 256 * 1024
MAXIMUM_REQUEST_BODY_BYTES: Final = 1024 * 1024
REMOTE_CONFIRMATION_VALUE: Final = "I_UNDERSTAND_THIS_GENERATES_LOAD"
_PROFILE_ENDPOINT_PATHS: Final = {
    "search": "/v1/products/search",
    "build": "/v1/builds/generate",
}


class LoadProfileError(ValueError):
    """Raised when a load profile or target cannot be used safely."""


@dataclass(frozen=True, slots=True)
class EndpointProfile:
    """One bounded HTTP endpoint task consumed by the Locust user."""

    method: str
    path: str
    body: dict[str, Any]
    expected_statuses: tuple[int, ...]
    task_weight: int

    @property
    def request_name(self) -> str:
        """Return the bounded Locust path label without query data.

        Locust records the HTTP method separately, so keeping only the path here avoids a
        duplicated ``POST POST /...`` table heading while retaining method-plus-path metrics.
        """

        return self.path


@dataclass(frozen=True, slots=True)
class LoadProfile:
    """A declared, checksummed search/build request mix."""

    profile_name: str
    reportability: str
    search: EndpointProfile
    build: EndpointProfile
    sha256: str


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LoadProfileError(f"{name} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], *, name: str, expected: set[str]) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("extra=" + ", ".join(extra))
        raise LoadProfileError(f"{name} fields do not match the contract: {'; '.join(details)}")


def _require_text(value: object, *, name: str, maximum_length: int = 200) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LoadProfileError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum_length:
        raise LoadProfileError(f"{name} exceeds {maximum_length} characters")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LoadProfileError("load profile must contain only finite JSON values") from exc


def _parse_endpoint(value: object, *, name: str, expected_path: str) -> EndpointProfile:
    endpoint = _require_mapping(value, name=name)
    _require_exact_keys(
        endpoint,
        name=name,
        expected={"method", "path", "body", "expected_statuses", "task_weight"},
    )
    method = _require_text(endpoint["method"], name=f"{name}.method").upper()
    if method != "POST":
        raise LoadProfileError(f"{name}.method must be POST")
    path = _require_text(endpoint["path"], name=f"{name}.path", maximum_length=500)
    parsed_path = urlsplit(path)
    if (
        not path.startswith("/")
        or path.startswith("//")
        or parsed_path.scheme
        or parsed_path.netloc
        or parsed_path.query
        or parsed_path.fragment
        or ".." in parsed_path.path.split("/")
    ):
        raise LoadProfileError(f"{name}.path must be one absolute API path without a query")
    if parsed_path.path != expected_path:
        raise LoadProfileError(f"{name}.path must be {expected_path!r}")
    body = _require_mapping(endpoint["body"], name=f"{name}.body")
    canonical_body = _canonical_json_bytes(body)
    if len(canonical_body) > MAXIMUM_REQUEST_BODY_BYTES:
        raise LoadProfileError(
            f"{name}.body exceeds {MAXIMUM_REQUEST_BODY_BYTES} serialised bytes"
        )
    statuses = endpoint["expected_statuses"]
    if not isinstance(statuses, list) or not statuses or len(statuses) > 10:
        raise LoadProfileError(f"{name}.expected_statuses must be a non-empty array of at most 10")
    if any(type(status) is not int or not 100 <= status <= 599 for status in statuses):
        raise LoadProfileError(f"{name}.expected_statuses must contain valid HTTP status codes")
    if len(set(statuses)) != len(statuses):
        raise LoadProfileError(f"{name}.expected_statuses must not contain duplicates")
    task_weight = endpoint["task_weight"]
    if type(task_weight) is not int or not 1 <= task_weight <= 100:
        raise LoadProfileError(f"{name}.task_weight must be an integer from 1 to 100")
    return EndpointProfile(
        method=method,
        path=parsed_path.path,
        body=dict(body),
        expected_statuses=tuple(statuses),
        task_weight=task_weight,
    )


def load_profile(path: str | Path) -> LoadProfile:
    """Load one bounded profile without accepting arbitrary request targets or headers."""

    profile_path = Path(path)
    try:
        if profile_path.is_symlink() or not profile_path.is_file():
            raise LoadProfileError("load profile must be a regular JSON file")
        raw = profile_path.read_bytes()
    except OSError as exc:
        raise LoadProfileError(f"load profile is unreadable: {exc}") from exc
    if len(raw) > MAXIMUM_PROFILE_BYTES:
        raise LoadProfileError(f"load profile exceeds {MAXIMUM_PROFILE_BYTES} bytes")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LoadProfileError(f"load profile is invalid JSON: {exc}") from exc
    root = _require_mapping(payload, name="load profile")
    _require_exact_keys(
        root,
        name="load profile",
        expected={"schema_version", "profile_name", "reportability", "search", "build"},
    )
    if root["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise LoadProfileError("load profile schema_version is unsupported")
    profile_name = _require_text(root["profile_name"], name="profile_name", maximum_length=120)
    reportability = _require_text(root["reportability"], name="reportability", maximum_length=80)
    if reportability not in {"development_only", "production_measurement_candidate"}:
        raise LoadProfileError(
            "reportability must be development_only or production_measurement_candidate"
        )
    return LoadProfile(
        profile_name=profile_name,
        reportability=reportability,
        search=_parse_endpoint(
            root["search"],
            name="search",
            expected_path=_PROFILE_ENDPOINT_PATHS["search"],
        ),
        build=_parse_endpoint(
            root["build"],
            name="build",
            expected_path=_PROFILE_ENDPOINT_PATHS["build"],
        ),
        sha256=hashlib.sha256(_canonical_json_bytes(root)).hexdigest(),
    )


def _normalise_host(value: str) -> str:
    try:
        host = value.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise LoadProfileError("load target host is invalid") from exc
    if not host:
        raise LoadProfileError("load target requires a host")
    return host


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalise_target_url(value: str, *, confirmation: str | None = None) -> str:
    """Validate a target origin and require explicit consent before remote load generation."""

    if not isinstance(value, str) or not value.strip():
        raise LoadProfileError("PCBR_LOAD_BASE_URL must be a non-empty URL")
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise LoadProfileError("PCBR_LOAD_BASE_URL has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise LoadProfileError(
            "PCBR_LOAD_BASE_URL must be an HTTP(S) origin without credentials, path, or query"
        )
    host = _normalise_host(parsed.hostname)
    is_loopback = _is_loopback_host(host)
    if not is_loopback:
        if parsed.scheme != "https":
            raise LoadProfileError("remote load targets must use HTTPS")
        if confirmation != REMOTE_CONFIRMATION_VALUE:
            raise LoadProfileError(
                "remote load targets require PCBR_LOAD_CONFIRM=" + REMOTE_CONFIRMATION_VALUE
            )
    display_host = f"[{host}]" if ":" in host else host
    netloc = display_host if port is None else f"{display_host}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


__all__ = [
    "EndpointProfile",
    "LoadProfile",
    "LoadProfileError",
    "MAXIMUM_PROFILE_BYTES",
    "PROFILE_SCHEMA_VERSION",
    "REMOTE_CONFIRMATION_VALUE",
    "load_profile",
    "normalise_target_url",
]
