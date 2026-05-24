"""Strict registry loader for scheduled governed-web retention."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

SOURCE_REGISTRY_SCHEMA_VERSION: Final = "pc-build-recommender.source-registry.v1"
GOVERNED_WEB_KIND: Final = "exact_url_schema_org_product_offer_crawl"
GOVERNED_WEB_RETENTION_ENGINE: Final = "governed_web_receipts_v2"
GOVERNED_WEB_RETENTION_CRON: Final = "0 * * * *"
MAXIMUM_REGISTRY_BYTES: Final = 1024 * 1024
MAXIMUM_GOVERNED_WEB_SOURCES: Final = 100
SCHEDULED_RETENTION_INTERVAL_MINUTES: Final = 60
_SOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_REQUIRED_ROOT_FIELDS: Final = frozenset({"schema_version", "sources"})
_ALLOWED_ROOT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "verified_on",
        "sources",
        "source_templates",
        "auxiliary_sources",
        "blocked_or_restricted_sources",
    }
)


@dataclass(frozen=True, slots=True)
class RestrictedWebSource:
    """A reviewed host restriction that the governed-web CLI must enforce.

    This is intentionally separate from retention selection: a source can be blocked before it
    has ever produced a governed-web receipt.  An operator must update this reviewed registry
    entry after obtaining a new written licence; a self-asserted crawl policy is not sufficient.
    """

    source_name: str
    hosts: tuple[str, ...]
    reason: str
    terms_url: str
    reviewed_on: date


@dataclass(frozen=True, slots=True)
class GovernedWebSourceAdmission:
    """Registry-bound host and scope contract for one crawlable web source."""

    source_name: str
    allowed_hosts: tuple[str, ...]
    usage_scope: str


class RetentionRegistryError(RuntimeError):
    """Raised when scheduled retention cannot derive a safe, complete source set."""


def _is_linklike(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction(path))


def _read_bounded_regular_file(path: Path) -> str:
    """Read one stable regular file without allocating beyond the registry limit."""

    if _is_linklike(path):
        raise RetentionRegistryError("source registry must be one regular, non-link file")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RetentionRegistryError("source registry must be one regular, non-link file")
            raw = handle.read(MAXIMUM_REGISTRY_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise RetentionRegistryError(f"source registry is unreadable: {exc}") from exc
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RetentionRegistryError("source registry changed while it was being read")
    if len(raw) > MAXIMUM_REGISTRY_BYTES:
        raise RetentionRegistryError(f"source registry exceeds {MAXIMUM_REGISTRY_BYTES} bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RetentionRegistryError(f"source registry is unreadable: {exc}") from exc


def _load_strict_yaml(raw: str) -> Any:
    """Load YAML lazily so importing pipeline helpers needs no pipeline extras."""

    try:
        import yaml  # type: ignore[import-untyped]
        from yaml.constructor import ConstructorError  # type: ignore[import-untyped]
        from yaml.events import AliasEvent  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        if exc.name not in {"yaml", "yaml.constructor", "yaml.events"}:
            raise
        raise RetentionRegistryError(
            "PyYAML is required to load the source registry; install the pipeline extra"
        ) from exc

    class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
        def compose_node(self, parent: Any, index: Any) -> Any:
            if self.check_event(AliasEvent):
                event = self.peek_event()
                raise ConstructorError(
                    "while composing the source registry",
                    None,
                    "YAML aliases are not permitted",
                    event.start_mark,
                )
            return super().compose_node(parent, index)

    def construct_unique_mapping(
        loader: _UniqueKeyLoader,
        node: Any,
        deep: bool = False,
    ) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "mapping keys must be scalar values",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping,
    )
    try:
        return yaml.load(raw, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise RetentionRegistryError(f"source registry is unreadable: {exc}") from exc


def _load_registry_payload(path: Path) -> dict[str, Any]:
    """Load the bounded registry once and reject a malformed root before use."""

    raw = _read_bounded_regular_file(Path(path))
    payload = _load_strict_yaml(raw)
    if not isinstance(payload, dict):
        raise RetentionRegistryError("source registry root must be an object")
    fields = set(payload)
    if not _REQUIRED_ROOT_FIELDS.issubset(fields) or not fields.issubset(_ALLOWED_ROOT_FIELDS):
        raise RetentionRegistryError("source registry root fields are incomplete or unknown")
    if payload.get("schema_version") != SOURCE_REGISTRY_SCHEMA_VERSION:
        raise RetentionRegistryError("unsupported source registry schema")
    if not isinstance(payload.get("sources"), dict):
        raise RetentionRegistryError("source registry sources must be an object")
    return payload


def _require_governed_web_lifecycle(raw_name: str, raw_source: dict[str, Any]) -> None:
    """Verify the retention contract shared by admission and scheduled maintenance."""

    if raw_source.get("template") != "governed_web_product":
        raise RetentionRegistryError(
            f"governed-web source {raw_name!r} has an invalid template"
        )
    lifecycle = raw_source.get("retention_maintenance")
    if not isinstance(lifecycle, dict) or set(lifecycle) != {
        "engine",
        "required",
        "maximum_interval_minutes",
    }:
        raise RetentionRegistryError(
            f"governed-web source {raw_name!r} has incomplete retention maintenance"
        )
    if lifecycle.get("engine") != GOVERNED_WEB_RETENTION_ENGINE:
        raise RetentionRegistryError(
            f"governed-web source {raw_name!r} has an unsupported retention engine"
        )
    if lifecycle.get("required") is not True:
        raise RetentionRegistryError(
            f"governed-web source {raw_name!r} must require scheduled retention"
        )
    interval = lifecycle.get("maximum_interval_minutes")
    if type(interval) is not int or interval != SCHEDULED_RETENTION_INTERVAL_MINUTES:
        raise RetentionRegistryError(
            f"governed-web source {raw_name!r} has an invalid maintenance interval; "
            f"it must match the {SCHEDULED_RETENTION_INTERVAL_MINUTES}-minute schedule"
        )


def _normalise_restricted_host(value: object) -> str:
    """Return one hostname suitable for exact-or-subdomain restriction checks."""

    if not isinstance(value, str) or not value.strip():
        raise RetentionRegistryError("restricted web-source host must be a non-empty string")
    raw = value.strip()
    parsed = urlsplit(f"//{raw}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise RetentionRegistryError("restricted web-source host must be a bare hostname") from exc
    if (
        parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RetentionRegistryError("restricted web-source host must be a bare hostname")
    try:
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise RetentionRegistryError("restricted web-source host is invalid") from exc
    if not host:
        raise RetentionRegistryError("restricted web-source host must not be empty")
    return host


def _restricted_terms_url(value: object, *, hosts: tuple[str, ...]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetentionRegistryError("restricted web-source terms_url must be a non-empty URL")
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or parsed.hostname is None or parsed.username or parsed.password:
        raise RetentionRegistryError("restricted web-source terms_url must be an HTTPS URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise RetentionRegistryError(
            "restricted web-source terms_url must use the default HTTPS port"
        ) from exc
    if port not in (None, 443) or parsed.fragment:
        raise RetentionRegistryError(
            "restricted web-source terms_url must use the default HTTPS port"
        )
    host = _normalise_restricted_host(parsed.hostname)
    if not any(host == allowed or host.endswith(f".{allowed}") for allowed in hosts):
        raise RetentionRegistryError(
            "restricted web-source terms_url host must be covered by its restricted hosts"
        )
    return value.strip()


def load_restricted_web_sources(path: Path) -> tuple[RestrictedWebSource, ...]:
    """Load reviewed web-source restrictions used to fail closed before a crawl.

    Historic restricted entries that record only ``reason`` remain documentation-only.  Any entry
    that names hosts must include a concrete terms URL and review date so the CLI can enforce it
    while still leaving a durable trail for a future licence re-review.
    """

    payload = _load_registry_payload(path)
    raw_sources = payload.get("blocked_or_restricted_sources", {})
    if not isinstance(raw_sources, dict):
        raise RetentionRegistryError("blocked_or_restricted_sources must be an object")
    restrictions: list[RestrictedWebSource] = []
    for raw_name, raw_source in raw_sources.items():
        if not isinstance(raw_name, str) or _SOURCE_NAME_PATTERN.fullmatch(raw_name) is None:
            raise RetentionRegistryError(f"unsafe restricted web-source name: {raw_name!r}")
        if not isinstance(raw_source, dict):
            raise RetentionRegistryError(
                f"restricted web-source {raw_name!r} must be an object"
            )
        reason = raw_source.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise RetentionRegistryError(
                f"restricted web-source {raw_name!r} requires a non-empty reason"
            )
        if "hosts" not in raw_source:
            if set(raw_source) != {"reason"}:
                raise RetentionRegistryError(
                    f"documentation-only restricted source {raw_name!r} has unknown fields"
                )
            continue
        if set(raw_source) != {"reason", "hosts", "terms_url", "reviewed_on"}:
            raise RetentionRegistryError(
                f"restricted web-source {raw_name!r} fields are incomplete or unknown"
            )
        raw_hosts = raw_source["hosts"]
        if not isinstance(raw_hosts, list) or not raw_hosts:
            raise RetentionRegistryError(
                f"restricted web-source {raw_name!r} hosts must be a non-empty array"
            )
        hosts = tuple(sorted({_normalise_restricted_host(item) for item in raw_hosts}))
        try:
            reviewed_on = date.fromisoformat(str(raw_source["reviewed_on"]))
        except (TypeError, ValueError) as exc:
            raise RetentionRegistryError(
                f"restricted web-source {raw_name!r} reviewed_on must be ISO-8601"
            ) from exc
        if reviewed_on > date.today():
            raise RetentionRegistryError(
                f"restricted web-source {raw_name!r} reviewed_on cannot be in the future"
            )
        restrictions.append(
            RestrictedWebSource(
                source_name=raw_name,
                hosts=hosts,
                reason=reason.strip(),
                terms_url=_restricted_terms_url(raw_source["terms_url"], hosts=hosts),
                reviewed_on=reviewed_on,
            )
        )
    return tuple(sorted(restrictions, key=lambda item: item.source_name))


def load_governed_web_source_admissions(path: Path) -> tuple[GovernedWebSourceAdmission, ...]:
    """Return the registry-bound source identities permitted to start a governed crawl.

    The source-policy JSON controls exact URLs, terms, rights, and resource limits.  This
    registry adds the independent operator-controlled binding: the policy cannot invent a new
    source name, widen its host set, change its scope, or evade scheduled retention.
    """

    payload = _load_registry_payload(Path(path))
    sources = payload["sources"]
    assert isinstance(sources, dict)
    admissions: list[GovernedWebSourceAdmission] = []
    for raw_name, raw_source in sources.items():
        if not isinstance(raw_name, str) or not isinstance(raw_source, dict):
            raise RetentionRegistryError("source registry contains a malformed source entry")
        if raw_source.get("kind") != GOVERNED_WEB_KIND:
            continue
        if _SOURCE_NAME_PATTERN.fullmatch(raw_name) is None:
            raise RetentionRegistryError(f"unsafe governed-web source name: {raw_name!r}")
        _require_governed_web_lifecycle(raw_name, raw_source)
        raw_hosts = raw_source.get("allowed_hosts")
        if not isinstance(raw_hosts, list) or not raw_hosts:
            raise RetentionRegistryError(
                f"governed-web source {raw_name!r} requires a non-empty allowed_hosts array"
            )
        hosts = tuple(sorted({_normalise_restricted_host(item) for item in raw_hosts}))
        source_url = raw_source.get("source_url")
        if not isinstance(source_url, str) or not source_url.strip():
            raise RetentionRegistryError(
                f"governed-web source {raw_name!r} requires a non-empty source_url"
            )
        parsed = urlsplit(source_url.strip())
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username
            or parsed.password
        ):
            raise RetentionRegistryError(
                f"governed-web source {raw_name!r} source_url must be an HTTPS URL"
            )
        try:
            port = parsed.port
        except ValueError as exc:
            raise RetentionRegistryError(
                f"governed-web source {raw_name!r} source_url must use the default HTTPS port"
            ) from exc
        source_host = _normalise_restricted_host(parsed.hostname)
        if port not in (None, 443) or source_host not in hosts:
            raise RetentionRegistryError(
                f"governed-web source {raw_name!r} source_url host is not an allowed host"
            )
        usage_scope = raw_source.get("usage_scope")
        if usage_scope not in {"internal_research", "production_catalog"}:
            raise RetentionRegistryError(
                f"governed-web source {raw_name!r} has an unsupported usage_scope"
            )
        admissions.append(
            GovernedWebSourceAdmission(
                source_name=raw_name,
                allowed_hosts=hosts,
                usage_scope=usage_scope,
            )
        )
    return tuple(sorted(admissions, key=lambda item: item.source_name))


def load_governed_web_retention_sources(path: Path) -> tuple[str, ...]:
    """Return the exact required source set from the immutable registry of record."""

    payload = _load_registry_payload(Path(path))
    sources = payload["sources"]
    assert isinstance(sources, dict)

    selected: list[str] = []
    for raw_name, raw_source in sources.items():
        if not isinstance(raw_name, str) or not isinstance(raw_source, dict):
            raise RetentionRegistryError("source registry contains a malformed source entry")
        if raw_source.get("kind") != GOVERNED_WEB_KIND:
            continue
        if _SOURCE_NAME_PATTERN.fullmatch(raw_name) is None:
            raise RetentionRegistryError(f"unsafe governed-web source name: {raw_name!r}")
        _require_governed_web_lifecycle(raw_name, raw_source)
        selected.append(raw_name)
        if len(selected) > MAXIMUM_GOVERNED_WEB_SOURCES:
            raise RetentionRegistryError(
                f"governed-web source count exceeds {MAXIMUM_GOVERNED_WEB_SOURCES}"
            )
    if not selected:
        raise RetentionRegistryError("source registry contains no managed governed-web source")
    return tuple(sorted(selected))


__all__ = [
    "GOVERNED_WEB_RETENTION_CRON",
    "GOVERNED_WEB_RETENTION_ENGINE",
    "GovernedWebSourceAdmission",
    "RestrictedWebSource",
    "RetentionRegistryError",
    "load_governed_web_source_admissions",
    "load_governed_web_retention_sources",
    "load_restricted_web_sources",
]
