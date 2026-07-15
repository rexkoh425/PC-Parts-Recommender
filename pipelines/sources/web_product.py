"""Policy-gated web crawler for schema.org Product and Offer documents.

This adapter deliberately does not contain retailer-specific URLs or selectors.  A crawl can
only start from an explicit policy that names the allowed hosts, an exact reviewed terms
document, machine-readable data-use rights, and conservative resource limits.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import socket
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpcore
import httpx

from pc_build_recommender.domain.enums import ComponentKind, ListingCondition, StockStatus
from pc_build_recommender.domain.models import PriceSample, RetailerListing
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION, stable_identifier
from pipelines.sources.base import ParsedBatch, RawSnapshot, rejected_record, sha256_bytes
from pipelines.sources.rights import DataUse, DataUseRights

WEB_PRODUCT_PARSER_VERSION = "schemaorg-product-offer-v3"
WEB_CRAWL_CACHE_SCHEMA_VERSION = "pc-build-recommender.web-crawl-cache.v1"
WEB_RAW_METADATA_SCHEMA_VERSION = "pc-build-recommender.web-raw-page.v2"
_SOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RAW_PAGE_FILE_PATTERN = re.compile(r"^[0-9a-f]{32}-[0-9a-f]{64}\.(?:html|terms|txt)$")
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_HTML_MEDIA_TYPES = {"text/html", "application/xhtml+xml"}
_CONTROL_MEDIA_TYPES = {
    "application/octet-stream",
    "application/json",
    "text/html",
    "text/plain",
}
_MAX_ROBOTS_INTERVAL_SECONDS = 60.0
_MONEY_PATTERN = re.compile(
    r"^\s*(?:(?P<prefix>[A-Z]{3}|S\$|\$)\s*)?"
    r"(?P<amount>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    r"(?:\s*(?P<suffix>[A-Z]{3}))?\s*$",
    re.IGNORECASE,
)


class WebCrawlError(RuntimeError):
    """Base error for a rejected or failed crawl."""


class WebCrawlPolicyError(WebCrawlError):
    """Raised when rights, terms, or robots policy does not authorize a crawl."""


class WebCrawlSecurityError(WebCrawlError):
    """Raised when a URL or redirect is unsafe."""


class WebCrawlLimitError(WebCrawlError):
    """Raised before a crawl can exceed an explicit resource budget."""


Resolver = Callable[[str], Iterable[str]]


class WebUsageScope(StrEnum):
    PRODUCTION_CATALOG = "production_catalog"
    INTERNAL_RESEARCH = "internal_research"


class UnknownShippingPolicy(StrEnum):
    REJECT = "reject"
    ZERO_CONFIRMED = "zero_confirmed"


@dataclass(frozen=True, slots=True)
class WebAcquisitionAuthority:
    """Separate evidence that automated retrieval and raw retention were reviewed.

    Robots directives are intentionally not represented here: respecting robots is mandatory,
    but robots.txt is not a licence or a grant of reuse rights.
    """

    authority_reference: str
    reviewed_on: date
    expires_on: date | None
    permits_automated_retrieval: bool
    permits_raw_snapshot_storage: bool
    permits_internal_analysis: bool
    retention_days: int
    deletion_required: bool

    def __post_init__(self) -> None:
        if not self.authority_reference.strip():
            raise ValueError("acquisition authority_reference is required")
        if not isinstance(self.reviewed_on, date):
            raise TypeError("acquisition reviewed_on must be a date")
        if self.reviewed_on > date.today():
            raise ValueError("acquisition reviewed_on cannot be in the future")
        if self.expires_on is not None and self.expires_on < self.reviewed_on:
            raise ValueError("acquisition expiry cannot precede review")
        for field_name in (
            "permits_automated_retrieval",
            "permits_raw_snapshot_storage",
            "permits_internal_analysis",
            "deletion_required",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"acquisition {field_name} must be a boolean")
        if type(self.retention_days) is not int:
            raise TypeError("acquisition retention_days must be an integer")
        self.assert_active()

    def assert_active(self, *, on_date: date | None = None) -> None:
        """Revalidate time-sensitive acquisition authority immediately before a crawl."""

        today = on_date or date.today()
        if self.reviewed_on > today:
            raise WebCrawlPolicyError("acquisition authority review is not yet effective")
        if self.expires_on is not None and self.expires_on < today:
            raise WebCrawlPolicyError("acquisition authority has expired")
        if not self.permits_automated_retrieval:
            raise WebCrawlPolicyError("acquisition authority does not permit automated retrieval")
        if not self.permits_raw_snapshot_storage:
            raise WebCrawlPolicyError("acquisition authority does not permit raw snapshot storage")
        if not 1 <= self.retention_days <= 3650:
            raise ValueError("acquisition retention_days must be between 1 and 3650")
        if not self.deletion_required:
            raise WebCrawlPolicyError("acquisition authority must require bounded deletion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_reference": self.authority_reference,
            "reviewed_on": self.reviewed_on.isoformat(),
            "expires_on": self.expires_on.isoformat() if self.expires_on is not None else None,
            "permits_automated_retrieval": self.permits_automated_retrieval,
            "permits_raw_snapshot_storage": self.permits_raw_snapshot_storage,
            "permits_internal_analysis": self.permits_internal_analysis,
            "retention_days": self.retention_days,
            "deletion_required": self.deletion_required,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WebAcquisitionAuthority:
        required = {
            "authority_reference",
            "reviewed_on",
            "expires_on",
            "permits_automated_retrieval",
            "permits_raw_snapshot_storage",
            "permits_internal_analysis",
            "retention_days",
            "deletion_required",
        }
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        if missing:
            raise ValueError(f"acquisition authority missing fields: {missing}")
        if extra:
            raise ValueError(f"acquisition authority contains unknown fields: {extra}")
        expiry = payload["expires_on"]
        return cls(
            authority_reference=str(payload["authority_reference"]),
            reviewed_on=date.fromisoformat(str(payload["reviewed_on"])),
            expires_on=date.fromisoformat(str(expiry)) if expiry is not None else None,
            permits_automated_retrieval=payload["permits_automated_retrieval"],
            permits_raw_snapshot_storage=payload["permits_raw_snapshot_storage"],
            permits_internal_analysis=payload["permits_internal_analysis"],
            retention_days=payload["retention_days"],
            deletion_required=payload["deletion_required"],
        )


def _normalise_host(host: str) -> str:
    value = host.rstrip(".").encode("idna").decode("ascii").lower()
    if not value:
        raise WebCrawlSecurityError("URL host is required")
    return value


def _default_resolver(host: str) -> Iterable[str]:
    try:
        results = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise WebCrawlSecurityError(f"DNS resolution failed for {host}") from exc
    return tuple(sorted({str(result[4][0]) for result in results}))


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%", maxsplit=1)[0])
    except ValueError:
        return False
    return address.is_global and not address.is_multicast


def _resolve_public_addresses(host: str, resolver: Resolver) -> tuple[str, ...]:
    """Resolve *host* once and return only validated literal public IP addresses."""

    addresses: list[str] = []
    for raw_address in resolver(host):
        value = str(raw_address).split("%", maxsplit=1)[0]
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise WebCrawlSecurityError(
                f"host {host} resolved to an invalid IP address and was rejected"
            ) from exc
        if not address.is_global or address.is_multicast:
            raise WebCrawlSecurityError(
                f"host {host} resolves to a non-public address and was rejected"
            )
        normalised = str(address)
        if normalised not in addresses:
            addresses.append(normalised)
    if not addresses:
        raise WebCrawlSecurityError(f"DNS resolution returned no addresses for {host}")
    return tuple(addresses)


class _PinnedDNSBackend(httpcore.NetworkBackend):
    """Resolve at connect time, then give the socket layer only a validated IP.

    httpcore still owns the original request origin, so it uses the hostname for the Host
    header, TLS SNI, and certificate verification.  The wrapped backend never receives that
    hostname and therefore cannot perform a second, potentially rebound DNS lookup.
    """

    def __init__(
        self,
        *,
        resolver: Resolver,
        backend: httpcore.NetworkBackend | None = None,
    ) -> None:
        self._resolver = resolver
        self._backend = backend or httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        addresses = _resolve_public_addresses(_normalise_host(host), self._resolver)
        last_error: httpcore.ConnectError | httpcore.ConnectTimeout | None = None
        for address in addresses:
            try:
                return self._backend.connect_tcp(
                    host=address,
                    port=port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        if last_error is None:  # pragma: no cover - addresses is guaranteed non-empty
            raise WebCrawlSecurityError("no validated public address was available")
        raise last_error

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        del path, timeout, socket_options
        raise WebCrawlSecurityError("Unix sockets are not permitted by the web crawler")

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


class _PinnedHTTPTransport(httpx.HTTPTransport):
    """HTTPX transport whose socket backend cannot re-resolve an origin hostname."""

    def __init__(
        self,
        *,
        resolver: Resolver,
        limits: httpx.Limits,
        network_backend: httpcore.NetworkBackend | None = None,
    ) -> None:
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpcore.default_ssl_context(),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            retries=0,
            network_backend=_PinnedDNSBackend(
                resolver=resolver,
                backend=network_backend,
            ),
        )


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise WebCrawlSecurityError("crawler targets must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise WebCrawlSecurityError("crawler targets must not contain credentials")
    if parsed.fragment:
        raise WebCrawlSecurityError("crawler targets must not contain URL fragments")
    try:
        port = parsed.port
    except ValueError as exc:
        raise WebCrawlSecurityError("crawler target has an invalid port") from exc
    if port not in (None, 443):
        raise WebCrawlSecurityError("crawler targets must use the default HTTPS port")
    if parsed.hostname is None:
        raise WebCrawlSecurityError("crawler target must have a host")
    host = _normalise_host(parsed.hostname)
    path = parsed.path or "/"
    return urlunsplit(("https", host, path, parsed.query, ""))


def _same_url_path(left: str, right: str) -> bool:
    left_value = urlsplit(left)
    right_value = urlsplit(right)
    return (
        left_value.scheme,
        left_value.hostname,
        left_value.port,
        left_value.path,
    ) == (
        right_value.scheme,
        right_value.hostname,
        right_value.port,
        right_value.path,
    )


@dataclass(frozen=True, slots=True)
class WebSourcePolicy:
    """Reviewed authority, scope, and resource limits for exactly one web source."""

    source_name: str
    retailer: str
    allowed_hosts: tuple[str, ...]
    terms_url: str
    terms_selector: str
    canonical_terms_sha256: str
    terms_verified_on: date
    licence_or_access_note: str
    rights: DataUseRights
    acquisition_authority: WebAcquisitionAuthority
    url_categories: Mapping[str, ComponentKind]
    usage_scope: WebUsageScope = WebUsageScope.PRODUCTION_CATALOG
    allowed_currencies: tuple[str, ...] = ("SGD",)
    unknown_shipping: UnknownShippingPolicy = UnknownShippingPolicy.REJECT
    user_agent: str = "BuildSignalProductBot/1.0 (+https://buildsignal.example/data-policy)"
    requests_per_second: float = 0.5
    max_concurrency: int = 2
    max_pages: int = 100
    maximum_page_bytes: int = 2 * 1024 * 1024
    maximum_total_bytes: int = 100 * 1024 * 1024
    maximum_control_bytes: int = 512 * 1024
    timeout_seconds: float = 20.0
    max_redirects: int = 2
    maximum_offers_per_page: int = 100
    maximum_records: int = 10_000
    maximum_jsonld_blocks: int = 1_000
    maximum_product_nodes: int = 10_000
    maximum_rejections: int = 10_000
    maximum_parse_events: int = 100_000
    allow_non_new: bool = False
    training_eligible: bool = False
    published_claims_eligible: bool = False

    def __post_init__(self) -> None:
        if _SOURCE_NAME_PATTERN.fullmatch(self.source_name) is None:
            raise ValueError("source_name must be a lowercase slug")
        if not self.retailer.strip():
            raise ValueError("retailer must not be empty")
        hosts = tuple(sorted({_normalise_host(host) for host in self.allowed_hosts}))
        if not hosts:
            raise ValueError("allowed_hosts must explicitly name at least one host")
        object.__setattr__(self, "allowed_hosts", hosts)
        canonical_terms_url = _canonical_url(self.terms_url)
        if _normalise_host(urlsplit(canonical_terms_url).hostname or "") not in hosts:
            raise WebCrawlSecurityError("terms_url host is outside allowed_hosts")
        object.__setattr__(self, "terms_url", canonical_terms_url)
        if re.fullmatch(r"[.#][A-Za-z][A-Za-z0-9_:-]*", self.terms_selector) is None:
            raise ValueError("terms_selector must be one exact element ID or class selector")
        terms_hash = self.canonical_terms_sha256.lower()
        if _SHA256_PATTERN.fullmatch(terms_hash) is None:
            raise ValueError("canonical_terms_sha256 must be an exact SHA-256 digest")
        object.__setattr__(self, "canonical_terms_sha256", terms_hash)
        if not isinstance(self.terms_verified_on, date):
            raise TypeError("terms_verified_on must be a date")
        if self.terms_verified_on > date.today():
            raise ValueError("terms_verified_on cannot be in the future")
        if not self.licence_or_access_note.strip():
            raise ValueError("licence_or_access_note must document the reviewed authority")
        if self.rights.contract_version_url != self.terms_url:
            raise WebCrawlPolicyError(
                "rights.contract_version_url must identify the exact reviewed terms_url"
            )
        categories: dict[str, ComponentKind] = {}
        for raw_url, raw_category in self.url_categories.items():
            canonical_url = _canonical_url(str(raw_url))
            host = _normalise_host(urlsplit(canonical_url).hostname or "")
            if host not in hosts:
                raise WebCrawlSecurityError(
                    f"category-mapped URL host {host} is outside allowed_hosts"
                )
            if canonical_url in categories:
                raise ValueError(f"duplicate canonical URL category mapping: {canonical_url}")
            if canonical_url == canonical_terms_url:
                raise WebCrawlPolicyError("terms_url cannot be mapped as a product URL")
            if urlsplit(canonical_url).path.casefold() == "/robots.txt":
                raise WebCrawlPolicyError("robots.txt cannot be mapped as a product URL")
            categories[canonical_url] = ComponentKind(raw_category)
        if not categories:
            raise ValueError("url_categories must explicitly map at least one URL to a category")
        object.__setattr__(self, "url_categories", dict(sorted(categories.items())))
        object.__setattr__(self, "usage_scope", WebUsageScope(self.usage_scope))
        currencies = tuple(
            sorted({str(currency).strip().upper() for currency in self.allowed_currencies})
        )
        if not currencies or any(re.fullmatch(r"[A-Z]{3}", value) is None for value in currencies):
            raise ValueError("allowed_currencies must contain ISO 4217-style codes")
        object.__setattr__(self, "allowed_currencies", currencies)
        object.__setattr__(self, "unknown_shipping", UnknownShippingPolicy(self.unknown_shipping))
        for field_name in (
            "allow_non_new",
            "training_eligible",
            "published_claims_eligible",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a boolean")
        self.assert_authorized_now()
        if not self.user_agent.strip() or any(char in self.user_agent for char in "\r\n"):
            raise ValueError("user_agent must be a non-empty single-line value")
        for field_name in (
            "max_concurrency",
            "max_pages",
            "maximum_page_bytes",
            "maximum_total_bytes",
            "maximum_control_bytes",
            "max_redirects",
            "maximum_offers_per_page",
            "maximum_records",
            "maximum_jsonld_blocks",
            "maximum_product_nodes",
            "maximum_rejections",
            "maximum_parse_events",
        ):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
        for field_name in ("requests_per_second", "timeout_seconds"):
            if isinstance(getattr(self, field_name), bool) or not isinstance(
                getattr(self, field_name), int | float
            ):
                raise TypeError(f"{field_name} must be a number")
        if not 0.01 <= self.requests_per_second <= 10:
            raise ValueError("requests_per_second must be between 0.01 and 10")
        if not 1 <= self.max_concurrency <= 8:
            raise ValueError("max_concurrency must be between 1 and 8")
        if not 1 <= self.max_pages <= 500:
            raise ValueError("max_pages must be between 1 and 500")
        if not 1_024 <= self.maximum_page_bytes <= 10 * 1024 * 1024:
            raise ValueError("maximum_page_bytes must be between 1 KiB and 10 MiB")
        if not self.maximum_page_bytes <= self.maximum_total_bytes <= 500 * 1024 * 1024:
            raise ValueError(
                "maximum_total_bytes must cover one page and be no greater than 500 MiB"
            )
        if not 1_024 <= self.maximum_control_bytes <= 2 * 1024 * 1024:
            raise ValueError("maximum_control_bytes must be between 1 KiB and 2 MiB")
        if not 1 <= self.timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be between 1 and 60")
        if not 0 <= self.max_redirects <= 3:
            raise ValueError("max_redirects must be between 0 and 3")
        if not 1 <= self.maximum_offers_per_page <= 1_000:
            raise ValueError("maximum_offers_per_page must be between 1 and 1000")
        if not 1 <= self.maximum_records <= 50_000:
            raise ValueError("maximum_records must be between 1 and 50000")
        if not 1 <= self.maximum_jsonld_blocks <= 10_000:
            raise ValueError("maximum_jsonld_blocks must be between 1 and 10000")
        if not 1 <= self.maximum_product_nodes <= 100_000:
            raise ValueError("maximum_product_nodes must be between 1 and 100000")
        if not 1 <= self.maximum_rejections <= 50_000:
            raise ValueError("maximum_rejections must be between 1 and 50000")
        if not 1 <= self.maximum_parse_events <= 1_000_000:
            raise ValueError("maximum_parse_events must be between 1 and 1000000")

    @property
    def normalized_record_limit(self) -> int:
        """Return the global offer/record cap after applying the page budget."""

        return min(self.maximum_records, self.max_pages * self.maximum_offers_per_page)

    def assert_authorized_now(self) -> None:
        """Revalidate acquisition and downstream-use authority for a new crawl."""

        self.acquisition_authority.assert_active()
        try:
            self.rights.assert_consent_active()
            if "SG" not in self.rights.territories:
                raise PermissionError("source rights do not permit territory SG")
            if self.usage_scope == WebUsageScope.PRODUCTION_CATALOG:
                self.rights.assert_catalog_serving_allowed(territory="SG")
                if self.training_eligible:
                    self.rights.assert_allowed(DataUse.TRAIN)
                if self.published_claims_eligible:
                    self.rights.assert_allowed(DataUse.DISPLAY)
                    self.rights.assert_allowed(DataUse.DERIVE)
            else:
                granted_uses = [
                    use.value for use in DataUse if getattr(self.rights, use.field_name)
                ]
                if granted_uses:
                    raise WebCrawlPolicyError(
                        "internal_research policies require all downstream DataUse grants false; "
                        f"received {granted_uses}"
                    )
                if self.training_eligible or self.published_claims_eligible:
                    raise WebCrawlPolicyError(
                        "internal_research records cannot be training or published-claims eligible"
                    )
                if not self.acquisition_authority.permits_internal_analysis:
                    raise WebCrawlPolicyError(
                        "acquisition authority does not permit internal analysis"
                    )
                if self.acquisition_authority.retention_days > 30:
                    raise WebCrawlPolicyError(
                        "internal_research raw snapshot retention cannot exceed 30 days"
                    )
        except PermissionError as exc:
            raise WebCrawlPolicyError(str(exc)) from exc
        if (
            self.rights.retention_days is not None
            and self.acquisition_authority.retention_days > self.rights.retention_days
        ):
            raise WebCrawlPolicyError(
                "raw snapshot retention exceeds the source rights retention period"
            )

    @property
    def development_only(self) -> bool:
        return self.usage_scope == WebUsageScope.INTERNAL_RESEARCH

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict()
        return sha256_bytes(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "retailer": self.retailer,
            "allowed_hosts": list(self.allowed_hosts),
            "terms_url": self.terms_url,
            "terms_selector": self.terms_selector,
            "canonical_terms_sha256": self.canonical_terms_sha256,
            "terms_verified_on": self.terms_verified_on.isoformat(),
            "licence_or_access_note": self.licence_or_access_note,
            "rights": self.rights.to_dict(),
            "acquisition_authority": self.acquisition_authority.to_dict(),
            "url_categories": {
                url: category.value for url, category in sorted(self.url_categories.items())
            },
            "usage_scope": self.usage_scope.value,
            "allowed_currencies": list(self.allowed_currencies),
            "unknown_shipping": self.unknown_shipping.value,
            "user_agent": self.user_agent,
            "requests_per_second": self.requests_per_second,
            "max_concurrency": self.max_concurrency,
            "max_pages": self.max_pages,
            "maximum_page_bytes": self.maximum_page_bytes,
            "maximum_total_bytes": self.maximum_total_bytes,
            "maximum_control_bytes": self.maximum_control_bytes,
            "timeout_seconds": self.timeout_seconds,
            "max_redirects": self.max_redirects,
            "maximum_offers_per_page": self.maximum_offers_per_page,
            "maximum_records": self.maximum_records,
            "maximum_jsonld_blocks": self.maximum_jsonld_blocks,
            "maximum_product_nodes": self.maximum_product_nodes,
            "maximum_rejections": self.maximum_rejections,
            "maximum_parse_events": self.maximum_parse_events,
            "allow_non_new": self.allow_non_new,
            "training_eligible": self.training_eligible,
            "published_claims_eligible": self.published_claims_eligible,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WebSourcePolicy:
        required = {
            "source_name",
            "retailer",
            "allowed_hosts",
            "terms_url",
            "terms_selector",
            "canonical_terms_sha256",
            "terms_verified_on",
            "licence_or_access_note",
            "rights",
            "acquisition_authority",
            "url_categories",
        }
        optional = {
            "user_agent",
            "requests_per_second",
            "max_concurrency",
            "max_pages",
            "maximum_page_bytes",
            "maximum_total_bytes",
            "maximum_control_bytes",
            "timeout_seconds",
            "max_redirects",
            "maximum_offers_per_page",
            "maximum_records",
            "maximum_jsonld_blocks",
            "maximum_product_nodes",
            "maximum_rejections",
            "maximum_parse_events",
            "allow_non_new",
            "training_eligible",
            "published_claims_eligible",
            "usage_scope",
            "allowed_currencies",
            "unknown_shipping",
        }
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required - optional)
        if missing:
            raise ValueError(f"web source policy missing fields: {missing}")
        if extra:
            raise ValueError(f"web source policy contains unknown fields: {extra}")
        raw_hosts = payload["allowed_hosts"]
        raw_rights = payload["rights"]
        raw_acquisition = payload["acquisition_authority"]
        raw_categories = payload["url_categories"]
        if not isinstance(raw_hosts, list | tuple):
            raise TypeError("allowed_hosts must be an array")
        if not isinstance(raw_rights, Mapping):
            raise TypeError("rights must be an object")
        if not isinstance(raw_acquisition, Mapping):
            raise TypeError("acquisition_authority must be an object")
        if not isinstance(raw_categories, Mapping):
            raise TypeError("url_categories must be an object")
        values = dict(payload)
        values["allowed_hosts"] = tuple(str(host) for host in raw_hosts)
        values["terms_verified_on"] = date.fromisoformat(str(payload["terms_verified_on"]))
        values["rights"] = DataUseRights.from_mapping(raw_rights)
        values["acquisition_authority"] = WebAcquisitionAuthority.from_mapping(raw_acquisition)
        values["url_categories"] = {
            str(url): ComponentKind(str(category)) for url, category in raw_categories.items()
        }
        if "allowed_currencies" in values:
            raw_currencies = values["allowed_currencies"]
            if not isinstance(raw_currencies, list | tuple):
                raise TypeError("allowed_currencies must be an array")
            values["allowed_currencies"] = tuple(str(value) for value in raw_currencies)
        if "usage_scope" in values:
            values["usage_scope"] = WebUsageScope(str(values["usage_scope"]))
        if "unknown_shipping" in values:
            values["unknown_shipping"] = UnknownShippingPolicy(str(values["unknown_shipping"]))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CrawledPage:
    requested_url: str
    final_url: str
    snapshot: RawSnapshot
    etag: str | None
    last_modified: str | None
    not_modified: bool


@dataclass(frozen=True, slots=True)
class WebCrawlResult:
    batch: ParsedBatch
    pages: tuple[CrawledPage, ...]
    retrieval_started_at: datetime
    retrieval_completed_at: datetime
    robots_sha256_by_host: Mapping[str, str]
    terms_snapshot_sha256: str
    terms_post_snapshot_sha256: str
    terms_canonical_sha256: str
    policy_fingerprint: str


class _JSONLDExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capturing = False
        self._chunks: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "script" or self._capturing:
            return
        values = {name.casefold(): (value or "") for name, value in attrs}
        media_type = values.get("type", "").split(";", maxsplit=1)[0].strip().casefold()
        if media_type == "application/ld+json":
            self._capturing = True
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._capturing:
            self.blocks.append("".join(self._chunks).strip())
            self._capturing = False
            self._chunks = []


class _PolicyTermsExtractor(HTMLParser):
    _HIDDEN_TAGS = {"script", "style", "template", "noscript", "svg"}
    _SEMANTIC_ATTRIBUTES = {
        "aria-hidden",
        "aria-label",
        "aria-labelledby",
        "content",
        "download",
        "hidden",
        "href",
        "itemprop",
        "rel",
        "role",
        "title",
    }
    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self, selector: str) -> None:
        super().__init__(convert_charrefs=True)
        self._selector_kind = selector[0]
        self._selector_value = selector[1:]
        self._target_depth = 0
        self._hidden_depth = 0
        self._stack: list[tuple[str, bool, bool]] = []
        self.match_count = 0
        self.chunks: list[str] = []
        self.semantic_tokens: list[str] = []

    def _matches(self, attrs: list[tuple[str, str | None]]) -> bool:
        values = {name.casefold(): value or "" for name, value in attrs}
        if self._selector_kind == "#":
            return values.get("id") == self._selector_value
        return self._selector_value in values.get("class", "").split()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        values = {name.casefold(): value or "" for name, value in attrs}
        matched = self._matches(attrs)
        if matched:
            self.match_count += 1
        active = self._target_depth > 0 or matched
        increments_target = active and tag not in self._VOID_TAGS
        starts_hidden = active and (
            tag in self._HIDDEN_TAGS
            or "hidden" in values
            or values.get("aria-hidden", "").strip().casefold() == "true"
        )
        if active:
            semantic_attrs = {
                name: unicodedata.normalize("NFKC", " ".join(value.split()))
                for name, value in sorted(values.items())
                if name in self._SEMANTIC_ATTRIBUTES
            }
            if semantic_attrs:
                self.semantic_tokens.append(
                    json.dumps(
                        {"attributes": semantic_attrs, "tag": tag},
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
        if increments_target:
            self._target_depth += 1
        if starts_hidden:
            self._hidden_depth += 1
        if tag not in self._VOID_TAGS:
            self._stack.append((tag, increments_target, starts_hidden))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        matched = self._matches(attrs)
        if matched:
            self.match_count += 1
        active = self._target_depth > 0 or matched
        if not active:
            return
        values = {name.casefold(): value or "" for name, value in attrs}
        semantic_attrs = {
            name: unicodedata.normalize("NFKC", " ".join(value.split()))
            for name, value in sorted(values.items())
            if name in self._SEMANTIC_ATTRIBUTES
        }
        if semantic_attrs:
            self.semantic_tokens.append(
                json.dumps(
                    {"attributes": semantic_attrs, "tag": tag},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        match_index = next(
            (
                index
                for index in range(len(self._stack) - 1, -1, -1)
                if self._stack[index][0] == tag
            ),
            None,
        )
        if match_index is None:
            return
        closing = self._stack[match_index:]
        del self._stack[match_index:]
        for _name, increments_target, starts_hidden in reversed(closing):
            if starts_hidden:
                self._hidden_depth = max(0, self._hidden_depth - 1)
            if increments_target:
                self._target_depth = max(0, self._target_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._target_depth > 0 and self._hidden_depth == 0:
            self.chunks.append(data)


def calculate_canonical_terms_sha256(
    body: bytes,
    *,
    media_type: str,
    selector: str,
) -> str:
    """Hash normalized wording and stable link/semantic attributes from one element."""

    if media_type.casefold() not in _HTML_MEDIA_TYPES:
        raise WebCrawlPolicyError("reviewed terms must be an HTML document")
    try:
        html = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WebCrawlPolicyError("reviewed terms are not valid UTF-8") from exc
    extractor = _PolicyTermsExtractor(selector)
    extractor.feed(html)
    extractor.close()
    if extractor.match_count != 1:
        raise WebCrawlPolicyError(
            f"terms_selector must match exactly one element; matched {extractor.match_count}"
        )
    visible_text = unicodedata.normalize("NFKC", " ".join("".join(extractor.chunks).split()))
    if not visible_text:
        raise WebCrawlPolicyError("terms_selector matched no human-visible wording")
    canonical_payload = (
        "pc-build-recommender.canonical-terms.v2\n"
        + json.dumps(
            {"semantic_tokens": extractor.semantic_tokens, "visible_text": visible_text},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    ).encode("utf-8")
    return sha256_bytes(canonical_payload)


class _HostRateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        self._policy_interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleeper = sleeper
        self._host_intervals: dict[str, float] = {}
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    def require_interval(self, host: str, interval_seconds: float) -> None:
        if interval_seconds < 0:
            raise ValueError("rate-limit interval cannot be negative")
        with self._lock:
            self._host_intervals[host] = max(
                self._host_intervals.get(host, self._policy_interval),
                self._policy_interval,
                interval_seconds,
            )

    def wait(self, host: str) -> None:
        with self._lock:
            now = self._clock()
            interval = self._host_intervals.get(host, self._policy_interval)
            last_request = self._last_request.get(host)
            delay = 0.0 if last_request is None else max(0.0, last_request + interval - now)
            if delay:
                self._sleeper(delay)
                now = self._clock()
            self._last_request[host] = now


class WebProductCrawlerAdapter:
    """Crawl explicitly approved product pages and emit normalized listing envelopes."""

    def __init__(
        self,
        *,
        raw_root: str | Path,
        policy: WebSourcePolicy,
        transport: httpx.MockTransport | None = None,
        resolver: Resolver = _default_resolver,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise TypeError("transport injection is restricted to httpx.MockTransport tests")
        self.raw_root = Path(raw_root)
        self.policy = policy
        self._transport = transport
        self._resolver = resolver
        self._limiter = _HostRateLimiter(
            policy.requests_per_second,
            clock=clock,
            sleeper=sleeper,
        )
        self._state_lock = threading.Lock()
        self._budget_lock = threading.Lock()
        self._total_bytes = 0
        self._state: dict[str, dict[str, Any]] = {}
        self._robots_parsers: dict[str, RobotFileParser] = {}

    @property
    def source_root(self) -> Path:
        return self.raw_root / self.policy.source_name

    @property
    def state_path(self) -> Path:
        return self.source_root / "http-cache.json"

    @property
    def pages_root(self) -> Path:
        return self.source_root / "pages"

    def validate_target_url(self, url: str) -> str:
        canonical = _canonical_url(url)
        host = _normalise_host(urlsplit(canonical).hostname or "")
        if host not in self.policy.allowed_hosts:
            raise WebCrawlSecurityError(f"host {host} is outside the explicit allowlist")
        _resolve_public_addresses(host, self._resolver)
        return canonical

    def _assert_public_connected_peer(self, response: httpx.Response) -> None:
        """Verify the actual socket peer so DNS cannot rebind to a private service."""

        network_stream = response.extensions.get("network_stream")
        if network_stream is None:
            if self._transport is None:
                raise WebCrawlSecurityError(
                    "HTTP transport did not expose the connected peer address"
                )
            # MockTransport is admitted only for deterministic tests and has no real socket.
            return
        get_extra_info = getattr(network_stream, "get_extra_info", None)
        if not callable(get_extra_info):
            raise WebCrawlSecurityError("HTTP transport exposed an invalid network stream")
        try:
            server_address = get_extra_info("server_addr")
        except Exception as exc:  # pragma: no cover - defensive transport boundary
            raise WebCrawlSecurityError(
                "HTTP transport could not report the connected peer address"
            ) from exc
        if not isinstance(server_address, tuple) or not server_address:
            raise WebCrawlSecurityError("HTTP transport returned an invalid peer address")
        if not _is_public_address(str(server_address[0])):
            raise WebCrawlSecurityError(
                "connected peer is not a public IP address; possible DNS rebinding rejected"
            )

    def crawl(self, urls: Sequence[str]) -> WebCrawlResult:
        """Fetch a bounded URL set after validating rights, terms, robots, and network scope."""

        self.policy.assert_authorized_now()
        unique_urls = tuple(dict.fromkeys(self.validate_target_url(url) for url in urls))
        if not unique_urls:
            raise WebCrawlPolicyError("at least one product URL is required")
        unmapped_urls = sorted(set(unique_urls) - set(self.policy.url_categories))
        if unmapped_urls:
            raise WebCrawlPolicyError(
                "every crawled URL requires an explicit component-category mapping: "
                f"{unmapped_urls}"
            )
        if len(unique_urls) > self.policy.max_pages:
            raise WebCrawlLimitError(
                f"crawl requested {len(unique_urls)} pages; policy limit is {self.policy.max_pages}"
            )
        self._total_bytes = 0
        self._robots_parsers = {}
        self._resolved_pages_root()
        self._state = self._load_state()
        self._retain_active_cache_entries()
        self._write_state()
        used_hosts = {
            _normalise_host(urlsplit(url).hostname or "")
            for url in (*unique_urls, self.policy.terms_url)
        }
        with self._client() as client:
            robots_pages: dict[str, CrawledPage] = {}
            for host in sorted(used_hosts):
                try:
                    robots_pages[host] = self._fetch_document(
                        client,
                        f"https://{host}/robots.txt",
                        maximum_bytes=self.policy.maximum_control_bytes,
                        suffix=".txt",
                        accepted_media_types=_CONTROL_MEDIA_TYPES,
                    )
                except WebCrawlError as exc:
                    raise WebCrawlPolicyError(
                        f"robots.txt for {host} was unavailable; failing closed"
                    ) from exc
            robots_parsers = {
                host: self._parse_robots(host, page) for host, page in robots_pages.items()
            }
            self._robots_parsers = robots_parsers
            for host, parser in robots_parsers.items():
                self._limiter.require_interval(host, self._robots_interval(host, parser))
            for url in (*unique_urls, self.policy.terms_url):
                host = _normalise_host(urlsplit(url).hostname or "")
                if not robots_parsers[host].can_fetch(self.policy.user_agent, url):
                    raise WebCrawlPolicyError(f"robots.txt does not permit {url}")
            try:
                terms_page = self._fetch_document(
                    client,
                    self.policy.terms_url,
                    maximum_bytes=self.policy.maximum_control_bytes,
                    suffix=".terms",
                    accepted_media_types=_CONTROL_MEDIA_TYPES,
                )
            except WebCrawlError as exc:
                raise WebCrawlPolicyError(
                    "reviewed terms were unavailable; failing closed"
                ) from exc
            terms_canonical_sha256 = calculate_canonical_terms_sha256(
                terms_page.snapshot.path.read_bytes(),
                media_type=terms_page.snapshot.media_type,
                selector=self.policy.terms_selector,
            )
            if terms_canonical_sha256 != self.policy.canonical_terms_sha256:
                raise WebCrawlPolicyError(
                    "canonical terms wording changed from the reviewed SHA-256; crawl stopped"
                )
            with ThreadPoolExecutor(
                max_workers=self.policy.max_concurrency,
                thread_name_prefix="web-product-crawl",
            ) as executor:
                pages = tuple(
                    executor.map(
                        lambda url: self._fetch_document(
                            client,
                            url,
                            maximum_bytes=self.policy.maximum_page_bytes,
                            suffix=".html",
                            accepted_media_types=_HTML_MEDIA_TYPES,
                        ),
                        unique_urls,
                    )
                )
            try:
                terms_post_page = self._fetch_document(
                    client,
                    self.policy.terms_url,
                    maximum_bytes=self.policy.maximum_control_bytes,
                    suffix=".terms",
                    accepted_media_types=_CONTROL_MEDIA_TYPES,
                    use_conditional_cache=False,
                )
            except WebCrawlError as exc:
                raise WebCrawlPolicyError(
                    "reviewed terms could not be revalidated after product retrieval; "
                    "failing closed"
                ) from exc
            terms_post_canonical_sha256 = calculate_canonical_terms_sha256(
                terms_post_page.snapshot.path.read_bytes(),
                media_type=terms_post_page.snapshot.media_type,
                selector=self.policy.terms_selector,
            )
            if (
                terms_post_canonical_sha256 != terms_canonical_sha256
                or terms_post_canonical_sha256 != self.policy.canonical_terms_sha256
            ):
                raise WebCrawlPolicyError(
                    "canonical terms wording changed during the crawl; crawl stopped"
                )
        self._write_state()
        robots_hashes = {host: page.snapshot.content_sha256 for host, page in robots_pages.items()}
        run_sha256 = self._run_sha256(
            urls=unique_urls,
            pages=pages,
            robots_hashes=robots_hashes,
            terms_pre_raw_sha256=terms_page.snapshot.content_sha256,
            terms_post_raw_sha256=terms_post_page.snapshot.content_sha256,
            terms_canonical_sha256=terms_canonical_sha256,
            terms_page=terms_page,
            terms_post_page=terms_post_page,
        )
        batch = self._parse_pages(
            pages,
            run_sha256=run_sha256,
            robots_hashes=robots_hashes,
            terms_page=terms_page,
            terms_post_page=terms_post_page,
        )
        all_observations = (
            *robots_pages.values(),
            terms_page,
            *pages,
            terms_post_page,
        )
        return WebCrawlResult(
            batch=batch,
            pages=pages,
            retrieval_started_at=min(page.snapshot.retrieved_at for page in all_observations),
            retrieval_completed_at=max(page.snapshot.retrieved_at for page in all_observations),
            robots_sha256_by_host=robots_hashes,
            terms_snapshot_sha256=terms_page.snapshot.content_sha256,
            terms_post_snapshot_sha256=terms_post_page.snapshot.content_sha256,
            terms_canonical_sha256=terms_canonical_sha256,
            policy_fingerprint=self.policy.fingerprint,
        )

    def _client(self) -> httpx.Client:
        limits = httpx.Limits(
            max_connections=self.policy.max_concurrency,
            max_keepalive_connections=self.policy.max_concurrency,
        )
        transport = self._transport or _PinnedHTTPTransport(
            resolver=self._resolver,
            limits=limits,
        )
        return httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(self.policy.timeout_seconds),
            follow_redirects=False,
            trust_env=False,
            limits=limits,
            headers={
                "User-Agent": self.policy.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8",
                "Accept-Encoding": "identity",
            },
        )

    def _assert_authorized_redirect(self, requested_url: str, redirected_url: str) -> None:
        requested_category = self.policy.url_categories.get(requested_url)
        if requested_category is None:
            if redirected_url != requested_url:
                raise WebCrawlSecurityError(
                    "control-document redirect target is not the exact authorized resource"
                )
            return
        redirected_category = self.policy.url_categories.get(redirected_url)
        if redirected_category != requested_category:
            raise WebCrawlSecurityError(
                "product redirect target is not explicitly authorized for the same category"
            )

    def _fetch_document(
        self,
        client: httpx.Client,
        requested_url: str,
        *,
        maximum_bytes: int,
        suffix: str,
        accepted_media_types: set[str],
        use_conditional_cache: bool = True,
    ) -> CrawledPage:
        requested_url = self.validate_target_url(requested_url)
        original_host = _normalise_host(urlsplit(requested_url).hostname or "")
        cached = self._state.get(requested_url) if use_conditional_cache else None
        if cached is not None and not self._cache_entry_is_active(cached):
            cached = None
        conditional_headers: dict[str, str] = {}
        if cached is not None:
            if value := self._safe_validator(cached.get("etag")):
                conditional_headers["If-None-Match"] = value
            if value := self._safe_validator(cached.get("last_modified")):
                conditional_headers["If-Modified-Since"] = value
        current_url = requested_url
        for redirect_number in range(self.policy.max_redirects + 1):
            current_url = self.validate_target_url(current_url)
            current_host = _normalise_host(urlsplit(current_url).hostname or "")
            if current_host != original_host:
                raise WebCrawlSecurityError("redirects to a different host are not permitted")
            self._limiter.wait(current_host)
            headers = conditional_headers if redirect_number == 0 else {}
            try:
                with client.stream("GET", current_url, headers=headers) as response:
                    self._assert_public_connected_peer(response)
                    if response.status_code == 304:
                        if redirect_number != 0 or cached is None:
                            raise WebCrawlError("received 304 without a usable cached response")
                        page = self._cached_page(
                            requested_url,
                            cached,
                            etag=(
                                self._safe_validator(response.headers.get("etag"))
                                or self._safe_validator(cached.get("etag"))
                            ),
                            last_modified=(
                                self._safe_validator(response.headers.get("last-modified"))
                                or self._safe_validator(cached.get("last_modified"))
                            ),
                        )
                        return CrawledPage(
                            requested_url=page.requested_url,
                            final_url=page.final_url,
                            snapshot=page.snapshot,
                            etag=page.etag,
                            last_modified=page.last_modified,
                            not_modified=True,
                        )
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise WebCrawlError("redirect response did not include Location")
                        if redirect_number >= self.policy.max_redirects:
                            raise WebCrawlLimitError("redirect limit exceeded")
                        redirected = self.validate_target_url(urljoin(current_url, location))
                        if _normalise_host(urlsplit(redirected).hostname or "") != original_host:
                            raise WebCrawlSecurityError(
                                "redirects to a different host are not permitted"
                            )
                        self._assert_authorized_redirect(requested_url, redirected)
                        redirect_parser = self._robots_parsers.get(original_host)
                        if redirect_parser is not None and not redirect_parser.can_fetch(
                            self.policy.user_agent, redirected
                        ):
                            raise WebCrawlPolicyError(
                                "robots.txt does not permit the authorized redirect target"
                            )
                        current_url = redirected
                        continue
                    if response.status_code != 200:
                        raise WebCrawlError(
                            f"GET {requested_url} returned HTTP {response.status_code}"
                        )
                    content_encoding = response.headers.get("content-encoding", "").strip()
                    if content_encoding and content_encoding.casefold() != "identity":
                        raise WebCrawlError(
                            f"GET {requested_url} returned unsupported Content-Encoding"
                        )
                    media_type = (
                        response.headers.get("content-type", "application/octet-stream")
                        .split(";", maxsplit=1)[0]
                        .strip()
                        .casefold()
                    )
                    if media_type not in accepted_media_types:
                        raise WebCrawlError(
                            f"GET {requested_url} returned unsupported media type {media_type!r}"
                        )
                    declared_length = response.headers.get("content-length")
                    if declared_length is not None:
                        try:
                            declared_bytes = int(declared_length)
                        except ValueError as exc:
                            raise WebCrawlError("response Content-Length is invalid") from exc
                        if declared_bytes < 0 or declared_bytes > maximum_bytes:
                            raise WebCrawlLimitError(
                                f"GET {requested_url} exceeds its {maximum_bytes}-byte limit"
                            )
                    body = self._bounded_body(
                        response,
                        maximum_bytes=maximum_bytes,
                        requested_url=requested_url,
                    )
                    return self._persist_page(
                        requested_url=requested_url,
                        final_url=current_url,
                        body=body,
                        media_type=media_type,
                        suffix=suffix,
                        etag=self._safe_validator(response.headers.get("etag")),
                        last_modified=self._safe_validator(response.headers.get("last-modified")),
                    )
            except httpx.HTTPError as exc:
                raise WebCrawlError(f"GET {requested_url} failed: {type(exc).__name__}") from exc
        raise WebCrawlLimitError("redirect limit exceeded")

    def _bounded_body(
        self,
        response: httpx.Response,
        *,
        maximum_bytes: int,
        requested_url: str,
    ) -> bytes:
        body = bytearray()
        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            if len(body) + len(chunk) > maximum_bytes:
                raise WebCrawlLimitError(
                    f"GET {requested_url} exceeded its {maximum_bytes}-byte limit"
                )
            self._charge_total_bytes(len(chunk))
            body.extend(chunk)
        return bytes(body)

    def _charge_total_bytes(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        with self._budget_lock:
            if self._total_bytes + byte_count > self.policy.maximum_total_bytes:
                raise WebCrawlLimitError(
                    f"crawl exceeded its {self.policy.maximum_total_bytes}-byte total limit"
                )
            self._total_bytes += byte_count

    def _persist_page(
        self,
        *,
        requested_url: str,
        final_url: str,
        body: bytes,
        media_type: str,
        suffix: str,
        etag: str | None,
        last_modified: str | None,
    ) -> CrawledPage:
        retrieved_at = datetime.now().astimezone()
        content_sha256 = sha256_bytes(body)
        url_sha256 = sha256_bytes(requested_url.encode("utf-8"))
        page_root = self.pages_root
        page_root.mkdir(parents=True, exist_ok=True)
        file_stem = f"{url_sha256[:32]}-{content_sha256}"
        raw_path = page_root / f"{file_stem}{suffix}"
        while True:
            receipt_id = secrets.token_hex(6)
            metadata_path = (
                page_root / f"{file_stem}-{self.policy.fingerprint[:16]}-{receipt_id}.json"
            )
            if not metadata_path.exists():
                break
        reused = raw_path.exists()
        if reused:
            if raw_path.stat().st_size != len(body) or sha256_bytes(raw_path.read_bytes()) != (
                content_sha256
            ):
                raise WebCrawlError(f"existing raw snapshot is corrupt: {raw_path}")
        else:
            self._write_bytes_atomic(raw_path, body)
        metadata = {
            "schema_version": WEB_RAW_METADATA_SCHEMA_VERSION,
            "source_name": self.policy.source_name,
            "source_url": requested_url,
            "source_url_sha256": url_sha256,
            "final_url": final_url,
            "source_type": "retailer",
            "retrieved_at": retrieved_at.isoformat(),
            "retention_expires_at": (
                retrieved_at + timedelta(days=self.policy.acquisition_authority.retention_days)
            ).isoformat(),
            "content_sha256": content_sha256,
            "byte_count": len(body),
            "media_type": media_type,
            "parser_version": WEB_PRODUCT_PARSER_VERSION,
            "licence_or_access_note": self.policy.licence_or_access_note,
            "policy_fingerprint": self.policy.fingerprint,
            "usage_scope": self.policy.usage_scope.value,
            "acquisition_authority": self.policy.acquisition_authority.to_dict(),
            "data_use_rights": self.policy.rights.to_dict(),
            "etag": etag,
            "last_modified": last_modified,
            "raw_file": raw_path.name,
        }
        self._write_json_atomic(metadata_path, metadata)
        snapshot = RawSnapshot(
            source_name=self.policy.source_name,
            source_url=requested_url,
            source_type="retailer",
            retrieved_at=retrieved_at,
            content_sha256=content_sha256,
            byte_count=len(body),
            media_type=media_type,
            parser_version=WEB_PRODUCT_PARSER_VERSION,
            licence_or_access_note=self.policy.licence_or_access_note,
            path=raw_path,
            metadata_path=metadata_path,
            reused=reused,
        )
        page = CrawledPage(
            requested_url=requested_url,
            final_url=final_url,
            snapshot=snapshot,
            etag=etag,
            last_modified=last_modified,
            not_modified=False,
        )
        with self._state_lock:
            self._state[requested_url] = self._state_entry(page)
        return page

    def _cached_page(
        self,
        requested_url: str,
        cached: Mapping[str, Any],
        *,
        etag: str | None,
        last_modified: str | None,
    ) -> CrawledPage:
        raw_path = self._safe_cached_path(str(cached.get("raw_path", "")))
        metadata_path = self._safe_cached_path(str(cached.get("metadata_path", "")))
        if not raw_path.is_file() or not metadata_path.is_file():
            raise WebCrawlError("conditional response referenced a missing cached snapshot")
        metadata = self._read_json(metadata_path)
        if self._metadata_raw_path(metadata) != raw_path:
            raise WebCrawlSecurityError("cached raw path does not match web-page metadata")
        self._validate_metadata(metadata, raw_path=raw_path)
        if metadata.get("policy_fingerprint") != self.policy.fingerprint:
            raise WebCrawlError("cached web-page policy fingerprint mismatch")
        final_url = self.validate_target_url(str(metadata["final_url"]))
        if final_url != requested_url:
            self._assert_authorized_redirect(requested_url, final_url)
        body = raw_path.read_bytes()
        self._charge_total_bytes(len(body))
        refreshed = self._persist_page(
            requested_url=requested_url,
            final_url=final_url,
            body=body,
            media_type=str(metadata["media_type"]),
            suffix=raw_path.suffix,
            etag=etag,
            last_modified=last_modified,
        )
        return CrawledPage(
            requested_url=refreshed.requested_url,
            final_url=refreshed.final_url,
            snapshot=refreshed.snapshot,
            etag=refreshed.etag,
            last_modified=refreshed.last_modified,
            not_modified=True,
        )

    def _parse_robots(self, host: str, page: CrawledPage) -> RobotFileParser:
        try:
            text = page.snapshot.path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WebCrawlPolicyError(f"robots.txt for {host} is not valid UTF-8") from exc
        if re.search(r"(?im)^\s*user-agent\s*:", text) is None:
            raise WebCrawlPolicyError(
                f"robots.txt for {host} has no User-agent directive; failing closed"
            )
        parser = RobotFileParser()
        parser.set_url(page.final_url)
        parser.parse(text.splitlines())
        return parser

    def _robots_interval(self, host: str, parser: RobotFileParser) -> float:
        intervals = [1.0 / self.policy.requests_per_second]
        crawl_delay = parser.crawl_delay(self.policy.user_agent)
        if crawl_delay is not None:
            intervals.append(float(crawl_delay))
        request_rate = parser.request_rate(self.policy.user_agent)
        if request_rate is not None:
            if request_rate.requests <= 0 or request_rate.seconds < 0:
                raise WebCrawlPolicyError(
                    f"robots.txt for {host} contains an invalid Request-rate directive"
                )
            intervals.append(request_rate.seconds / request_rate.requests)
        interval = max(intervals)
        if interval > _MAX_ROBOTS_INTERVAL_SECONDS:
            raise WebCrawlPolicyError(
                f"robots.txt for {host} requires an extreme request interval; failing closed"
            )
        return interval

    def _parse_pages(
        self,
        pages: Sequence[CrawledPage],
        *,
        run_sha256: str,
        robots_hashes: Mapping[str, str],
        terms_page: CrawledPage,
        terms_post_page: CrawledPage,
    ) -> ParsedBatch:
        batch = ParsedBatch(source_name=self.policy.source_name, snapshot_sha256=run_sha256)
        seen_listing_ids: set[str] = set()
        jsonld_blocks = 0
        product_nodes = 0
        offers_seen = 0
        rejections_seen = 0
        parse_events = 0

        def consume_parse_events(count: int, kind: str) -> None:
            nonlocal parse_events
            if parse_events + count > self.policy.maximum_parse_events:
                raise WebCrawlLimitError(
                    "crawl exceeded its "
                    f"{self.policy.maximum_parse_events}-event parse limit while parsing {kind}"
                )
            parse_events += count

        def reject(record: dict[str, object]) -> None:
            nonlocal rejections_seen
            if rejections_seen >= self.policy.maximum_rejections:
                raise WebCrawlLimitError(
                    f"crawl exceeded its {self.policy.maximum_rejections}-rejection limit"
                )
            rejections_seen += 1
            batch.rejected.append(record)

        for page in pages:
            try:
                html = page.snapshot.path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                reject(rejected_record(page.requested_url, "invalid_html_encoding", error=str(exc)))
                continue
            extractor = _JSONLDExtractor()
            extractor.feed(html)
            if jsonld_blocks + len(extractor.blocks) > self.policy.maximum_jsonld_blocks:
                raise WebCrawlLimitError(
                    f"crawl exceeded its {self.policy.maximum_jsonld_blocks}-JSON-LD-block limit"
                )
            jsonld_blocks += len(extractor.blocks)
            consume_parse_events(len(extractor.blocks), "JSON-LD blocks")
            page_product_count = 0
            page_offers_seen = 0
            for block_number, block in enumerate(extractor.blocks, start=1):
                try:
                    document = json.loads(self._clean_jsonld(block))
                except (json.JSONDecodeError, ValueError) as exc:
                    reject(
                        rejected_record(
                            f"{page.requested_url}#jsonld-{block_number}",
                            "invalid_jsonld",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                try:
                    nodes = self._jsonld_nodes(document)
                except WebCrawlLimitError as exc:
                    reject(
                        rejected_record(
                            f"{page.requested_url}#jsonld-{block_number}",
                            "jsonld_node_limit",
                            error=str(exc),
                        )
                    )
                    continue
                consume_parse_events(len(nodes), "JSON-LD nodes")
                for node_number, product in enumerate(nodes, start=1):
                    if not self._has_type(product, "Product"):
                        continue
                    page_product_count += 1
                    if product_nodes >= self.policy.maximum_product_nodes:
                        raise WebCrawlLimitError(
                            "crawl exceeded its "
                            f"{self.policy.maximum_product_nodes}-product-node limit"
                        )
                    product_nodes += 1
                    offers = product.get("offers")
                    offer_nodes = offers if isinstance(offers, list) else [offers]
                    if not offers:
                        reject(
                            rejected_record(
                                f"{page.requested_url}#product-{node_number}",
                                "product_has_no_offer",
                            )
                        )
                        continue
                    if page_offers_seen + len(offer_nodes) > self.policy.maximum_offers_per_page:
                        raise WebCrawlLimitError(
                            f"page exceeded its {self.policy.maximum_offers_per_page}-offer limit"
                        )
                    if offers_seen + len(offer_nodes) > self.policy.normalized_record_limit:
                        raise WebCrawlLimitError(
                            "crawl exceeded its "
                            f"{self.policy.normalized_record_limit}-offer/record limit"
                        )
                    page_offers_seen += len(offer_nodes)
                    offers_seen += len(offer_nodes)
                    consume_parse_events(len(offer_nodes), "offers")
                    if len(offer_nodes) > 1:
                        supported_offers = [
                            offer
                            for offer in offer_nodes
                            if isinstance(offer, Mapping)
                            and self._has_type(offer, "Offer")
                            and not self._has_type(offer, "AggregateOffer")
                        ]
                        offer_identities = [
                            self._offer_level_identity(offer) for offer in supported_offers
                        ]
                        if supported_offers and (
                            any(identity is None for identity in offer_identities)
                            or len(set(offer_identities)) != len(offer_identities)
                        ):
                            reject(
                                rejected_record(
                                    f"{page.requested_url}#product-{node_number}",
                                    "ambiguous_multiple_offers",
                                )
                            )
                            continue
                    for offer_number, offer in enumerate(offer_nodes, start=1):
                        record_id = (
                            f"{page.requested_url}#product-{node_number}-offer-{offer_number}"
                        )
                        if (
                            not isinstance(offer, Mapping)
                            or not self._has_type(offer, "Offer")
                            or self._has_type(offer, "AggregateOffer")
                        ):
                            reject(rejected_record(record_id, "unsupported_offer_shape"))
                            continue
                        try:
                            record = self._normalise_offer(
                                product=product,
                                offer=offer,
                                page=page,
                                terms_snapshot_sha256=terms_page.snapshot.content_sha256,
                                terms_post_snapshot_sha256=(
                                    terms_post_page.snapshot.content_sha256
                                ),
                                multiple_offers=len(offer_nodes) > 1,
                            )
                        except (InvalidOperation, TypeError, ValueError, WebCrawlError) as exc:
                            reject(
                                rejected_record(
                                    record_id,
                                    "invalid_product_offer",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                            )
                            continue
                        source_listing_id = str(record["source_record_id"])
                        if source_listing_id in seen_listing_ids:
                            reject(
                                rejected_record(
                                    source_listing_id,
                                    "duplicate_source_listing_id",
                                    page_url=page.requested_url,
                                )
                            )
                            continue
                        seen_listing_ids.add(source_listing_id)
                        batch.records.append(record)
            if page_product_count == 0:
                reject(rejected_record(page.requested_url, "no_schemaorg_product"))
        batch.statistics = {
            "retailer": self.policy.retailer,
            "policy_fingerprint": self.policy.fingerprint,
            "usage_scope": self.policy.usage_scope.value,
            "development_only": self.policy.development_only,
            "allowed_hosts": list(self.policy.allowed_hosts),
            "pages_requested": len(pages),
            "jsonld_blocks": jsonld_blocks,
            "product_nodes": product_nodes,
            "offers_seen": offers_seen,
            "parse_events": parse_events,
            "rejections_seen": rejections_seen,
            "maximum_offers_per_page": self.policy.maximum_offers_per_page,
            "maximum_records": self.policy.normalized_record_limit,
            "maximum_jsonld_blocks": self.policy.maximum_jsonld_blocks,
            "maximum_product_nodes": self.policy.maximum_product_nodes,
            "maximum_rejections": self.policy.maximum_rejections,
            "maximum_parse_events": self.policy.maximum_parse_events,
            "unique_source_listing_ids": len(seen_listing_ids),
            "terms_url": self.policy.terms_url,
            "terms_selector": self.policy.terms_selector,
            "terms_snapshot_sha256": terms_page.snapshot.content_sha256,
            "terms_post_snapshot_sha256": terms_post_page.snapshot.content_sha256,
            "terms_receipt_sha256": self._receipt_sha256(terms_page),
            "terms_post_receipt_sha256": self._receipt_sha256(terms_post_page),
            "terms_canonical_sha256": self.policy.canonical_terms_sha256,
            "terms_verified_on": self.policy.terms_verified_on.isoformat(),
            "robots_sha256_by_host": dict(sorted(robots_hashes.items())),
            "robots_compliance_checked": True,
            "acquisition_authority": self.policy.acquisition_authority.to_dict(),
            "training_eligible": self.policy.training_eligible,
            "published_claims_eligible": self.policy.published_claims_eligible,
            "data_use_rights": self.policy.rights.to_dict(),
        }
        return batch

    def _normalise_offer(
        self,
        *,
        product: Mapping[str, Any],
        offer: Mapping[str, Any],
        page: CrawledPage,
        terms_snapshot_sha256: str,
        terms_post_snapshot_sha256: str,
        multiple_offers: bool,
    ) -> dict[str, Any]:
        title = self._text(product.get("name"), "product.name")
        raw_currency = offer.get("priceCurrency") or self._nested(
            offer, "priceSpecification", "priceCurrency"
        )
        currency = self._text(
            raw_currency,
            "offer.priceCurrency",
        ).upper()
        if re.fullmatch(r"[A-Z]{3}", currency) is None:
            raise ValueError(f"invalid ISO currency: {currency!r}")
        if currency not in self.policy.allowed_currencies:
            raise ValueError(f"currency {currency!r} is outside source policy")
        raw_price = offer.get("price") or self._nested(offer, "priceSpecification", "price")
        base_price = self._money(raw_price, positive=True, expected_currency=currency)
        shipping_value = self._shipping_value(offer, currency=currency)
        if shipping_value is not None:
            shipping_known = True
            shipping_basis = "schema_org_shipping_rate"
            shipping_price = self._money(
                shipping_value,
                positive=False,
                expected_currency=currency,
            )
        elif self.policy.usage_scope == WebUsageScope.INTERNAL_RESEARCH:
            shipping_known = False
            shipping_basis = "unknown_development_only"
            shipping_price = Decimal("0.00")
        elif self.policy.unknown_shipping == UnknownShippingPolicy.ZERO_CONFIRMED:
            shipping_known = True
            shipping_basis = "policy_confirmed_zero"
            shipping_price = Decimal("0.00")
        else:
            raise ValueError("shipping price is unknown and source policy requires rejection")
        listing_url = self.validate_target_url(str(offer.get("url") or page.final_url))
        category = self.policy.url_categories[page.requested_url]
        same_mapped_path = any(
            _same_url_path(listing_url, mapped_url)
            for mapped_url in (page.requested_url, page.final_url)
        )
        if not same_mapped_path and self.policy.url_categories.get(listing_url) != category:
            raise WebCrawlPolicyError(
                "offer URL requires an explicit matching component-category mapping"
            )
        condition = self._condition(offer.get("itemCondition"))
        if (
            self.policy.usage_scope == WebUsageScope.PRODUCTION_CATALOG
            and condition == ListingCondition.UNKNOWN
        ):
            raise ValueError("unknown item condition is not permitted for production")
        if not self.policy.allow_non_new and condition in {
            ListingCondition.OPEN_BOX,
            ListingCondition.REFURBISHED,
            ListingCondition.USED,
        }:
            raise ValueError(f"non-new condition is outside source policy: {condition.value}")
        stock_status = self._stock_status(offer.get("availability"))
        identifiers = self._identifiers(product, offer)
        seller = offer.get("seller")
        seller_name = (
            str(seller.get("name") or "").strip()
            if isinstance(seller, Mapping)
            else str(seller or "").strip()
        ) or None
        seller_identity = self._seller_identity(seller)
        offer_identity = self._offer_level_identity(offer)
        if multiple_offers and offer_identity is None:
            raise ValueError("multiple offers require unique offer-level identifiers")
        source_listing_id = stable_identifier(
            "web_listing",
            self.policy.source_name,
            self.policy.retailer,
            seller_identity,
            listing_url,
            *(offer_identity or ("listing_url", listing_url, seller_identity)),
            length=32,
        )
        product_id = stable_identifier("unmatched_product", self.policy.retailer, source_listing_id)
        listing_id = stable_identifier(
            "listing", self.policy.retailer, source_listing_id, length=32
        )
        observed_at = page.snapshot.retrieved_at
        listing = RetailerListing(
            listing_id=listing_id,
            product_id=product_id,
            retailer=self.policy.retailer,
            source_listing_id=source_listing_id,
            title=title,
            condition=condition,
            currency=currency,
            base_price=base_price,
            shipping_price=shipping_price,
            stock_status=stock_status,
            seller_name=seller_name,
            listing_url=listing_url,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
        )
        price_snapshot = PriceSample(
            snapshot_id=stable_identifier(
                "price",
                listing_id,
                observed_at.isoformat(),
                base_price,
                shipping_price,
                stock_status.value,
                length=32,
            ),
            listing_id=listing_id,
            observed_at=observed_at,
            base_price=base_price,
            shipping_price=shipping_price,
            stock_status=stock_status,
            promotion_text=self._optional_text(offer.get("description")),
        )
        raw_record_bytes = json.dumps(
            {"product": product, "offer": offer},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        host = _normalise_host(urlsplit(page.final_url).hostname or "")
        return {
            "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
            "record_type": "retailer_listing",
            "source_record_id": source_listing_id,
            "archive_snapshot_sha256": page.snapshot.content_sha256,
            "raw_record_sha256": sha256_bytes(raw_record_bytes),
            "training_eligible": self.policy.training_eligible,
            "published_claims_eligible": self.policy.published_claims_eligible,
            "development_only": self.policy.development_only,
            "data_use_rights": self.policy.rights.to_dict(),
            "provenance": {
                "source_name": self.policy.source_name,
                "source_url": listing_url,
                "source_type": "retailer",
                "retrieved_at": observed_at.isoformat(),
                "parser_version": WEB_PRODUCT_PARSER_VERSION,
                "licence_or_access_note": self.policy.licence_or_access_note,
                "extraction_confidence": 0.9,
                "raw_page_url": page.requested_url,
                "raw_page_final_url": page.final_url,
                "raw_page_sha256": page.snapshot.content_sha256,
                "raw_page_receipt_sha256": self._receipt_sha256(page),
            },
            "normalisation_metadata": {
                "extraction_method": "schema_org_jsonld",
                "canonical_mapping_status": "unmatched",
                "category": category.value,
                "identifiers": identifiers,
                "shipping_price_known": shipping_known,
                "shipping_price_basis": shipping_basis,
                "terms_url": self.policy.terms_url,
                "terms_selector": self.policy.terms_selector,
                "canonical_terms_sha256": self.policy.canonical_terms_sha256,
                "raw_terms_snapshot_sha256": terms_snapshot_sha256,
                "raw_terms_post_snapshot_sha256": terms_post_snapshot_sha256,
                "terms_verified_on": self.policy.terms_verified_on.isoformat(),
                "robots_host": host,
                "robots_compliance_checked": True,
                "usage_scope": self.policy.usage_scope.value,
                "development_only": self.policy.development_only,
                "acquisition_authority_reference": (
                    self.policy.acquisition_authority.authority_reference
                ),
            },
            "data": {
                "listing": listing.model_dump(mode="json"),
                "price_snapshot": price_snapshot.model_dump(mode="json"),
            },
        }

    @staticmethod
    def _clean_jsonld(value: str) -> str:
        cleaned = value.strip()
        if cleaned.startswith("<!--") and cleaned.endswith("-->"):
            cleaned = cleaned[4:-3].strip()
        if not cleaned:
            raise ValueError("empty JSON-LD block")
        return cleaned

    @staticmethod
    def _jsonld_nodes(document: object) -> list[Mapping[str, Any]]:
        pending: list[object] = [document]
        nodes: list[Mapping[str, Any]] = []
        visited = 0
        while pending:
            item = pending.pop()
            visited += 1
            if visited > 10_000:
                raise WebCrawlLimitError("JSON-LD document exceeds 10,000 nodes")
            if isinstance(item, list):
                pending.extend(reversed(item))
            elif isinstance(item, Mapping):
                nodes.append(item)
                graph = item.get("@graph")
                if isinstance(graph, list | Mapping):
                    pending.append(graph)
        return nodes

    @staticmethod
    def _has_type(node: Mapping[str, Any], expected: str) -> bool:
        raw_type = node.get("@type")
        values = raw_type if isinstance(raw_type, list) else [raw_type]
        return any(str(value).rsplit("/", maxsplit=1)[-1] == expected for value in values)

    @staticmethod
    def _text(value: object, field_name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field_name} is required")
        return text

    @staticmethod
    def _optional_text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _money(
        value: object,
        *,
        positive: bool,
        expected_currency: str,
    ) -> Decimal:
        match = _MONEY_PATTERN.fullmatch(str(value or ""))
        if match is None:
            raise ValueError("offer price is required")
        if match.group("prefix") and match.group("suffix"):
            raise ValueError("offer price cannot contain two currency markers")
        marker = match.group("prefix") or match.group("suffix")
        if marker:
            normalized_marker = marker.upper()
            marker_currency = "SGD" if normalized_marker == "S$" else normalized_marker
            if marker_currency != "$" and marker_currency != expected_currency:
                raise ValueError(
                    f"money currency marker {marker_currency!r} contradicts declared "
                    f"currency {expected_currency!r}"
                )
        amount = Decimal(match.group("amount").replace(",", "")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        if amount < 0 or (positive and amount == 0):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"money must be {qualifier}")
        return amount

    def _shipping_value(
        self,
        offer: Mapping[str, Any],
        *,
        currency: str,
    ) -> object | None:
        details = offer.get("shippingDetails")
        if details is None:
            return None
        candidates = details if isinstance(details, list) else [details]
        sg_rates: list[Decimal] = []
        saw_rate = False
        ambiguous = False
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                ambiguous = True
                continue
            countries = self._shipping_countries(candidate.get("shippingDestination"))
            does_not_ship = candidate.get("doesNotShip")
            if does_not_ship is not None and type(does_not_ship) is not bool:
                ambiguous = True
                continue
            if does_not_ship is True:
                if "SG" in countries:
                    raise ValueError("offer explicitly does not ship to Singapore")
                if not countries:
                    ambiguous = True
                continue
            rate = candidate.get("shippingRate")
            if rate is None:
                continue
            saw_rate = True
            if not isinstance(rate, Mapping) or rate.get("value") is None:
                ambiguous = True
                continue
            rate_currency = (
                str(rate.get("currency") or rate.get("priceCurrency") or "").strip().upper()
            )
            if not rate_currency or not countries:
                ambiguous = True
                continue
            if "SG" not in countries:
                continue
            if rate_currency != currency:
                ambiguous = True
                continue
            sg_rates.append(
                self._money(
                    rate["value"],
                    positive=False,
                    expected_currency=rate_currency,
                )
            )
        if ambiguous:
            if self.policy.usage_scope == WebUsageScope.INTERNAL_RESEARCH:
                return None
            raise ValueError("shipping rate has ambiguous currency, value, or destination")
        if not sg_rates:
            if saw_rate and self.policy.usage_scope == WebUsageScope.PRODUCTION_CATALOG:
                raise ValueError("offer has no shipping rate applicable to Singapore")
            return None
        if len(set(sg_rates)) != 1:
            if self.policy.usage_scope == WebUsageScope.INTERNAL_RESEARCH:
                return None
            raise ValueError("offer has multiple conflicting Singapore shipping rates")
        return sg_rates[0]

    @staticmethod
    def _shipping_countries(value: object) -> set[str]:
        pending = value if isinstance(value, list) else [value]
        countries: set[str] = set()
        for destination in pending:
            if isinstance(destination, Mapping):
                raw_country = destination.get("addressCountry")
                if isinstance(raw_country, Mapping):
                    raw_country = (
                        raw_country.get("name")
                        or raw_country.get("alternateName")
                        or raw_country.get("@id")
                    )
            else:
                raw_country = destination
            token = str(raw_country or "").strip().rstrip("/").rsplit("/", maxsplit=1)[-1]
            normalized = token.casefold()
            if normalized in {"sg", "sgp", "singapore"}:
                countries.add("SG")
            elif token:
                countries.add(token.upper())
        return countries

    @staticmethod
    def _nested(value: Mapping[str, Any], *path: str) -> object | None:
        current: object = value
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    @staticmethod
    def _stock_status(value: object) -> StockStatus:
        token = str(value or "").rstrip("/").rsplit("/", maxsplit=1)[-1].casefold()
        return {
            "instock": StockStatus.IN_STOCK,
            "limitedavailability": StockStatus.IN_STOCK,
            "onlineonly": StockStatus.IN_STOCK,
            "outofstock": StockStatus.OUT_OF_STOCK,
            "soldout": StockStatus.OUT_OF_STOCK,
            "backorder": StockStatus.BACKORDER,
            "preorder": StockStatus.PREORDER,
            "presale": StockStatus.PREORDER,
        }.get(token, StockStatus.UNKNOWN)

    @staticmethod
    def _condition(value: object) -> ListingCondition:
        token = str(value or "").rstrip("/").rsplit("/", maxsplit=1)[-1].casefold()
        if token == "damagedcondition":
            raise ValueError("damaged item condition is never eligible")
        return {
            "newcondition": ListingCondition.NEW,
            "usedcondition": ListingCondition.USED,
            "refurbishedcondition": ListingCondition.REFURBISHED,
        }.get(token, ListingCondition.UNKNOWN)

    def _offer_level_identity(self, offer: Mapping[str, Any]) -> tuple[str, str, str] | None:
        seller_identity = self._seller_identity(offer.get("seller"))
        for field_name in ("sku", "@id", "url"):
            value = str(offer.get(field_name) or "").strip()
            if value:
                return field_name, value, seller_identity
        return None

    def _seller_identity(self, seller: object) -> str:
        values: list[str] = []
        if isinstance(seller, Mapping):
            for field_name in ("@id", "identifier", "url", "name"):
                raw_value = seller.get(field_name)
                if isinstance(raw_value, Mapping):
                    raw_value = (
                        raw_value.get("value") or raw_value.get("@id") or raw_value.get("name")
                    )
                text = self._normalized_identity_text(raw_value)
                if text:
                    values.append(f"{field_name}:{text}")
        else:
            text = self._normalized_identity_text(seller)
            if text:
                values.append(f"name:{text}")
        if not values:
            values.append(f"retailer:{self._normalized_identity_text(self.policy.retailer)}")
        return "|".join(values)

    @staticmethod
    def _normalized_identity_text(value: object) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()

    @staticmethod
    def _identifiers(product: Mapping[str, Any], offer: Mapping[str, Any]) -> dict[str, str | None]:
        gtin = next(
            (
                str(product[key]).strip()
                for key in ("gtin", "gtin14", "gtin13", "gtin12", "gtin8")
                if product.get(key)
            ),
            None,
        )
        return {
            "offer_sku": str(offer.get("sku") or "").strip() or None,
            "sku": str(product.get("sku") or "").strip() or None,
            "mpn": str(product.get("mpn") or "").strip() or None,
            "gtin": gtin,
        }

    def _run_sha256(
        self,
        *,
        urls: Sequence[str],
        pages: Sequence[CrawledPage],
        robots_hashes: Mapping[str, str],
        terms_pre_raw_sha256: str,
        terms_post_raw_sha256: str,
        terms_canonical_sha256: str,
        terms_page: CrawledPage,
        terms_post_page: CrawledPage,
    ) -> str:
        payload = {
            "policy_fingerprint": self.policy.fingerprint,
            "urls": list(urls),
            "pages": [
                {
                    "content_sha256": page.snapshot.content_sha256,
                    "receipt_sha256": self._receipt_sha256(page),
                    "retrieved_at": page.snapshot.retrieved_at.isoformat(),
                }
                for page in pages
            ],
            "robots": dict(sorted(robots_hashes.items())),
            "terms_pre_raw": terms_pre_raw_sha256,
            "terms_post_raw": terms_post_raw_sha256,
            "terms_pre_receipt": self._receipt_sha256(terms_page),
            "terms_post_receipt": self._receipt_sha256(terms_post_page),
            "terms_canonical": terms_canonical_sha256,
        }
        return sha256_bytes(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )

    @staticmethod
    def _receipt_sha256(page: CrawledPage) -> str:
        return sha256_bytes(page.snapshot.metadata_path.read_bytes())

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if not self.state_path.exists():
            return {}
        payload = self._read_json(self.state_path)
        if payload.get("schema_version") != WEB_CRAWL_CACHE_SCHEMA_VERSION:
            raise WebCrawlError("unsupported web crawl cache schema")
        if payload.get("source_name") != self.policy.source_name:
            raise WebCrawlError("web crawl cache source does not match policy")
        if payload.get("policy_fingerprint") != self.policy.fingerprint:
            return {}
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise WebCrawlError("web crawl cache entries must be an object")
        return {
            str(url): dict(entry) for url, entry in entries.items() if isinstance(entry, Mapping)
        }

    def _retain_active_cache_entries(self) -> None:
        """Drop expired cache references without deleting governed raw evidence.

        The independent retention engine is the only destructive path.  Acquisition only
        decides whether a cached response remains usable for conditional retrieval.
        """

        self._state = {
            url: cached
            for url, cached in self._state.items()
            if self._cache_entry_is_active(cached)
        }

    def _write_state(self) -> None:
        self._write_json_atomic(
            self.state_path,
            {
                "schema_version": WEB_CRAWL_CACHE_SCHEMA_VERSION,
                "source_name": self.policy.source_name,
                "policy_fingerprint": self.policy.fingerprint,
                "entries": dict(sorted(self._state.items())),
            },
        )

    def _state_entry(self, page: CrawledPage) -> dict[str, Any]:
        return {
            "final_url": page.final_url,
            "etag": page.etag,
            "last_modified": page.last_modified,
            "content_sha256": page.snapshot.content_sha256,
            "raw_path": str(page.snapshot.path.resolve().relative_to(self.raw_root.resolve())),
            "metadata_path": str(
                page.snapshot.metadata_path.resolve().relative_to(self.raw_root.resolve())
            ),
        }

    def _safe_cached_path(self, relative_path: str) -> Path:
        if not relative_path or Path(relative_path).is_absolute():
            raise WebCrawlSecurityError("web crawl cache path must be relative")
        return self._safe_page_path(self.raw_root / relative_path)

    def _resolved_pages_root(self) -> Path:
        raw_root = self.raw_root.resolve()
        source_root = self.source_root.resolve()
        if source_root == raw_root or raw_root not in source_root.parents:
            raise WebCrawlSecurityError("web source root escaped the raw-data root")
        page_root = self.pages_root.resolve()
        if page_root == source_root or source_root not in page_root.parents:
            raise WebCrawlSecurityError("web page store escaped the policy source root")
        return page_root

    def _safe_page_path(self, path: Path) -> Path:
        candidate = path.resolve()
        page_root = self._resolved_pages_root()
        if candidate == page_root or page_root not in candidate.parents:
            raise WebCrawlSecurityError("web page path escaped the policy source page store")
        return candidate

    def _metadata_raw_path(self, metadata: Mapping[str, Any]) -> Path:
        raw_file = metadata.get("raw_file")
        if not isinstance(raw_file, str) or not raw_file:
            raise WebCrawlError("raw web-page metadata has no raw file")
        if _RAW_PAGE_FILE_PATTERN.fullmatch(raw_file) is None:
            raise WebCrawlSecurityError("raw web-page metadata contains an unsafe raw file")
        return self._safe_page_path(self.pages_root / raw_file)

    @staticmethod
    def _metadata_expiry(metadata: Mapping[str, Any]) -> datetime:
        raw_expiry = metadata.get("retention_expires_at")
        if not isinstance(raw_expiry, str):
            raise WebCrawlError("cached web-page metadata has no retention expiry")
        try:
            expiry = datetime.fromisoformat(raw_expiry)
        except ValueError as exc:
            raise WebCrawlError("cached web-page retention expiry is invalid") from exc
        if expiry.tzinfo is None:
            raise WebCrawlError("cached web-page retention expiry must be timezone aware")
        return expiry

    def _cache_entry_is_active(self, cached: Mapping[str, Any]) -> bool:
        raw_path = self._safe_cached_path(str(cached.get("raw_path", "")))
        metadata_path = self._safe_cached_path(str(cached.get("metadata_path", "")))
        if not metadata_path.is_file():
            return False
        metadata = self._read_json(metadata_path)
        if self._metadata_raw_path(metadata) != raw_path:
            raise WebCrawlSecurityError("web crawl cache raw path does not match its metadata")
        self._validate_metadata(metadata, raw_path=raw_path)
        return datetime.now().astimezone() <= self._metadata_expiry(metadata)

    @staticmethod
    def _safe_validator(value: object) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        if len(text) > 512 or any(char in text for char in "\r\n"):
            raise WebCrawlError("unsafe HTTP cache validator")
        return text

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WebCrawlError(f"invalid crawler metadata: {path}") from exc
        if not isinstance(payload, dict):
            raise WebCrawlError(f"crawler metadata must be an object: {path}")
        return payload

    @staticmethod
    def _validate_metadata(payload: Mapping[str, Any], *, raw_path: Path) -> None:
        if payload.get("schema_version") != WEB_RAW_METADATA_SCHEMA_VERSION:
            raise WebCrawlError("unsupported raw web-page metadata schema")
        content_sha256 = str(payload.get("content_sha256", ""))
        if _SHA256_PATTERN.fullmatch(content_sha256) is None:
            raise WebCrawlError("raw web-page metadata has an invalid SHA-256")
        if _SHA256_PATTERN.fullmatch(str(payload.get("source_url_sha256", ""))) is None:
            raise WebCrawlError("raw web-page metadata has an invalid URL SHA-256")
        if not raw_path.is_file():
            raise WebCrawlError("raw web-page file is missing")
        if raw_path.stat().st_size != int(payload.get("byte_count", -1)):
            raise WebCrawlError("raw web-page byte count does not match metadata")
        if sha256_bytes(raw_path.read_bytes()) != content_sha256:
            raise WebCrawlError("raw web-page content hash does not match metadata")

    @staticmethod
    def _write_bytes_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=".write.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                temporary_path = Path(handle.name)
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        body = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        WebProductCrawlerAdapter._write_bytes_atomic(path, body)


__all__ = [
    "WEB_PRODUCT_PARSER_VERSION",
    "CrawledPage",
    "WebAcquisitionAuthority",
    "WebCrawlError",
    "WebCrawlLimitError",
    "WebCrawlPolicyError",
    "WebCrawlResult",
    "WebCrawlSecurityError",
    "WebProductCrawlerAdapter",
    "WebSourcePolicy",
    "WebUsageScope",
    "calculate_canonical_terms_sha256",
]
