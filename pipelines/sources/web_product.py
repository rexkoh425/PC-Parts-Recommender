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

from pc_build_recommender.domain.enums import ComponentKind, ListingCondition, StockState
from pc_build_recommender.domain.models import PriceSample, RetailerOffering
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION, stable_identifier
from pipelines.sources.base import ParseResult, FetchedSnapshot, rejected_record, sha256_bytes
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
    snapshot: FetchedSnapshot
    etag: str | None
    last_modified: str | None
    not_modified: bool


@dataclass(frozen=True, slots=True)
class WebCrawlResult:
    batch: ParseResult
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

# TODO: rest of this module still to come.
