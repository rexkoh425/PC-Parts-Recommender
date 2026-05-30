from __future__ import annotations

import json
import ssl
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date

import httpcore
import httpx
import pytest
from pipelines.sources.web_product import (
    UnknownShippingPolicy,
    WebAcquisitionAuthority,
    WebCrawlLimitError,
    WebCrawlPolicyError,
    WebCrawlSecurityError,
    WebProductCrawlerAdapter,
    WebSourcePolicy,
    WebUsageScope,
    _PinnedDNSBackend,
    _PinnedHTTPTransport,
    calculate_canonical_terms_sha256,
)
from scripts.fetch_open_data import main as fetch_open_data_main

from pc_build_recommender.data_rights import DataUse, DataUseRights
from pc_build_recommender.domain.enums import ComponentCategory

HOST = "shop.example.test"
PRODUCT_URL = f"https://{HOST}/products/fixture-gpu"
TERMS_URL = f"https://{HOST}/terms/data-use"
TERMS_BODY = b"""
<html><body><header>Dynamic storefront</header>
<div class="reviewed-terms">Automated retrieval and internal processing are permitted.</div>
<script nonce="first">window.theme = 'light';</script></body></html>
"""
ROBOTS_BODY = b"User-agent: *\nAllow: /\n"


class VirtualClock:
    def __init__(self) -> None:
        self.value = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds


class RecordingNetworkStream(httpcore.MockStream):
    def __init__(self) -> None:
        super().__init__([b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK"])
        self.writes: list[bytes] = []
        self.tls_contexts: list[ssl.SSLContext] = []
        self.tls_server_names: list[str | None] = []

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        del timeout
        self.writes.append(buffer)

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        del timeout
        self.tls_contexts.append(ssl_context)
        self.tls_server_names.append(server_hostname)
        return self

    def get_extra_info(self, info: str):
        if info == "server_addr":
            return ("1.1.1.1", 443)
        return super().get_extra_info(info)


class RecordingNetworkBackend(httpcore.NetworkBackend):
    def __init__(self, stream: httpcore.NetworkStream | None = None) -> None:
        self.stream = stream
        self.tcp_hosts: list[str] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        del port, timeout, local_address, socket_options
        self.tcp_hosts.append(host)
        if self.stream is None:
            raise AssertionError("socket backend must not be reached")
        return self.stream

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        del path, timeout, socket_options
        raise AssertionError("Unix socket backend must not be reached")


def _rights(*, enabled: bool = True) -> DataUseRights:
    return DataUseRights(
        contract_reference="fixture-web-policy-v1",
        contract_version_url=TERMS_URL,
        consent_effective_on=date(2020, 1, 1),
        consent_expires_on=None,
        retention_days=30,
        deletion_required_on_termination=True,
        deletion_sla_days=7,
        territories=("SG",),
        may_display=enabled,
        may_cache=enabled,
        may_store_history=enabled,
        may_redistribute=False,
        may_embed=False,
        may_train=False,
        may_derive=enabled,
    )


def _authority(*, internal: bool = True) -> WebAcquisitionAuthority:
    return WebAcquisitionAuthority(
        authority_reference="fixture-legal-review-2026-01",
        reviewed_on=date(2020, 1, 1),
        expires_on=None,
        permits_automated_retrieval=True,
        permits_raw_snapshot_storage=True,
        permits_internal_analysis=internal,
        retention_days=30,
        deletion_required=True,
    )


def _policy(
    *,
    usage_scope: WebUsageScope = WebUsageScope.PRODUCTION_CATALOG,
    rights: DataUseRights | None = None,
    allowed_hosts: tuple[str, ...] = (HOST,),
    terms_sha256: str | None = None,
    url_categories: dict[str, ComponentCategory] | None = None,
    unknown_shipping: UnknownShippingPolicy = UnknownShippingPolicy.REJECT,
) -> WebSourcePolicy:
    return WebSourcePolicy(
        source_name="fixture_web",
        retailer="Fixture Retailer",
        allowed_hosts=allowed_hosts,
        terms_url=TERMS_URL,
        terms_selector=".reviewed-terms",
        canonical_terms_sha256=terms_sha256
        or calculate_canonical_terms_sha256(
            TERMS_BODY,
            media_type="text/html",
            selector=".reviewed-terms",
        ),
        terms_verified_on=date(2020, 1, 1),
        licence_or_access_note="Fixture legal review; robots is compliance, not a licence.",
        rights=rights or _rights(enabled=usage_scope == WebUsageScope.PRODUCTION_CATALOG),
        acquisition_authority=_authority(),
        url_categories=url_categories or {PRODUCT_URL: ComponentCategory.GPU},
        usage_scope=usage_scope,
        unknown_shipping=unknown_shipping,
        requests_per_second=10,
        max_concurrency=2,
        max_pages=2,
        maximum_page_bytes=16 * 1024,
        maximum_total_bytes=64 * 1024,
        maximum_control_bytes=4 * 1024,
        timeout_seconds=5,
    )


def _product_html(
    *,
    offer_url: str = PRODUCT_URL,
    include_shipping: bool = True,
    item_condition: str | None = "https://schema.org/NewCondition",
    offer_count: int = 1,
    offer_level_ids: bool = True,
    price: str = "S$ 1,299.90",
    price_currency: str = "SGD",
    shipping_country: str = "SG",
    shipping_currency: str = "SGD",
    shipping_price: str = "15.00",
    does_not_ship: object | None = None,
) -> bytes:
    offer = {
        "@type": "Offer",
        "url": offer_url,
        "priceCurrency": price_currency,
        "price": price,
        "availability": "https://schema.org/InStock",
        "seller": {"@type": "Organization", "name": "Fixture Retailer"},
    }
    if item_condition is not None:
        offer["itemCondition"] = item_condition
    if include_shipping:
        offer["shippingDetails"] = {
            "@type": "OfferShippingDetails",
            "shippingDestination": {
                "@type": "DefinedRegion",
                "addressCountry": shipping_country,
            },
            "shippingRate": {
                "@type": "MonetaryAmount",
                "currency": shipping_currency,
                "value": shipping_price,
            },
        }
        if does_not_ship is not None:
            offer["shippingDetails"]["doesNotShip"] = does_not_ship
    payload = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "Fixture RTX GPU 16 GB",
        "sku": "FIXTURE-GPU-16",
        "mpn": "MPN-FIXTURE-16",
        "brand": {"@type": "Brand", "name": "Fixture"},
        "offers": (
            offer
            if offer_count == 1
            else [
                (dict(offer, sku=f"FIXTURE-OFFER-{index}") if offer_level_ids else dict(offer))
                for index in range(offer_count)
            ]
        ),
    }
    return (
        '<!doctype html><html><head><script type="application/ld+json">'
        + json.dumps(payload)
        + "</script></head></html>"
    ).encode()


def _transport(
    *,
    robots: bytes = ROBOTS_BODY,
    terms: bytes = TERMS_BODY,
    product: bytes | None = None,
    redirect: str | None = None,
    conditional: bool = False,
) -> httpx.MockTransport:
    bodies = {
        "/robots.txt": (robots, "text/plain", '"robots-v1"'),
        "/terms/data-use": (terms, "text/html", '"terms-v1"'),
        "/products/fixture-gpu": (product or _product_html(), "text/html", '"product-v1"'),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if redirect is not None and request.url.path == "/products/fixture-gpu":
            return httpx.Response(302, headers={"Location": redirect})
        body, media_type, etag = bodies[request.url.path]
        if conditional and request.headers.get("if-none-match") == etag:
            return httpx.Response(304)
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": media_type, "ETag": etag},
        )

    return httpx.MockTransport(handler)


def _adapter(tmp_path, policy: WebSourcePolicy, transport: httpx.MockTransport):
    clock = VirtualClock()
    return WebProductCrawlerAdapter(
        raw_root=tmp_path / "raw",
        policy=policy,
        transport=transport,
        resolver=lambda _host: ("1.1.1.1",),
        clock=clock,
        sleeper=clock.sleep,
    )


def test_schemaorg_product_offer_is_normalised_with_category_and_provenance(tmp_path) -> None:
    result = _adapter(tmp_path, _policy(), _transport()).crawl([PRODUCT_URL])

    assert result.batch.accepted_count == 1
    assert result.batch.rejected_count == 0
    record = result.batch.records[0]
    assert record["development_only"] is False
    assert record["source_record_id"].startswith("web_listing_")
    assert record["source_record_id"] != "FIXTURE-GPU-16"
    assert record["normalisation_metadata"]["category"] == "gpu"
    assert record["normalisation_metadata"]["shipping_price_known"] is True
    assert record["normalisation_metadata"]["robots_compliance_checked"] is True
    assert record["data_use_rights"][DataUse.DISPLAY.field_name] is True
    assert record["provenance"]["raw_page_sha256"] == record["archive_snapshot_sha256"]
    listing = record["data"]["listing"]
    assert listing["base_price"] == "1299.90"
    assert listing["shipping_price"] == "15.00"
    assert listing["currency"] == "SGD"
    assert listing["stock_status"] == "in_stock"
    assert listing["condition"] == "new"
    assert record["normalisation_metadata"]["identifiers"]["sku"] == "FIXTURE-GPU-16"


def test_internal_research_is_explicit_and_cannot_be_served_or_trained(tmp_path) -> None:
    policy = _policy(usage_scope=WebUsageScope.INTERNAL_RESEARCH, rights=_rights(enabled=False))
    result = _adapter(tmp_path, policy, _transport()).crawl([PRODUCT_URL])

    record = result.batch.records[0]
    assert record["development_only"] is True
    assert record["training_eligible"] is False
    assert record["published_claims_eligible"] is False
    assert record["normalisation_metadata"]["usage_scope"] == "internal_research"
    assert all(record["data_use_rights"][use.field_name] is False for use in DataUse)


def test_shopify_variant_query_stays_within_the_explicit_product_path(tmp_path) -> None:
    variant_url = f"{PRODUCT_URL}?variant=123456789"
    result = _adapter(
        tmp_path,
        _policy(),
        _transport(product=_product_html(offer_url=variant_url)),
    ).crawl([PRODUCT_URL])

    assert result.batch.accepted_count == 1
    assert result.batch.records[0]["data"]["listing"]["listing_url"] == variant_url
    assert result.batch.records[0]["normalisation_metadata"]["category"] == "gpu"


def test_unknown_shipping_is_fail_closed_for_production_and_quarantined_for_research(
    tmp_path,
) -> None:
    product = _product_html(include_shipping=False)
    production = _adapter(tmp_path / "production", _policy(), _transport(product=product)).crawl(
        [PRODUCT_URL]
    )
    assert production.batch.accepted_count == 0
    assert production.batch.rejected[0]["reason"] == "invalid_product_offer"
    assert "shipping price is unknown" in production.batch.rejected[0]["details"]["error"]

    confirmed = _adapter(
        tmp_path / "confirmed",
        _policy(unknown_shipping=UnknownShippingPolicy.ZERO_CONFIRMED),
        _transport(product=product),
    ).crawl([PRODUCT_URL])
    confirmed_metadata = confirmed.batch.records[0]["normalisation_metadata"]
    assert confirmed_metadata["shipping_price_known"] is True
    assert confirmed_metadata["shipping_price_basis"] == "policy_confirmed_zero"

    research_policy = _policy(
        usage_scope=WebUsageScope.INTERNAL_RESEARCH,
        rights=_rights(enabled=False),
    )
    research = _adapter(
        tmp_path / "research",
        research_policy,
        _transport(product=product),
    ).crawl([PRODUCT_URL])
    research_record = research.batch.records[0]
    assert research_record["development_only"] is True
    assert research_record["normalisation_metadata"]["shipping_price_known"] is False
    assert (
        research_record["normalisation_metadata"]["shipping_price_basis"]
        == "unknown_development_only"
    )


def test_production_and_research_policies_fail_closed_on_wrong_rights() -> None:
    with pytest.raises(WebCrawlPolicyError, match="does not permit display"):
        _policy(rights=_rights(enabled=False))

    with pytest.raises(WebCrawlPolicyError, match="all downstream DataUse grants false"):
        _policy(usage_scope=WebUsageScope.INTERNAL_RESEARCH, rights=_rights(enabled=True))


@pytest.mark.parametrize(
    "usage_scope",
    [WebUsageScope.PRODUCTION_CATALOG, WebUsageScope.INTERNAL_RESEARCH],
)
def test_acquisition_retention_cannot_exceed_rights_for_any_scope(
    usage_scope: WebUsageScope,
) -> None:
    payload = _policy(
        usage_scope=usage_scope,
        rights=_rights(enabled=usage_scope == WebUsageScope.PRODUCTION_CATALOG),
    ).to_dict()
    payload["rights"]["retention_days"] = 7
    payload["acquisition_authority"]["retention_days"] = 8

    with pytest.raises(
        WebCrawlPolicyError,
        match="raw snapshot retention exceeds the source rights retention period",
    ):
        WebSourcePolicy.from_mapping(payload)


def test_policy_mapping_requires_terms_rights_authority_and_exact_categories() -> None:
    payload = _policy().to_dict()
    restored = WebSourcePolicy.from_mapping(payload)
    assert restored.url_categories[PRODUCT_URL] == ComponentCategory.GPU

    for required_field in ("rights", "acquisition_authority", "canonical_terms_sha256"):
        incomplete = dict(payload)
        del incomplete[required_field]
        with pytest.raises(ValueError, match="missing fields"):
            WebSourcePolicy.from_mapping(incomplete)

    for field_name in (
        "allow_non_new",
        "training_eligible",
        "published_claims_eligible",
    ):
        invalid_boolean = dict(payload)
        invalid_boolean[field_name] = "false"
        with pytest.raises(TypeError, match=rf"{field_name} must be a boolean"):
            WebSourcePolicy.from_mapping(invalid_boolean)


@pytest.mark.parametrize("invalid_value", [True, "30", 30.0])
def test_acquisition_retention_days_requires_a_strict_integer(invalid_value: object) -> None:
    payload = _authority().to_dict()
    payload["retention_days"] = invalid_value

    with pytest.raises(TypeError, match="retention_days must be an integer"):
        WebAcquisitionAuthority.from_mapping(payload)


@pytest.mark.parametrize("invalid_value", [True, "2", 2.0])
def test_integer_policy_caps_reject_boolean_string_and_float(invalid_value: object) -> None:
    integral_caps = (
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
    )
    for field_name in integral_caps:
        payload = _policy().to_dict()
        payload[field_name] = invalid_value
        with pytest.raises(TypeError, match=rf"{field_name} must be an integer"):
            WebSourcePolicy.from_mapping(payload)


@pytest.mark.parametrize("field_name", ["requests_per_second", "timeout_seconds"])
@pytest.mark.parametrize("invalid_value", [True, "2"])
def test_numeric_policy_caps_reject_boolean_and_string(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _policy().to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(TypeError, match=rf"{field_name} must be a number"):
        WebSourcePolicy.from_mapping(payload)


@pytest.mark.parametrize(
    ("control_url", "expected_error"),
    [
        (TERMS_URL, "terms_url cannot be mapped"),
        (f"https://{HOST}/robots.txt", "robots.txt cannot be mapped"),
        (f"https://{HOST}/robots.txt?variant=product", "robots.txt cannot be mapped"),
    ],
)
def test_control_documents_cannot_be_category_mapped_as_products(
    control_url: str,
    expected_error: str,
) -> None:
    with pytest.raises(WebCrawlPolicyError, match=expected_error):
        _policy(url_categories={control_url: ComponentCategory.GPU})


@pytest.mark.parametrize("expired_gate", ["acquisition", "consent"])
def test_each_crawl_start_revalidates_time_sensitive_authority_before_io(
    tmp_path,
    expired_gate: str,
) -> None:
    policy = _policy()
    if expired_gate == "acquisition":
        object.__setattr__(policy.acquisition_authority, "expires_on", date(2020, 1, 2))
        expected_error = "acquisition authority has expired"
    else:
        object.__setattr__(policy.rights, "consent_expires_on", date(2020, 1, 2))
        expected_error = "source consent has expired"
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    adapter = _adapter(tmp_path, policy, httpx.MockTransport(handler))
    with pytest.raises(WebCrawlPolicyError, match=expected_error):
        adapter.crawl([PRODUCT_URL])
    assert called is False


def test_unknown_condition_is_rejected_in_production_but_quarantined_for_research(
    tmp_path,
) -> None:
    product = _product_html(item_condition=None)
    production = _adapter(tmp_path / "production", _policy(), _transport(product=product)).crawl(
        [PRODUCT_URL]
    )
    assert production.batch.accepted_count == 0
    assert "unknown item condition" in production.batch.rejected[0]["details"]["error"]

    research_policy = _policy(
        usage_scope=WebUsageScope.INTERNAL_RESEARCH,
        rights=_rights(enabled=False),
    )
    research = _adapter(tmp_path / "research", research_policy, _transport(product=product)).crawl(
        [PRODUCT_URL]
    )
    assert research.batch.accepted_count == 1
    assert research.batch.records[0]["data"]["listing"]["condition"] == "unknown"


def test_damaged_condition_is_rejected_even_when_non_new_items_are_allowed(tmp_path) -> None:
    policy = replace(_policy(), allow_non_new=True)
    result = _adapter(
        tmp_path,
        policy,
        _transport(product=_product_html(item_condition="https://schema.org/DamagedCondition")),
    ).crawl([PRODUCT_URL])

    assert result.batch.accepted_count == 0
    assert "damaged item condition" in result.batch.rejected[0]["details"]["error"]


@pytest.mark.parametrize(
    ("product", "expected_error"),
    [
        (_product_html(price="1e3"), "offer price is required"),
        (_product_html(price="USD 1,299.90"), "contradicts declared currency"),
        (_product_html(price="1,299.90 USD"), "contradicts declared currency"),
        (
            _product_html(shipping_price="USD 15.00"),
            "contradicts declared currency",
        ),
        (_product_html(shipping_currency="USD"), "ambiguous currency"),
        (_product_html(shipping_currency=""), "ambiguous currency"),
        (_product_html(shipping_country="MY"), "no shipping rate applicable to Singapore"),
        (_product_html(shipping_country=""), "ambiguous currency"),
    ],
)
def test_prices_and_shipping_are_parsed_with_strict_sg_semantics(
    tmp_path,
    product: bytes,
    expected_error: str,
) -> None:
    result = _adapter(tmp_path, _policy(), _transport(product=product)).crawl([PRODUCT_URL])

    assert result.batch.accepted_count == 0
    assert expected_error in result.batch.rejected[0]["details"]["error"]


@pytest.mark.parametrize("usage_scope", list(WebUsageScope))
def test_does_not_ship_to_singapore_overrides_a_present_sg_rate(
    tmp_path,
    usage_scope: WebUsageScope,
) -> None:
    rights = _rights(enabled=usage_scope == WebUsageScope.PRODUCTION_CATALOG)
    result = _adapter(
        tmp_path,
        _policy(usage_scope=usage_scope, rights=rights),
        _transport(product=_product_html(does_not_ship=True)),
    ).crawl([PRODUCT_URL])

    assert result.batch.accepted_count == 0
    assert "explicitly does not ship to Singapore" in result.batch.rejected[0]["details"]["error"]


def test_non_boolean_does_not_ship_is_not_treated_as_false(tmp_path) -> None:
    result = _adapter(
        tmp_path,
        _policy(),
        _transport(product=_product_html(does_not_ship="false")),
    ).crawl([PRODUCT_URL])

    assert result.batch.accepted_count == 0
    assert (
        "ambiguous currency, value, or destination" in result.batch.rejected[0]["details"]["error"]
    )


def test_listing_identity_is_namespaced_by_source_offer_url_and_seller(tmp_path) -> None:
    base_offer = {
        "@type": "Offer",
        "url": PRODUCT_URL,
        "sku": "SHARED-PRODUCT-SKU",
        "priceCurrency": "SGD",
        "price": "S$ 1299.90",
        "availability": "https://schema.org/InStock",
        "itemCondition": "https://schema.org/NewCondition",
        "shippingDetails": {
            "@type": "OfferShippingDetails",
            "shippingDestination": {"addressCountry": "SG"},
            "shippingRate": {"currency": "SGD", "value": "15.00"},
        },
    }
    product = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "Fixture RTX GPU 16 GB",
        "sku": "SHARED-PRODUCT-SKU",
        "mpn": "SHARED-MPN",
        "gtin": "0123456789012",
        "offers": [
            dict(base_offer, seller={"@type": "Organization", "name": "Seller Alpha"}),
            dict(base_offer, seller={"@type": "Organization", "name": "Seller Beta"}),
        ],
    }
    html = ('<script type="application/ld+json">' + json.dumps(product) + "</script>").encode()

    result = _adapter(tmp_path, _policy(), _transport(product=html)).crawl([PRODUCT_URL])

    assert result.batch.accepted_count == 2
    source_listing_ids = {record["source_record_id"] for record in result.batch.records}
    assert len(source_listing_ids) == 2
    assert all(identifier.startswith("web_listing_") for identifier in source_listing_ids)
    assert not source_listing_ids.intersection(
        {"SHARED-PRODUCT-SKU", "SHARED-MPN", "0123456789012"}
    )
    assert {record["data"]["listing"]["seller_name"] for record in result.batch.records} == {
        "Seller Alpha",
        "Seller Beta",
    }


def test_multiple_offers_require_unique_offer_level_identifiers(tmp_path) -> None:
    accepted = _adapter(
        tmp_path / "accepted",
        _policy(),
        _transport(product=_product_html(offer_count=2)),
    ).crawl([PRODUCT_URL])
    assert accepted.batch.accepted_count == 2
    assert len({record["source_record_id"] for record in accepted.batch.records}) == 2

    ambiguous = _adapter(
        tmp_path / "ambiguous",
        _policy(),
        _transport(product=_product_html(offer_count=2, offer_level_ids=False)),
    ).crawl([PRODUCT_URL])
    assert ambiguous.batch.accepted_count == 0
    assert ambiguous.batch.rejected[0]["reason"] == "ambiguous_multiple_offers"


def test_offer_and_record_budgets_fail_before_unbounded_normalisation(tmp_path) -> None:
    product = _product_html(offer_count=2)
    per_page_policy = replace(_policy(), maximum_offers_per_page=1)
    with pytest.raises(WebCrawlLimitError, match="1-offer limit"):
        _adapter(tmp_path / "per-page", per_page_policy, _transport(product=product)).crawl(
            [PRODUCT_URL]
        )

    global_policy = replace(
        _policy(),
        maximum_offers_per_page=10,
        maximum_records=1,
    )
    assert global_policy.normalized_record_limit == 1
    with pytest.raises(WebCrawlLimitError, match="1-offer/record limit"):
        _adapter(tmp_path / "global", global_policy, _transport(product=product)).crawl(
            [PRODUCT_URL]
        )


@pytest.mark.parametrize(
    ("limit_name", "expected"),
    [
        ("jsonld", "1-JSON-LD-block limit"),
        ("products", "1-product-node limit"),
        ("rejections", "1-rejection limit"),
        ("events", "1-event parse limit"),
    ],
)
def test_parser_resource_caps_bound_all_untrusted_structures(
    tmp_path,
    limit_name: str,
    expected: str,
) -> None:
    product = _product_html()
    policy = _policy()
    if limit_name == "jsonld":
        policy = replace(policy, maximum_jsonld_blocks=1)
        body = product + product
    elif limit_name == "products":
        policy = replace(policy, maximum_jsonld_blocks=2, maximum_product_nodes=1)
        body = product + product
    elif limit_name == "rejections":
        policy = replace(policy, maximum_jsonld_blocks=2, maximum_rejections=1)
        body = (
            b'<script type="application/ld+json">{</script>'
            b'<script type="application/ld+json">{</script>'
        )
    else:
        policy = replace(policy, maximum_parse_events=1)
        body = product

    with pytest.raises(WebCrawlLimitError, match=expected):
        _adapter(tmp_path, policy, _transport(product=body)).crawl([PRODUCT_URL])


def test_unmapped_url_is_rejected_before_network_access(tmp_path) -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    adapter = _adapter(tmp_path, _policy(), httpx.MockTransport(handler))
    with pytest.raises(WebCrawlPolicyError, match="component-category mapping"):
        adapter.crawl([f"https://{HOST}/products/unmapped"])
    assert called is False


@pytest.mark.parametrize(
    "url",
    [
        f"http://{HOST}/products/fixture-gpu",
        f"https://user:password@{HOST}/products/fixture-gpu",
        f"https://{HOST}:8443/products/fixture-gpu",
        "https://unlisted.example.test/products/fixture-gpu",
    ],
)
def test_unsafe_or_out_of_scope_urls_are_rejected(tmp_path, url: str) -> None:
    adapter = _adapter(tmp_path, _policy(), _transport())
    with pytest.raises(WebCrawlSecurityError):
        adapter.validate_target_url(url)


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "127.0.0.1",
        "fc00::1",
        "fe80::1",
    ],
)
def test_private_and_link_local_dns_resolution_is_rejected(tmp_path, address: str) -> None:
    adapter = WebProductCrawlerAdapter(
        raw_root=tmp_path / "raw",
        policy=_policy(),
        transport=_transport(),
        resolver=lambda _host: (address,),
    )
    with pytest.raises(WebCrawlSecurityError, match="non-public"):
        adapter.validate_target_url(PRODUCT_URL)


@pytest.mark.parametrize("address", ["224.0.0.1", "ff02::1"])
def test_multicast_dns_resolution_is_rejected(tmp_path, address: str) -> None:
    adapter = WebProductCrawlerAdapter(
        raw_root=tmp_path / "raw",
        policy=_policy(),
        transport=_transport(),
        resolver=lambda _host: (address,),
    )

    with pytest.raises(WebCrawlSecurityError, match="non-public"):
        adapter.validate_target_url(PRODUCT_URL)


def test_non_mock_transport_injection_cannot_bypass_pinning(tmp_path) -> None:
    with (
        httpx.HTTPTransport() as unpinned_transport,
        pytest.raises(TypeError, match="restricted to httpx.MockTransport"),
    ):
        WebProductCrawlerAdapter(
            raw_root=tmp_path / "raw",
            policy=_policy(),
            transport=unpinned_transport,  # type: ignore[arg-type]
            resolver=lambda _host: ("1.1.1.1",),
        )


def test_default_client_blocks_private_rebind_before_socket_or_request(
    tmp_path, monkeypatch
) -> None:
    resolved_address = ["1.1.1.1"]
    adapter = WebProductCrawlerAdapter(
        raw_root=tmp_path / "raw",
        policy=_policy(),
        resolver=lambda _host: tuple(resolved_address),
    )
    assert adapter.validate_target_url(PRODUCT_URL) == PRODUCT_URL
    resolved_address[0] = "127.0.0.1"
    socket_calls = 0

    def forbidden_connect(*_args, **_kwargs):
        nonlocal socket_calls
        socket_calls += 1
        raise AssertionError("socket backend must not be reached after DNS rebind")

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", forbidden_connect)

    with (
        adapter._client() as client,
        pytest.raises(WebCrawlSecurityError, match="non-public"),
    ):
        client.get(PRODUCT_URL)

    assert socket_calls == 0


def test_pinned_transport_uses_public_ip_but_preserves_host_and_tls_name() -> None:
    stream = RecordingNetworkStream()
    backend = RecordingNetworkBackend(stream)
    transport = _PinnedHTTPTransport(
        resolver=lambda _host: ("1.1.1.1",),
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        network_backend=backend,
    )

    with httpx.Client(transport=transport, trust_env=False) as client:
        response = client.get(f"https://{HOST}/products/fixture-gpu?source=test")

    wire_request = b"".join(stream.writes)
    assert response.text == "OK"
    assert backend.tcp_hosts == ["1.1.1.1"]
    assert stream.tls_server_names == [HOST]
    assert stream.tls_contexts[0].check_hostname is True
    assert stream.tls_contexts[0].verify_mode is ssl.CERT_REQUIRED
    assert wire_request.startswith(b"GET /products/fixture-gpu?source=test HTTP/1.1\r\n")
    assert f"Host: {HOST}\r\n".encode() in wire_request


def test_pinned_backend_revalidates_every_new_connection() -> None:
    responses = iter((("1.1.1.1",), ("192.168.1.10",)))
    stream = RecordingNetworkStream()
    backend = RecordingNetworkBackend(stream)
    pinned = _PinnedDNSBackend(
        resolver=lambda _host: next(responses),
        backend=backend,
    )

    assert pinned.connect_tcp(HOST, 443) is stream
    with pytest.raises(WebCrawlSecurityError, match="non-public"):
        pinned.connect_tcp(HOST, 443)

    assert backend.tcp_hosts == ["1.1.1.1"]


@pytest.mark.parametrize("address", ["224.0.0.1", "ff02::1"])
def test_pinned_backend_rejects_multicast_before_socket_connect(address: str) -> None:
    backend = RecordingNetworkBackend()
    pinned = _PinnedDNSBackend(
        resolver=lambda _host: (address,),
        backend=backend,
    )

    with pytest.raises(WebCrawlSecurityError, match="non-public"):
        pinned.connect_tcp(HOST, 443)

    assert backend.tcp_hosts == []


@pytest.mark.parametrize(
    "address",
    ["10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254", "fc00::1", "fe80::1"],
)
def test_pinned_backend_rejects_private_and_link_local_before_socket_connect(
    address: str,
) -> None:
    backend = RecordingNetworkBackend()
    pinned = _PinnedDNSBackend(
        resolver=lambda _host: (address,),
        backend=backend,
    )

    with pytest.raises(WebCrawlSecurityError, match="non-public"):
        pinned.connect_tcp(HOST, 443)

    assert backend.tcp_hosts == []


def test_pinned_transport_supports_concurrent_public_connections() -> None:
    class ConcurrentBackend(RecordingNetworkBackend):
        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()

        def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
        ) -> httpcore.NetworkStream:
            del port, timeout, local_address, socket_options
            with self._lock:
                self.tcp_hosts.append(host)
            return RecordingNetworkStream()

    backend = ConcurrentBackend()
    transport = _PinnedHTTPTransport(
        resolver=lambda _host: ("1.1.1.1",),
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        network_backend=backend,
    )
    urls = [f"https://{HOST}/products/item-{index}" for index in range(4)]

    with (
        httpx.Client(transport=transport, trust_env=False) as client,
        ThreadPoolExecutor(max_workers=4) as executor,
    ):
        responses = tuple(executor.map(client.get, urls))

    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    assert backend.tcp_hosts == ["1.1.1.1"] * 4


def test_cross_host_redirect_is_rejected_even_when_both_hosts_are_allowlisted(tmp_path) -> None:
    other_host = "cdn.example.test"
    policy = _policy(allowed_hosts=(HOST, other_host))
    adapter = _adapter(
        tmp_path,
        policy,
        _transport(redirect=f"https://{other_host}/products/fixture-gpu"),
    )
    with pytest.raises(WebCrawlSecurityError, match="different host"):
        adapter.crawl([PRODUCT_URL])


def test_same_host_redirect_requires_an_explicit_same_category_target(tmp_path) -> None:
    target_url = f"https://{HOST}/products/redirect-target"
    requested_paths: list[str] = []
    base_transport = _transport()

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/products/fixture-gpu":
            return httpx.Response(302, headers={"Location": target_url})
        if request.url.path == "/products/redirect-target":
            raise AssertionError("unauthorized redirect target was fetched")
        return base_transport.handle_request(request)

    for categories in (
        {PRODUCT_URL: ComponentCategory.GPU},
        {
            PRODUCT_URL: ComponentCategory.GPU,
            target_url: ComponentCategory.CPU,
        },
    ):
        adapter = _adapter(
            tmp_path / str(len(categories)),
            _policy(url_categories=categories),
            httpx.MockTransport(handler),
        )
        with pytest.raises(WebCrawlSecurityError, match="same category"):
            adapter.crawl([PRODUCT_URL])
    assert "/products/redirect-target" not in requested_paths


def test_same_host_redirect_to_explicit_same_category_target_is_followed(tmp_path) -> None:
    target_url = f"https://{HOST}/products/redirect-target"
    base_transport = _transport()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/products/fixture-gpu":
            return httpx.Response(302, headers={"Location": target_url})
        if request.url.path == "/products/redirect-target":
            return httpx.Response(
                200,
                content=_product_html(offer_url=target_url),
                headers={"Content-Type": "text/html"},
            )
        return base_transport.handle_request(request)

    policy = _policy(
        url_categories={
            PRODUCT_URL: ComponentCategory.GPU,
            target_url: ComponentCategory.GPU,
        }
    )
    result = _adapter(tmp_path, policy, httpx.MockTransport(handler)).crawl([PRODUCT_URL])

    assert result.batch.accepted_count == 1
    assert result.pages[0].final_url == target_url


@pytest.mark.parametrize(
    "directive",
    [b"Crawl-delay: 5\n", b"Request-rate: 2/10\n"],
)
def test_robots_rate_directives_apply_the_more_restrictive_host_interval(
    tmp_path,
    directive: bytes,
) -> None:
    clock = VirtualClock()
    request_times: list[tuple[str, float]] = []
    base_transport = _transport(robots=b"User-agent: *\nAllow: /\n" + directive)

    def handler(request: httpx.Request) -> httpx.Response:
        request_times.append((request.url.path, clock()))
        return base_transport.handle_request(request)

    adapter = WebProductCrawlerAdapter(
        raw_root=tmp_path / "raw",
        policy=_policy(),
        transport=httpx.MockTransport(handler),
        resolver=lambda _host: ("1.1.1.1",),
        clock=clock,
        sleeper=clock.sleep,
    )
    adapter.crawl([PRODUCT_URL])

    assert [path for path, _at in request_times] == [
        "/robots.txt",
        "/terms/data-use",
        "/products/fixture-gpu",
        "/terms/data-use",
    ]
    assert all(
        later - earlier >= 5
        for (_path, earlier), (_next_path, later) in zip(
            request_times, request_times[1:], strict=False
        )
    )


def test_extreme_robots_interval_fails_closed_before_terms_or_products(tmp_path) -> None:
    requested_paths: list[str] = []
    base_transport = _transport(robots=b"User-agent: *\nAllow: /\nCrawl-delay: 999\n")

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return base_transport.handle_request(request)

    with pytest.raises(WebCrawlPolicyError, match="extreme request interval"):
        _adapter(tmp_path, _policy(), httpx.MockTransport(handler)).crawl([PRODUCT_URL])
    assert requested_paths == ["/robots.txt"]


def test_robots_denial_and_missing_directive_fail_closed(tmp_path) -> None:
    denied = _transport(robots=b"User-agent: *\nDisallow: /products/\n")
    with pytest.raises(WebCrawlPolicyError, match="robots.txt does not permit"):
        _adapter(tmp_path / "denied", _policy(), denied).crawl([PRODUCT_URL])

    missing = _transport(robots=b"# no declared crawler policy\n")
    with pytest.raises(WebCrawlPolicyError, match="no User-agent directive"):
        _adapter(tmp_path / "missing", _policy(), missing).crawl([PRODUCT_URL])

    def missing_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise AssertionError("crawler continued after missing robots.txt")

    with pytest.raises(WebCrawlPolicyError, match="was unavailable"):
        _adapter(
            tmp_path / "not-found",
            _policy(),
            httpx.MockTransport(missing_handler),
        ).crawl([PRODUCT_URL])


def test_terms_hash_change_stops_before_product_fetch(tmp_path) -> None:
    requested_paths: list[str] = []
    changed_terms = TERMS_BODY.replace(b"are permitted", b"are prohibited")
    base_transport = _transport(terms=changed_terms)

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return base_transport.handle_request(request)

    adapter = _adapter(tmp_path, _policy(), httpx.MockTransport(handler))
    with pytest.raises(WebCrawlPolicyError, match="terms wording changed"):
        adapter.crawl([PRODUCT_URL])
    assert "/products/fixture-gpu" not in requested_paths


def test_terms_are_rehashed_after_products_and_mid_run_wording_change_fails(tmp_path) -> None:
    requested_paths: list[str] = []
    changed_terms = TERMS_BODY.replace(b"are permitted", b"are prohibited")
    product_seen = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal product_seen
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=ROBOTS_BODY, headers={"Content-Type": "text/plain"})
        if request.url.path == "/terms/data-use":
            return httpx.Response(
                200,
                content=changed_terms if product_seen else TERMS_BODY,
                headers={"Content-Type": "text/html"},
            )
        product_seen = True
        return httpx.Response(
            200,
            content=_product_html(),
            headers={"Content-Type": "text/html"},
        )

    with pytest.raises(WebCrawlPolicyError, match="changed during the crawl"):
        _adapter(tmp_path, _policy(), httpx.MockTransport(handler)).crawl([PRODUCT_URL])
    assert requested_paths.count("/terms/data-use") == 2
    assert requested_paths.index("/products/fixture-gpu") < len(requested_paths) - 1
    assert len(list((tmp_path / "raw" / "fixture_web" / "pages").glob("*.terms"))) == 2


def test_declared_page_size_limit_is_enforced(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=ROBOTS_BODY, headers={"Content-Type": "text/plain"})
        if request.url.path == "/terms/data-use":
            return httpx.Response(200, content=TERMS_BODY, headers={"Content-Type": "text/html"})
        return httpx.Response(
            200,
            content=b"small",
            headers={"Content-Type": "text/html", "Content-Length": "20000"},
        )

    with pytest.raises(WebCrawlLimitError, match="byte limit"):
        _adapter(tmp_path, _policy(), httpx.MockTransport(handler)).crawl([PRODUCT_URL])


@pytest.mark.parametrize("peer_address", ["127.0.0.1", "224.0.0.1", "ff02::1"])
def test_connected_non_public_peer_is_rejected_after_dns_validation(
    tmp_path, peer_address: str
) -> None:
    class NonPublicPeerStream:
        @staticmethod
        def get_extra_info(name: str):
            assert name == "server_addr"
            return (peer_address, 443)

    base_transport = _transport()

    def handler(request: httpx.Request) -> httpx.Response:
        response = base_transport.handle_request(request)
        response.extensions["network_stream"] = NonPublicPeerStream()
        return response

    adapter = _adapter(tmp_path, _policy(), httpx.MockTransport(handler))
    with pytest.raises(WebCrawlPolicyError, match="robots.txt.*unavailable") as caught:
        adapter.crawl([PRODUCT_URL])
    assert isinstance(caught.value.__cause__, WebCrawlSecurityError)
    assert "DNS rebinding" in str(caught.value.__cause__)


def test_cli_requires_an_explicit_policy_and_url(capsys) -> None:
    exit_code = fetch_open_data_main(["--source", "web_product"])
    error = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert "--web-policy-json" in error["error"]


def test_canonical_terms_hash_ignores_script_nonce_but_detects_wording_changes() -> None:
    dynamic_variant = TERMS_BODY.replace(
        b"nonce=\"first\">window.theme = 'light'",
        b"nonce=\"second\">window.theme = 'dark'",
    ).replace(b"Dynamic storefront", b"Different header")
    wording_change = TERMS_BODY.replace(b"are permitted", b"are prohibited")

    expected = calculate_canonical_terms_sha256(
        TERMS_BODY, media_type="text/html", selector=".reviewed-terms"
    )
    assert (
        calculate_canonical_terms_sha256(
            dynamic_variant,
            media_type="text/html",
            selector=".reviewed-terms",
        )
        == expected
    )
    assert (
        calculate_canonical_terms_sha256(
            wording_change,
            media_type="text/html",
            selector=".reviewed-terms",
        )
        != expected
    )


def test_canonical_terms_hash_detects_link_and_semantic_attribute_changes() -> None:
    original = b"""
    <div class="reviewed-terms">Use is permitted.
      <a href="/policy/v1" rel="license" aria-label="Licence">Details</a>
    </div>
    """
    changed_href = original.replace(b"/policy/v1", b"/policy/v2")
    changed_outside_selector = b'<a href="/unrelated">Other</a>' + original

    expected = calculate_canonical_terms_sha256(
        original, media_type="text/html", selector=".reviewed-terms"
    )
    assert (
        calculate_canonical_terms_sha256(
            changed_href,
            media_type="text/html",
            selector=".reviewed-terms",
        )
        != expected
    )
    assert (
        calculate_canonical_terms_sha256(
            changed_outside_selector,
            media_type="text/html",
            selector=".reviewed-terms",
        )
        == expected
    )


def test_requests_identity_encoding_and_rejects_compressed_responses(tmp_path) -> None:
    seen_identity = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_identity
        seen_identity = request.headers.get("accept-encoding") == "identity"
        return httpx.Response(
            200,
            content=ROBOTS_BODY,
            headers={"Content-Encoding": "gzip", "Content-Type": "text/plain"},
        )

    with pytest.raises(WebCrawlPolicyError, match="robots.txt.*unavailable"):
        _adapter(tmp_path, _policy(), httpx.MockTransport(handler)).crawl([PRODUCT_URL])
    assert seen_identity is True
