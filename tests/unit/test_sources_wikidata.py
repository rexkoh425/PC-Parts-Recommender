from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pipelines.sources.wikidata import (
    WIKIDATA_LICENSE_URL,
    WIKIDATA_PARSER_VERSION,
    WIKIDATA_USER_AGENT,
    WikidataAPIError,
    WikidataCandidate,
    WikidataEnrichmentAdapter,
    WikidataResponseTooLargeError,
    load_wikidata_candidates,
)


def _string_claim(value: str) -> dict[str, object]:
    return {
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {"type": "string", "value": value},
        },
        "rank": "normal",
    }


def _entity_claim(entity_id: str) -> dict[str, object]:
    return {
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {
                "type": "wikibase-entityid",
                "value": {"entity-type": "item", "id": entity_id},
            },
        },
        "rank": "normal",
    }


def _time_claim(value: str, precision: int) -> dict[str, object]:
    return {
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {
                "type": "time",
                "value": {
                    "time": value,
                    "precision": precision,
                    "timezone": 0,
                    "before": 0,
                    "after": 0,
                    "calendarmodel": "http://www.wikidata.org/entity/Q1985727",
                },
            },
        },
        "rank": "preferred",
    }


def _fixture_entity(*, mpn: str = "100-000000910") -> dict[str, object]:
    return {
        "id": "Q12345",
        "type": "item",
        "labels": {
            "en": {"language": "en", "value": "AMD Ryzen 7 7800X3D"},
        },
        "aliases": {
            "en": [
                {"language": "en", "value": "Ryzen 7 7800X3D"},
                {"language": "en", "value": "AMD Ryzen 7 7800X3D"},
            ]
        },
        "claims": {
            "P13802": [_string_claim(mpn)],
            "P3962": [_string_claim("00123456789012")],
            "P176": [_entity_claim("Q128896")],
            "P31": [_entity_claim("Q122967152")],
            "P577": [_time_claim("+2023-04-06T00:00:00Z", 11)],
        },
        "sitelinks": {
            "enwiki": {"site": "enwiki", "title": "Ryzen 7 7800X3D", "badges": []},
        },
    }


def _candidate() -> WikidataCandidate:
    return WikidataCandidate(
        candidate_id="prod_fixture_cpu",
        canonical_name="AMD Ryzen 7 7800X3D",
        category="cpu",
        brand="AMD",
        manufacturer_part_number="100-000000910",
        gtin="00123456789012",
    )


def _mock_transport(*, entity: dict[str, object] | None = None) -> httpx.MockTransport:
    fixture_entity = entity or _fixture_entity()

    def handler(request: httpx.Request) -> httpx.Response:
        action = request.url.params.get("action")
        if action == "wbsearchentities":
            assert request.url.host == "www.wikidata.org"
            assert request.url.params["maxlag"] == "5"
            assert request.url.params["limit"] == "3"
            return httpx.Response(
                200,
                request=request,
                json={
                    "searchinfo": {"search": "AMD Ryzen 7 7800X3D"},
                    "search": [
                        {
                            "id": "Q12345",
                            "title": "Q12345",
                            "pageid": 12345,
                            "label": "AMD Ryzen 7 7800X3D",
                            "description": "desktop processor",
                        }
                    ],
                    "success": 1,
                },
            )
        if action == "wbgetentities":
            assert request.url.params["maxlag"] == "5"
            assert request.url.params["ids"] == "Q12345"
            return httpx.Response(
                200,
                request=request,
                json={"entities": {"Q12345": fixture_entity}, "success": 1},
            )
        raise AssertionError(f"unexpected Wikidata action: {action}")

    return httpx.MockTransport(handler)


def _mock_transport_for_entities(
    entities: dict[str, dict[str, object]],
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        action = request.url.params.get("action")
        if action == "wbsearchentities":
            return httpx.Response(
                200,
                request=request,
                json={"search": [{"id": entity_id} for entity_id in entities], "success": 1},
            )
        if action == "wbgetentities":
            requested_ids = request.url.params["ids"].split("|")
            return httpx.Response(
                200,
                request=request,
                json={
                    "entities": {
                        entity_id: entities[entity_id]
                        for entity_id in requested_ids
                        if entity_id in entities
                    },
                    "success": 1,
                },
            )
        raise AssertionError(f"unexpected Wikidata action: {action}")

    return httpx.MockTransport(handler)


def test_wikidata_fetch_is_bounded_content_addressed_and_enrichment_only(tmp_path) -> None:
    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "raw",
        transport=_mock_transport(),
        sleeper=lambda _seconds: None,
    )
    first = adapter.fetch([_candidate()], max_records=1)
    second = adapter.fetch([_candidate()], max_records=1)
    batch = adapter.parse(first, max_records=1)

    assert first.reused is False
    assert second.reused is True
    assert first.content_sha256 == second.content_sha256
    assert first.licence_or_access_note.startswith(
        "Wikidata structured data is dedicated to the public domain under CC0 1.0"
    )
    assert batch.accepted_count == 1
    assert batch.rejected_count == 0
    assert batch.statistics["contains_retailer_prices"] is False
    assert batch.statistics["training_eligible"] is False
    assert batch.statistics["downstream_consumer_validation_required"] is True
    assert batch.statistics["downstream_consumer_validation_status"] == "pending"

    record = batch.records[0]
    assert record["record_type"] == "catalogue_identity_enrichment"
    assert record["training_eligible"] is False
    assert record["training_scope"] == "quarantined_until_downstream_consumer_validation"
    assert record["development_only"] is True
    assert record["redistribution_eligible"] is True
    assert record["rights_metadata"]["may_train"] is True
    assert record["published_claims_eligible"] is False
    assert record["provenance"]["licence"] == "CC0-1.0"
    assert record["provenance"]["licence_url"] == WIKIDATA_LICENSE_URL
    assert record["provenance"]["parser_version"] == WIKIDATA_PARSER_VERSION
    assert record["normalisation_metadata"]["match_method"] == "exact_gtin_and_mpn"
    assert record["normalisation_metadata"]["contains_retailer_prices"] is False
    assert record["normalisation_metadata"]["downstream_consumer_validation_required"] is True
    assert record["normalisation_metadata"]["downstream_consumer_validation_status"] == "pending"
    enrichment = record["data"]
    assert enrichment["product_id"] == "prod_fixture_cpu"
    assert enrichment["wikidata_entity_id"] == "Q12345"
    assert enrichment["aliases"] == ["Ryzen 7 7800X3D"]
    assert enrichment["manufacturer_part_numbers"] == ["100-000000910"]
    assert enrichment["gtins"] == ["00123456789012"]
    assert enrichment["manufacturer_entity_ids"] == ["Q128896"]
    assert enrichment["release_date"] == {
        "value": "2023-04-06",
        "precision": "day",
        "property_id": "P577",
    }
    assert enrichment["wikipedia_url"].endswith("/wiki/Ryzen_7_7800X3D")
    assert (
        not {
            "base_price",
            "shipping_price",
            "stock_status",
            "retailer",
            "listing_url",
        }
        & enrichment.keys()
    )


def test_wikidata_parser_rejects_identifier_conflict_even_when_name_matches(tmp_path) -> None:
    adapter = WikidataEnrichmentAdapter(raw_root=tmp_path / "raw")
    fixture = {
        "schema_version": "pc-build-recommender.wikidata-raw.v2",
        "acquisition_mode": "official_api",
        "api_url": "https://www.wikidata.org/w/api.php",
        "licence": "CC0-1.0",
        "licence_url": WIKIDATA_LICENSE_URL,
        "language": "en",
        "candidates": [
            {
                "candidate_id": "prod_fixture_cpu",
                "canonical_name": "AMD Ryzen 7 7800X3D",
                "category": "cpu",
                "brand": "AMD",
                "manufacturer_part_number": "EXPECTED-MPN",
                "gtin": None,
            }
        ],
        "search_responses": [
            {
                "candidate_id": "prod_fixture_cpu",
                "payload": {"search": [{"id": "Q12345"}]},
            }
        ],
        "entity_responses": [
            {
                "ids": ["Q12345"],
                "payload": {"entities": {"Q12345": _fixture_entity(mpn="CONFLICTING-MPN")}},
            }
        ],
    }
    fixture_path = tmp_path / "wikidata.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    batch = adapter.parse(adapter.fetch(response_path=fixture_path), max_records=1)

    assert batch.accepted_count == 0
    assert batch.rejected_count == 1
    assert batch.rejected[0]["reason"] == "no_exact_wikidata_identity_match"


def test_wikidata_candidate_loader_streams_a_bounded_canonical_product_file(tmp_path) -> None:
    candidate_path = tmp_path / "products.jsonl"
    records = []
    for index in range(3):
        records.append(
            {
                "record_type": "canonical_product",
                "data": {
                    "product_id": f"prod_{index}",
                    "canonical_name": f"Fixture CPU {index}",
                    "category": "gpu" if index == 1 else "cpu",
                    "brand": "Fixture",
                    "manufacturer_part_number": f"MPN-{index}",
                    "gtin": None,
                },
            }
        )
    candidate_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    candidates = load_wikidata_candidates(candidate_path, max_records=2)
    assert [candidate.candidate_id for candidate in candidates] == ["prod_0", "prod_1"]

    gpu_candidates = load_wikidata_candidates(
        candidate_path,
        max_records=2,
        categories=("gpu",),
    )
    assert [candidate.candidate_id for candidate in gpu_candidates] == ["prod_1"]

    with pytest.raises(ValueError, match="did not contain products in categories: motherboard"):
        load_wikidata_candidates(candidate_path, categories=("motherboard",))

    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        load_wikidata_candidates(candidate_path, max_records=2, maximum_line_bytes=10)


def test_wikidata_candidate_loader_uses_bounded_readline_before_allocation(
    tmp_path, monkeypatch
) -> None:
    class BoundedHandle:
        def __init__(self) -> None:
            self.chunks = [b"x" * 11, b"tail\n"]
            self.requested_sizes: list[int] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            raise AssertionError("unbounded file iteration must not be used")

        def readline(self, size: int = -1) -> bytes:
            self.requested_sizes.append(size)
            return self.chunks.pop(0) if self.chunks else b""

    bounded = BoundedHandle()
    monkeypatch.setattr(Path, "open", lambda _self, _mode: bounded)

    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        load_wikidata_candidates(tmp_path / "candidates.jsonl", maximum_line_bytes=10)

    assert bounded.requested_sizes == [11, 11]


def test_wikidata_network_and_response_budget_fail_clearly(tmp_path) -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("fixture offline", request=request)

    offline_adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "offline",
        transport=httpx.MockTransport(offline),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(WikidataAPIError, match="Wikidata API request failed"):
        offline_adapter.fetch([_candidate()], max_records=1)

    def oversized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=b'{"search":[],' + b'"padding":"' + (b"x" * 512) + b'"}',
        )

    bounded_adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "bounded",
        transport=httpx.MockTransport(oversized),
        sleeper=lambda _seconds: None,
        maximum_response_bytes=128,
        maximum_total_bytes=256,
    )
    with pytest.raises(WikidataResponseTooLargeError, match="declared"):
        bounded_adapter.fetch([_candidate()], max_records=1)


def test_wikidata_retries_429_with_retry_after_and_maxlag(tmp_path) -> None:
    request_count = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.url.params["maxlag"] == "5"
        assert request.headers["accept-encoding"] == "gzip, deflate"
        assert request.headers["user-agent"] == WIKIDATA_USER_AGENT
        assert request.headers["api-user-agent"] == WIKIDATA_USER_AGENT
        if request_count == 1:
            return httpx.Response(
                429,
                request=request,
                headers={"Retry-After": "1"},
                json={"error": {"code": "ratelimited", "info": "slow down"}},
            )
        if request.url.params["action"] == "wbsearchentities":
            return httpx.Response(
                200,
                request=request,
                json={"search": [{"id": "Q12345"}], "success": 1},
            )
        return httpx.Response(
            200,
            request=request,
            json={"entities": {"Q12345": _fixture_entity()}, "success": 1},
        )

    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "raw",
        transport=httpx.MockTransport(handler),
        maximum_retries=1,
        sleeper=sleeps.append,
    )
    batch = adapter.parse(adapter.fetch([_candidate()], max_records=1), max_records=1)

    assert batch.accepted_count == 1
    assert request_count == 3
    assert sleeps == [1.0, 0.2, 0.2]


def test_wikidata_hard_caps_and_pc_categories_fail_before_network(tmp_path) -> None:
    adapter = WikidataEnrichmentAdapter(raw_root=tmp_path / "raw")
    with pytest.raises(ValueError, match="between 1 and 500"):
        adapter.fetch([_candidate()], max_records=501)
    with pytest.raises(ValueError, match="between 1 and 5"):
        adapter.fetch([_candidate()], max_records=1, search_limit=6)
    with pytest.raises(ValueError, match="at least 0.2 seconds"):
        WikidataEnrichmentAdapter(
            raw_root=tmp_path / "too-fast",
            request_delay_seconds=0.1,
        )
    rate_limited = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "rate-limited",
        request_delay_seconds=0.2,
    )
    assert rate_limited._retry_delay("0") == 0.2
    with pytest.raises(ValueError):
        WikidataCandidate(
            candidate_id="prod_bad",
            canonical_name="Not a component",
            category="laptop",
        )


def test_wikidata_p1628_is_never_treated_as_a_part_number(tmp_path) -> None:
    entity = _fixture_entity()
    claims = entity["claims"]
    assert isinstance(claims, dict)
    claims["P13802"] = []
    claims["P3962"] = []
    claims["P1628"] = [_string_claim(_candidate().manufacturer_part_number or "")]
    entity["labels"] = {"en": {"language": "en", "value": "Different product"}}
    entity["aliases"] = {}
    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "raw",
        transport=_mock_transport(entity=entity),
        sleeper=lambda _seconds: None,
    )

    batch = adapter.parse(adapter.fetch([_candidate()], max_records=1), max_records=1)

    assert batch.accepted_count == 0
    assert batch.rejected[0]["reason"] == "no_exact_wikidata_identity_match"


def test_wikidata_exact_mpn_requires_and_accepts_reviewed_identity_context(tmp_path) -> None:
    candidate = WikidataCandidate(
        candidate_id="prod_mpn_only",
        canonical_name="AMD Ryzen 7 7800X3D",
        category="cpu",
        brand="AMD",
        manufacturer_part_number="100-000000910",
    )
    entity = _fixture_entity()
    claims = entity["claims"]
    assert isinstance(claims, dict)
    claims["P3962"] = []
    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "raw",
        transport=_mock_transport(entity=entity),
        sleeper=lambda _seconds: None,
    )

    batch = adapter.parse(adapter.fetch([candidate], max_records=1), max_records=1)

    assert batch.accepted_count == 1
    assert batch.rejected_count == 0
    assert batch.records[0]["normalisation_metadata"]["match_method"] == "exact_mpn"


@pytest.mark.parametrize(
    ("manufacturer_id", "instance_id"),
    [
        ("Q248", "Q122967152"),
        ("Q128896", "Q5"),
        (None, "Q122967152"),
    ],
    ids=["manufacturer-mismatch", "category-mismatch", "manufacturer-missing"],
)
def test_wikidata_exact_mpn_rejects_unreviewed_identity_context(
    tmp_path,
    manufacturer_id: str | None,
    instance_id: str,
) -> None:
    candidate = WikidataCandidate(
        candidate_id="prod_mpn_only",
        canonical_name="AMD Ryzen 7 7800X3D",
        category="cpu",
        brand="AMD",
        manufacturer_part_number="100-000000910",
    )
    entity = _fixture_entity()
    claims = entity["claims"]
    assert isinstance(claims, dict)
    claims["P3962"] = []
    claims["P176"] = [_entity_claim(manufacturer_id)] if manufacturer_id else []
    claims["P31"] = [_entity_claim(instance_id)]
    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / manufacturer_id if manufacturer_id else tmp_path / "missing",
        transport=_mock_transport(entity=entity),
        sleeper=lambda _seconds: None,
    )

    batch = adapter.parse(adapter.fetch([candidate], max_records=1), max_records=1)

    assert batch.accepted_count == 0
    assert batch.rejected_count == 1
    assert batch.rejected[0]["reason"] == "no_exact_wikidata_identity_match"


def test_wikidata_mpn_collision_selects_only_matching_reviewed_manufacturer(tmp_path) -> None:
    candidate = WikidataCandidate(
        candidate_id="prod_mpn_collision",
        canonical_name="AMD Ryzen 7 7800X3D",
        category="cpu",
        brand="AMD",
        manufacturer_part_number="COLLIDING-MPN",
    )
    matching = _fixture_entity(mpn="COLLIDING-MPN")
    conflicting = _fixture_entity(mpn="COLLIDING-MPN")
    for entity in (matching, conflicting):
        claims = entity["claims"]
        assert isinstance(claims, dict)
        claims["P3962"] = []
    conflicting_claims = conflicting["claims"]
    assert isinstance(conflicting_claims, dict)
    conflicting_claims["P176"] = [_entity_claim("Q248")]
    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "raw",
        transport=_mock_transport_for_entities({"Q12345": matching, "Q54321": conflicting}),
        sleeper=lambda _seconds: None,
    )

    batch = adapter.parse(adapter.fetch([candidate], max_records=1), max_records=1)

    assert batch.accepted_count == 1
    assert batch.rejected_count == 0
    assert batch.records[0]["data"]["wikidata_entity_id"] == "Q12345"
    assert batch.records[0]["normalisation_metadata"]["match_method"] == "exact_mpn"


def test_wikidata_gtin_remains_global_when_mpn_context_disagrees(tmp_path) -> None:
    entity = _fixture_entity()
    claims = entity["claims"]
    assert isinstance(claims, dict)
    claims["P176"] = [_entity_claim("Q248")]
    claims["P31"] = [_entity_claim("Q5")]
    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "raw",
        transport=_mock_transport(entity=entity),
        sleeper=lambda _seconds: None,
    )

    batch = adapter.parse(adapter.fetch([_candidate()], max_records=1), max_records=1)

    assert batch.accepted_count == 1
    assert batch.rejected_count == 0
    assert batch.records[0]["normalisation_metadata"]["match_method"] == "exact_gtin"


def test_wikidata_exact_mpn_rejects_a_reviewed_class_from_another_category(tmp_path) -> None:
    candidate = WikidataCandidate(
        candidate_id="prod_gpu_with_cpu_identity",
        canonical_name="AMD Fixture GPU",
        category="gpu",
        brand="AMD",
        manufacturer_part_number="100-000000910",
    )
    entity = _fixture_entity()
    claims = entity["claims"]
    assert isinstance(claims, dict)
    claims["P3962"] = []
    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "raw",
        transport=_mock_transport(entity=entity),
        sleeper=lambda _seconds: None,
    )

    batch = adapter.parse(adapter.fetch([candidate], max_records=1), max_records=1)

    assert batch.accepted_count == 0
    assert batch.rejected_count == 1
    assert batch.rejected[0]["reason"] == "no_exact_wikidata_identity_match"


def test_wikidata_exact_name_requires_reviewed_type_and_manufacturer_context(tmp_path) -> None:
    entity = _fixture_entity(mpn="")
    claims = entity["claims"]
    assert isinstance(claims, dict)
    claims["P13802"] = []
    claims["P3962"] = []
    claims["P31"] = [_entity_claim("Q5")]
    adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "wrong-type",
        transport=_mock_transport(entity=entity),
        sleeper=lambda _seconds: None,
    )

    batch = adapter.parse(adapter.fetch([_candidate()], max_records=1), max_records=1)

    assert batch.accepted_count == 0
    assert batch.rejected[0]["reason"] == "no_exact_wikidata_identity_match"


def test_wikidata_exact_name_requires_reviewed_manufacturer_qid(tmp_path) -> None:
    candidate = WikidataCandidate(
        candidate_id="prod_name_only",
        canonical_name="AMD Ryzen 7 7800X3D",
        category="cpu",
        brand="AMD",
    )
    mismatched = _fixture_entity(mpn="")
    mismatched_claims = mismatched["claims"]
    assert isinstance(mismatched_claims, dict)
    mismatched_claims["P13802"] = []
    mismatched_claims["P3962"] = []
    mismatched_claims["P176"] = [_entity_claim("Q248")]
    rejected_adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "mismatched",
        transport=_mock_transport(entity=mismatched),
        sleeper=lambda _seconds: None,
    )
    rejected = rejected_adapter.parse(
        rejected_adapter.fetch([candidate], max_records=1),
        max_records=1,
    )
    assert rejected.accepted_count == 0

    matched = _fixture_entity(mpn="")
    matched_claims = matched["claims"]
    assert isinstance(matched_claims, dict)
    matched_claims["P13802"] = []
    matched_claims["P3962"] = []
    matched_claims["P176"] = [_entity_claim("Q128896")]
    accepted_adapter = WikidataEnrichmentAdapter(
        raw_root=tmp_path / "matched",
        transport=_mock_transport(entity=matched),
        sleeper=lambda _seconds: None,
    )
    accepted = accepted_adapter.parse(
        accepted_adapter.fetch([candidate], max_records=1),
        max_records=1,
    )
    assert accepted.accepted_count == 1
    assert accepted.records[0]["normalisation_metadata"]["match_method"] == "exact_name"


def test_wikidata_local_fixture_is_quarantined_from_training_and_redistribution(tmp_path) -> None:
    fixture = {
        "schema_version": "pc-build-recommender.wikidata-raw.v2",
        "acquisition_mode": "official_api",
        "api_url": "https://www.wikidata.org/w/api.php",
        "licence": "CC0-1.0",
        "licence_url": WIKIDATA_LICENSE_URL,
        "language": "en",
        "candidates": [
            {
                "candidate_id": _candidate().candidate_id,
                "canonical_name": _candidate().canonical_name,
                "category": _candidate().category,
                "brand": _candidate().brand,
                "manufacturer_part_number": _candidate().manufacturer_part_number,
                "gtin": _candidate().gtin,
            }
        ],
        "search_responses": [
            {
                "candidate_id": _candidate().candidate_id,
                "payload": {"search": [{"id": "Q12345"}]},
            }
        ],
        "entity_responses": [
            {"ids": ["Q12345"], "payload": {"entities": {"Q12345": _fixture_entity()}}}
        ],
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    adapter = WikidataEnrichmentAdapter(raw_root=tmp_path / "raw")

    snapshot = adapter.fetch(response_path=fixture_path)
    batch = adapter.parse(snapshot, max_records=1)

    assert snapshot.source_name == "wikidata_fixture"
    assert batch.accepted_count == 1
    assert batch.statistics["acquisition_mode"] == "local_fixture"
    assert batch.statistics["training_eligible"] is False
    record = batch.records[0]
    assert record["development_only"] is True
    assert record["training_eligible"] is False
    assert record["redistribution_eligible"] is False
    assert record["rights_metadata"]["rights_basis"] == "unverified_local_fixture"
    assert record["provenance"]["source_type"] == "controlled_fixture"
