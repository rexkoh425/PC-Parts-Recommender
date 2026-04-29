"""Bounded CC0 Wikidata identity enrichment for canonical PC components.

The adapter deliberately does not collect offers, stock, or prices.  It searches a
bounded set of existing catalogue candidates through Wikidata's official Action API,
then resolves only exact identifier or exact-name matches.  Ambiguous and weak search
results remain auditable rejections instead of silently changing canonical products.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from pc_build_recommender.domain.enums import ComponentKind
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION
from pipelines.sources.base import (
    ParsedBatch,
    RawSnapshot,
    rejected_record,
    sha256_bytes,
    snapshot_local_file,
)

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/entity/{entity_id}"
WIKIDATA_LICENSE_URL = "https://www.wikidata.org/wiki/Wikidata:Licensing"
WIKIDATA_USER_AGENT = (
    "BuildSignalPCRecommender/0.1 "
    "(https://buildsignal-pc-recommender.tendra425.chatgpt.site; "
    "Wikidata CC0 catalogue enrichment)"
)
WIKIDATA_LICENSE_NOTE = (
    "Wikidata structured data is dedicated to the public domain under CC0 1.0. "
    "This adapter uses it only for catalogue identity, alias, identifier, release-date, "
    "and entity-link enrichment; it does not collect retailer prices, offers, or stock."
)
WIKIDATA_PARSER_VERSION = "wikidata-catalogue-enrichment-v3"
WIKIDATA_RAW_SCHEMA_VERSION = "pc-build-recommender.wikidata-raw.v2"

DEFAULT_MAX_RECORDS = 100
HARD_MAX_RECORDS = 500
DEFAULT_SEARCH_LIMIT = 3
HARD_MAX_SEARCH_LIMIT = 5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_REQUEST_DELAY_SECONDS = 0.2
DEFAULT_MAXIMUM_RETRIES = 2
DEFAULT_MAXIMUM_RETRY_AFTER_SECONDS = 30.0
DEFAULT_MAXIMUM_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_MAXIMUM_TOTAL_BYTES = 16 * 1024 * 1024
DEFAULT_MAXIMUM_LINE_BYTES = 2 * 1024 * 1024
_ENTITY_ID_PATTERN = re.compile(r"^Q[1-9][0-9]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OFFICIAL_ACQUISITION_MODE = "official_api"
_FIXTURE_ACQUISITION_MODE = "local_fixture"

# Exact-name and manufacturer-part-number matches are intentionally limited by reviewed
# Wikidata identity context. GTIN remains globally unique and therefore does not require this
# context. Add new classes or manufacturers only with an official Wikidata evidence link and
# a regression fixture.
_EXACT_NAME_INSTANCE_IDS: dict[ComponentKind, frozenset[str]] = {
    ComponentKind.CPU: frozenset({"Q122967152"}),  # CPU model
}
_REVIEWED_MANUFACTURER_ENTITY_IDS: dict[str, frozenset[str]] = {
    "amd": frozenset({"Q128896"}),
    "advanced micro devices": frozenset({"Q128896"}),
    "intel": frozenset({"Q248"}),
}


class WikidataAPIError(RuntimeError):
    """Raised when the official Wikidata API cannot provide a valid bounded response."""


class WikidataResponseTooLargeError(WikidataAPIError):
    """Raised before a Wikidata response can exceed the configured memory budget."""


@dataclass(frozen=True, slots=True)
class WikidataCandidate:
    """Minimal canonical-product identity used to request safe enrichment."""

    candidate_id: str
    canonical_name: str
    category: str
    brand: str | None = None
    manufacturer_part_number: str | None = None
    gtin: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be blank")
        if not self.canonical_name.strip():
            raise ValueError("canonical_name must not be blank")
        ComponentKind(self.category)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WikidataCandidate:
        """Read either a compact candidate or a canonical-product mapping."""

        candidate_id = value.get("candidate_id", value.get("product_id"))
        return cls(
            candidate_id=_required_text(candidate_id, "candidate_id or product_id"),
            canonical_name=_required_text(value.get("canonical_name"), "canonical_name"),
            category=_required_text(value.get("category"), "category"),
            brand=_optional_text(value.get("brand")),
            manufacturer_part_number=_optional_text(value.get("manufacturer_part_number")),
            gtin=_optional_text(value.get("gtin")),
        )


def load_wikidata_candidates(
    path: str | Path,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
    maximum_line_bytes: int = DEFAULT_MAXIMUM_LINE_BYTES,
    categories: Sequence[ComponentKind | str] | None = None,
) -> list[WikidataCandidate]:
    """Stream a bounded, optionally category-targeted candidate file into memory.

    The category filter is applied while streaming, before the request budget is
    consumed. This permits a small, reviewed acquisition cohort (for example,
    CPUs with known Wikidata identity coverage) without materialising or
    sending unrelated catalogue records to the official API.
    """

    _validate_max_records(max_records)
    if maximum_line_bytes <= 0:
        raise ValueError("maximum_line_bytes must be positive")
    selected_categories: frozenset[ComponentKind] | None = None
    if categories is not None:
        try:
            selected_categories = frozenset(ComponentKind(value) for value in categories)
        except ValueError as exc:
            raise ValueError(
                "Wikidata candidate categories must be supported PC categories"
            ) from exc
        if not selected_categories:
            raise ValueError("Wikidata candidate categories must not be empty when supplied")
    candidates: list[WikidataCandidate] = []
    seen_ids: set[str] = set()
    with Path(path).open("rb") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline(maximum_line_bytes + 1)
            if not raw_line:
                break
            line_number += 1
            if len(raw_line) > maximum_line_bytes:
                # Drain the remainder in bounded fragments so even a malicious single line
                # cannot be allocated in full before the configured limit is enforced.
                while raw_line and not raw_line.endswith(b"\n"):
                    raw_line = handle.readline(maximum_line_bytes + 1)
                raise ValueError(
                    f"Wikidata candidate line {line_number} exceeds {maximum_line_bytes} bytes"
                )
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid Wikidata candidate JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"Wikidata candidate line {line_number} must be an object")
            data = (
                payload.get("data")
                if payload.get("record_type") == "canonical_product"
                else payload
            )
            if not isinstance(data, Mapping):
                raise ValueError(
                    f"Wikidata candidate line {line_number} is missing a product data object"
                )
            candidate = WikidataCandidate.from_mapping(data)
            if selected_categories is not None and ComponentKind(candidate.category) not in (
                selected_categories
            ):
                continue
            if candidate.candidate_id in seen_ids:
                raise ValueError(f"duplicate Wikidata candidate_id: {candidate.candidate_id}")
            seen_ids.add(candidate.candidate_id)
            candidates.append(candidate)
            if len(candidates) >= max_records:
                break
    if not candidates:
        if selected_categories is None:
            raise ValueError("Wikidata candidate file did not contain any usable products")
        names = ", ".join(sorted(category.value for category in selected_categories))
        raise ValueError(f"Wikidata candidate file did not contain products in categories: {names}")
    return candidates


class WikidataEnrichmentAdapter:
    """Fetch and parse bounded PC-component identity evidence from Wikidata."""

    def __init__(
        self,
        *,
        raw_root: str | Path,
        transport: httpx.BaseTransport | None = None,
        request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
        maximum_retries: int = DEFAULT_MAXIMUM_RETRIES,
        maximum_retry_after_seconds: float = DEFAULT_MAXIMUM_RETRY_AFTER_SECONDS,
        sleeper: Callable[[float], None] = time.sleep,
        maximum_response_bytes: int = DEFAULT_MAXIMUM_RESPONSE_BYTES,
        maximum_total_bytes: int = DEFAULT_MAXIMUM_TOTAL_BYTES,
    ) -> None:
        if maximum_response_bytes <= 0 or maximum_total_bytes <= 0:
            raise ValueError("Wikidata byte limits must be positive")
        if maximum_response_bytes > maximum_total_bytes:
            raise ValueError("maximum_response_bytes cannot exceed maximum_total_bytes")
        if request_delay_seconds < 0.2:
            raise ValueError("request_delay_seconds must be at least 0.2 seconds")
        if not 0 <= maximum_retries <= 3:
            raise ValueError("maximum_retries must be between 0 and 3")
        if maximum_retry_after_seconds <= 0:
            raise ValueError("maximum_retry_after_seconds must be positive")
        self.raw_root = Path(raw_root)
        self.transport = transport
        self.request_delay_seconds = request_delay_seconds
        self.maximum_retries = maximum_retries
        self.maximum_retry_after_seconds = maximum_retry_after_seconds
        self.sleeper = sleeper
        self.maximum_response_bytes = maximum_response_bytes
        self.maximum_total_bytes = maximum_total_bytes

    def fetch(
        self,
        candidates: Sequence[WikidataCandidate] | None = None,
        *,
        response_path: str | Path | None = None,
        max_records: int = DEFAULT_MAX_RECORDS,
        search_limit: int = DEFAULT_SEARCH_LIMIT,
        language: str = "en",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> RawSnapshot:
        """Fetch official API responses or snapshot a controlled response fixture.

        ``response_path`` is intended for deterministic reproduction and tests.  Live
        requests use only the official Action API and fail before configured byte limits.
        """

        _validate_max_records(max_records)
        _validate_search_limit(search_limit)
        _validate_language(language)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if response_path is not None:
            response = Path(response_path)
            if response.stat().st_size > self.maximum_total_bytes:
                raise WikidataResponseTooLargeError(
                    f"Wikidata fixture exceeds {self.maximum_total_bytes} bytes"
                )
            fixture_bytes = response.read_bytes()
            try:
                fixture_payload = json.loads(fixture_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WikidataAPIError(f"Wikidata fixture is not valid JSON: {exc}") from exc
            if not isinstance(fixture_payload, dict):
                raise WikidataAPIError("Wikidata fixture root must be an object")
            fixture_sha256 = sha256_bytes(fixture_bytes)
            fixture_payload = {
                **fixture_payload,
                "acquisition_mode": _FIXTURE_ACQUISITION_MODE,
                "fixture_source_sha256": fixture_sha256,
                "fixture_quarantined": True,
            }
            self.raw_root.mkdir(parents=True, exist_ok=True)
            fixture_temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=self.raw_root,
                    prefix=".wikidata-fixture.",
                    suffix=".json",
                    delete=False,
                ) as handle:
                    json.dump(
                        fixture_payload,
                        handle,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    handle.write("\n")
                    fixture_temporary_path = Path(handle.name)
                assert fixture_temporary_path is not None
                return snapshot_local_file(
                    source_name="wikidata_fixture",
                    source_url=f"local-fixture-sha256:{fixture_sha256}",
                    source_type="controlled_fixture",
                    source_path=fixture_temporary_path,
                    raw_root=self.raw_root,
                    parser_version=WIKIDATA_PARSER_VERSION,
                    licence_or_access_note=(
                        "Unverified local Wikidata-shaped fixture; quarantined from training, "
                        "redistribution, and published claims."
                    ),
                    suffix=".json",
                    media_type="application/json",
                )
            finally:
                if fixture_temporary_path is not None and fixture_temporary_path.exists():
                    fixture_temporary_path.unlink()
        if candidates is None or not candidates:
            raise ValueError("at least one Wikidata candidate is required for a live fetch")
        if len(candidates) > max_records:
            raise ValueError(
                f"received {len(candidates)} Wikidata candidates; max_records is {max_records}"
            )
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Wikidata candidate IDs must be unique")

        remaining_bytes = [self.maximum_total_bytes]
        search_responses: list[dict[str, Any]] = []
        entity_ids: list[str] = []
        seen_entity_ids: set[str] = set()
        request_headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Api-User-Agent": WIKIDATA_USER_AGENT,
            "User-Agent": WIKIDATA_USER_AGENT,
        }
        timeout = httpx.Timeout(timeout_seconds)
        try:
            with httpx.Client(
                timeout=timeout,
                headers=request_headers,
                transport=self.transport,
                follow_redirects=False,
            ) as client:
                for candidate in candidates:
                    payload = self._request_json(
                        client,
                        params={
                            "action": "wbsearchentities",
                            "format": "json",
                            "formatversion": "2",
                            "language": language,
                            "uselang": language,
                            "type": "item",
                            "limit": str(search_limit),
                            "search": candidate.canonical_name,
                        },
                        remaining_bytes=remaining_bytes,
                    )
                    search_responses.append(
                        {"candidate_id": candidate.candidate_id, "payload": payload}
                    )
                    for result in _sequence(payload.get("search")):
                        if not isinstance(result, Mapping):
                            continue
                        entity_id = _optional_text(result.get("id"))
                        if (
                            entity_id is not None
                            and _ENTITY_ID_PATTERN.fullmatch(entity_id)
                            and entity_id not in seen_entity_ids
                        ):
                            seen_entity_ids.add(entity_id)
                            entity_ids.append(entity_id)

                entity_responses: list[dict[str, Any]] = []
                for start in range(0, len(entity_ids), 50):
                    requested_ids = entity_ids[start : start + 50]
                    payload = self._request_json(
                        client,
                        params={
                            "action": "wbgetentities",
                            "format": "json",
                            "formatversion": "2",
                            "languages": language,
                            "props": "labels|aliases|claims|sitelinks",
                            "sitefilter": f"{language}wiki",
                            "ids": "|".join(requested_ids),
                        },
                        remaining_bytes=remaining_bytes,
                    )
                    entity_responses.append({"ids": requested_ids, "payload": payload})
        except WikidataAPIError:
            raise
        except httpx.HTTPError as exc:
            raise WikidataAPIError(f"Wikidata API request failed: {exc}") from exc

        raw_payload = {
            "schema_version": WIKIDATA_RAW_SCHEMA_VERSION,
            "acquisition_mode": _OFFICIAL_ACQUISITION_MODE,
            "api_url": WIKIDATA_API_URL,
            "licence": "CC0-1.0",
            "licence_url": WIKIDATA_LICENSE_URL,
            "language": language,
            "candidates": [asdict(candidate) for candidate in candidates],
            "search_responses": search_responses,
            "entity_responses": entity_responses,
        }
        self.raw_root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.raw_root,
                prefix=".wikidata.",
                suffix=".json",
                delete=False,
            ) as handle:
                json.dump(
                    raw_payload,
                    handle,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                handle.write("\n")
                temporary_path = Path(handle.name)
            if temporary_path.stat().st_size > self.maximum_total_bytes:
                raise WikidataResponseTooLargeError(
                    f"combined Wikidata response exceeds {self.maximum_total_bytes} bytes"
                )
            snapshot = snapshot_local_file(
                source_name="wikidata_cc0",
                source_url=WIKIDATA_API_URL,
                source_type="import",
                source_path=temporary_path,
                raw_root=self.raw_root,
                parser_version=WIKIDATA_PARSER_VERSION,
                licence_or_access_note=WIKIDATA_LICENSE_NOTE,
                suffix=".json",
                media_type="application/json",
            )
            return snapshot
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _request_json(
        self,
        client: httpx.Client,
        *,
        params: Mapping[str, str],
        remaining_bytes: list[int],
    ) -> dict[str, Any]:
        request_params = {"maxlag": "5", **params}
        for attempt in range(self.maximum_retries + 1):
            response_bytes = bytearray()
            with client.stream("GET", WIKIDATA_API_URL, params=request_params) as response:
                if response.status_code in {429, 503}:
                    if attempt >= self.maximum_retries:
                        raise WikidataAPIError(
                            "Wikidata throttled or deferred the request after bounded retries"
                        )
                    self.sleeper(self._retry_delay(response.headers.get("retry-after")))
                    continue
                response.raise_for_status()
                declared_length = response.headers.get("content-length")
                if declared_length is not None:
                    try:
                        declared_bytes = int(declared_length)
                    except ValueError as exc:
                        raise WikidataAPIError(
                            f"Wikidata returned invalid Content-Length: {declared_length!r}"
                        ) from exc
                    allowed = min(self.maximum_response_bytes, remaining_bytes[0])
                    if declared_bytes > allowed:
                        raise WikidataResponseTooLargeError(
                            f"Wikidata declared {declared_bytes} bytes; "
                            f"remaining limit is {allowed}"
                        )
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    response_bytes.extend(chunk)
                    if len(response_bytes) > self.maximum_response_bytes:
                        raise WikidataResponseTooLargeError(
                            f"Wikidata response exceeded {self.maximum_response_bytes} bytes"
                        )
                    if len(response_bytes) > remaining_bytes[0]:
                        raise WikidataResponseTooLargeError(
                            "Wikidata responses exceeded combined byte budget"
                        )
            remaining_bytes[0] -= len(response_bytes)
            try:
                payload = json.loads(response_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WikidataAPIError(f"Wikidata returned invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise WikidataAPIError("Wikidata JSON response root must be an object")
            error = payload.get("error")
            if isinstance(error, Mapping):
                code = _optional_text(error.get("code")) or "unknown"
                info = _optional_text(error.get("info")) or "no detail"
                if code == "maxlag" and attempt < self.maximum_retries:
                    self.sleeper(self._retry_delay(response.headers.get("retry-after")))
                    continue
                raise WikidataAPIError(f"Wikidata API error {code}: {info}")
            if error is not None:
                raise WikidataAPIError(f"Wikidata API error: {error}")
            self.sleeper(self.request_delay_seconds)
            return payload
        raise AssertionError("bounded Wikidata retry loop did not return or raise")

    def _retry_delay(self, retry_after: str | None) -> float:
        delay = max(self.request_delay_seconds, 1.0)
        if retry_after is not None:
            try:
                requested_delay = float(retry_after)
            except ValueError as exc:
                raise WikidataAPIError(
                    f"Wikidata returned invalid Retry-After: {retry_after!r}"
                ) from exc
            if requested_delay < 0:
                raise WikidataAPIError(f"Wikidata returned invalid Retry-After: {retry_after!r}")
            delay = max(self.request_delay_seconds, requested_delay)
        if delay > self.maximum_retry_after_seconds:
            raise WikidataAPIError(
                "Wikidata Retry-After exceeds configured wait budget: "
                f"{delay} > {self.maximum_retry_after_seconds} seconds"
            )
        return delay

    def parse(
        self,
        snapshot: RawSnapshot,
        *,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> ParsedBatch:
        """Normalise unique, exact Wikidata matches into enrichment records."""

        _validate_max_records(max_records)
        if snapshot.source_name not in {"wikidata_cc0", "wikidata_fixture"}:
            raise ValueError(f"unexpected Wikidata source: {snapshot.source_name}")
        if snapshot.byte_count > self.maximum_total_bytes:
            raise WikidataResponseTooLargeError(
                f"Wikidata snapshot exceeds {self.maximum_total_bytes} bytes"
            )
        with snapshot.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("Wikidata raw response root must be an object")
        if payload.get("schema_version") != WIKIDATA_RAW_SCHEMA_VERSION:
            raise ValueError("unsupported Wikidata raw response schema")
        if payload.get("api_url") != WIKIDATA_API_URL:
            raise ValueError("Wikidata raw response has an unexpected API URL")
        if payload.get("licence") != "CC0-1.0":
            raise ValueError("Wikidata raw response must declare CC0-1.0")
        acquisition_mode = _required_text(payload.get("acquisition_mode"), "acquisition_mode")
        if acquisition_mode not in {
            _OFFICIAL_ACQUISITION_MODE,
            _FIXTURE_ACQUISITION_MODE,
        }:
            raise ValueError("unsupported Wikidata acquisition mode")
        is_official = acquisition_mode == _OFFICIAL_ACQUISITION_MODE
        expected_source_name = "wikidata_cc0" if is_official else "wikidata_fixture"
        if snapshot.source_name != expected_source_name:
            raise ValueError("Wikidata acquisition mode does not match its snapshot source")
        if not is_official:
            fixture_hash = _required_text(
                payload.get("fixture_source_sha256"), "fixture_source_sha256"
            )
            if _SHA256_PATTERN.fullmatch(fixture_hash) is None:
                raise ValueError("Wikidata fixture source hash must be a SHA-256 digest")
            if payload.get("fixture_quarantined") is not True:
                raise ValueError("Wikidata local fixtures must remain quarantined")
        language = _required_text(payload.get("language"), "language")
        _validate_language(language)

        candidates = self._parse_candidates(payload.get("candidates"), max_records=max_records)
        search_ids_by_candidate = self._parse_search_results(payload.get("search_responses"))
        entities = self._parse_entities(payload.get("entity_responses"))
        batch = ParsedBatch(
            source_name=snapshot.source_name,
            snapshot_sha256=snapshot.content_sha256,
        )
        match_methods: dict[str, int] = {}
        for candidate in candidates:
            entity_ids = search_ids_by_candidate.get(candidate.candidate_id, [])
            ranked: list[tuple[int, str, str, dict[str, Any]]] = []
            for entity_id in entity_ids:
                entity = entities.get(entity_id)
                if entity is None:
                    continue
                match = _match_candidate(candidate, entity, language=language)
                if match is not None:
                    rank, method = match
                    ranked.append((rank, entity_id, method, entity))
            if not ranked:
                batch.rejected.append(
                    rejected_record(
                        candidate.candidate_id,
                        "no_exact_wikidata_identity_match",
                        canonical_name=candidate.canonical_name,
                        searched_entity_ids=entity_ids,
                    )
                )
                continue
            ranked.sort(key=lambda value: (-value[0], value[1]))
            best_rank = ranked[0][0]
            best = [value for value in ranked if value[0] == best_rank]
            if len(best) != 1:
                batch.rejected.append(
                    rejected_record(
                        candidate.candidate_id,
                        "ambiguous_wikidata_identity_match",
                        canonical_name=candidate.canonical_name,
                        entity_ids=[value[1] for value in best],
                    )
                )
                continue
            _, entity_id, method, entity = best[0]
            record = self._normalise_entity(
                candidate=candidate,
                entity_id=entity_id,
                entity=entity,
                method=method,
                language=language,
                snapshot=snapshot,
                is_official=is_official,
            )
            batch.records.append(record)
            match_methods[method] = match_methods.get(method, 0) + 1

        batch.statistics = {
            "candidate_count": len(candidates),
            "entity_count": len(entities),
            "matched_count": batch.accepted_count,
            "unmatched_or_ambiguous_count": batch.rejected_count,
            "match_methods": dict(sorted(match_methods.items())),
            "language": language,
            "max_records": max_records,
            "contains_retailer_prices": False,
            "licence": "CC0-1.0",
            "acquisition_mode": acquisition_mode,
            "training_eligible": False,
            "standalone_enrichment_only": True,
            "downstream_consumer_validation_required": True,
            "downstream_consumer_validation_status": "pending",
        }
        return batch

    @staticmethod
    def _parse_candidates(value: object, *, max_records: int) -> list[WikidataCandidate]:
        candidates: list[WikidataCandidate] = []
        seen_ids: set[str] = set()
        for raw_candidate in _sequence(value):
            if len(candidates) >= max_records:
                break
            if not isinstance(raw_candidate, Mapping):
                raise TypeError("Wikidata candidate entries must be objects")
            candidate = WikidataCandidate.from_mapping(raw_candidate)
            if candidate.candidate_id in seen_ids:
                raise ValueError(f"duplicate Wikidata candidate_id: {candidate.candidate_id}")
            seen_ids.add(candidate.candidate_id)
            candidates.append(candidate)
        if not candidates:
            raise ValueError("Wikidata response contains no candidates")
        return candidates

    @staticmethod
    def _parse_search_results(value: object) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for response in _sequence(value):
            if not isinstance(response, Mapping):
                raise TypeError("Wikidata search response entries must be objects")
            candidate_id = _required_text(response.get("candidate_id"), "candidate_id")
            payload = response.get("payload")
            if not isinstance(payload, Mapping):
                raise TypeError("Wikidata search response payload must be an object")
            entity_ids: list[str] = []
            for search_hit in _sequence(payload.get("search")):
                if not isinstance(search_hit, Mapping):
                    continue
                entity_id = _optional_text(search_hit.get("id"))
                if entity_id and _ENTITY_ID_PATTERN.fullmatch(entity_id):
                    entity_ids.append(entity_id)
            if candidate_id in result:
                raise ValueError(f"duplicate Wikidata search response: {candidate_id}")
            result[candidate_id] = list(dict.fromkeys(entity_ids))
        return result

    @staticmethod
    def _parse_entities(value: object) -> dict[str, dict[str, Any]]:
        entities: dict[str, dict[str, Any]] = {}
        for response in _sequence(value):
            if not isinstance(response, Mapping):
                raise TypeError("Wikidata entity response entries must be objects")
            payload = response.get("payload")
            if not isinstance(payload, Mapping):
                raise TypeError("Wikidata entity response payload must be an object")
            response_entities = payload.get("entities")
            if not isinstance(response_entities, Mapping):
                raise TypeError("Wikidata entity response is missing entities")
            for entity_id, raw_entity in response_entities.items():
                entity_id_text = str(entity_id)
                if not _ENTITY_ID_PATTERN.fullmatch(entity_id_text):
                    continue
                if not isinstance(raw_entity, Mapping) or raw_entity.get("missing") is not None:
                    continue
                if entity_id_text in entities:
                    raise ValueError(f"duplicate Wikidata entity response: {entity_id_text}")
                entities[entity_id_text] = dict(raw_entity)
        return entities

    @staticmethod
    def _normalise_entity(
        *,
        candidate: WikidataCandidate,
        entity_id: str,
        entity: dict[str, Any],
        method: str,
        language: str,
        snapshot: RawSnapshot,
        is_official: bool,
    ) -> dict[str, Any]:
        labels = _language_values(entity.get("labels"), language)
        aliases = _language_values(entity.get("aliases"), language)
        label = labels[0] if labels else candidate.canonical_name
        mpns = _string_claim_values(entity, "P13802")
        gtins = _string_claim_values(entity, "P3962")
        manufacturer_ids = _entity_claim_values(entity, "P176")
        instance_ids = _entity_claim_values(entity, "P31")
        release = _release_date(entity)
        wikipedia_url = _wikipedia_url(entity, language=language)
        confidence = {
            "exact_gtin_and_mpn": 1.0,
            "exact_gtin": 0.99,
            "exact_mpn": 0.98,
            "exact_name": 0.90,
        }[method]
        raw_entity_bytes = json.dumps(
            entity,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return {
            "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
            "record_type": "catalogue_identity_enrichment",
            "source_record_id": f"{candidate.candidate_id}:{entity_id}",
            "archive_snapshot_sha256": snapshot.content_sha256,
            "raw_record_sha256": sha256_bytes(raw_entity_bytes),
            "training_eligible": False,
            "training_scope": "quarantined_until_downstream_consumer_validation",
            "published_claims_eligible": False,
            "redistribution_eligible": is_official,
            "development_only": True,
            "rights_metadata": {
                "rights_basis": "open_licence" if is_official else "unverified_local_fixture",
                "licence": "CC0-1.0" if is_official else "unverified-fixture-claim",
                "may_display": is_official,
                "may_cache": is_official,
                "may_redistribute": is_official,
                "may_derive": is_official,
                "may_embed": is_official,
                "may_train": is_official,
                "use_scope": "identity_and_alias_features_only",
            },
            "provenance": {
                "source_name": snapshot.source_name,
                "source_url": (
                    WIKIDATA_ENTITY_URL.format(entity_id=entity_id)
                    if is_official
                    else snapshot.source_url
                ),
                "source_type": "official_api" if is_official else "controlled_fixture",
                "retrieved_at": snapshot.retrieved_at.isoformat(),
                "raw_content_hash": sha256_bytes(raw_entity_bytes),
                "parser_version": WIKIDATA_PARSER_VERSION,
                "licence": "CC0-1.0" if is_official else "unverified-fixture-claim",
                "licence_url": WIKIDATA_LICENSE_URL if is_official else None,
                "licence_or_access_note": snapshot.licence_or_access_note,
                "extraction_confidence": confidence,
            },
            "normalisation_metadata": {
                "match_method": method,
                "match_confidence": confidence,
                "language": language,
                "contains_retailer_prices": False,
                "applies_to_product_id": candidate.candidate_id,
                "standalone_enrichment_only": True,
                "downstream_consumer_validation_required": True,
                "downstream_consumer_validation_status": "pending",
            },
            "data": {
                "product_id": candidate.candidate_id,
                "category": candidate.category,
                "canonical_name": candidate.canonical_name,
                "wikidata_entity_id": entity_id,
                "entity_url": WIKIDATA_ENTITY_URL.format(entity_id=entity_id),
                "wikipedia_url": wikipedia_url,
                "label": label,
                "aliases": [
                    value for value in aliases if _normalise_text(value) != _normalise_text(label)
                ],
                "manufacturer_part_numbers": mpns,
                "gtins": gtins,
                "manufacturer_entity_ids": manufacturer_ids,
                "instance_of_entity_ids": instance_ids,
                "release_date": release,
            },
        }


def _match_candidate(
    candidate: WikidataCandidate,
    entity: Mapping[str, Any],
    *,
    language: str,
) -> tuple[int, str] | None:
    candidate_mpn = _normalise_identifier(candidate.manufacturer_part_number)
    candidate_gtin = _normalise_gtin(candidate.gtin)
    entity_mpns = {_normalise_identifier(value) for value in _string_claim_values(entity, "P13802")}
    entity_gtins = {_normalise_gtin(value) for value in _string_claim_values(entity, "P3962")}
    entity_mpns.discard(None)
    entity_gtins.discard(None)
    mpn_identifier_match = candidate_mpn is not None and candidate_mpn in entity_mpns
    mpn_match = mpn_identifier_match and _exact_mpn_context_matches(candidate, entity)
    gtin_match = candidate_gtin is not None and candidate_gtin in entity_gtins
    if mpn_match and gtin_match:
        return 4, "exact_gtin_and_mpn"
    if gtin_match:
        return 3, "exact_gtin"
    if mpn_match:
        return 2, "exact_mpn"

    # Conflicting known identifiers override a coincidental label match.
    if candidate_mpn is not None and entity_mpns and not mpn_match:
        return None
    if candidate_gtin is not None and entity_gtins and not gtin_match:
        return None
    names = _language_values(entity.get("labels"), language) + _language_values(
        entity.get("aliases"), language
    )
    candidate_name = _normalise_text(candidate.canonical_name)
    if (
        candidate_name
        and any(_normalise_text(name) == candidate_name for name in names)
        and _exact_name_context_matches(candidate, entity, names=names)
    ):
        return 1, "exact_name"
    return None


def _exact_name_context_matches(
    candidate: WikidataCandidate,
    entity: Mapping[str, Any],
    *,
    names: Sequence[str],
) -> bool:
    """Require reviewed category/type and manufacturer context for a name-only match."""

    if not _reviewed_identity_context_matches(candidate, entity, require_known_category=True):
        return False

    brand = _normalise_text(candidate.brand or "")
    return any(
        (normalised_name := _normalise_text(name)) == brand
        or normalised_name.startswith(f"{brand} ")
        for name in names
    )


def _exact_mpn_context_matches(
    candidate: WikidataCandidate,
    entity: Mapping[str, Any],
) -> bool:
    """Require manufacturer agreement and any reviewed category evidence for an MPN."""

    return _reviewed_identity_context_matches(candidate, entity, require_known_category=False)


def _reviewed_identity_context_matches(
    candidate: WikidataCandidate,
    entity: Mapping[str, Any],
    *,
    require_known_category: bool,
) -> bool:
    """Validate a candidate against reviewed Wikidata manufacturer and type identifiers."""

    if entity.get("type") != "item":
        return False
    brand = _normalise_text(candidate.brand or "")
    expected_manufacturers = _REVIEWED_MANUFACTURER_ENTITY_IDS.get(brand, frozenset())
    manufacturer_ids = set(_entity_claim_values(entity, "P176"))
    if not expected_manufacturers or manufacturer_ids.isdisjoint(expected_manufacturers):
        return False

    category = ComponentKind(candidate.category)
    allowed_instance_ids = _EXACT_NAME_INSTANCE_IDS.get(category)
    if allowed_instance_ids is None:
        instance_ids = set(_entity_claim_values(entity, "P31"))
        reviewed_other_category_ids = set().union(*_EXACT_NAME_INSTANCE_IDS.values())
        if not instance_ids.isdisjoint(reviewed_other_category_ids):
            return False
        return not require_known_category
    instance_ids = set(_entity_claim_values(entity, "P31"))
    return not instance_ids.isdisjoint(allowed_instance_ids)


def _string_claim_values(entity: Mapping[str, Any], property_id: str) -> list[str]:
    values: list[str] = []
    for value in _claim_values(entity, property_id):
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return sorted(dict.fromkeys(values), key=str.casefold)


def _entity_claim_values(entity: Mapping[str, Any], property_id: str) -> list[str]:
    values: list[str] = []
    for value in _claim_values(entity, property_id):
        if not isinstance(value, Mapping):
            continue
        entity_id = _optional_text(value.get("id"))
        if entity_id and _ENTITY_ID_PATTERN.fullmatch(entity_id):
            values.append(entity_id)
    return sorted(dict.fromkeys(values))


def _claim_values(entity: Mapping[str, Any], property_id: str) -> list[object]:
    claims = entity.get("claims")
    if not isinstance(claims, Mapping):
        return []
    values: list[object] = []
    for claim in _sequence(claims.get(property_id)):
        if not isinstance(claim, Mapping) or claim.get("rank") == "deprecated":
            continue
        mainsnak = claim.get("mainsnak")
        if not isinstance(mainsnak, Mapping) or mainsnak.get("snaktype") != "value":
            continue
        datavalue = mainsnak.get("datavalue")
        if isinstance(datavalue, Mapping) and "value" in datavalue:
            values.append(datavalue["value"])
    return values


def _release_date(entity: Mapping[str, Any]) -> dict[str, object] | None:
    for property_id in ("P577", "P571"):
        for value in _claim_values(entity, property_id):
            if not isinstance(value, Mapping):
                continue
            raw_time = _optional_text(value.get("time"))
            precision = value.get("precision")
            if raw_time is None or not isinstance(precision, int):
                continue
            match = re.fullmatch(r"\+([0-9]{4,})-([0-9]{2})-([0-9]{2})T.*", raw_time)
            if match is None:
                continue
            year, month, day = match.groups()
            if precision >= 11:
                text = f"{year}-{month}-{day}"
                precision_name = "day"
            elif precision == 10:
                text = f"{year}-{month}"
                precision_name = "month"
            elif precision == 9:
                text = year
                precision_name = "year"
            else:
                continue
            return {
                "value": text,
                "precision": precision_name,
                "property_id": property_id,
            }
    return None


def _wikipedia_url(entity: Mapping[str, Any], *, language: str) -> str | None:
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, Mapping):
        return None
    sitelink = sitelinks.get(f"{language}wiki")
    if not isinstance(sitelink, Mapping):
        return None
    explicit_url = _optional_text(sitelink.get("url"))
    if explicit_url is not None:
        return explicit_url
    title = _optional_text(sitelink.get("title"))
    if title is None:
        return None
    return f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def _language_values(value: object, language: str) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    language_value = value.get(language)
    raw_values = language_value if isinstance(language_value, list) else [language_value]
    values: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, Mapping):
            continue
        text = _optional_text(raw.get("value"))
        if text is not None:
            values.append(text)
    return list(dict.fromkeys(values))


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} must not be blank")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _normalise_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _normalise_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalised = "".join(character for character in value.casefold() if character.isalnum())
    return normalised or None


def _normalise_gtin(value: str | None) -> str | None:
    if value is None:
        return None
    normalised = "".join(character for character in value if character.isdigit())
    return normalised or None


def _validate_max_records(max_records: int) -> None:
    if not 1 <= max_records <= HARD_MAX_RECORDS:
        raise ValueError(f"max_records must be between 1 and {HARD_MAX_RECORDS}")


def _validate_search_limit(search_limit: int) -> None:
    if not 1 <= search_limit <= HARD_MAX_SEARCH_LIMIT:
        raise ValueError(f"search_limit must be between 1 and {HARD_MAX_SEARCH_LIMIT}")


def _validate_language(language: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,11}", language):
        raise ValueError("language must be a short lowercase Wikimedia language code")


__all__ = [
    "WIKIDATA_API_URL",
    "WIKIDATA_LICENSE_NOTE",
    "WIKIDATA_LICENSE_URL",
    "WIKIDATA_PARSER_VERSION",
    "WIKIDATA_USER_AGENT",
    "WikidataAPIError",
    "WikidataCandidate",
    "WikidataEnrichmentAdapter",
    "WikidataResponseTooLargeError",
    "load_wikidata_candidates",
]
