"""SQLAlchemy-backed persistence for generated results and interaction events."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from pc_build_recommender.application import (
    ApplicationBuildGenerationResponse,
    ApplicationError,
    RequestConflictError,
    ResultNotFoundError,
    StoredGeneration,
)
from pc_build_recommender.catalog import (
    BuildShareRecord,
    CanonicalProductRecord,
    CatalogRepository,
    GeneratedBuildRecord,
    InteractionEventRecord,
    RetailerListingRecord,
    SearchQueryRecord,
    create_db_engine,
    create_session_factory,
    session_scope,
)
from pc_build_recommender.domain import (
    BuildGenerationRequest,
    BuildRecommendation,
    CanonicalProduct,
    InteractionEvent,
    RetailerListing,
    SearchQuery,
)
from pc_build_recommender.retrieval.embedding_index import (
    TEXT_BUILDER_VERSION,
    build_product_embedding_text,
)

_ENVELOPE_KEY = "_pcbr_application_generation"
_ENVELOPE_VERSION = 2
_REQUIRED_TABLES = frozenset(
    {
        "canonical_products",
        "retailer_listings",
        "search_queries",
        "generated_builds",
        "build_components",
        "build_shares",
        "interaction_events",
    }
)
_REQUIRED_COLUMNS = {
    "interaction_events": frozenset(
        {
            "impression_id",
            "trust_level",
            "idempotency_key_sha256",
            "idempotency_payload_sha256",
        }
    )
}


class DurableStorageError(ApplicationError):
    """The configured durable backend could not safely complete an operation."""


@dataclass(frozen=True, slots=True)
class StoredBuildShare:
    """Storage-neutral representation of a public snapshot, never including its revoke secret."""

    share_id: str
    snapshot: dict[str, Any]
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class CreatedBuildShare(StoredBuildShare):
    revocation_token: str


@dataclass(frozen=True, slots=True)
class InteractionWriteResult:
    """Outcome of an idempotent interaction write."""

    event: InteractionEvent
    replayed: bool


def _is_unique_violation(error: IntegrityError) -> bool:
    original = error.orig
    sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    if sqlstate == "23505":
        return True
    message = str(original).casefold()
    return "unique constraint failed" in message or "duplicate key value" in message


def _batches(values: Iterable[str], size: int = 500) -> Iterable[tuple[str, ...]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _canonical_product_from_record(record: CanonicalProductRecord) -> CanonicalProduct:
    return CanonicalProduct.model_validate(
        {
            "product_id": record.product_id,
            "category": record.category,
            "brand": record.brand,
            "model": record.model,
            "manufacturer_part_number": record.manufacturer_part_number,
            "gtin": record.gtin,
            "canonical_name": record.canonical_name,
            "release_date": record.release_date,
            "status": record.status,
            "common_attributes": record.common_attributes or {},
            "category_attributes": record.category_attributes or {},
            "source_confidence": record.source_confidence,
            "created_at": _as_utc(record.created_at),
            "updated_at": _as_utc(record.updated_at),
        }
    )


def _interaction_from_record(record: InteractionEventRecord) -> InteractionEvent:
    return InteractionEvent.model_validate(
        {
            "event_id": record.event_id,
            "session_id": record.session_id,
            "user_id": record.user_id,
            "query_id": record.query_id,
            "product_id": record.product_id,
            "build_id": record.build_id,
            "event_type": record.event_type,
            "rank_position": record.rank_position,
            "model_version": record.model_version,
            "data_version": record.data_version,
            "rule_version": record.rule_version,
            "metadata": record.event_metadata or {},
            "impression_id": record.impression_id,
            "trust_level": record.trust_level,
            "idempotency_key_sha256": record.idempotency_key_sha256,
            "idempotency_payload_sha256": record.idempotency_payload_sha256,
            "created_at": _as_utc(record.created_at),
        }
    )


def _idempotent_replay(
    existing: InteractionEventRecord, proposed: InteractionEvent
) -> InteractionWriteResult:
    if (
        proposed.idempotency_key_sha256 is None
        or existing.session_id != proposed.session_id
        or existing.idempotency_key_sha256 != proposed.idempotency_key_sha256
        or existing.idempotency_payload_sha256 != proposed.idempotency_payload_sha256
    ):
        raise RequestConflictError(
            "Idempotency-Key was already used for a different interaction"
        )
    return InteractionWriteResult(event=_interaction_from_record(existing), replayed=True)


def _impression_semantic_replay(
    existing: InteractionEventRecord, proposed: InteractionEvent
) -> InteractionWriteResult:
    stored = _interaction_from_record(existing)
    if (
        proposed.impression_id is None
        or stored.impression_id != proposed.impression_id
        or stored.event_type != proposed.event_type
    ):
        raise RequestConflictError("interaction already exists")
    excluded = {
        "created_at",
        "event_id",
        "idempotency_key_sha256",
        "idempotency_payload_sha256",
    }
    if stored.model_dump(mode="json", exclude=excluded) != proposed.model_dump(
        mode="json", exclude=excluded
    ):
        raise RequestConflictError(
            "Impression was already used for a different interaction payload"
        )
    return InteractionWriteResult(event=stored, replayed=True)


def _resolve_existing_interaction(
    existing: InteractionEventRecord, proposed: InteractionEvent
) -> InteractionWriteResult:
    if (
        proposed.idempotency_key_sha256 is not None
        and existing.session_id == proposed.session_id
        and existing.idempotency_key_sha256 == proposed.idempotency_key_sha256
    ):
        return _idempotent_replay(existing, proposed)
    if (
        proposed.impression_id is not None
        and existing.impression_id == proposed.impression_id
        and existing.event_type == proposed.event_type.value
    ):
        return _impression_semantic_replay(existing, proposed)
    raise RequestConflictError("interaction already exists")


def _retailer_listing_from_record(record: RetailerListingRecord) -> RetailerListing:
    return RetailerListing.model_validate(
        {
            "listing_id": record.listing_id,
            "product_id": record.product_id,
            "retailer": record.retailer,
            "source_listing_id": record.source_listing_id,
            "title": record.title,
            "condition": record.condition,
            "currency": record.currency,
            "base_price": record.base_price,
            "shipping_price": record.shipping_price,
            "stock_status": record.stock_status,
            "seller_name": record.seller_name,
            "listing_url": record.listing_url,
            "first_seen_at": _as_utc(record.first_seen_at),
            "last_seen_at": _as_utc(record.last_seen_at),
        }
    )


def _canonical_product_row_sha256(product: CanonicalProduct) -> str:
    return _sha256_json(product.model_dump(mode="json", exclude={"provenance"}))


def _retailer_listing_row_sha256(listing: RetailerListing) -> str:
    return _sha256_json(listing.model_dump(mode="json"))


def _search_document_identity(product: CanonicalProduct) -> tuple[str, str]:
    search_document = build_product_embedding_text(product.model_dump(mode="json"))
    content = f"{TEXT_BUILDER_VERSION}\0{product.product_id}\0{search_document}".encode()
    return search_document, hashlib.sha256(content).hexdigest()


class SqlAlchemyDurableStore:
    """Persist complete application generations and feedback with database transactions.

    The existing search/build tables remain queryable for analytics. An exact, versioned
    application envelope is additionally stored on ``search_queries`` so refresh and replacement
    behavior survives process restarts without reconstructing optimizer metadata heuristically.
    """

    def __init__(
        self,
        engine: Engine,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self.engine = engine
        self.session_factory = session_factory or create_session_factory(engine)

    @classmethod
    def from_url(cls, database_url: str) -> SqlAlchemyDurableStore:
        engine = create_db_engine(database_url)
        return cls(engine)

    def verify_schema(self) -> None:
        """Fail fast when the database is unreachable or migrations are missing."""

        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            inspector = inspect(self.engine)
            available = set(inspector.get_table_names())
        except SQLAlchemyError as error:
            raise DurableStorageError("durable database is unavailable") from error
        missing = sorted(_REQUIRED_TABLES - available)
        if missing:
            raise DurableStorageError(
                "durable database is missing required migrated tables: " + ", ".join(missing)
            )
        missing_columns = {
            table: sorted(required - {str(item["name"]) for item in inspector.get_columns(table)})
            for table, required in _REQUIRED_COLUMNS.items()
            if table in available
        }
        missing_columns = {
            table: columns for table, columns in missing_columns.items() if columns
        }
        if missing_columns:
            detail = "; ".join(
                f"{table}: {', '.join(columns)}"
                for table, columns in sorted(missing_columns.items())
            )
            raise DurableStorageError(
                "durable database is missing required migrated columns (revision "
                f"20260815_0008): {detail}"
            )

    def verify_no_unexpected_catalog_ids(
        self,
        *,
        product_ids: Iterable[str],
        listing_ids: Iterable[str],
    ) -> None:
        """Reject stale catalogue rows before a production release mutates the database.

        Missing expected rows are permitted because the subsequent idempotent import creates
        them. Rows outside the pinned product/listing universe require a separate, explicitly
        authorized reconciliation workflow; this check never deletes them.
        """

        expected_products = frozenset(product_ids)
        expected_listings = frozenset(listing_ids)
        try:
            with self.session_factory() as session:
                known_products = frozenset(
                    session.scalars(select(CanonicalProductRecord.product_id))
                )
                known_listings = frozenset(
                    session.scalars(select(RetailerListingRecord.listing_id))
                )
        except SQLAlchemyError as error:
            raise DurableStorageError(
                "durable catalogue stale-row preflight failed"
            ) from error

        unexpected_products = sorted(known_products - expected_products)
        unexpected_listings = sorted(known_listings - expected_listings)
        if not unexpected_products and not unexpected_listings:
            return

        details: list[str] = []
        if unexpected_products:
            details.append(
                f"{len(unexpected_products)} stale canonical products, including "
                + ", ".join(unexpected_products[:3])
            )
        if unexpected_listings:
            details.append(
                f"{len(unexpected_listings)} stale retailer listings, including "
                + ", ".join(unexpected_listings[:3])
            )
        raise DurableStorageError(
            "production catalogue release requires explicit audited stale-row "
            "reconciliation: " + "; ".join(details)
        )

    def verify_catalog_identity(
        self,
        *,
        product_ids: Iterable[str],
        listing_ids: Iterable[str],
        canonical_products: Iterable[CanonicalProduct] | None = None,
        retailer_listings: Iterable[RetailerListing] | None = None,
    ) -> None:
        """Ensure durable catalogue rows match the catalogue loaded for serving.

        The ID-only form remains available to development import tooling. Production startup
        supplies the pinned canonical products and retailer listings, which activates exact set,
        canonical-row, listing-row, and search-document verification.
        """

        expected_products = frozenset(product_ids)
        expected_listings = frozenset(listing_ids)
        strict_identity = canonical_products is not None or retailer_listings is not None
        if strict_identity and (canonical_products is None or retailer_listings is None):
            raise DurableStorageError(
                "strict durable catalogue verification requires products and listings together"
            )

        release_products = tuple(canonical_products or ())
        release_listings = tuple(retailer_listings or ())
        if strict_identity:
            release_product_ids = [product.product_id for product in release_products]
            release_listing_ids = [listing.listing_id for listing in release_listings]
            if (
                len(release_product_ids) != len(set(release_product_ids))
                or frozenset(release_product_ids) != expected_products
            ):
                raise DurableStorageError(
                    "strict durable catalogue product identities do not match the expected IDs"
                )
            if (
                len(release_listing_ids) != len(set(release_listing_ids))
                or frozenset(release_listing_ids) != expected_listings
            ):
                raise DurableStorageError(
                    "strict durable catalogue listing identities do not match the expected IDs"
                )

        try:
            with self.session_factory() as session:
                product_records: tuple[CanonicalProductRecord, ...] = ()
                listing_records: tuple[RetailerListingRecord, ...] = ()
                known_products: set[str] = set()
                known_listings: set[str] = set()
                if strict_identity:
                    product_records = tuple(
                        session.scalars(
                            select(CanonicalProductRecord).order_by(
                                CanonicalProductRecord.product_id
                            )
                        )
                    )
                    listing_records = tuple(
                        session.scalars(
                            select(RetailerListingRecord).order_by(
                                RetailerListingRecord.listing_id
                            )
                        )
                    )
                    known_products = {record.product_id for record in product_records}
                    known_listings = {record.listing_id for record in listing_records}
                else:
                    for batch in _batches(sorted(expected_products)):
                        known_products.update(
                            session.scalars(
                                select(CanonicalProductRecord.product_id).where(
                                    CanonicalProductRecord.product_id.in_(batch)
                                )
                            )
                        )
                    for batch in _batches(sorted(expected_listings)):
                        known_listings.update(
                            session.scalars(
                                select(RetailerListingRecord.listing_id).where(
                                    RetailerListingRecord.listing_id.in_(batch)
                                )
                            )
                        )
        except SQLAlchemyError as error:
            raise DurableStorageError("durable catalogue identity check failed") from error

        missing_products = sorted(expected_products - known_products)
        missing_listings = sorted(expected_listings - known_listings)
        unexpected_products = (
            sorted(known_products - expected_products) if strict_identity else []
        )
        unexpected_listings = (
            sorted(known_listings - expected_listings) if strict_identity else []
        )
        if (
            missing_products
            or missing_listings
            or unexpected_products
            or unexpected_listings
        ):
            details: list[str] = []
            if missing_products:
                details.append(
                    f"{len(missing_products)} canonical products, including "
                    + ", ".join(missing_products[:3])
                )
            if missing_listings:
                details.append(
                    f"{len(missing_listings)} retailer listings, including "
                    + ", ".join(missing_listings[:3])
                )
            if unexpected_products:
                details.append(
                    f"{len(unexpected_products)} unexpected canonical products, including "
                    + ", ".join(unexpected_products[:3])
                )
            if unexpected_listings:
                details.append(
                    f"{len(unexpected_listings)} unexpected retailer listings, including "
                    + ", ".join(unexpected_listings[:3])
                )
            raise DurableStorageError(
                "durable database catalogue does not match serving artifacts: " + "; ".join(details)
            )

        if not strict_identity:
            return

        expected_product_hashes = {
            product.product_id: _canonical_product_row_sha256(product)
            for product in release_products
        }
        expected_listing_hashes = {
            listing.listing_id: _retailer_listing_row_sha256(listing)
            for listing in release_listings
        }
        expected_search_documents = {
            product.product_id: _search_document_identity(product)
            for product in release_products
        }
        try:
            actual_product_hashes = {
                record.product_id: _canonical_product_row_sha256(
                    _canonical_product_from_record(record)
                )
                for record in product_records
            }
            actual_listing_hashes = {
                record.listing_id: _retailer_listing_row_sha256(
                    _retailer_listing_from_record(record)
                )
                for record in listing_records
            }
        except (TypeError, ValueError) as error:
            raise DurableStorageError(
                "durable database catalogue contains an invalid canonical row"
            ) from error

        changed_products = sorted(
            product_id
            for product_id, expected_hash in expected_product_hashes.items()
            if actual_product_hashes[product_id] != expected_hash
        )
        changed_listings = sorted(
            listing_id
            for listing_id, expected_hash in expected_listing_hashes.items()
            if actual_listing_hashes[listing_id] != expected_hash
        )
        changed_search_documents = sorted(
            record.product_id
            for record in product_records
            if (record.search_document, record.search_document_hash)
            != expected_search_documents[record.product_id]
        )
        if changed_products or changed_listings or changed_search_documents:
            details = []
            if changed_products:
                details.append(
                    f"{len(changed_products)} modified canonical product rows, including "
                    + ", ".join(changed_products[:3])
                )
            if changed_listings:
                details.append(
                    f"{len(changed_listings)} modified retailer listing rows, including "
                    + ", ".join(changed_listings[:3])
                )
            if changed_search_documents:
                details.append(
                    f"{len(changed_search_documents)} modified search documents, including "
                    + ", ".join(changed_search_documents[:3])
                )
            raise DurableStorageError(
                "durable database catalogue hashes do not match serving artifacts: "
                + "; ".join(details)
            )

    def is_ready(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            return False
        return True

    @staticmethod
    def _search_query_from_record(record: SearchQueryRecord) -> SearchQuery:
        created_at = record.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return SearchQuery(
            query_id=record.query_id,
            raw_query=record.raw_query,
            structured_constraints=dict(record.structured_constraints or {}),
            created_at=created_at,
        )

    @staticmethod
    def _same_search_identity(left: SearchQuery, right: SearchQuery) -> bool:
        # The normalized, versioned constraints define identity. Preserve the first submitted
        # spelling in ``raw_query`` when equivalent case/whitespace variants are retried.
        return (
            left.query_id == right.query_id
            and left.structured_constraints == right.structured_constraints
        )

    def get_search_query(self, query_id: str) -> SearchQuery | None:
        try:
            with self.session_factory() as session:
                record = session.get(SearchQueryRecord, query_id)
                return None if record is None else self._search_query_from_record(record)
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to read durable search query") from error

    def save_search_query(self, query: SearchQuery) -> SearchQuery:
        """Idempotently bind a stable product-search identity before returning it to clients."""

        proposed = query.model_copy(deep=True)
        try:
            with session_scope(self.session_factory) as session:
                record = session.get(SearchQueryRecord, proposed.query_id)
                if record is not None:
                    existing = self._search_query_from_record(record)
                    if not self._same_search_identity(existing, proposed):
                        raise RequestConflictError(
                            "query_id is already bound to another search: "
                            f"{proposed.query_id}"
                        )
                    return existing.model_copy(deep=True)
                CatalogRepository(session).save_search_query(proposed)
        except RequestConflictError:
            raise
        except IntegrityError as error:
            # A concurrent identical search may have won the insert race.
            prior = self.get_search_query(proposed.query_id)
            if prior is not None and self._same_search_identity(prior, proposed):
                return prior.model_copy(deep=True)
            raise RequestConflictError(
                f"durable search query identity conflict for {proposed.query_id}"
            ) from error
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to persist product search query") from error
        return proposed.model_copy(deep=True)

    @staticmethod
    def _envelope(
        request: BuildGenerationRequest,
        response: ApplicationBuildGenerationResponse,
        no_cost_product_ids: frozenset[str],
        owned_product_ids: frozenset[str],
        stored_at: datetime,
    ) -> dict[str, object]:
        return {
            "schema_version": _ENVELOPE_VERSION,
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
            "no_cost_product_ids": sorted(no_cost_product_ids),
            "owned_product_ids": sorted(owned_product_ids),
            "stored_at": stored_at.isoformat(),
        }

    @staticmethod
    def _decode(record: SearchQueryRecord) -> StoredGeneration | None:
        payload = record.structured_constraints or {}
        raw = payload.get(_ENVELOPE_KEY)
        if not isinstance(raw, dict):
            return None
        schema_version = raw.get("schema_version")
        if schema_version not in {1, _ENVELOPE_VERSION}:
            raise DurableStorageError(
                f"unsupported durable result envelope for request {record.query_id}"
            )
        try:
            request = BuildGenerationRequest.model_validate(raw["request"])
            response = ApplicationBuildGenerationResponse.model_validate(raw["response"])
            no_cost_product_ids = frozenset(str(value) for value in raw["no_cost_product_ids"])
            owned_product_ids = (
                frozenset(str(value) for value in raw["owned_product_ids"])
                if schema_version == _ENVELOPE_VERSION
                else frozenset(item.product_id for item in request.existing_products)
            )
            stored_at = datetime.fromisoformat(str(raw["stored_at"]))
        except (KeyError, TypeError, ValueError) as error:
            raise DurableStorageError(
                f"invalid durable result envelope for request {record.query_id}"
            ) from error
        if response.request_id != record.query_id:
            raise DurableStorageError(
                f"durable request identity mismatch for request {record.query_id}"
            )
        return StoredGeneration(
            request=request,
            response=response,
            no_cost_product_ids=no_cost_product_ids,
            owned_product_ids=owned_product_ids,
            stored_at=stored_at,
        )

    @staticmethod
    def _copy(stored: StoredGeneration) -> StoredGeneration:
        return StoredGeneration(
            request=stored.request.model_copy(deep=True),
            response=stored.response.model_copy(deep=True),
            no_cost_product_ids=stored.no_cost_product_ids,
            owned_product_ids=stored.owned_product_ids,
            stored_at=stored.stored_at,
        )

    @staticmethod
    def _same_generation(left: StoredGeneration, right: StoredGeneration) -> bool:
        return (
            left.request == right.request
            and left.response == right.response
            and left.no_cost_product_ids == right.no_cost_product_ids
            and left.owned_product_ids == right.owned_product_ids
        )

    def save(
        self,
        request: BuildGenerationRequest,
        response: ApplicationBuildGenerationResponse,
        *,
        no_cost_product_ids: frozenset[str] | None = None,
        owned_product_ids: frozenset[str] | None = None,
    ) -> ApplicationBuildGenerationResponse:
        request_copy = request.model_copy(deep=True)
        response_copy = response.model_copy(deep=True)
        free_ids = (
            frozenset(item.product_id for item in request.existing_products)
            if no_cost_product_ids is None
            else frozenset(no_cost_product_ids)
        )
        owner_ids = (
            frozenset(item.product_id for item in request.existing_products)
            if owned_product_ids is None
            else frozenset(owned_product_ids)
        )
        stored_at = datetime.now(UTC)
        proposed = StoredGeneration(
            request=request_copy,
            response=response_copy,
            no_cost_product_ids=free_ids,
            owned_product_ids=owner_ids,
            stored_at=stored_at,
        )
        try:
            with session_scope(self.session_factory) as session:
                existing_record = session.get(SearchQueryRecord, response.request_id)
                if existing_record is not None:
                    existing = self._decode(existing_record)
                    if existing is None or not self._same_generation(existing, proposed):
                        raise RequestConflictError(
                            "request_id is already bound to another durable result: "
                            f"{response.request_id}"
                        )
                    return existing.response.model_copy(deep=True)

                build_ids = [build.build_id for build in response.builds]
                if build_ids:
                    duplicates = sorted(
                        session.scalars(
                            select(GeneratedBuildRecord.build_id).where(
                                GeneratedBuildRecord.build_id.in_(build_ids)
                            )
                        )
                    )
                    if duplicates:
                        raise RequestConflictError(
                            "build IDs are already stored: " + ", ".join(duplicates)
                        )

                repository = CatalogRepository(session)
                repository.save_search_query(
                    SearchQuery(
                        query_id=response.request_id,
                        raw_query=request.raw_query,
                        structured_constraints={
                            _ENVELOPE_KEY: self._envelope(
                                request_copy,
                                response_copy,
                                free_ids,
                                owner_ids,
                                stored_at,
                            )
                        },
                        created_at=stored_at,
                    )
                )
                for build in response.builds:
                    repository.save_build(
                        query_id=response.request_id,
                        build=build,
                        optimizer_status=response.optimizer_status.value,
                        rule_version=response.rule_version,
                        model_version=response.ranking_model,
                        data_version=response.data_version,
                    )
        except RequestConflictError:
            raise
        except IntegrityError as error:
            if not _is_unique_violation(error):
                raise DurableStorageError(
                    "generated recommendation violated durable database integrity"
                ) from error
            # A concurrent writer may have committed the same idempotent result first.
            prior = self.get_generation(response.request_id)
            if prior is not None and self._same_generation(prior, proposed):
                return prior.response.model_copy(deep=True)
            raise RequestConflictError(
                f"durable result identity conflict for request {response.request_id}"
            ) from error
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to persist generated recommendation") from error
        return response_copy.model_copy(deep=True)

    def get_generation(self, request_id: str) -> StoredGeneration | None:
        try:
            with self.session_factory() as session:
                record = session.get(SearchQueryRecord, request_id)
                if record is None:
                    return None
                stored = self._decode(record)
        except DurableStorageError:
            raise
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to read generated recommendation") from error
        return None if stored is None else self._copy(stored)

    def require_generation(self, request_id: str) -> StoredGeneration:
        stored = self.get_generation(request_id)
        if stored is None:
            raise ResultNotFoundError(f"request result not found: {request_id}")
        return stored

    def get_response(self, request_id: str) -> ApplicationBuildGenerationResponse | None:
        stored = self.get_generation(request_id)
        return None if stored is None else stored.response

    def get_request(self, request_id: str) -> BuildGenerationRequest | None:
        stored = self.get_generation(request_id)
        return None if stored is None else stored.request

    def request_id_for_build(self, build_id: str) -> str | None:
        try:
            with self.session_factory() as session:
                return session.scalar(
                    select(GeneratedBuildRecord.query_id).where(
                        GeneratedBuildRecord.build_id == build_id
                    )
                )
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to resolve durable build identity") from error

    def get_build(self, build_id: str) -> BuildRecommendation | None:
        request_id = self.request_id_for_build(build_id)
        if request_id is None:
            return None
        stored = self.require_generation(request_id)
        for build in stored.response.builds:
            if build.build_id == build_id:
                return build.model_copy(deep=True)
        raise DurableStorageError(f"durable build envelope is incomplete for {build_id}")

    def require_build(self, build_id: str) -> BuildRecommendation:
        build = self.get_build(build_id)
        if build is None:
            raise ResultNotFoundError(f"build not found: {build_id}")
        return build

    def generation_for_build(self, build_id: str) -> StoredGeneration:
        request_id = self.request_id_for_build(build_id)
        if request_id is None:
            raise ResultNotFoundError(f"build not found: {build_id}")
        return self.require_generation(request_id)

    @staticmethod
    def _share_from_record(record: BuildShareRecord) -> StoredBuildShare:
        created_at = record.created_at
        expires_at = record.expires_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return StoredBuildShare(
            share_id=record.share_id,
            snapshot=dict(record.snapshot or {}),
            created_at=created_at,
            expires_at=expires_at,
        )

    def create_build_share(
        self,
        *,
        build_id: str,
        snapshot: Mapping[str, Any],
        expires_at: datetime,
    ) -> CreatedBuildShare:
        """Persist an immutable public projection and a hashed revocation capability."""

        now = datetime.now(UTC)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= now:
            raise DurableStorageError("build share expiry must be in the future")
        share_id = f"share_{secrets.token_hex(20)}"
        revocation_token = secrets.token_urlsafe(32)
        revocation_digest = hashlib.sha256(revocation_token.encode("utf-8")).hexdigest()
        try:
            with session_scope(self.session_factory) as session:
                if session.get(GeneratedBuildRecord, build_id) is None:
                    raise ResultNotFoundError(f"build not found: {build_id}")
                record = BuildShareRecord(
                    share_id=share_id,
                    build_id=build_id,
                    snapshot=dict(snapshot),
                    revocation_token_sha256=revocation_digest,
                    created_at=now,
                    expires_at=expires_at,
                )
                session.add(record)
                session.flush()
                stored = self._share_from_record(record)
        except ResultNotFoundError:
            raise
        except IntegrityError as error:
            raise DurableStorageError(
                "failed to allocate a durable build-share identifier"
            ) from error
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to persist public build share") from error
        return CreatedBuildShare(
            share_id=stored.share_id,
            snapshot=stored.snapshot,
            created_at=stored.created_at,
            expires_at=stored.expires_at,
            revocation_token=revocation_token,
        )

    def get_active_build_share(self, share_id: str) -> StoredBuildShare | None:
        now = datetime.now(UTC)
        try:
            with self.session_factory() as session:
                record = session.get(BuildShareRecord, share_id)
                if record is None or record.revoked_at is not None:
                    return None
                expires_at = record.expires_at
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
                if expires_at <= now:
                    return None
                return self._share_from_record(record)
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to read public build share") from error

    def revoke_build_share(self, share_id: str, revocation_token: str) -> datetime | None:
        """Revoke only when a current, unexpired share's exact token is supplied."""

        now = datetime.now(UTC)
        digest = hashlib.sha256(revocation_token.encode("utf-8")).hexdigest()
        try:
            with session_scope(self.session_factory) as session:
                record = session.get(BuildShareRecord, share_id)
                if record is None or record.revoked_at is not None:
                    return None
                expires_at = record.expires_at
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
                if expires_at <= now or not hmac.compare_digest(
                    record.revocation_token_sha256, digest
                ):
                    return None
                record.revoked_at = now
                session.flush()
                return now
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to revoke public build share") from error

    def put_interaction(self, event: InteractionEvent) -> InteractionWriteResult:
        """Commit once, replay exactly, and reject conflicting idempotency-key reuse."""

        try:
            with session_scope(self.session_factory) as session:
                existing = session.get(InteractionEventRecord, event.event_id)
                if existing is None and event.idempotency_key_sha256 is not None:
                    existing = session.scalar(
                        select(InteractionEventRecord).where(
                            InteractionEventRecord.session_id == event.session_id,
                            InteractionEventRecord.idempotency_key_sha256
                            == event.idempotency_key_sha256,
                        )
                    )
                if existing is None and event.impression_id is not None:
                    existing = session.scalar(
                        select(InteractionEventRecord).where(
                            InteractionEventRecord.impression_id == event.impression_id,
                            InteractionEventRecord.event_type == event.event_type.value,
                        )
                    )
                if existing is not None:
                    return _resolve_existing_interaction(existing, event)
                query = (
                    session.get(SearchQueryRecord, event.query_id)
                    if event.query_id is not None
                    else None
                )
                product = (
                    session.get(CanonicalProductRecord, event.product_id)
                    if event.product_id is not None
                    else None
                )
                build = (
                    session.get(GeneratedBuildRecord, event.build_id)
                    if event.build_id is not None
                    else None
                )
                if event.query_id is not None and query is None:
                    raise RequestConflictError(
                        f"interaction references unknown query: {event.query_id}"
                    )
                if event.product_id is not None and product is None:
                    raise RequestConflictError(
                        f"interaction references unknown product: {event.product_id}"
                    )
                if event.build_id is not None and build is None:
                    raise RequestConflictError(
                        f"interaction references unknown build: {event.build_id}"
                    )
                if build is not None and query is not None and build.query_id != query.query_id:
                    raise RequestConflictError(
                        "interaction build_id and query_id refer to different generations"
                    )
                CatalogRepository(session).add_interaction(event)
        except RequestConflictError:
            raise
        except IntegrityError as error:
            if _is_unique_violation(error):
                try:
                    with self.session_factory() as session:
                        existing = session.scalar(
                            select(InteractionEventRecord).where(
                                InteractionEventRecord.session_id == event.session_id,
                                InteractionEventRecord.idempotency_key_sha256
                                == event.idempotency_key_sha256,
                            )
                        )
                        if existing is not None:
                            return _resolve_existing_interaction(existing, event)
                        if event.impression_id is not None:
                            existing = session.scalar(
                                select(InteractionEventRecord).where(
                                    InteractionEventRecord.impression_id
                                    == event.impression_id,
                                    InteractionEventRecord.event_type
                                    == event.event_type.value,
                                )
                            )
                            if existing is not None:
                                return _resolve_existing_interaction(existing, event)
                except RequestConflictError:
                    raise
                except SQLAlchemyError as reread_error:
                    raise DurableStorageError(
                        "failed to resolve a concurrent interaction retry"
                    ) from reread_error
                raise RequestConflictError("interaction already exists") from error
            raise DurableStorageError(
                "interaction violated durable database integrity"
            ) from error
        except SQLAlchemyError as error:
            raise DurableStorageError("failed to persist interaction event") from error
        return InteractionWriteResult(event=event, replayed=False)

    def add_interaction(self, event: InteractionEvent) -> InteractionEvent:
        """Backward-compatible wrapper returning the stored canonical event."""

        return self.put_interaction(event).event
