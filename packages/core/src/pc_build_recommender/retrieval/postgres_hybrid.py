"""Production PostgreSQL full-text plus pgvector hybrid retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import func, literal_column, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from pc_build_recommender.catalog.orm import (
    CanonicalProductRecord,
    ProductEmbeddingRecord,
)

from .bm25 import BM25ProductIndex
from .fusion import reciprocal_rank_fusion
from .models import ProductDocument, RetrievedCandidate, SearchHit, StructuredFilters
from .postgres import PgVectorSearchBackend
from .postgres_filters import (
    cheapest_price_expression,
    cheapest_stock_expression,
    normalize_postgres_category,
    postgres_structured_predicates,
)
from .vector import EmbeddingEncoder

MAX_DATABASE_TOP_K = 1000
MAX_QUERY_CHARACTERS = 4096


@dataclass(frozen=True, slots=True)
class PostgresRetrievalRelease:
    """Exact immutable release identity exposed by the online retriever."""

    retrieval_model: str
    embedding_model: str
    data_version: str
    index_version: str
    encoder_fingerprint: str
    dataset_content_hash: str
    rrf_k: int


def _validated_query(query: str) -> str:
    normalized = " ".join(query.split())
    if len(normalized) > MAX_QUERY_CHARACTERS:
        raise ValueError(f"query must not exceed {MAX_QUERY_CHARACTERS} characters")
    return normalized


def _validated_top_k(top_k: int) -> int:
    if top_k < 1:
        return 0
    if top_k > MAX_DATABASE_TOP_K:
        raise ValueError(f"top_k must not exceed {MAX_DATABASE_TOP_K}")
    return top_k


class PostgresFullTextSearchBackend:
    """PostgreSQL cover-density ranking over the indexed canonical search document.

    PostgreSQL does not implement BM25 natively. ``ts_rank_cd`` is the
    production lexical analogue here; RRF consumes rank positions rather than
    pretending its score is calibrated to vector cosine similarity.
    """

    source_name = "postgres_fts"
    lexical_model = "postgresql_ts_rank_cd"

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        embedding_model: str,
        data_version: str,
        index_version: str,
        encoder_fingerprint: str,
        dataset_content_hash: str,
    ) -> None:
        self._session_factory = session_factory
        self.embedding_model = embedding_model
        self.data_version = data_version
        self.index_version = index_version
        self.encoder_fingerprint = encoder_fingerprint
        self.dataset_content_hash = dataset_content_hash

    def _search_session(
        self,
        session: Session,
        query: str,
        *,
        category: str,
        top_k: int,
        filters: StructuredFilters | None,
        candidate_ids: set[str] | frozenset[str] | None,
    ) -> list[SearchHit]:
        normalized_query = _validated_query(query)
        limit = _validated_top_k(top_k)
        if not normalized_query or limit == 0 or candidate_ids is not None and not candidate_ids:
            return []
        config: ColumnElement[Any] = literal_column("'english'::regconfig")
        document_vector = func.to_tsvector(config, CanonicalProductRecord.search_document)
        parsed_query = func.websearch_to_tsquery(config, normalized_query)
        score = func.ts_rank_cd(document_vector, parsed_query, 32).label("score")
        statement = (
            select(CanonicalProductRecord.product_id, score)
            .join(
                ProductEmbeddingRecord,
                (ProductEmbeddingRecord.product_id == CanonicalProductRecord.product_id)
                & (
                    ProductEmbeddingRecord.content_hash
                    == CanonicalProductRecord.search_document_hash
                ),
            )
            .where(
                ProductEmbeddingRecord.embedding_model == self.embedding_model,
                ProductEmbeddingRecord.data_version == self.data_version,
                ProductEmbeddingRecord.index_version == self.index_version,
                ProductEmbeddingRecord.encoder_fingerprint == self.encoder_fingerprint,
                ProductEmbeddingRecord.dataset_content_hash == self.dataset_content_hash,
                document_vector.op("@@")(parsed_query),
                *postgres_structured_predicates(
                    category=category,
                    filters=filters,
                    candidate_ids=candidate_ids,
                ),
            )
            .order_by(score.desc(), CanonicalProductRecord.product_id)
            .limit(limit)
        )
        rows = session.execute(statement).all()
        return [
            SearchHit(
                product_id=product_id,
                score=float(raw_score),
                rank=rank,
                source=PostgresFullTextSearchBackend.source_name,
            )
            for rank, (product_id, raw_score) in enumerate(rows, start=1)
        ]

    def search(
        self,
        query: str,
        *,
        category: str,
        top_k: int = 50,
        candidate_ids: set[str] | frozenset[str] | None = None,
        filters: StructuredFilters | None = None,
    ) -> list[SearchHit]:
        with self._session_factory() as session:
            return self._search_session(
                session,
                query,
                category=category,
                top_k=top_k,
                filters=filters,
                candidate_ids=candidate_ids,
            )


class PostgresBm25SearchBackend:
    """Release-bound BM25 with database-authoritative structured filtering.

    BM25 scores an immutable, startup-built corpus from the validated embedding
    artifact.  The database determines which of those product IDs are active,
    indexed by the exact semantic release, and eligible under the request's
    current price, stock, brand, and specification filters.  Both lexical and
    vector searches therefore use the same read-only transaction snapshot.
    """

    source_name = "bm25"
    lexical_model = "bm25_okapi"

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        index: BM25ProductIndex,
        embedding_model: str,
        data_version: str,
        index_version: str,
        encoder_fingerprint: str,
        dataset_content_hash: str,
    ) -> None:
        self._session_factory = session_factory
        self._index = index
        self.embedding_model = embedding_model
        self.data_version = data_version
        self.index_version = index_version
        self.encoder_fingerprint = encoder_fingerprint
        self.dataset_content_hash = dataset_content_hash

    def _eligible_product_ids_session(
        self,
        session: Session,
        *,
        category: str,
        filters: StructuredFilters | None,
        candidate_ids: set[str] | frozenset[str] | None,
    ) -> set[str]:
        statement = (
            select(CanonicalProductRecord.product_id)
            .join(
                ProductEmbeddingRecord,
                (ProductEmbeddingRecord.product_id == CanonicalProductRecord.product_id)
                & (
                    ProductEmbeddingRecord.content_hash
                    == CanonicalProductRecord.search_document_hash
                ),
            )
            .where(
                ProductEmbeddingRecord.embedding_model == self.embedding_model,
                ProductEmbeddingRecord.data_version == self.data_version,
                ProductEmbeddingRecord.index_version == self.index_version,
                ProductEmbeddingRecord.encoder_fingerprint == self.encoder_fingerprint,
                ProductEmbeddingRecord.dataset_content_hash == self.dataset_content_hash,
                *postgres_structured_predicates(
                    category=category,
                    filters=filters,
                    candidate_ids=candidate_ids,
                ),
            )
        )
        return {str(row[0]) for row in session.execute(statement).all()}

    def _search_session(
        self,
        session: Session,
        query: str,
        *,
        category: str,
        top_k: int,
        filters: StructuredFilters | None,
        candidate_ids: set[str] | frozenset[str] | None,
    ) -> list[SearchHit]:
        normalized_query = _validated_query(query)
        limit = _validated_top_k(top_k)
        if not normalized_query or limit == 0 or candidate_ids is not None and not candidate_ids:
            return []
        eligible_ids = self._eligible_product_ids_session(
            session,
            category=category,
            filters=filters,
            candidate_ids=candidate_ids,
        )
        if not eligible_ids:
            return []
        return self._index.search(
            normalized_query,
            category=category,
            top_k=limit,
            candidate_ids=eligible_ids,
        )

    def search(
        self,
        query: str,
        *,
        category: str,
        top_k: int = 50,
        candidate_ids: set[str] | frozenset[str] | None = None,
        filters: StructuredFilters | None = None,
    ) -> list[SearchHit]:
        with self._session_factory() as session:
            return self._search_session(
                session,
                query,
                category=category,
                top_k=top_k,
                filters=filters,
                candidate_ids=candidate_ids,
            )


class PostgresHybridRetriever:
    """One-transaction lexical/vector retrieval with deterministic rank fusion."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        encoder: EmbeddingEncoder,
        data_version: str,
        index_version: str,
        encoder_fingerprint: str,
        dataset_content_hash: str,
        embedding_model: str | None = None,
        rrf_k: int = 60,
        bm25_index: BM25ProductIndex | None = None,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self._session_factory = session_factory
        configured_model = embedding_model or encoder.model_name
        if bm25_index is None:
            self._lexical: PostgresFullTextSearchBackend | PostgresBm25SearchBackend = (
                PostgresFullTextSearchBackend(
                    session_factory,
                    embedding_model=configured_model,
                    data_version=data_version,
                    index_version=index_version,
                    encoder_fingerprint=encoder_fingerprint,
                    dataset_content_hash=dataset_content_hash,
                )
            )
        else:
            self._lexical = PostgresBm25SearchBackend(
                session_factory,
                index=bm25_index,
                embedding_model=configured_model,
                data_version=data_version,
                index_version=index_version,
                encoder_fingerprint=encoder_fingerprint,
                dataset_content_hash=dataset_content_hash,
            )
        self._vector = PgVectorSearchBackend(
            session_factory,
            encoder=encoder,
            data_version=data_version,
            index_version=index_version,
            encoder_fingerprint=encoder_fingerprint,
            dataset_content_hash=dataset_content_hash,
            embedding_model=embedding_model,
        )
        self.rrf_k = rrf_k

    @property
    def release(self) -> PostgresRetrievalRelease:
        return PostgresRetrievalRelease(
            retrieval_model=(
                "bm25-okapi+pgvector-cosine+rrf-v1"
                if isinstance(self._lexical, PostgresBm25SearchBackend)
                else "postgres-fts-ts-rank-cd+pgvector-cosine+rrf-v1"
            ),
            embedding_model=self._vector.embedding_model,
            data_version=self._vector.data_version,
            index_version=self._vector.index_version,
            encoder_fingerprint=self._vector.encoder_fingerprint,
            dataset_content_hash=self._vector.dataset_content_hash,
            rrf_k=self.rrf_k,
        )

    @property
    def retrieval_model_version(self) -> str:
        return f"{self.release.retrieval_model}@{self.release.index_version}"

    @staticmethod
    def _load_documents(
        session: Session,
        product_ids: Sequence[str],
        *,
        category: str,
        filters: StructuredFilters,
    ) -> dict[str, ProductDocument]:
        if not product_ids:
            return {}
        price = cheapest_price_expression(in_stock_only=filters.in_stock_only).label("price_sgd")
        stock = cheapest_stock_expression(in_stock_only=filters.in_stock_only).label("stock_status")
        statement = select(
            CanonicalProductRecord.product_id,
            CanonicalProductRecord.category,
            CanonicalProductRecord.search_document,
            CanonicalProductRecord.canonical_name,
            CanonicalProductRecord.brand,
            CanonicalProductRecord.model,
            CanonicalProductRecord.manufacturer_part_number,
            CanonicalProductRecord.common_attributes,
            CanonicalProductRecord.category_attributes,
            price,
            stock,
        ).where(
            *postgres_structured_predicates(
                category=category,
                filters=filters,
                candidate_ids=set(product_ids),
            )
        )
        documents: dict[str, ProductDocument] = {}
        for row in session.execute(statement):
            attributes = dict(row.common_attributes or {})
            attributes.update(row.category_attributes or {})
            attributes["model"] = row.model
            attributes["manufacturer_part_number"] = row.manufacturer_part_number
            documents[row.product_id] = ProductDocument(
                product_id=row.product_id,
                category=row.category,
                text=row.search_document or row.canonical_name,
                brand=row.brand,
                price_sgd=float(row.price_sgd) if row.price_sgd is not None else None,
                stock_status=row.stock_status,
                attributes=attributes,
            )
        return documents

    def _retrieve_encoded(
        self,
        query: str,
        query_vector: NDArray[np.float32],
        *,
        category: str,
        filters: StructuredFilters | None = None,
        top_k: int = 50,
        per_source_k: int | None = None,
    ) -> list[RetrievedCandidate]:
        final_k = _validated_top_k(top_k)
        if final_k == 0:
            return []
        source_k = per_source_k if per_source_k is not None else max(50, final_k)
        if source_k < 1:
            raise ValueError("per_source_k must be positive")
        _validated_top_k(source_k)
        active_filters = filters or StructuredFilters()
        category_key = normalize_postgres_category(category)
        with self._session_factory() as session, session.begin():
            session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            lexical_hits = self._lexical._search_session(
                session,
                query,
                category=category_key,
                top_k=source_k,
                filters=active_filters,
                candidate_ids=None,
            )
            vector_hits = self._vector._search_vector_session(
                session,
                query_vector,
                category=category_key,
                top_k=source_k,
                filters=active_filters,
                candidate_ids=None,
            )
            fused = reciprocal_rank_fusion(
                {
                    self._lexical.source_name: lexical_hits,
                    self._vector.source_name: vector_hits,
                },
                k=self.rrf_k,
                limit=final_k,
            )
            documents = self._load_documents(
                session,
                [hit.product_id for hit in fused],
                category=category_key,
                filters=active_filters,
            )

        lexical_by_id = {hit.product_id: hit for hit in lexical_hits}
        vector_by_id = {hit.product_id: hit for hit in vector_hits}
        candidates: list[RetrievedCandidate] = []
        for hit in fused:
            product = documents.get(hit.product_id)
            if product is None:
                continue
            lexical = lexical_by_id.get(hit.product_id)
            vector = vector_by_id.get(hit.product_id)
            candidates.append(
                RetrievedCandidate(
                    product=product,
                    rank=len(candidates) + 1,
                    rrf_score=hit.score,
                    lexical_score=lexical.score if lexical else 0.0,
                    lexical_rank=lexical.rank if lexical else None,
                    lexical_model=self._lexical.lexical_model,
                    bm25_score=(
                        lexical.score
                        if lexical is not None
                        and isinstance(self._lexical, PostgresBm25SearchBackend)
                        else 0.0
                    ),
                    vector_similarity=vector.score if vector else 0.0,
                    bm25_rank=(
                        lexical.rank
                        if lexical is not None
                        and isinstance(self._lexical, PostgresBm25SearchBackend)
                        else None
                    ),
                    vector_rank=vector.rank if vector else None,
                )
            )
        return candidates

    def retrieve(
        self,
        query: str,
        *,
        category: str,
        filters: StructuredFilters | None = None,
        top_k: int = 50,
        per_source_k: int | None = None,
    ) -> list[RetrievedCandidate]:
        query_vector = self._vector.encode_query(query)
        if query_vector is None:
            return []
        return self._retrieve_encoded(
            query,
            query_vector,
            category=category,
            filters=filters,
            top_k=top_k,
            per_source_k=per_source_k,
        )

    def retrieve_categories(
        self,
        query: str,
        categories: Sequence[str],
        *,
        filters_by_category: Mapping[str, StructuredFilters] | None = None,
        top_k_per_category: int = 50,
    ) -> dict[str, list[RetrievedCandidate]]:
        configured_filters = filters_by_category or {}
        result: dict[str, list[RetrievedCandidate]] = {}
        query_vector = self._vector.encode_query(query)
        for category in categories:
            category_key = normalize_postgres_category(category)
            result[category_key] = (
                []
                if query_vector is None
                else self._retrieve_encoded(
                    query,
                    query_vector,
                    category=category_key,
                    filters=configured_filters.get(category, configured_filters.get(category_key)),
                    top_k=top_k_per_category,
                )
            )
        return result
