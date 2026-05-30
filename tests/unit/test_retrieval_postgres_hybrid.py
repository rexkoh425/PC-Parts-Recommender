from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from sqlalchemy.dialects import postgresql

from pc_build_recommender.retrieval.bm25 import BM25ProductIndex
from pc_build_recommender.retrieval.embedding_index import embedding_encoder_fingerprint
from pc_build_recommender.retrieval.models import (
    ProductDocument,
    SearchHit,
    StructuredFilters,
)
from pc_build_recommender.retrieval.postgres import PgVectorSearchBackend
from pc_build_recommender.retrieval.postgres_filters import (
    postgres_structured_predicates,
)
from pc_build_recommender.retrieval.postgres_hybrid import (
    PostgresBm25SearchBackend,
    PostgresFullTextSearchBackend,
    PostgresHybridRetriever,
)
from pc_build_recommender.retrieval.vector import FloatMatrix


class _Encoder:
    model_name = "fixture-encoder"
    dimension = 384

    def __init__(self) -> None:
        self.call_count = 0

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        self.call_count += 1
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        matrix[:, 0] = 1.0
        return matrix


class _Result:
    def __init__(self, rows: list[tuple[str, float]] | None = None) -> None:
        self._rows = rows or []

    def all(self) -> list[tuple[str, float]]:
        return self._rows


class _SqlSession:
    def __init__(self, expected_operator: str) -> None:
        self.expected_operator = expected_operator
        self.compiled = ""
        self.iterative_scan_enabled = False

    def __enter__(self) -> _SqlSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: Any) -> _Result:
        if str(statement).startswith("SET LOCAL hnsw.iterative_scan"):
            self.iterative_scan_enabled = True
            return _Result()
        self.compiled = str(
            statement.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]
        )
        assert self.expected_operator in self.compiled
        return _Result([("prod_b", 0.8), ("prod_a", 0.6)])


class _Bm25SqlSession:
    def __init__(self) -> None:
        self.compiled = ""

    def __enter__(self) -> _Bm25SqlSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: Any) -> _Result:
        self.compiled = str(
            statement.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]
        )
        return _Result([("prod_a", 0.0), ("prod_b", 0.0)])


def _strict_gpu_filters() -> StructuredFilters:
    return StructuredFilters(
        maximum_price_sgd=900,
        minimum_gpu_vram_gb=16,
        excluded_brands=frozenset({"blocked"}),
        in_stock_only=True,
        attribute_equals={"architecture": "Ada Lovelace"},
        attribute_minimums={"memory_bandwidth_gbps": 500},
    )


def _full_text_backend(session: _SqlSession) -> PostgresFullTextSearchBackend:
    return PostgresFullTextSearchBackend(
        lambda: session,  # type: ignore[arg-type]
        embedding_model="fixture-encoder",
        data_version="data-v1",
        index_version="index-v1",
        encoder_fingerprint="a" * 64,
        dataset_content_hash="b" * 64,
    )


def _bm25_backend(session: _Bm25SqlSession) -> PostgresBm25SearchBackend:
    index = BM25ProductIndex(
        (
            ProductDocument("prod_a", "gpu", "NVIDIA RTX quiet local AI GPU"),
            ProductDocument("prod_b", "gpu", "AMD gaming GPU"),
            ProductDocument("prod_c", "gpu", "Intel Arc creator GPU"),
        )
    )
    return PostgresBm25SearchBackend(
        lambda: session,  # type: ignore[arg-type]
        index=index,
        embedding_model="fixture-encoder",
        data_version="data-v1",
        index_version="index-v1",
        encoder_fingerprint="a" * 64,
        dataset_content_hash="b" * 64,
    )


def test_postgres_full_text_compiles_rank_and_fail_closed_filters() -> None:
    session = _SqlSession("@@")
    backend = _full_text_backend(session)

    hits = backend.search(
        'quiet "local AI" GPU',
        category="gpu",
        top_k=2,
        filters=_strict_gpu_filters(),
    )

    assert [hit.product_id for hit in hits] == ["prod_b", "prod_a"]
    assert "ts_rank_cd" in session.compiled
    assert "websearch_to_tsquery" in session.compiled
    assert "EXISTS" in session.compiled
    assert "retailer_listings" in session.compiled
    assert "category_attributes" in session.compiled
    assert "canonical_products.brand" in session.compiled
    assert "canonical_products.search_document_hash" in session.compiled
    assert "product_embeddings.data_version" in session.compiled


def test_postgres_bm25_uses_release_bound_filter_population() -> None:
    session = _Bm25SqlSession()
    backend = _bm25_backend(session)

    hits = backend.search(
        "quiet RTX local AI",
        category="gpu",
        top_k=2,
        filters=_strict_gpu_filters(),
    )

    assert [hit.product_id for hit in hits] == ["prod_a", "prod_b"]
    assert hits[0].source == "bm25"
    assert hits[0].score > hits[1].score
    assert "product_embeddings.data_version" in session.compiled
    assert "canonical_products.search_document_hash" in session.compiled
    assert "EXISTS" in session.compiled
    assert "retailer_listings" in session.compiled
    assert "category_attributes" in session.compiled
    assert "ts_rank_cd" not in session.compiled


def test_pgvector_uses_the_same_structured_candidate_filters() -> None:
    session = _SqlSession("<=>")
    backend = PgVectorSearchBackend(
        lambda: session,  # type: ignore[arg-type]
        encoder=_Encoder(),
        data_version="data-v1",
        index_version="index-v1",
        encoder_fingerprint=embedding_encoder_fingerprint(_Encoder()),
        dataset_content_hash="b" * 64,
    )

    hits = backend.search(
        "16 GB local AI GPU",
        category="gpu",
        top_k=2,
        filters=_strict_gpu_filters(),
    )

    assert len(hits) == 2
    assert "product_embeddings.data_version" in session.compiled
    assert "product_embeddings.encoder_fingerprint" in session.compiled
    assert "product_embeddings.dataset_content_hash" in session.compiled
    assert "EXISTS" in session.compiled
    assert "retailer_listings" in session.compiled
    assert "category_attributes" in session.compiled
    assert session.iterative_scan_enabled is True
    assert "canonical_products.search_document_hash" in session.compiled


def test_pgvector_rejects_a_caller_supplied_encoder_fingerprint() -> None:
    session = _SqlSession("<=>")

    with pytest.raises(ValueError, match="does not match"):
        PgVectorSearchBackend(
            lambda: session,  # type: ignore[arg-type]
            encoder=_Encoder(),
            data_version="data-v1",
            index_version="index-v1",
            encoder_fingerprint="0" * 64,
            dataset_content_hash="b" * 64,
        )


def test_structured_filters_reject_non_scalar_dynamic_equality() -> None:
    filters = StructuredFilters(
        in_stock_only=False,
        attribute_equals={"tags": ["ai", "quiet"]},
    )

    with pytest.raises(ValueError, match="must be scalar"):
        postgres_structured_predicates(category="gpu", filters=filters)


class _HydrationSession:
    compiled = ""

    def execute(self, statement: Any) -> list[Any]:
        self.compiled = str(
            statement.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]
        )
        return []


def test_fused_document_hydration_reapplies_structured_filters() -> None:
    session = _HydrationSession()

    documents = PostgresHybridRetriever._load_documents(
        session,  # type: ignore[arg-type]
        ["prod_a", "prod_b"],
        category="gpu",
        filters=_strict_gpu_filters(),
    )

    assert documents == {}
    assert "canonical_products.product_id IN" in session.compiled
    assert "EXISTS" in session.compiled
    assert "retailer_listings" in session.compiled
    assert "category_attributes" in session.compiled


class _Transaction:
    def __enter__(self) -> _Transaction:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _HybridSession:
    began = False
    repeatable_read_enabled = False

    def __enter__(self) -> _HybridSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def begin(self) -> _Transaction:
        self.began = True
        return _Transaction()

    def execute(self, statement: Any) -> _Result:
        if str(statement).startswith("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"):
            self.repeatable_read_enabled = True
            return _Result()
        raise AssertionError(f"unexpected statement: {statement}")


def _document(product_id: str) -> ProductDocument:
    return ProductDocument(
        product_id=product_id,
        category="gpu",
        text=f"GPU {product_id}",
        brand="Example",
        price_sgd=700,
        stock_status="in_stock",
        attributes={"vram_gb": 16},
    )


def test_postgres_hybrid_applies_deterministic_rrf_and_preserves_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _HybridSession()
    retriever = PostgresHybridRetriever(
        lambda: session,  # type: ignore[arg-type]
        encoder=_Encoder(),
        data_version="data-v1",
        index_version="index-v1",
        encoder_fingerprint=embedding_encoder_fingerprint(_Encoder()),
        dataset_content_hash="b" * 64,
    )
    lexical = [
        SearchHit("prod_a", 8.0, 1, "postgres_fts"),
        SearchHit("prod_b", 7.0, 2, "postgres_fts"),
    ]
    vector = [
        SearchHit("prod_b", 0.95, 1, "pgvector"),
        SearchHit("prod_c", 0.90, 2, "pgvector"),
    ]
    monkeypatch.setattr(
        retriever._lexical,
        "_search_session",
        lambda *_args, **_kwargs: lexical,
    )
    monkeypatch.setattr(
        retriever._vector,
        "_search_vector_session",
        lambda *_args, **_kwargs: vector,
    )
    documents = {product_id: _document(product_id) for product_id in ("prod_a", "prod_b", "prod_c")}
    monkeypatch.setattr(
        retriever,
        "_load_documents",
        lambda *_args, **_kwargs: documents,
    )

    candidates = retriever.retrieve("local AI", category="gpu", top_k=3)

    assert session.began is True
    assert session.repeatable_read_enabled is True
    assert [candidate.product_id for candidate in candidates] == [
        "prod_b",
        "prod_a",
        "prod_c",
    ]
    assert candidates[0].lexical_score == 7.0
    assert candidates[0].lexical_model == "postgresql_ts_rank_cd"
    assert candidates[0].bm25_score == 0.0
    assert candidates[0].vector_similarity == 0.95
    assert candidates[0].lexical_rank == 2
    assert candidates[0].bm25_rank is None
    assert candidates[0].vector_rank == 1
    assert "ts-rank-cd" in retriever.retrieval_model_version
    assert "bm25" not in retriever.retrieval_model_version


def test_postgres_hybrid_uses_bm25_when_a_release_index_is_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _HybridSession()
    index = BM25ProductIndex((_document("prod_a"), _document("prod_b"), _document("prod_c")))
    retriever = PostgresHybridRetriever(
        lambda: session,  # type: ignore[arg-type]
        encoder=_Encoder(),
        data_version="data-v1",
        index_version="index-v1",
        encoder_fingerprint=embedding_encoder_fingerprint(_Encoder()),
        dataset_content_hash="b" * 64,
        bm25_index=index,
    )
    lexical = [
        SearchHit("prod_a", 8.0, 1, "bm25"),
        SearchHit("prod_b", 7.0, 2, "bm25"),
    ]
    vector = [
        SearchHit("prod_b", 0.95, 1, "pgvector"),
        SearchHit("prod_c", 0.90, 2, "pgvector"),
    ]
    monkeypatch.setattr(retriever._lexical, "_search_session", lambda *_args, **_kwargs: lexical)
    monkeypatch.setattr(
        retriever._vector,
        "_search_vector_session",
        lambda *_args, **_kwargs: vector,
    )
    documents = {product_id: _document(product_id) for product_id in ("prod_a", "prod_b", "prod_c")}
    monkeypatch.setattr(retriever, "_load_documents", lambda *_args, **_kwargs: documents)

    candidates = retriever.retrieve("local AI", category="gpu", top_k=3)

    assert [candidate.product_id for candidate in candidates] == ["prod_b", "prod_a", "prod_c"]
    assert candidates[0].lexical_model == "bm25_okapi"
    assert candidates[0].bm25_score == 7.0
    assert candidates[0].bm25_rank == 2
    assert "bm25-okapi" in retriever.retrieval_model_version


def test_database_retrieval_caps_query_and_result_size() -> None:
    session = _SqlSession("@@")
    backend = _full_text_backend(session)

    with pytest.raises(ValueError, match="top_k must not exceed"):
        backend.search("gpu", category="gpu", top_k=1001)
    with pytest.raises(ValueError, match="query must not exceed"):
        backend.search("x" * 4097, category="gpu")


def test_postgres_bm25_applies_the_same_query_and_result_bounds() -> None:
    session = _Bm25SqlSession()
    backend = _bm25_backend(session)

    with pytest.raises(ValueError, match="top_k must not exceed"):
        backend.search("gpu", category="gpu", top_k=1001)
    with pytest.raises(ValueError, match="query must not exceed"):
        backend.search("x" * 4097, category="gpu")


def test_hybrid_requires_a_positive_per_source_limit() -> None:
    session = _HybridSession()
    retriever = PostgresHybridRetriever(
        lambda: session,  # type: ignore[arg-type]
        encoder=_Encoder(),
        data_version="data-v1",
        index_version="index-v1",
        encoder_fingerprint=embedding_encoder_fingerprint(_Encoder()),
        dataset_content_hash="b" * 64,
    )

    with pytest.raises(ValueError, match="per_source_k must be positive"):
        retriever.retrieve("gpu", category="gpu", per_source_k=0)


def test_multi_category_retrieval_encodes_once_before_transactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _HybridSession()
    encoder = _Encoder()
    retriever = PostgresHybridRetriever(
        lambda: session,  # type: ignore[arg-type]
        encoder=encoder,
        data_version="data-v1",
        index_version="index-v1",
        encoder_fingerprint=embedding_encoder_fingerprint(encoder),
        dataset_content_hash="b" * 64,
    )
    monkeypatch.setattr(
        retriever,
        "_retrieve_encoded",
        lambda *_args, **_kwargs: [],
    )

    results = retriever.retrieve_categories("local AI", ["gpu", "cpu", "memory"])

    assert results == {"gpu": [], "cpu": [], "memory": []}
    assert encoder.call_count == 1
