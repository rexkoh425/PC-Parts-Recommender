"""Hybrid, category-scoped product candidate retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from .bm25 import BM25ProductIndex
from .filters import product_matches_filters
from .fusion import reciprocal_rank_fusion
from .models import ProductDocument, RetrievedCandidate, SearchHit, StructuredFilterSpec
from .vector import InMemoryVectorIndex, VectorSearchBackend


@runtime_checkable
class ProductRetriever(Protocol):
    """Storage-agnostic retrieval contract consumed by the candidate pipeline."""

    @property
    def retrieval_model_version(self) -> str: ...

    def retrieve(
        self,
        query: str,
        *,
        category: str,
        filters: StructuredFilterSpec | None = None,
        top_k: int = 50,
        per_source_k: int | None = None,
    ) -> list[RetrievedCandidate]: ...


class HybridProductRetriever:
    """BM25 + vector discovery followed by RRF with direct pre-filters."""

    def __init__(
        self,
        documents: Iterable[ProductDocument | Mapping[str, object]],
        *,
        vector_backend: VectorSearchBackend | None = None,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        converted = tuple(
            item if isinstance(item, ProductDocument) else ProductDocument.from_mapping(item)
            for item in documents
        )
        ids = [document.product_id for document in converted]
        if len(ids) != len(set(ids)):
            raise ValueError("product_id values must be unique")
        self._documents = {document.product_id: document for document in converted}
        self._bm25 = BM25ProductIndex(converted)
        self._vector = vector_backend or InMemoryVectorIndex(converted)
        self.rrf_k = rrf_k

    @property
    def retrieval_model_version(self) -> str:
        if isinstance(self._vector, InMemoryVectorIndex):
            return "bm25+stable-hash-vector+rrf-development-v1"
        vector_name = type(self._vector).__name__.casefold()
        return f"bm25+{vector_name}+rrf-development-v1"

    @property
    def product_count(self) -> int:
        return len(self._documents)

    def retrieve(
        self,
        query: str,
        *,
        category: str,
        filters: StructuredFilterSpec | None = None,
        top_k: int = 50,
        per_source_k: int | None = None,
    ) -> list[RetrievedCandidate]:
        """Retrieve products from exactly one component category.

        Structured filters are evaluated *before* either top-k operation.  This
        prevents a long list of disallowed products from starving the final
        candidate pool.
        """

        candidates, _ = self.retrieve_with_source_pools(
            query,
            category=category,
            filters=filters,
            top_k=top_k,
            per_source_k=per_source_k,
        )
        return candidates

    def retrieve_with_source_pools(
        self,
        query: str,
        *,
        category: str,
        filters: StructuredFilterSpec | None = None,
        top_k: int = 50,
        per_source_k: int | None = None,
    ) -> tuple[list[RetrievedCandidate], Mapping[str, tuple[SearchHit, ...]]]:
        """Retrieve products from exactly one component category.

        Structured filters are evaluated *before* either top-k operation.  This
        prevents a long list of disallowed products from starving the final
        candidate pool.  The returned source pools retain only candidates that
        passed the same category and structured-filter gate as the fused
        results.  Callers that expose results to end users must decide whether
        per-source ranks and scores are appropriate to disclose.
        """

        empty_source_pools: dict[str, tuple[SearchHit, ...]] = {"bm25": (), "vector": ()}
        if top_k < 1:
            return [], empty_source_pools
        source_k = per_source_k if per_source_k is not None else max(50, top_k)
        if source_k < 1:
            raise ValueError("per_source_k must be positive")
        active_filters = filters or StructuredFilterSpec()
        category_key = category.casefold()
        allowed_ids = {
            product_id
            for product_id, document in self._documents.items()
            if document.category == category_key
            and product_matches_filters(document, active_filters)
        }
        if not allowed_ids:
            return [], empty_source_pools

        bm25_hits = [
            hit
            for hit in self._bm25.search(
                query,
                category=category_key,
                top_k=source_k,
                candidate_ids=allowed_ids,
            )
            if hit.product_id in allowed_ids
        ]
        vector_hits = [
            hit
            for hit in self._vector.search(
                query,
                category=category_key,
                top_k=source_k,
                candidate_ids=allowed_ids,
            )
            if hit.product_id in allowed_ids
        ]
        source_pools: dict[str, tuple[SearchHit, ...]] = {
            "bm25": tuple(bm25_hits),
            "vector": tuple(vector_hits),
        }
        fused = reciprocal_rank_fusion(
            source_pools,
            k=self.rrf_k,
            limit=top_k,
        )
        bm25_by_id = {hit.product_id: hit for hit in bm25_hits}
        vector_by_id = {hit.product_id: hit for hit in vector_hits}

        result: list[RetrievedCandidate] = []
        for hit in fused:
            # A remote vector backend is not trusted to honour candidate_ids;
            # enforce category and filters again at this boundary.
            if hit.product_id not in allowed_ids or hit.product_id not in self._documents:
                continue
            bm25 = bm25_by_id.get(hit.product_id)
            vector = vector_by_id.get(hit.product_id)
            result.append(
                RetrievedCandidate(
                    product=self._documents[hit.product_id],
                    rank=len(result) + 1,
                    rrf_score=hit.score,
                    lexical_score=bm25.score if bm25 else 0.0,
                    lexical_rank=bm25.rank if bm25 else None,
                    lexical_model="bm25",
                    bm25_score=bm25.score if bm25 else 0.0,
                    vector_similarity=vector.score if vector else 0.0,
                    bm25_rank=bm25.rank if bm25 else None,
                    vector_rank=vector.rank if vector else None,
                )
            )
        return result, source_pools

    def retrieve_categories(
        self,
        query: str,
        categories: Sequence[str],
        *,
        filters_by_category: Mapping[str, StructuredFilterSpec] | None = None,
        top_k_per_category: int = 50,
    ) -> dict[str, list[RetrievedCandidate]]:
        """Retrieve independent candidate pools for build optimisation."""

        result: dict[str, list[RetrievedCandidate]] = {}
        configured_filters = filters_by_category or {}
        for category in categories:
            category_key = category.casefold()
            result[category_key] = self.retrieve(
                query,
                category=category_key,
                filters=configured_filters.get(category, configured_filters.get(category_key)),
                top_k=top_k_per_category,
            )
        return result
