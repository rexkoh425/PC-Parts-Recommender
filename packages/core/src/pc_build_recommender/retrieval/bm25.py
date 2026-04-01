"""Category-scoped BM25 retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from .models import ProductDocument, SearchHit
from .text import tokenize


class BM25ProductIndex:
    """An in-memory BM25 index with one corpus per component category."""

    source_name = "bm25"

    def __init__(self, documents: Iterable[ProductDocument]) -> None:
        by_category: dict[str, list[ProductDocument]] = defaultdict(list)
        seen: set[str] = set()
        for document in documents:
            if document.product_id in seen:
                raise ValueError(f"duplicate product_id: {document.product_id}")
            seen.add(document.product_id)
            by_category[document.category].append(document)

        self._documents = {
            category: tuple(sorted(items, key=lambda item: item.product_id))
            for category, items in by_category.items()
        }
        self._indices = {
            category: BM25Okapi([tokenize(item.text) for item in items])
            for category, items in self._documents.items()
            if items
        }

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted(self._documents))

    def search(
        self,
        query: str,
        *,
        category: str,
        top_k: int = 50,
        candidate_ids: set[str] | frozenset[str] | None = None,
    ) -> list[SearchHit]:
        """Return deterministic BM25 hits from only the requested category."""

        if top_k < 1:
            return []
        category_key = category.casefold()
        documents = self._documents.get(category_key, ())
        index = self._indices.get(category_key)
        if not documents or index is None:
            return []

        raw_scores = np.asarray(index.get_scores(tokenize(query)), dtype=np.float64)
        eligible = [
            (document.product_id, float(raw_scores[position]))
            for position, document in enumerate(documents)
            if candidate_ids is None or document.product_id in candidate_ids
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchHit(product_id=product_id, score=score, rank=rank, source=self.source_name)
            for rank, (product_id, score) in enumerate(eligible[:top_k], start=1)
        ]
