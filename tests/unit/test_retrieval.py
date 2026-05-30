from __future__ import annotations

import numpy as np
import pytest

from pc_build_recommender.retrieval import (
    BM25ProductIndex,
    HybridProductRetriever,
    InMemoryVectorIndex,
    ProductDocument,
    SearchHit,
    StableHashEmbeddingEncoder,
    StructuredFilters,
    product_matches_filters,
    reciprocal_rank_fusion,
)


@pytest.fixture
def products() -> list[ProductDocument]:
    return [
        ProductDocument(
            product_id="gpu-4070s",
            category="gpu",
            text="NVIDIA GeForce RTX 4070 Super 12 GB quiet 1440p gaming",
            brand="NVIDIA",
            price_sgd=899,
            stock_status="in_stock",
            attributes={"model": "RTX 4070 Super", "vram_gb": 12},
        ),
        ProductDocument(
            product_id="gpu-7900xt",
            category="gpu",
            text="AMD Radeon RX 7900 XT 20 GB high performance 4K gaming",
            brand="AMD",
            price_sgd=1099,
            stock_status="limited_stock",
            attributes={"model": "RX 7900 XT", "vram_gb": 20},
        ),
        ProductDocument(
            product_id="gpu-4090-oos",
            category="gpu",
            text="NVIDIA GeForce RTX 4090 24 GB local AI inference",
            brand="NVIDIA",
            price_sgd=2999,
            stock_status="out_of_stock",
            attributes={"model": "RTX 4090", "vram_gb": 24},
        ),
        ProductDocument(
            product_id="cpu-7950x",
            category="cpu",
            text="AMD Ryzen 9 7950X compilation multicore development",
            brand="AMD",
            price_sgd=749,
            stock_status="in_stock",
        ),
    ]


def test_product_document_builds_search_text_from_mapping() -> None:
    document = ProductDocument.from_mapping(
        {
            "product_id": "mem-1",
            "category": "memory",
            "brand": "G.Skill",
            "canonical_name": "Trident Z5",
            "category_attributes": {"memory_type": "DDR5", "capacity_gb": 32},
            "current_price_sgd": 159,
            "stock_status": "available",
        }
    )

    assert document.category == "memory"
    assert document.get("memory_type") == "DDR5"
    assert "Trident Z5" in document.text


def test_bm25_and_vector_search_are_category_scoped(
    products: list[ProductDocument],
) -> None:
    bm25 = BM25ProductIndex(products)
    vector = InMemoryVectorIndex(products)

    bm25_hits = bm25.search("compilation development", category="cpu")
    vector_hits = vector.search("compilation development", category="cpu")

    assert [hit.product_id for hit in bm25_hits] == ["cpu-7950x"]
    assert [hit.product_id for hit in vector_hits] == ["cpu-7950x"]
    assert all(hit.source == "bm25" for hit in bm25_hits)
    assert all(hit.source == "vector" for hit in vector_hits)


def test_hash_embedding_is_deterministic_and_lexically_useful() -> None:
    encoder = StableHashEmbeddingEncoder(dimension=256)
    first = encoder.encode(["RTX 4070 Super local AI", "Radeon graphics"])
    second = encoder.encode(["RTX 4070 Super local AI", "Radeon graphics"])
    query = encoder.encode(["RTX4070 Super AI"])[0]

    np.testing.assert_array_equal(first, second)
    assert np.linalg.norm(first[0]) == pytest.approx(1.0)
    assert float(first[0] @ query) > float(first[1] @ query)
    assert encoder.model_name.startswith("stable-lexical-hash-v1")


def test_reciprocal_rank_fusion_uses_rank_not_raw_score() -> None:
    bm25 = [
        SearchHit("a", score=1000.0, rank=1, source="bm25"),
        SearchHit("b", score=999.0, rank=2, source="bm25"),
    ]
    vector = [
        SearchHit("b", score=0.51, rank=1, source="vector"),
        SearchHit("c", score=0.50, rank=2, source="vector"),
    ]

    fused = reciprocal_rank_fusion({"bm25": bm25, "vector": vector})

    assert [hit.product_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].source_ranks == {"bm25": 2, "vector": 1}


def test_structured_filters_are_strict_for_applicable_category(
    products: list[ProductDocument],
) -> None:
    filters = StructuredFilters(
        maximum_price_sgd=1200,
        minimum_gpu_vram_gb=16,
        excluded_brands=frozenset({"intel"}),
        allowed_product_ids=frozenset({"gpu-7900xt", "gpu-4090-oos"}),
    )

    assert product_matches_filters(products[1], filters)
    assert not product_matches_filters(products[0], filters)  # too little VRAM / allow-list
    assert not product_matches_filters(products[2], filters)  # price and stock
    # A GPU-only minimum must not remove unrelated CPU products.
    assert product_matches_filters(
        products[3], StructuredFilters(minimum_gpu_vram_gb=16, in_stock_only=True)
    )


def test_hybrid_retrieval_filters_before_source_top_k(
    products: list[ProductDocument],
) -> None:
    retriever = HybridProductRetriever(products)

    candidates = retriever.retrieve(
        "24 GB local AI inference",
        category="gpu",
        filters=StructuredFilters(maximum_price_sgd=1200, minimum_gpu_vram_gb=16),
        top_k=5,
    )

    assert [candidate.product_id for candidate in candidates] == ["gpu-7900xt"]
    assert candidates[0].bm25_rank == 1
    assert candidates[0].vector_rank == 1


def test_hybrid_retrieval_source_pools_use_the_same_filter_gate(
    products: list[ProductDocument],
) -> None:
    retriever = HybridProductRetriever(products)

    candidates, source_pools = retriever.retrieve_with_source_pools(
        "24 GB local AI inference",
        category="gpu",
        filters=StructuredFilters(maximum_price_sgd=1200, minimum_gpu_vram_gb=16),
        top_k=5,
        per_source_k=5,
    )

    assert [candidate.product_id for candidate in candidates] == ["gpu-7900xt"]
    assert set(source_pools) == {"bm25", "vector"}
    assert all(
        hit.product_id == "gpu-7900xt"
        for source_hits in source_pools.values()
        for hit in source_hits
    )


def test_retrieve_categories_keeps_independent_candidate_pools(
    products: list[ProductDocument],
) -> None:
    retriever = HybridProductRetriever(products)

    result = retriever.retrieve_categories(
        "development and 1440p gaming",
        ["cpu", "gpu"],
        filters_by_category={
            "gpu": StructuredFilters(maximum_price_sgd=1000),
            "cpu": StructuredFilters(maximum_price_sgd=800),
        },
    )

    assert [item.product_id for item in result["cpu"]] == ["cpu-7950x"]
    assert [item.product_id for item in result["gpu"]] == ["gpu-4070s"]
