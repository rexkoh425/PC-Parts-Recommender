from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import capture_relevance_annotation_candidates as capture
from scripts import prepare_relevance_annotation_batch as compiler

from pc_build_recommender.evaluation.manifest import sha256_file
from pc_build_recommender.retrieval import ProductDocument, RetrievedCandidate, SearchHit


def _source(product_id: str) -> dict[str, object]:
    return {
        "source_name": "BuildCores OpenDB",
        "source_url": f"https://manufacturer.example.test/{product_id}",
        "licence_or_access_note": "ODC-By 1.0; attribution required.",
        "retrieved_at": "2026-07-23T00:00:00Z",
    }


def _product(product_id: str, name: str, vram_gb: int) -> dict[str, object]:
    return {
        "record_type": "canonical_product",
        "training_eligible": True,
        "published_claims_eligible": True,
        "data": {
            "product_id": product_id,
            "category": "gpu",
            "canonical_name": name,
            "brand": "Example",
            "model": name,
            "manufacturer_part_number": f"MPN-{product_id}",
            "common_attributes": {"colour": "black"},
            "category_attributes": {"vram_gb": vram_gb, "board_power_watts": 220},
            "source_confidence": 0.9,
            "provenance": [_source(product_id)],
        },
    }


def _query_set() -> dict[str, object]:
    return {
        "schema_version": capture.QUERY_SET_SCHEMA_VERSION,
        "dataset_name": "buildcores-relevance-collection",
        "dataset_version": "capture-2026-07-23",
        "rubric_version": "relevance-rubric-v1",
        "data_version": "buildcores-test-snapshot",
        "source_policy": {
            "training_eligible": True,
            "published_metrics_eligible": True,
            "model_serving_eligible": False,
            "scope_note": "Rights-cleared catalogue candidate discovery for human review.",
        },
        "queries": [
            {
                "query_id": "gpu-ai",
                "query_group_id": "intent-ai",
                "category": "gpu",
                "query_text": "GPU with at least 16 GB for local AI",
                "structured_constraints": {"minimum_gpu_vram_gb": 16},
                "is_synthetic": False,
            },
            {
                "query_id": "gpu-gaming",
                "query_group_id": "intent-gaming",
                "category": "gpu",
                "query_text": "quiet GPU for 1440p gaming",
                "structured_constraints": {"noise": "low"},
                "is_synthetic": False,
            },
            {
                "query_id": "gpu-efficient",
                "query_group_id": "intent-efficiency",
                "category": "gpu",
                "query_text": "power efficient graphics card",
                "structured_constraints": {"power_efficiency": "high"},
                "is_synthetic": False,
            },
        ],
    }


def _write_catalogue(root: Path) -> tuple[Path, Path]:
    records = root / "records.jsonl"
    rows = [
        _product("gpu-1", "Example Alpha 16", 16),
        _product("gpu-2", "Example Bravo 12", 12),
        _product("gpu-3", "Example Charlie 24", 24),
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"files": {"records.jsonl": {"sha256": sha256_file(records)}}}),
        encoding="utf-8",
    )
    return records, manifest


def _write_query_set(path: Path, payload: dict[str, object] | None = None) -> None:
    path.write_text(json.dumps(payload or _query_set()), encoding="utf-8")


def test_capture_creates_score_blinded_candidates_that_feed_the_annotation_compiler(
    tmp_path: Path,
) -> None:
    records, manifest = _write_catalogue(tmp_path)
    query_set = tmp_path / "queries.json"
    _write_query_set(query_set)

    result = capture.capture_relevance_annotation_candidates(
        query_set_path=query_set,
        catalog_records_path=records,
        catalog_manifest_path=manifest,
        output_dir=tmp_path / "capture",
        top_k=2,
    )

    capture_dir = tmp_path / "capture"
    candidates = json.loads((capture_dir / "candidates.json").read_text(encoding="utf-8"))
    capture_manifest = json.loads((capture_dir / "manifest.json").read_text(encoding="utf-8"))
    prelabel_rows = [
        json.loads(line)
        for line in (capture_dir / capture.PRELABEL_FEATURE_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["query_count"] == 3
    assert result["candidate_count"] == 9
    assert capture_manifest["reviewer_blinding"]["retrieval_scores_excluded"] is True
    assert capture_manifest["reviewer_blinding"]["retrieval_ranks_excluded"] is True
    assert capture_manifest["reviewer_blinding"]["retrieval_source_membership_excluded"] is True
    assert capture_manifest["retrieval_pooling"]["strategy"] == "rrf-plus-source-union"
    assert capture_manifest["retrieval_pooling"]["source_names"] == ["bm25", "vector", "rrf"]
    snapshot = capture_manifest["prelabel_ranking_snapshot"]
    assert snapshot["schema_version"] == capture.PRELABEL_SNAPSHOT_SCHEMA_VERSION
    assert snapshot["label_state"] == "absent"
    assert snapshot["feature_contract"]["label_free_by_construction"] is True
    assert snapshot["file_sha256"] == sha256_file(
        capture_dir / capture.PRELABEL_FEATURE_FILENAME
    )
    assert len(prelabel_rows) == 3
    assert all(
        row["schema_version"] == capture.PRELABEL_QUERY_SCHEMA_VERSION
        for row in prelabel_rows
    )
    assert all(
        set(candidate["retrieval_scores"])
        == {"bm25_score", "lexical_score", "rrf_score", "vector_similarity"}
        for row in prelabel_rows
        for candidate in row["candidates"]
    )
    assert "relevance_grade" not in json.dumps(prelabel_rows).casefold()
    assert "reviewer" not in json.dumps(prelabel_rows).casefold()
    assert "adjudication" not in json.dumps(prelabel_rows).casefold()
    assert "rank" not in json.dumps(candidates).casefold()
    assert "score" not in json.dumps(candidates).casefold()
    assert "bm25" not in json.dumps(candidates).casefold()
    assert "vector" not in json.dumps(candidates).casefold()
    assert all(
        candidate["provenance"]["source_url"].startswith("https://")
        for query in candidates["queries"]
        for candidate in query["candidates"]
    )

    compiled = compiler.compile_relevance_annotation_batch(
        input_path=capture_dir / "candidates.json",
        capture_manifest_path=capture_dir / "manifest.json",
        output_dir=tmp_path / "annotation-batch",
        seed=23,
    )
    assert compiled["query_count"] == 3
    assert (tmp_path / "annotation-batch" / "groups.jsonl").is_file()


def test_capture_pool_adds_source_exclusive_candidates_without_exposing_membership() -> None:
    fused_candidates = [
        RetrievedCandidate(
            product=ProductDocument(
                product_id="gpu-fused",
                category="gpu",
                text="fused candidate",
            ),
            rank=1,
            rrf_score=0.03,
        )
    ]
    selected_ids, counts = capture._select_pooled_candidate_ids(
        fused_candidates=fused_candidates,
        source_pools={
            "bm25": (
                SearchHit("gpu-fused", score=10.0, rank=1, source="bm25"),
                SearchHit("gpu-lexical-only", score=9.0, rank=2, source="bm25"),
            ),
            "vector": (
                SearchHit("gpu-fused", score=0.8, rank=1, source="vector"),
                SearchHit("gpu-semantic-only", score=0.7, rank=2, source="vector"),
            ),
        },
        max_candidates=3,
    )

    assert selected_ids == ("gpu-fused", "gpu-lexical-only", "gpu-semantic-only")
    assert counts == {
        "rrf_candidates": 1,
        "bm25_candidates": 2,
        "vector_candidates": 2,
        "bm25_source_exclusive_added": 1,
        "vector_source_exclusive_added": 1,
        "pooled_candidates": 3,
    }


def test_capture_rejects_catalogue_bytes_that_do_not_match_the_supplied_manifest(
    tmp_path: Path,
) -> None:
    records, manifest = _write_catalogue(tmp_path)
    records.write_text(records.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    query_set = tmp_path / "queries.json"
    _write_query_set(query_set)

    with pytest.raises(capture.RelevanceCandidateCaptureError, match="SHA-256"):
        capture.capture_relevance_annotation_candidates(
            query_set_path=query_set,
            catalog_records_path=records,
            catalog_manifest_path=manifest,
            output_dir=tmp_path / "capture",
        )


def test_capture_rejects_a_total_pool_smaller_than_the_fused_pool(tmp_path: Path) -> None:
    records, manifest = _write_catalogue(tmp_path)
    query_set = tmp_path / "queries.json"
    _write_query_set(query_set)

    with pytest.raises(ValueError, match="at least top_k"):
        capture.capture_relevance_annotation_candidates(
            query_set_path=query_set,
            catalog_records_path=records,
            catalog_manifest_path=manifest,
            output_dir=tmp_path / "capture",
            top_k=2,
            max_candidates_per_query=1,
        )


def test_capture_rejects_source_policy_that_cannot_support_later_metrics(tmp_path: Path) -> None:
    records, manifest = _write_catalogue(tmp_path)
    query_set = tmp_path / "queries.json"
    payload = _query_set()
    policy = payload["source_policy"]
    assert isinstance(policy, dict)
    policy["published_metrics_eligible"] = False
    _write_query_set(query_set, payload)

    with pytest.raises(capture.RelevanceCandidateCaptureError, match="published_metrics_eligible"):
        capture.capture_relevance_annotation_candidates(
            query_set_path=query_set,
            catalog_records_path=records,
            catalog_manifest_path=manifest,
            output_dir=tmp_path / "capture",
        )


def test_capture_requires_attribution_when_source_policy_allows_model_serving(
    tmp_path: Path,
) -> None:
    records, manifest = _write_catalogue(tmp_path)
    query_set = tmp_path / "queries.json"
    payload = _query_set()
    policy = payload["source_policy"]
    assert isinstance(policy, dict)
    policy["model_serving_eligible"] = True
    _write_query_set(query_set, payload)

    with pytest.raises(capture.RelevanceCandidateCaptureError, match="serving_attribution_notice"):
        capture.capture_relevance_annotation_candidates(
            query_set_path=query_set,
            catalog_records_path=records,
            catalog_manifest_path=manifest,
            output_dir=tmp_path / "capture",
        )


def test_capture_refuses_to_overwrite_an_immutable_capture(tmp_path: Path) -> None:
    records, manifest = _write_catalogue(tmp_path)
    query_set = tmp_path / "queries.json"
    _write_query_set(query_set)
    output = tmp_path / "capture"
    capture.capture_relevance_annotation_candidates(
        query_set_path=query_set,
        catalog_records_path=records,
        catalog_manifest_path=manifest,
        output_dir=output,
    )

    with pytest.raises(capture.RelevanceCandidateCaptureError, match="will not be overwritten"):
        capture.capture_relevance_annotation_candidates(
            query_set_path=query_set,
            catalog_records_path=records,
            catalog_manifest_path=manifest,
            output_dir=output,
        )


def test_capture_rejects_non_human_query_leakage_field(tmp_path: Path) -> None:
    records, manifest = _write_catalogue(tmp_path)
    query_set = tmp_path / "queries.json"
    payload = _query_set()
    queries = payload["queries"]
    assert isinstance(queries, list)
    first = queries[0]
    assert isinstance(first, dict)
    first["rank_position"] = 1
    _write_query_set(query_set, payload)

    with pytest.raises(capture.RelevanceCandidateCaptureError, match="unsupported fields"):
        capture.capture_relevance_annotation_candidates(
            query_set_path=query_set,
            catalog_records_path=records,
            catalog_manifest_path=manifest,
            output_dir=tmp_path / "capture",
        )
