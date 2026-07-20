from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts import manage_annotations as annotation_cli
from scripts import prepare_relevance_annotation_batch as compiler

from pc_build_recommender.evaluation.manifest import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)


def _candidate(product_id: str) -> dict[str, object]:
    return {
        "product_id": product_id,
        "evidence_payload": {
            "canonical_name": f"Example {product_id}",
            "vram_gb": 16,
            "observed_benchmark_score": 123.4,
        },
        "provenance": {
            "source_name": "Official manufacturer specification",
            "source_url": f"https://manufacturer.example.test/{product_id}",
            "license_or_access_note": "Official public specification page.",
            "retrieved_at": "2026-07-23T00:00:00Z",
        },
        "is_synthetic": False,
    }


def _query(query_id: str, query_group_id: str, *product_ids: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_group_id": query_group_id,
        "category": "GPU",
        "query_text": "Quiet GPU with at least 16 GB VRAM for local AI",
        "structured_constraints": {"minimum_gpu_vram_gb": 16, "noise": "low"},
        "candidates": [_candidate(product_id) for product_id in product_ids],
        "is_synthetic": False,
    }


def _input_payload() -> dict[str, object]:
    return {
        "schema_version": compiler.INPUT_SCHEMA_VERSION,
        "dataset_name": "singapore-pc-relevance",
        "dataset_version": "candidate-capture-2026-07-23",
        "rubric_version": "relevance-rubric-v1",
        "data_version": "catalogue-2026-07-23",
        "source_policy": {
            "training_eligible": True,
            "published_metrics_eligible": True,
            "model_serving_eligible": False,
            "scope_note": "Official product evidence cleared for human relevance collection.",
        },
        # Intentionally unordered: the compiler writes a stable query-ID order.
        "queries": [
            _query("query-4", "intent-gaming", "gpu-4b", "gpu-4a"),
            _query("query-1", "intent-ai", "gpu-1b", "gpu-1a"),
            _query("query-3", "intent-development", "gpu-3a"),
            _query("query-2", "intent-ai", "gpu-2a"),
        ],
    }


def _write_input(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_capture_manifest(input_path: Path) -> Path:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for query in sorted(payload["queries"], key=lambda item: item["query_id"]):
        product_ids = sorted(
            candidate["product_id"] for candidate in query["candidates"]
        )
        rows.append(
            {
                "schema_version": compiler.PRELABEL_QUERY_SCHEMA_VERSION,
                "query_id": query["query_id"],
                "query_group_id": query["query_group_id"],
                "category": str(query["category"]).casefold(),
                "context": {
                    "query_id": query["query_id"],
                    "query_text": query["query_text"],
                    "requirements": query["structured_constraints"],
                },
                "candidates": [
                    {
                        "product_id": product_id,
                        "category": str(query["category"]).casefold(),
                        "retrieval_scores": {
                            "bm25_score": 0.0,
                            "rrf_score": 0.0,
                            "vector_similarity": 0.0,
                        },
                    }
                    for product_id in product_ids
                ],
                "candidate_ids_sha256": sha256_json(product_ids),
                "feature_matrix_sha256": sha256_json([]),
            }
        )
    feature_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    feature_path = input_path.parent / "prelabel-features.jsonl"
    feature_path.write_bytes(feature_bytes)
    feature_contract = {
        "contains_relevance_labels": False,
        "label_free_by_construction": True,
    }
    candidate_universe = [
        {
            "query_id": row["query_id"],
            "query_group_id": row["query_group_id"],
            "category": row["category"],
            "candidate_ids_sha256": row["candidate_ids_sha256"],
        }
        for row in rows
    ]
    snapshot: dict[str, object] = {
        "schema_version": compiler.PRELABEL_SNAPSHOT_SCHEMA_VERSION,
        "file_name": feature_path.name,
        "file_sha256": sha256_file(feature_path),
        "size_bytes": feature_path.stat().st_size,
        "query_row_sha256": {
            str(row["query_id"]): sha256_json(row) for row in rows
        },
        "candidate_universe_sha256": sha256_json(candidate_universe),
        "feature_contract": feature_contract,
        "feature_contract_sha256": sha256_json(feature_contract),
        "label_state": "absent",
    }
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    manifest_path = input_path.parent / "capture-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": compiler.CAPTURE_MANIFEST_SCHEMA_VERSION,
                "candidate_file_sha256": sha256_file(input_path),
                "prelabel_ranking_snapshot": snapshot,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_compiler_creates_deterministic_blinded_jsonl_compatible_with_annotation_cli(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "candidates.json"
    _write_input(input_path, _input_payload())
    capture_manifest = _write_capture_manifest(input_path)

    first = compiler.compile_relevance_annotation_batch(
        input_path=input_path,
        capture_manifest_path=capture_manifest,
        output_dir=tmp_path / "batch-a",
        seed=19,
    )
    second = compiler.compile_relevance_annotation_batch(
        input_path=input_path,
        capture_manifest_path=capture_manifest,
        output_dir=tmp_path / "batch-b",
        seed=19,
    )

    first_dir = tmp_path / "batch-a"
    second_dir = tmp_path / "batch-b"
    assert first["query_count"] == 4
    assert first["item_count"] == 6
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (first_dir / "groups.jsonl").read_bytes() == (second_dir / "groups.jsonl").read_bytes()

    groups = tuple(
        annotation_cli._iter_jsonl_group_records(first_dir / "groups.jsonl", max_line_bytes=4096)
    )
    assert [group["group_key"] for group in groups] == ["query-1", "query-2", "query-3", "query-4"]
    split_by_query = {str(group["group_key"]): group["split_name"] for group in groups}
    assert split_by_query["query-1"] == split_by_query["query-2"]
    assert {str(group["split_name"]) for group in groups} == {"train", "validation", "test"}
    assert all(item["priority"] == 0 for group in groups for item in group["items"])
    assert groups[0]["items"][0]["target_id"] == "gpu-1a"
    assert groups[0]["items"][0]["evidence_payload"]["provenance"]["source_url"].startswith(
        "https://"
    )
    assert all(
        group["context_payload"]["ranking_prelabel_binding"]["snapshot_sha256"]
        for group in groups
    )
    assert "bm25_score" not in json.dumps(groups)
    assert "rrf_score" not in json.dumps(groups)

    project = json.loads((first_dir / "project-spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    assert project["schema_version"] == annotation_cli.PROJECT_SCHEMA_VERSION
    assert project["task_type"] == "relevance"
    assert manifest["query_group_count"] == 3
    assert manifest["files"]["groups.jsonl"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda payload: payload["queries"][0]["candidates"][0]["evidence_payload"].update(
                {"relevance_grade": 4}
            ),
            "reviewer bias",
        ),
        (
            lambda payload: payload["queries"][0]["candidates"][0]["evidence_payload"].update(
                {"label": 4}
            ),
            "reviewer bias",
        ),
        (
            lambda payload: payload["queries"][0]["candidates"][0].update({"rank_position": 1}),
            "unsupported fields",
        ),
        (
            lambda payload: payload["queries"][0]["candidates"][0].update({"is_synthetic": True}),
            "synthetic collection evidence",
        ),
        (
            lambda payload: payload["source_policy"].update({"published_metrics_eligible": False}),
            "published_metrics_eligible must be true",
        ),
        (
            lambda payload: payload["source_policy"].update({"model_serving_eligible": True}),
            "serving_attribution_notice",
        ),
    ],
)
def test_compiler_rejects_label_leakage_synthetic_inputs_and_ineligible_rights(
    tmp_path: Path,
    mutation: Any,
    expected: str,
) -> None:
    payload = _input_payload()
    mutation(payload)
    input_path = tmp_path / "unsafe.json"
    _write_input(input_path, payload)

    with pytest.raises(compiler.RelevanceAnnotationBatchInputError, match=expected):
        compiler.compile_relevance_annotation_batch(
            input_path=input_path,
            capture_manifest_path=tmp_path / "not-needed-for-rejected-input.json",
            output_dir=tmp_path / "unsafe-batch",
        )

    assert not (tmp_path / "unsafe-batch").exists()


def test_compiler_refuses_to_overwrite_an_existing_batch(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.json"
    output_dir = tmp_path / "batch"
    _write_input(input_path, _input_payload())
    capture_manifest = _write_capture_manifest(input_path)
    compiler.compile_relevance_annotation_batch(
        input_path=input_path,
        capture_manifest_path=capture_manifest,
        output_dir=output_dir,
    )

    with pytest.raises(
        compiler.RelevanceAnnotationBatchInputError, match="will not be overwritten"
    ):
        compiler.compile_relevance_annotation_batch(
            input_path=input_path,
            capture_manifest_path=capture_manifest,
            output_dir=output_dir,
        )


def test_compiler_rejects_the_documented_template_until_real_evidence_replaces_it(
    tmp_path: Path,
) -> None:
    template = Path("evals/retrieval/relevance-annotation-candidates.template.json")

    with pytest.raises(compiler.RelevanceAnnotationBatchInputError, match="template placeholder"):
        compiler.compile_relevance_annotation_batch(
            input_path=template,
            capture_manifest_path=tmp_path / "missing-capture-manifest.json",
            output_dir=tmp_path / "template-batch",
        )


def test_compiler_requires_enough_distinct_leakage_groups_for_a_freezable_release(
    tmp_path: Path,
) -> None:
    payload = _input_payload()
    payload["queries"] = [
        _query("query-1", "intent-ai", "gpu-1"),
        _query("query-2", "intent-gaming", "gpu-2"),
    ]
    input_path = tmp_path / "too-small.json"
    _write_input(input_path, payload)

    with pytest.raises(compiler.RelevanceAnnotationBatchInputError, match="three distinct"):
        compiler.compile_relevance_annotation_batch(
            input_path=input_path,
            capture_manifest_path=tmp_path / "not-needed-for-small-input.json",
            output_dir=tmp_path / "batch",
        )
