from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import capture_relevance_annotation_candidates as capture
from training import materialize_ranking_snapshot as materializer
from training.train_ranking import main as ranking_main

from pc_build_recommender.evaluation.manifest import sha256_file, sha256_json
from pc_build_recommender.ranking import LambdaMARTRanker
from pc_build_recommender.retrieval import (
    QueryGroupSplit,
    HumanJudgmentSet,
    LabelingQuery,
    ReviewerJudgment,
    write_human_judgment_set,
)


def _catalog_product(product_id: str, vram_gb: int) -> dict[str, object]:
    return {
        "record_type": "canonical_product",
        "training_eligible": True,
        "published_claims_eligible": True,
        "data": {
            "product_id": product_id,
            "category": "gpu",
            "canonical_name": f"Example {product_id}",
            "brand": "Example",
            "model": product_id,
            "manufacturer_part_number": f"MPN-{product_id}",
            "category_attributes": {"vram_gb": vram_gb},
            "provenance": [
                {
                    "source_name": "Official specification",
                    "source_url": f"https://manufacturer.example.test/{product_id}",
                    "licence_or_access_note": "Official public product specification.",
                    "retrieved_at": "2026-07-29T00:00:00Z",
                }
            ],
        },
    }


def _query_set() -> dict[str, object]:
    return {
        "schema_version": capture.QUERY_SET_SCHEMA_VERSION,
        "dataset_name": "human-ranking-test",
        "dataset_version": "human-ranking-v1",
        "rubric_version": "relevance-rubric-v1",
        "data_version": "catalog-test-v1",
        "source_policy": {
            "training_eligible": True,
            "published_metrics_eligible": True,
            "model_serving_eligible": False,
            "scope_note": "Rights-cleared product evidence for human relevance evaluation.",
        },
        "queries": [
            {
                "query_id": f"q{index}",
                "query_group_id": f"intent-{index}",
                "category": "gpu",
                "query_text": f"local AI GPU request {index}",
                "structured_constraints": {
                    "minimum_gpu_vram_gb": 16,
                    "workload": "local_ai",
                },
                "is_synthetic": False,
            }
            for index in range(3)
        ],
    }


def _create_capture(tmp_path: Path) -> Path:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    records_path = catalog_dir / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(_catalog_product(f"gpu-{index}", 12 + index * 4)) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    catalog_manifest = catalog_dir / "manifest.json"
    catalog_manifest.write_text(
        json.dumps({"files": {"records.jsonl": {"sha256": sha256_file(records_path)}}}),
        encoding="utf-8",
    )
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(_query_set()), encoding="utf-8")
    capture_dir = tmp_path / "capture"
    capture.capture_relevance_annotation_candidates(
        query_set_path=query_path,
        catalog_records_path=records_path,
        catalog_manifest_path=catalog_manifest,
        output_dir=capture_dir,
        top_k=2,
        per_source_top_k=3,
        max_candidates_per_query=3,
    )
    return capture_dir


def _create_annotation_release(tmp_path: Path, capture_dir: Path) -> Path:
    candidates = json.loads((capture_dir / "candidates.json").read_text(encoding="utf-8"))
    capture_manifest = json.loads(
        (capture_dir / "manifest.json").read_text(encoding="utf-8")
    )
    snapshot = capture_manifest["prelabel_ranking_snapshot"]
    queries = tuple(
        LabelingQuery(
            query_id=query["query_id"],
            query_group_id=query["query_group_id"],
            query_text=query["query_text"],
            category=query["category"],
            candidate_ids=tuple(
                candidate["product_id"] for candidate in query["candidates"]
            ),
        )
        for query in candidates["queries"]
    )
    judgments = tuple(
        ReviewerJudgment(
            query_id=query.query_id,
            product_id=product_id,
            reviewer_id=reviewer_id,
            grade=4 if position == 0 else 1,
            rationale=f"{reviewer_id} independently graded {product_id}",
            reviewed_at_utc="2026-07-29T01:00:00Z",
        )
        for query in queries
        for position, product_id in enumerate(query.candidate_ids)
        for reviewer_id in ("reviewer-a", "reviewer-b")
    )
    human = HumanJudgmentSet(
        dataset_name=candidates["dataset_name"],
        dataset_version=candidates["dataset_version"],
        queries=queries,
        judgments=judgments,
        adjudications=(),
    )
    qrels = human.adjudicate().frozen_candidates
    split = QueryGroupSplit.create(
        qrels,
        version=f"{qrels.version}:split-v1",
        weights={"train": 0.34, "validation": 0.33, "test": 0.33},
        seed=29,
    )
    candidate_rows = {query["query_id"]: query for query in candidates["queries"]}
    prelabel_rows = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (capture_dir / capture.PRELABEL_FEATURE_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    evidence = {
        "schema_version": materializer.EVIDENCE_SNAPSHOT_SCHEMA_VERSION,
        "task_type": "relevance",
        "dataset_name": candidates["dataset_name"],
        "dataset_version": candidates["dataset_version"],
        "rubric_version": candidates["rubric_version"],
        "data_version": candidates["data_version"],
        "source_policy": candidates["source_policy"],
        "source_policy_sha256": capture_manifest["source_policy_sha256"],
        "groups": [
            {
                "group_key": query.query_id,
                "leakage_group_id": query.query_group_id,
                "category": query.category,
                "split_name": split.assignments[query.query_id],
                "context_payload": {
                    "query_text": query.query_text,
                    "structured_constraints": candidate_rows[query.query_id][
                        "structured_constraints"
                    ],
                    "ranking_prelabel_binding": {
                        "snapshot_sha256": snapshot["snapshot_sha256"],
                        "query_row_sha256": snapshot["query_row_sha256"][
                            query.query_id
                        ],
                        "candidate_ids_sha256": prelabel_rows[query.query_id][
                            "candidate_ids_sha256"
                        ],
                    },
                },
                "items": [
                    {"target_id": product_id} for product_id in query.candidate_ids
                ],
            }
            for query in queries
        ],
    }
    release_dir = tmp_path / "annotation-release"
    release_dir.mkdir()
    write_human_judgment_set(human, release_dir / "human-judgments.json")
    qrels.save(release_dir / "qrels.json")
    split.save(release_dir / "query-split.json")
    (release_dir / "evidence-snapshots.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {
        name: {
            "sha256": sha256_file(release_dir / name),
            "size_bytes": (release_dir / name).stat().st_size,
        }
        for name in sorted(materializer._RELEVANCE_RELEASE_FILES)
    }
    identity = {
        "schema_version": materializer.ANNOTATION_RELEASE_SCHEMA_VERSION,
        "task_type": "relevance",
        "dataset_name": candidates["dataset_name"],
        "dataset_version": candidates["dataset_version"],
        "rubric_version": candidates["rubric_version"],
        "data_version": candidates["data_version"],
        "source_policy_sha256": capture_manifest["source_policy_sha256"],
        "required_independent_reviews": 2,
        "files": files,
    }
    manifest = {**identity, "release_sha256": sha256_json(identity)}
    (release_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_dir


def test_materializer_appends_only_adjudicated_grades_and_binds_lineage(
    tmp_path: Path,
) -> None:
    capture_dir = _create_capture(tmp_path)
    release_dir = _create_annotation_release(tmp_path, capture_dir)

    result = materializer.materialize_ranking_snapshot(
        capture_dir=capture_dir,
        annotation_release_dir=release_dir,
        output_dir=tmp_path / "labeled",
    )

    prelabel = [
        json.loads(line)
        for line in (capture_dir / capture.PRELABEL_FEATURE_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    labeled = [
        json.loads(line)
        for line in (tmp_path / "labeled" / materializer.RANKING_DATA_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["query_count"] == 3
    assert result["row_count"] == 9
    for before, after in zip(prelabel, labeled, strict=True):
        stripped = {
            **after,
            "candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "relevance_grade"
                }
                for candidate in after["candidates"]
            ],
        }
        assert stripped == before
        assert all(
            candidate["relevance_grade"] in {1, 4}
            for candidate in after["candidates"]
        )

    manifest = json.loads(
        (tmp_path / "labeled" / "manifest.json").read_text(encoding="utf-8")
    )
    semantic = dict(manifest)
    manifest_sha256 = semantic.pop("manifest_sha256")
    assert sha256_json(semantic) == manifest_sha256
    assert manifest_sha256 == result["manifest_sha256"]
    assert manifest["prelabel"]["snapshot_sha256"]
    assert manifest["annotation_release"]["release_sha256"]


def test_materializer_rejects_prelabel_tampering_and_existing_output(
    tmp_path: Path,
) -> None:
    capture_dir = _create_capture(tmp_path)
    release_dir = _create_annotation_release(tmp_path, capture_dir)
    output_dir = tmp_path / "labeled"
    materializer.materialize_ranking_snapshot(
        capture_dir=capture_dir,
        annotation_release_dir=release_dir,
        output_dir=output_dir,
    )

    with pytest.raises(
        materializer.RankingSnapshotMaterializationError,
        match="will not be overwritten",
    ):
        materializer.materialize_ranking_snapshot(
            capture_dir=capture_dir,
            annotation_release_dir=release_dir,
            output_dir=output_dir,
        )

    feature_path = capture_dir / capture.PRELABEL_FEATURE_FILENAME
    feature_path.write_bytes(feature_path.read_bytes() + b"\n")
    with pytest.raises(
        materializer.RankingSnapshotMaterializationError,
        match="size mismatch",
    ):
        materializer.materialize_ranking_snapshot(
            capture_dir=capture_dir,
            annotation_release_dir=release_dir,
            output_dir=tmp_path / "tampered",
        )


@pytest.mark.model
def test_capture_to_human_release_to_materialized_lambdamart_is_bound_end_to_end(
    tmp_path: Path,
) -> None:
    capture_dir = _create_capture(tmp_path)
    release_dir = _create_annotation_release(tmp_path, capture_dir)
    labeled_dir = tmp_path / "labeled"
    materializer.materialize_ranking_snapshot(
        capture_dir=capture_dir,
        annotation_release_dir=release_dir,
        output_dir=labeled_dir,
    )
    artifact_dir = tmp_path / "ranker"
    qrels = json.loads((release_dir / "qrels.json").read_text(encoding="utf-8"))

    assert (
        ranking_main(
            [
                "--input",
                str(labeled_dir / materializer.RANKING_DATA_FILENAME),
                "--dataset-manifest",
                str(labeled_dir / "manifest.json"),
                "--artifact-dir",
                str(artifact_dir),
                "--candidate-set-version",
                qrels["version"],
                "--label-provenance",
                "human",
                "--human-judgments",
                str(release_dir / "human-judgments.json"),
                "--qrels",
                str(release_dir / "qrels.json"),
                "--frozen-query-split",
                str(release_dir / "query-split.json"),
                "--early-stopping-rounds",
                "1",
                "--parameters",
                json.dumps(
                    {
                        "n_estimators": 2,
                        "num_leaves": 3,
                        "min_child_samples": 1,
                        "n_jobs": 1,
                    }
                ),
            ]
        )
        == 0
    )

    ranker = LambdaMARTRanker.load(
        artifact_dir / "ranker-artifact" / "ranker.txt"
    )
    labeled_manifest = json.loads(
        (labeled_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        ranker.metadata.training_dataset_manifest_sha256
        == labeled_manifest["manifest_sha256"]
    )
    assert (
        ranker.metadata.training_prelabel_snapshot_sha256
        == labeled_manifest["prelabel"]["snapshot_sha256"]
    )
    assert (
        ranker.metadata.training_feature_contract_sha256
        == labeled_manifest["prelabel"]["feature_contract_sha256"]
    )
