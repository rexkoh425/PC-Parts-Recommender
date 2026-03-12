from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pc_build_recommender.evaluation.manifest import canonical_json_bytes, sha256_file
from pc_build_recommender.evaluation.retrieval_silver_pilot import (
    SILVER_QUERY_SCHEMA_VERSION,
    SilverConstraint,
    SilverQuery,
    SilverQuerySet,
    build_frozen_silver_dataset,
    run_retrieval_silver_pilot,
    silver_grade,
)
from pc_build_recommender.retrieval import RelevanceLabelSource


def _gpu(product_id: str, *, vram_gb: int) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "category": "gpu",
        "brand": "Example",
        "model": product_id,
        "canonical_name": f"Example GPU {product_id} {vram_gb} GB",
        "common_attributes": {"tags": ["GPU"]},
        "category_attributes": {
            "vram_gb": vram_gb,
            "vram_type": "GDDR6",
            "board_power_watts": 180,
        },
    }


def _query() -> SilverQuery:
    return SilverQuery(
        query_id="gpu-ai",
        query_text="local AI graphics card with at least 16 GB video memory",
        category="GPU",
        must=(SilverConstraint("category_attributes.vram_gb", "ge", 16),),
        excellent=(SilverConstraint("category_attributes.vram_gb", "ge", 24),),
    )


def test_silver_constraints_are_strict_and_case_insensitive() -> None:
    record = {
        "category_attributes": {
            "socket": "AM5",
            "supported_sockets": ["AM5", "LGA 1700"],
            "tdp_watts": 65,
        }
    }

    assert SilverConstraint("category_attributes.socket", "eq", "am5").matches(record)
    assert SilverConstraint("category_attributes.tdp_watts", "le", 65).matches(record)
    assert SilverConstraint(
        "category_attributes.supported_sockets", "contains_all", ["am5", "lga 1700"]
    ).matches(record)
    assert not SilverConstraint("category_attributes.missing", "not_null", True).matches(record)
    assert not SilverConstraint("category_attributes.tdp_watts", "ge", 66).matches(record)


def test_frozen_silver_dataset_grades_and_freezes_complete_category() -> None:
    records = [_gpu("gpu-a", vram_gb=8), _gpu("gpu-b", vram_gb=16), _gpu("gpu-c", vram_gb=24)]
    query_set = SilverQuerySet(
        dataset_name="test",
        judgment_method="test-only silver predicates",
        queries=(_query(),),
        source_sha256="1" * 64,
    )

    dataset = build_frozen_silver_dataset(records, query_set, catalog_sha256="2" * 64)

    assert dataset.queries[0].candidate_ids == ("gpu-a", "gpu-b", "gpu-c")
    assert dataset.queries[0].relevance_labels == {"gpu-b": 3, "gpu-c": 4}
    assert dataset.label_source is RelevanceLabelSource.SILVER
    assert not dataset.eligible_for_promotion
    assert silver_grade(records[0], _query()) == 0
    assert silver_grade(records[1], _query()) == 3
    assert silver_grade(records[2], _query()) == 4


def _write_embedding_artifact(root: Path, records: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True)
    matrix = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    embeddings_path = root / "embeddings.npy"
    np.save(embeddings_path, matrix, allow_pickle=False)
    ids_path = root / "ids.jsonl"
    ids_path.write_text(
        "".join(
            json.dumps(
                {
                    "row_index": index,
                    "product_id": record["product_id"],
                    "category": record["category"],
                    "content_hash": str(index) * 64,
                },
                sort_keys=True,
            )
            + "\n"
            for index, record in enumerate(records)
        ),
        encoding="utf-8",
    )
    manifest = {
        "content_hash": "3" * 64,
        "index_version": "test-index-v1",
        "encoder": {
            "kind": "sentence_transformer",
            "model_name": "test-encoder",
            "resolved_device": "cuda",
        },
        "artifacts": {
            "embeddings": {
                "path": embeddings_path.name,
                "sha256": sha256_file(embeddings_path),
                "bytes": embeddings_path.stat().st_size,
            },
            "id_map": {
                "path": ids_path.name,
                "sha256": sha256_file(ids_path),
                "bytes": ids_path.stat().st_size,
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_end_to_end_pilot_is_explicitly_non_reportable(tmp_path, monkeypatch) -> None:
    records = [_gpu("gpu-a", vram_gb=8), _gpu("gpu-b", vram_gb=16), _gpu("gpu-c", vram_gb=24)]
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    query_path = tmp_path / "queries.json"
    query_path.write_text(
        json.dumps(
            {
                "schema_version": SILVER_QUERY_SCHEMA_VERSION,
                "dataset_name": "test-silver-pilot",
                "judgment_method": "deterministic predicates, not human labels",
                "queries": [
                    {
                        "query_id": "gpu-ai",
                        "query_text": "local AI graphics card with at least 16 GB video memory",
                        "category": "gpu",
                        "must": [
                            {
                                "field": "category_attributes.vram_gb",
                                "operator": "ge",
                                "value": 16,
                            }
                        ],
                        "excellent": [
                            {
                                "field": "category_attributes.vram_gb",
                                "operator": "ge",
                                "value": 24,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    embedding_dir = tmp_path / "embeddings"
    _write_embedding_artifact(embedding_dir, records)

    def fake_vector_rankings(dataset, **_kwargs):  # type: ignore[no-untyped-def]
        query = dataset.queries[0]
        return (
            {query.query_id: ["gpu-c", "gpu-b", "gpu-a"]},
            {
                "model_name": "test-encoder",
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "batch_size": 2,
                "dimension": 2,
                "cosine_score_rounding_decimals": 8,
            },
        )

    monkeypatch.setattr(
        "pc_build_recommender.evaluation.retrieval_silver_pilot._complete_vector_rankings",
        fake_vector_rankings,
    )
    output_dir = tmp_path / "output"
    summary = run_retrieval_silver_pilot(
        catalog_path=catalog_path,
        query_set_path=query_path,
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        device="cpu",
        batch_size=2,
    )

    artifact = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    stored_hash = artifact.pop("artifact_sha256")
    assert stored_hash == hashlib.sha256(canonical_json_bytes(artifact)).hexdigest()
    assert summary["eligible_for_production_or_resume_metric_claims"] is False
    assert artifact["human_relevance_judgments"] is False
    assert artifact["training_performed"] is False
    assert artifact["learning_to_rank_status"]["trained"] is False
    assert artifact["dataset"]["positive_qrel_count"] == 2
    assert artifact["models"]["vector_only"]["metrics"]["aggregate"]["ndcg_at_10"] == 1.0
    assert (output_dir / "frozen-candidates.json").is_file()
