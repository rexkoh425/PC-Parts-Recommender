from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scripts.import_vector_catalog import main as import_main
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql

from pc_build_recommender.catalog.orm import ProductEmbeddingRecord
from pc_build_recommender.retrieval.embedding_index import (
    MANIFEST_SCHEMA_VERSION,
    TEXT_BUILDER_VERSION,
    build_product_embedding_text,
    embedding_encoder_fingerprint,
)
from pc_build_recommender.retrieval.postgres import (
    EMBEDDING_DIMENSION,
    MAX_BM25_DOCUMENTS,
    EmbeddingArtifactValidationError,
    PgVectorSearchBackend,
    PostgresVectorCatalogRepository,
    _batched,
    _pgvector_version_tuple,
    bm25_index_from_embedding_artifact,
    validate_embedding_artifact,
)
from pc_build_recommender.retrieval.vector import FloatMatrix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _product(product_id: str, model: str) -> dict[str, Any]:
    timestamp = "2026-07-22T00:00:00Z"
    return {
        "product_id": product_id,
        "category": "gpu",
        "brand": "Example",
        "model": model,
        "manufacturer_part_number": model,
        "gtin": None,
        "canonical_name": f"Example {model}",
        "release_date": None,
        "status": "active",
        "common_attributes": {"tags": ["GPU"]},
        "category_attributes": {"vram_gb": 16},
        "source_confidence": 0.9,
        "provenance": [
            {
                "provenance_id": f"src_{product_id}",
                "product_id": product_id,
                "listing_id": None,
                "source_name": "fixture",
                "source_url": "https://example.invalid/catalog",
                "source_type": "import",
                "retrieved_at": timestamp,
                "raw_content_hash": "b" * 64,
                "parser_version": "fixture-v1",
                "licence_or_access_note": "Test fixture only",
                "last_verified_at": timestamp,
                "extraction_confidence": 0.9,
            }
        ],
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def _write_artifact(tmp_path: Path) -> tuple[Path, Path]:
    catalog_path = tmp_path / "records.jsonl"
    records = [_product("prod_a", "GPU-A"), _product("prod_b", "GPU-B")]
    catalog_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    id_rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        text = build_product_embedding_text(record)
        content_hash = hashlib.sha256(
            f"{TEXT_BUILDER_VERSION}\0{record['product_id']}\0{text}".encode()
        ).hexdigest()
        id_rows.append(
            {
                "row_index": row_index,
                "product_id": record["product_id"],
                "category": record["category"],
                "content_hash": content_hash,
            }
        )
    id_map_path = artifact_dir / "ids.jsonl"
    id_map_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in id_rows),
        encoding="utf-8",
    )

    vectors = np.zeros((2, EMBEDDING_DIMENSION), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    embeddings_path = artifact_dir / "embeddings.npy"
    np.save(embeddings_path, vectors, allow_pickle=False)
    content_payload = "\n".join(
        f"{row['product_id']}:{row['content_hash']}" for row in id_rows
    ).encode()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "index_version": "fixture-index-v1",
        "data_version": "fixture-data-v1",
        "text_builder_version": TEXT_BUILDER_VERSION,
        "created_at_utc": "2026-07-22T00:00:00+00:00",
        "content_hash": hashlib.sha256(content_payload).hexdigest(),
        "product_count": 2,
        "source": {
            "input_path": str(catalog_path.resolve()),
            "files": [
                {
                    "path": str(catalog_path.resolve()),
                    "relative_path": catalog_path.name,
                    "sha256": _sha256(catalog_path),
                    "bytes": catalog_path.stat().st_size,
                }
            ],
            "file_count": 1,
        },
        "encoder": {
            "kind": "sentence_transformer",
            "model_name": "fixture-encoder",
            "fingerprint": "a" * 64,
            "dimension": EMBEDDING_DIMENSION,
            "normalised": True,
        },
        "matrix": {
            "dtype": "float32",
            "shape": [2, EMBEDDING_DIMENSION],
            "l2_normalised": True,
        },
        "artifacts": {
            "embeddings": {
                "path": embeddings_path.name,
                "sha256": _sha256(embeddings_path),
                "bytes": embeddings_path.stat().st_size,
            },
            "id_map": {
                "path": id_map_path.name,
                "sha256": _sha256(id_map_path),
                "bytes": id_map_path.stat().st_size,
            },
        },
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog_path, artifact_dir


def test_validate_embedding_artifact_cross_checks_every_row(tmp_path: Path) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)

    artifact = validate_embedding_artifact(catalog_path, artifact_dir)

    assert artifact.product_count == 2
    assert artifact.vectors.shape == (2, EMBEDDING_DIMENSION)
    assert artifact.embedding_model == "fixture-encoder"
    assert artifact.data_version == "fixture-data-v1"
    assert [row.product_id for row in artifact.id_rows] == ["prod_a", "prod_b"]
    assert isinstance(artifact.vectors.base, np.memmap)


def test_bm25_index_uses_the_validated_embedding_search_documents(tmp_path: Path) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)
    artifact = validate_embedding_artifact(catalog_path, artifact_dir)

    index = bm25_index_from_embedding_artifact(artifact)
    hits = index.search("GPU-A", category="gpu", top_k=2)

    assert [hit.product_id for hit in hits] == ["prod_a", "prod_b"]
    assert all(hit.source == "bm25" for hit in hits)


def test_bm25_index_rejects_an_oversized_release_corpus_before_indexing(tmp_path: Path) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)
    artifact = validate_embedding_artifact(catalog_path, artifact_dir)
    repeats = MAX_BM25_DOCUMENTS // artifact.product_count + 1
    oversized = replace(
        artifact,
        products=artifact.products * repeats,
        search_documents=artifact.search_documents * repeats,
    )

    with pytest.raises(EmbeddingArtifactValidationError, match="document serving limit"):
        bm25_index_from_embedding_artifact(oversized)


def test_database_batches_consume_import_rows_lazily() -> None:
    consumed: list[int] = []

    def rows() -> Iterator[int]:
        for value in range(5):
            consumed.append(value)
            yield value

    batches = _batched(rows(), 2)
    assert consumed == []
    assert next(batches) == [0, 1]
    assert consumed == [0, 1]
    assert next(batches) == [2, 3]
    assert consumed == [0, 1, 2, 3]


def test_pgvector_version_parser_supports_filtered_hnsw_requirement() -> None:
    assert _pgvector_version_tuple("0.8.5") == (0, 8, 5)
    with pytest.raises(RuntimeError, match="cannot parse"):
        _pgvector_version_tuple("development")


def test_embedding_release_identity_is_rollback_safe() -> None:
    assert {column.name for column in ProductEmbeddingRecord.__table__.primary_key} == {
        "product_id",
        "embedding_model",
        "data_version",
        "index_version",
        "encoder_fingerprint",
        "dataset_content_hash",
    }


def test_validated_artifact_can_be_relocated_without_weakening_hash_checks(
    tmp_path: Path,
) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["files"][0]["path"] = "/immutable-export/records.jsonl"
    manifest["source"]["input_path"] = "/immutable-export/records.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact = validate_embedding_artifact(catalog_path, artifact_dir)

    assert artifact.product_count == 2


def test_source_catalog_hash_catches_non_embedding_provenance_changes(
    tmp_path: Path,
) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)
    catalog_path.write_text(
        catalog_path.read_text(encoding="utf-8").replace("fixture-v1", "fixture-v2"),
        encoding="utf-8",
    )

    with pytest.raises(EmbeddingArtifactValidationError, match="source catalog SHA-256"):
        validate_embedding_artifact(catalog_path, artifact_dir)


def test_duplicate_provenance_ids_are_rejected_before_import(tmp_path: Path) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)
    catalog_path.write_text(
        catalog_path.read_text(encoding="utf-8").replace("src_prod_b", "src_prod_a"),
        encoding="utf-8",
    )
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_entry = manifest["source"]["files"][0]
    source_entry["sha256"] = _sha256(catalog_path)
    source_entry["bytes"] = catalog_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EmbeddingArtifactValidationError, match="duplicate source provenance"):
        validate_embedding_artifact(catalog_path, artifact_dir)


def test_validate_embedding_artifact_rejects_tampered_id_map(tmp_path: Path) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)
    id_map_path = artifact_dir / "ids.jsonl"
    id_map_path.write_text(
        id_map_path.read_text(encoding="utf-8").replace("prod_a", "prod_x"),
        encoding="utf-8",
    )

    with pytest.raises(EmbeddingArtifactValidationError, match="id-map artifact SHA-256"):
        validate_embedding_artifact(catalog_path, artifact_dir)


def test_validate_embedding_artifact_rejects_resigned_id_mismatch(tmp_path: Path) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)
    id_map_path = artifact_dir / "ids.jsonl"
    id_map_path.write_text(
        id_map_path.read_text(encoding="utf-8").replace("prod_a", "prod_x"),
        encoding="utf-8",
    )
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["id_map"]["sha256"] = _sha256(id_map_path)
    manifest["artifacts"]["id_map"]["bytes"] = id_map_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EmbeddingArtifactValidationError, match="catalog/id-map mismatch"):
        validate_embedding_artifact(catalog_path, artifact_dir)


def test_validate_embedding_artifact_rejects_wrong_dimension(tmp_path: Path) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)
    embeddings_path = artifact_dir / "embeddings.npy"
    vectors = np.zeros((2, 32), dtype=np.float32)
    vectors[:, 0] = 1.0
    np.save(embeddings_path, vectors, allow_pickle=False)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["embeddings"]["sha256"] = _sha256(embeddings_path)
    manifest["artifacts"]["embeddings"]["bytes"] = embeddings_path.stat().st_size
    manifest["matrix"]["shape"] = [2, 32]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EmbeddingArtifactValidationError, match="embedding dimension mismatch"):
        validate_embedding_artifact(catalog_path, artifact_dir)


def test_verify_only_command_never_needs_a_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog_path, artifact_dir = _write_artifact(tmp_path)

    exit_code = import_main(
        [
            "--catalog",
            str(catalog_path),
            "--artifact-dir",
            str(artifact_dir),
            "--verify-only",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["validation"] == "passed"
    assert output["product_count"] == 2
    assert output["matrix_shape"] == [2, EMBEDDING_DIMENSION]


class _FakeEncoder:
    model_name = "fixture-encoder"
    dimension = EMBEDDING_DIMENSION

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        matrix = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        matrix[:, 0] = 1.0
        return matrix


class _FakeResult:
    def all(self) -> list[tuple[str, float]]:
        return [("prod_a", 0.91), ("prod_b", 0.72)]


class _FakeSession:
    def __init__(self) -> None:
        self.statement: Any = None
        self.iterative_scan_enabled = False

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: Any) -> _FakeResult:
        if str(statement).startswith("SET LOCAL hnsw.iterative_scan"):
            self.iterative_scan_enabled = True
            return _FakeResult()
        self.statement = statement
        compiled = str(
            statement.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]
        )
        assert "<=>" in compiled
        assert "canonical_products.category" in compiled
        assert "product_embeddings.data_version" in compiled
        return _FakeResult()


def test_pgvector_backend_compiles_cosine_category_and_version_filters() -> None:
    session = _FakeSession()
    backend = PgVectorSearchBackend(
        lambda: session,  # type: ignore[arg-type]
        encoder=_FakeEncoder(),
        data_version="fixture-data-v1",
        index_version="fixture-index-v1",
        encoder_fingerprint=embedding_encoder_fingerprint(_FakeEncoder()),
        dataset_content_hash="b" * 64,
    )

    hits = backend.search("local AI 16 GB", category="gpu", top_k=2)

    assert [(hit.product_id, hit.rank, hit.source) for hit in hits] == [
        ("prod_a", 1, "pgvector"),
        ("prod_b", 2, "pgvector"),
    ]
    assert session.iterative_scan_enabled is True


def test_postgres_repository_rejects_non_postgres_engine() -> None:
    engine = create_engine("sqlite://")
    try:
        with pytest.raises(ValueError, match="requires PostgreSQL"):
            PostgresVectorCatalogRepository(engine)
    finally:
        engine.dispose()
