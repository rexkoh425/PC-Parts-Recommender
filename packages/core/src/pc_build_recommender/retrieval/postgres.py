"""Strict PostgreSQL/pgvector catalog import and semantic retrieval.

This module is deliberately separate from API orchestration.  It validates the
complete disk artifact contract before opening a write transaction, imports the
catalog and vectors idempotently, and implements the storage-agnostic
``VectorSearchBackend`` protocol with PostgreSQL cosine distance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from pc_build_recommender.catalog.orm import (
    CanonicalProductRecord,
    ProductEmbeddingRecord,
    SourceProvenanceRecord,
)
from pc_build_recommender.domain import MasterProduct

from .bm25 import BM25ProductIndex
from .embedding_index import (
    MANIFEST_SCHEMA_VERSION,
    TEXT_BUILDER_VERSION,
    build_product_embedding_text,
    embedding_encoder_fingerprint,
    load_normalized_product_jsonl,
)
from .models import ProductDocument, SearchHit, StructuredFilterSpec
from .postgres_filters import normalize_postgres_category, postgres_structured_predicates
from .vector import EmbeddingEncoder

EMBEDDING_DIMENSION = 384
_SHA256_LENGTH = 64
_MAX_DATABASE_TOP_K = 1000
_MAX_QUERY_CHARACTERS = 4096
_MINIMUM_PGVECTOR_VERSION = (0, 8, 0)
MAX_BM25_DOCUMENTS = 50_000
MAX_BM25_TEXT_BYTES = 64 * 1024 * 1024


class EmbeddingArtifactValidationError(ValueError):
    """Raised before database mutation when an artifact contract is inconsistent."""


@dataclass(frozen=True, slots=True)
class EmbeddingIdRow:
    row_index: int
    product_id: str
    category: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class ValidatedEmbeddingArtifact:
    """An immutable, fully cross-checked catalog/vector import payload."""

    artifact_dir: Path
    catalog_path: Path
    products: tuple[MasterProduct, ...]
    search_documents: tuple[str, ...]
    id_rows: tuple[EmbeddingIdRow, ...]
    vectors: NDArray[np.float32]
    embedding_model: str
    data_version: str
    index_version: str
    encoder_fingerprint: str
    dataset_content_hash: str
    embeddings_artifact_sha256: str
    id_map_artifact_sha256: str
    created_at: datetime
    manifest: Mapping[str, Any]

    @property
    def product_count(self) -> int:
        return len(self.products)


def bm25_index_from_embedding_artifact(
    artifact: ValidatedEmbeddingArtifact,
) -> BM25ProductIndex:
    """Build a bounded lexical index from the exact release-bound search corpus.

    The database import verifier establishes that these product IDs and search
    documents are identical to the rows joined by pgvector.  Keeping BM25's
    immutable corpus bound to that same validated artifact prevents a serving
    process from silently mixing lexical and semantic catalogue versions.
    """

    if len(artifact.products) != len(artifact.search_documents):
        raise EmbeddingArtifactValidationError(
            "embedding artifact product and search-document counts do not match"
        )
    if not artifact.products:
        raise EmbeddingArtifactValidationError("embedding artifact contains no products")
    if len(artifact.products) > MAX_BM25_DOCUMENTS:
        raise EmbeddingArtifactValidationError(
            f"BM25 corpus exceeds the {MAX_BM25_DOCUMENTS} document serving limit"
        )

    documents: list[ProductDocument] = []
    text_bytes = 0
    for product, search_document in zip(artifact.products, artifact.search_documents, strict=True):
        encoded_size = len(search_document.encode("utf-8"))
        text_bytes += encoded_size
        if text_bytes > MAX_BM25_TEXT_BYTES:
            raise EmbeddingArtifactValidationError(
                "BM25 corpus exceeds the configured text-byte serving limit"
            )
        documents.append(
            ProductDocument(
                product_id=product.product_id,
                category=product.category.value,
                text=search_document,
                brand=product.brand,
            )
        )
    return BM25ProductIndex(documents)


@dataclass(frozen=True, slots=True)
class VectorCatalogImportResult:
    """Measured database state after an idempotent import transaction."""

    product_count: int
    provenance_count: int
    embedding_count: int
    database_product_count: int
    database_embedding_count: int
    embedding_model: str
    data_version: str
    index_version: str
    dataset_content_hash: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "product_count": self.product_count,
            "provenance_count": self.provenance_count,
            "embedding_count": self.embedding_count,
            "database_product_count": self.database_product_count,
            "database_embedding_count": self.database_embedding_count,
            "embedding_model": self.embedding_model,
            "data_version": self.data_version,
            "index_version": self.index_version,
            "dataset_content_hash": self.dataset_content_hash,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pgvector_version_tuple(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"([0-9]+)[.]([0-9]+)[.]([0-9]+)", version.strip())
    if match is None:
        raise RuntimeError(f"cannot parse PostgreSQL vector extension version: {version!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmbeddingArtifactValidationError(f"{path} must be an object")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EmbeddingArtifactValidationError(f"{path}.{key} must be a non-empty string")
    return value.strip()


def _require_sha256(value: str, path: str) -> str:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise EmbeddingArtifactValidationError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _safe_artifact_path(artifact_dir: Path, relative_path: str, label: str) -> Path:
    root = artifact_dir.resolve()
    candidate = (root / relative_path).resolve()
    if candidate.parent != root:
        raise EmbeddingArtifactValidationError(f"{label} must be a file directly in artifact_dir")
    if not candidate.is_file():
        raise EmbeddingArtifactValidationError(f"{label} does not exist: {candidate}")
    return candidate


def _read_manifest(artifact_dir: Path) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise EmbeddingArtifactValidationError(f"manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EmbeddingArtifactValidationError(f"manifest is invalid JSON: {exc.msg}") from exc
    return manifest_path, _require_mapping(payload, "manifest")


def _load_id_rows(path: Path) -> tuple[EmbeddingIdRow, ...]:
    rows: list[EmbeddingIdRow] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmbeddingArtifactValidationError(
                    f"invalid id-map JSON at line {line_number}: {exc.msg}"
                ) from exc
            item = _require_mapping(raw, f"id_map[{line_number - 1}]")
            row_index = item.get("row_index")
            product_id = item.get("product_id")
            category = item.get("category")
            content_hash = item.get("content_hash")
            if not isinstance(row_index, int) or row_index != len(rows):
                raise EmbeddingArtifactValidationError(
                    f"id-map row_index must be contiguous; expected {len(rows)}, got {row_index!r}"
                )
            if not isinstance(product_id, str) or not product_id.strip():
                raise EmbeddingArtifactValidationError(
                    f"id-map line {line_number} has no product_id"
                )
            if product_id in seen_ids:
                raise EmbeddingArtifactValidationError(f"duplicate id-map product_id: {product_id}")
            if not isinstance(category, str) or not category.strip():
                raise EmbeddingArtifactValidationError(f"id-map line {line_number} has no category")
            if not isinstance(content_hash, str):
                raise EmbeddingArtifactValidationError(
                    f"id-map line {line_number} has no content_hash"
                )
            rows.append(
                EmbeddingIdRow(
                    row_index=row_index,
                    product_id=product_id.strip(),
                    category=category.strip().casefold(),
                    content_hash=_require_sha256(
                        content_hash, f"id_map[{line_number - 1}].content_hash"
                    ),
                )
            )
            seen_ids.add(product_id)
    if not rows:
        raise EmbeddingArtifactValidationError("id map contains no rows")
    return tuple(rows)


def _batched[T](items: Iterable[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("batch_size must be positive")
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def validate_embedding_artifact(
    catalog_path: str | Path,
    artifact_dir: str | Path,
    *,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> ValidatedEmbeddingArtifact:
    """Cross-check catalog text, row IDs, content hashes, matrix, and manifest hashes."""

    catalog = Path(catalog_path).resolve()
    artifacts = Path(artifact_dir).resolve()
    if expected_dimension < 1:
        raise ValueError("expected_dimension must be positive")
    _, manifest = _read_manifest(artifacts)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EmbeddingArtifactValidationError(
            f"unsupported manifest schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("text_builder_version") != TEXT_BUILDER_VERSION:
        raise EmbeddingArtifactValidationError(
            f"unsupported text builder: {manifest.get('text_builder_version')!r}"
        )

    artifact_entries = _require_mapping(manifest.get("artifacts"), "manifest.artifacts")
    embedding_entry = _require_mapping(
        artifact_entries.get("embeddings"), "manifest.artifacts.embeddings"
    )
    id_map_entry = _require_mapping(artifact_entries.get("id_map"), "manifest.artifacts.id_map")
    embeddings_path = _safe_artifact_path(
        artifacts,
        _require_string(embedding_entry, "path", "manifest.artifacts.embeddings"),
        "embeddings artifact",
    )
    id_map_path = _safe_artifact_path(
        artifacts,
        _require_string(id_map_entry, "path", "manifest.artifacts.id_map"),
        "id-map artifact",
    )
    embeddings_sha = _require_sha256(
        _require_string(embedding_entry, "sha256", "manifest.artifacts.embeddings"),
        "manifest.artifacts.embeddings.sha256",
    )
    id_map_sha = _require_sha256(
        _require_string(id_map_entry, "sha256", "manifest.artifacts.id_map"),
        "manifest.artifacts.id_map.sha256",
    )
    if _sha256_file(embeddings_path) != embeddings_sha:
        raise EmbeddingArtifactValidationError("embeddings artifact SHA-256 mismatch")
    if _sha256_file(id_map_path) != id_map_sha:
        raise EmbeddingArtifactValidationError("id-map artifact SHA-256 mismatch")
    if embedding_entry.get("bytes") != embeddings_path.stat().st_size:
        raise EmbeddingArtifactValidationError("embeddings artifact byte count mismatch")
    if id_map_entry.get("bytes") != id_map_path.stat().st_size:
        raise EmbeddingArtifactValidationError("id-map artifact byte count mismatch")

    try:
        vectors_raw = np.load(embeddings_path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise EmbeddingArtifactValidationError(f"cannot load embedding matrix: {exc}") from exc
    if not isinstance(vectors_raw, np.ndarray) or vectors_raw.ndim != 2:
        raise EmbeddingArtifactValidationError("embedding matrix must be two-dimensional")
    if vectors_raw.dtype != np.float32:
        raise EmbeddingArtifactValidationError(
            f"embedding matrix dtype must be float32, got {vectors_raw.dtype}"
        )
    vectors = np.asarray(vectors_raw, dtype=np.float32)
    if vectors.shape[1] != expected_dimension:
        raise EmbeddingArtifactValidationError(
            f"embedding dimension mismatch: expected {expected_dimension}, got {vectors.shape[1]}"
        )
    for start in range(0, vectors.shape[0], 1024):
        vector_batch = vectors[start : start + 1024]
        if not np.isfinite(vector_batch).all():
            raise EmbeddingArtifactValidationError("embedding matrix contains non-finite values")
        norms = np.linalg.norm(vector_batch, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-3):
            raise EmbeddingArtifactValidationError("embedding matrix is not L2-normalised")

    matrix_contract = _require_mapping(manifest.get("matrix"), "manifest.matrix")
    declared_shape = matrix_contract.get("shape")
    if declared_shape != [int(vectors.shape[0]), int(vectors.shape[1])]:
        raise EmbeddingArtifactValidationError(
            f"manifest matrix shape {declared_shape!r} does not match {list(vectors.shape)!r}"
        )
    if (
        matrix_contract.get("dtype") != "float32"
        or matrix_contract.get("l2_normalised") is not True
    ):
        raise EmbeddingArtifactValidationError(
            "manifest matrix dtype/normalisation contract is invalid"
        )

    id_rows = _load_id_rows(id_map_path)
    records, source_files = load_normalized_product_jsonl(catalog)
    products = tuple(MasterProduct.model_validate(record) for record in records)
    if len(id_rows) != len(products) or vectors.shape[0] != len(products):
        raise EmbeddingArtifactValidationError(
            "catalog, id-map, and embedding matrix row counts do not match"
        )
    if manifest.get("product_count") != len(products):
        raise EmbeddingArtifactValidationError("manifest product_count does not match catalog")
    provenance_owners: dict[str, str] = {}
    for product in products:
        for provenance in product.provenance:
            previous_owner = provenance_owners.setdefault(
                provenance.provenance_id, product.product_id
            )
            if (
                previous_owner != product.product_id
                or sum(
                    item.provenance_id == provenance.provenance_id for item in product.provenance
                )
                > 1
            ):
                raise EmbeddingArtifactValidationError(
                    f"duplicate source provenance ID in catalog: {provenance.provenance_id}"
                )

    search_documents: list[str] = []
    expected_rows: list[EmbeddingIdRow] = []
    for row_index, record in enumerate(records):
        product_id = str(record["product_id"]).strip()
        category = str(record["category"]).strip().casefold()
        search_document = build_product_embedding_text(record)
        content_payload = f"{TEXT_BUILDER_VERSION}\0{product_id}\0{search_document}".encode()
        expected_rows.append(
            EmbeddingIdRow(
                row_index=row_index,
                product_id=product_id,
                category=category,
                content_hash=hashlib.sha256(content_payload).hexdigest(),
            )
        )
        search_documents.append(search_document)
    if tuple(expected_rows) != id_rows:
        for expected, actual in zip(expected_rows, id_rows, strict=True):
            if expected != actual:
                raise EmbeddingArtifactValidationError(
                    "catalog/id-map mismatch at row "
                    f"{expected.row_index}: expected {expected}, got {actual}"
                )
        raise EmbeddingArtifactValidationError("catalog/id-map mismatch")

    dataset_payload = "\n".join(
        f"{row.product_id}:{row.content_hash}" for row in expected_rows
    ).encode("utf-8")
    dataset_content_hash = hashlib.sha256(dataset_payload).hexdigest()
    if manifest.get("content_hash") != dataset_content_hash:
        raise EmbeddingArtifactValidationError("manifest dataset content hash mismatch")

    encoder = _require_mapping(manifest.get("encoder"), "manifest.encoder")
    if encoder.get("dimension") != expected_dimension:
        raise EmbeddingArtifactValidationError("encoder dimension does not match database schema")
    if encoder.get("normalised") is not True:
        raise EmbeddingArtifactValidationError("encoder does not declare normalised vectors")
    embedding_model = _require_string(encoder, "model_name", "manifest.encoder")
    encoder_fingerprint = _require_sha256(
        _require_string(encoder, "fingerprint", "manifest.encoder"),
        "manifest.encoder.fingerprint",
    )
    data_version = _require_string(manifest, "data_version", "manifest")
    index_version = _require_string(manifest, "index_version", "manifest")
    created_at_text = _require_string(manifest, "created_at_utc", "manifest")
    try:
        created_at = datetime.fromisoformat(created_at_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EmbeddingArtifactValidationError(
            "manifest.created_at_utc must be an ISO-8601 timestamp"
        ) from exc
    if created_at.tzinfo is None:
        raise EmbeddingArtifactValidationError("manifest.created_at_utc must include a timezone")

    source = _require_mapping(manifest.get("source"), "manifest.source")
    declared_source_files = source.get("files")
    if not isinstance(declared_source_files, list) or not all(
        isinstance(item, Mapping) for item in declared_source_files
    ):
        raise EmbeddingArtifactValidationError(
            "manifest.source.files must contain content-addressed file objects"
        )
    if source.get("file_count") != len(declared_source_files) or len(source_files) != len(
        declared_source_files
    ):
        raise EmbeddingArtifactValidationError(
            "manifest source file count does not match the supplied catalog input"
        )
    loaded_by_relative = {
        str(path.relative_to(catalog)) if catalog.is_dir() else path.name: path
        for path in source_files
    }
    declared_relative: set[str] = set()
    for index, raw_entry in enumerate(declared_source_files):
        entry = _require_mapping(raw_entry, f"manifest.source.files[{index}]")
        relative_path = _require_string(entry, "relative_path", f"manifest.source.files[{index}]")
        if relative_path in declared_relative:
            raise EmbeddingArtifactValidationError(
                f"duplicate manifest source relative path: {relative_path}"
            )
        declared_relative.add(relative_path)
        loaded_path = loaded_by_relative.get(relative_path)
        if loaded_path is None:
            raise EmbeddingArtifactValidationError(
                f"manifest source file is absent from supplied catalog: {relative_path}"
            )
        declared_hash = _require_sha256(
            _require_string(entry, "sha256", f"manifest.source.files[{index}]"),
            f"manifest.source.files[{index}].sha256",
        )
        if _sha256_file(loaded_path) != declared_hash:
            raise EmbeddingArtifactValidationError(
                f"source catalog SHA-256 mismatch: {relative_path}"
            )
        if entry.get("bytes") != loaded_path.stat().st_size:
            raise EmbeddingArtifactValidationError(
                f"source catalog byte count mismatch: {relative_path}"
            )
    if declared_relative != set(loaded_by_relative):
        raise EmbeddingArtifactValidationError(
            "manifest source files do not match the supplied catalog input"
        )

    return ValidatedEmbeddingArtifact(
        artifact_dir=artifacts,
        catalog_path=catalog,
        products=products,
        search_documents=tuple(search_documents),
        id_rows=id_rows,
        vectors=vectors,
        embedding_model=embedding_model,
        data_version=data_version,
        index_version=index_version,
        encoder_fingerprint=encoder_fingerprint,
        dataset_content_hash=dataset_content_hash,
        embeddings_artifact_sha256=embeddings_sha,
        id_map_artifact_sha256=id_map_sha,
        created_at=created_at,
        manifest=manifest,
    )


class PostgresVectorCatalogRepository:
    """PostgreSQL-only writer for a validated canonical catalog and pgvector index."""

    def __init__(self, engine: Engine) -> None:
        if engine.dialect.name != "postgresql":
            raise ValueError("PostgresVectorCatalogRepository requires PostgreSQL")
        self.engine = engine

    @staticmethod
    def _product_values(
        product: MasterProduct,
        search_document: str,
        search_document_hash: str,
    ) -> dict[str, Any]:
        return {
            "product_id": product.product_id,
            "category": product.category.value,
            "brand": product.brand,
            "model": product.model,
            "manufacturer_part_number": product.manufacturer_part_number,
            "gtin": product.gtin,
            "canonical_name": product.canonical_name,
            "release_date": product.release_date,
            "status": product.status.value,
            "common_attributes": product.common_attributes.model_dump(mode="json"),
            "category_attributes": product.category_attributes.model_dump(mode="json"),
            "source_confidence": product.source_confidence,
            "search_document": search_document,
            "search_document_hash": search_document_hash,
            "created_at": product.created_at,
            "updated_at": product.updated_at,
        }

    @staticmethod
    def _provenance_values(product: MasterProduct) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for provenance in product.provenance:
            row = provenance.model_dump(mode="python")
            row["product_id"] = product.product_id
            row["source_type"] = provenance.source_type.value
            values.append(row)
        return values

    def import_artifact(
        self,
        artifact: ValidatedEmbeddingArtifact,
        *,
        batch_size: int = 250,
        reconcile_stale_provenance: bool = True,
    ) -> VectorCatalogImportResult:
        """Upsert all rows atomically and verify the exact live versioned set.

        Production release orchestration disables stale-provenance reconciliation so an
        unexpected row rolls back this transaction instead of being deleted implicitly.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        expected_provenance = {
            provenance.provenance_id: (product.product_id, provenance.source_name)
            for product in artifact.products
            for provenance in product.provenance
        }
        imported_source_names = {source_name for _, source_name in expected_provenance.values()}
        with Session(self.engine) as session, session.begin():
            vector_version = session.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            if not isinstance(vector_version, str) or not vector_version:
                raise RuntimeError("PostgreSQL vector extension is not installed")
            if _pgvector_version_tuple(vector_version) < _MINIMUM_PGVECTOR_VERSION:
                raise RuntimeError(
                    "PostgreSQL vector extension 0.8.0 or newer is required for "
                    "filtered iterative HNSW scans"
                )

            product_rows = (
                self._product_values(
                    product,
                    artifact.search_documents[index],
                    artifact.id_rows[index].content_hash,
                )
                for index, product in enumerate(artifact.products)
            )
            for batch in _batched(product_rows, batch_size):
                insert_products = pg_insert(CanonicalProductRecord).values(batch)
                session.execute(
                    insert_products.on_conflict_do_update(
                        index_elements=[CanonicalProductRecord.product_id],
                        set_={
                            name: getattr(insert_products.excluded, name)
                            for name in batch[0]
                            if name not in {"product_id", "created_at"}
                        },
                    )
                )

            provenance_rows = (
                row for product in artifact.products for row in self._provenance_values(product)
            )
            for batch in _batched(provenance_rows, batch_size):
                insert_provenance = pg_insert(SourceProvenanceRecord).values(batch)
                session.execute(
                    insert_provenance.on_conflict_do_update(
                        index_elements=[SourceProvenanceRecord.provenance_id],
                        set_={
                            name: getattr(insert_provenance.excluded, name)
                            for name in batch[0]
                            if name != "provenance_id"
                        },
                    )
                )

            expected_ids = {row.product_id for row in artifact.id_rows}
            if imported_source_names and reconcile_stale_provenance:
                stale_provenance = delete(SourceProvenanceRecord).where(
                    SourceProvenanceRecord.product_id.in_(expected_ids),
                    SourceProvenanceRecord.source_name.in_(imported_source_names),
                )
                if expected_provenance:
                    stale_provenance = stale_provenance.where(
                        SourceProvenanceRecord.provenance_id.not_in(expected_provenance)
                    )
                session.execute(stale_provenance)

            embedding_rows = (
                {
                    "product_id": id_row.product_id,
                    "embedding_model": artifact.embedding_model,
                    "content_hash": id_row.content_hash,
                    "embedding": artifact.vectors[id_row.row_index].tolist(),
                    "data_version": artifact.data_version,
                    "index_version": artifact.index_version,
                    "encoder_fingerprint": artifact.encoder_fingerprint,
                    "dataset_content_hash": artifact.dataset_content_hash,
                    "embeddings_artifact_sha256": artifact.embeddings_artifact_sha256,
                    "id_map_artifact_sha256": artifact.id_map_artifact_sha256,
                    "updated_at": artifact.created_at.astimezone(UTC),
                }
                for id_row in artifact.id_rows
            )
            for batch in _batched(embedding_rows, batch_size):
                insert_embeddings = pg_insert(ProductEmbeddingRecord).values(batch)
                session.execute(
                    insert_embeddings.on_conflict_do_nothing(
                        index_elements=[
                            ProductEmbeddingRecord.product_id,
                            ProductEmbeddingRecord.embedding_model,
                            ProductEmbeddingRecord.data_version,
                            ProductEmbeddingRecord.index_version,
                            ProductEmbeddingRecord.encoder_fingerprint,
                            ProductEmbeddingRecord.dataset_content_hash,
                        ],
                    )
                )

            live_product_rows = session.execute(
                select(
                    CanonicalProductRecord.product_id,
                    CanonicalProductRecord.category,
                    CanonicalProductRecord.search_document,
                    CanonicalProductRecord.search_document_hash,
                ).where(CanonicalProductRecord.product_id.in_(expected_ids))
            ).all()
            live_product_map = {
                product_id: (category, search_document, search_document_hash)
                for product_id, category, search_document, search_document_hash in live_product_rows
            }
            for index, product in enumerate(artifact.products):
                if live_product_map.get(product.product_id) != (
                    product.category.value,
                    artifact.search_documents[index],
                    artifact.id_rows[index].content_hash,
                ):
                    raise RuntimeError(
                        f"database product verification failed: {product.product_id}"
                    )

            live_embedding_rows = session.execute(
                select(
                    ProductEmbeddingRecord.product_id,
                    ProductEmbeddingRecord.content_hash,
                    ProductEmbeddingRecord.embeddings_artifact_sha256,
                    ProductEmbeddingRecord.id_map_artifact_sha256,
                ).where(
                    ProductEmbeddingRecord.embedding_model == artifact.embedding_model,
                    ProductEmbeddingRecord.data_version == artifact.data_version,
                    ProductEmbeddingRecord.index_version == artifact.index_version,
                    ProductEmbeddingRecord.encoder_fingerprint == artifact.encoder_fingerprint,
                    ProductEmbeddingRecord.dataset_content_hash == artifact.dataset_content_hash,
                )
            ).all()
            live_embedding_map: dict[str, tuple[str, str, str]] = {
                product_id: (content_hash, embeddings_sha, id_map_sha)
                for product_id, content_hash, embeddings_sha, id_map_sha in live_embedding_rows
            }
            expected_embedding_map = {
                row.product_id: (
                    row.content_hash,
                    artifact.embeddings_artifact_sha256,
                    artifact.id_map_artifact_sha256,
                )
                for row in artifact.id_rows
            }
            if live_embedding_map != expected_embedding_map:
                raise RuntimeError("database embedding IDs/content hashes do not match artifact")

            database_product_count = int(
                session.scalar(select(func.count()).select_from(CanonicalProductRecord)) or 0
            )
            database_embedding_count = int(
                session.scalar(select(func.count()).select_from(ProductEmbeddingRecord)) or 0
            )
            if imported_source_names:
                live_provenance_rows = session.execute(
                    select(
                        SourceProvenanceRecord.provenance_id,
                        SourceProvenanceRecord.product_id,
                        SourceProvenanceRecord.source_name,
                    ).where(
                        SourceProvenanceRecord.product_id.in_(expected_ids),
                        SourceProvenanceRecord.source_name.in_(imported_source_names),
                    )
                ).all()
                live_provenance = {
                    provenance_id: (product_id, source_name)
                    for provenance_id, product_id, source_name in live_provenance_rows
                }
                if live_provenance != expected_provenance:
                    raise RuntimeError(
                        "database provenance IDs/owners do not match imported catalog"
                    )
            provenance_count = len(expected_provenance)

        return VectorCatalogImportResult(
            product_count=len(live_product_map),
            provenance_count=provenance_count,
            embedding_count=len(live_embedding_map),
            database_product_count=database_product_count,
            database_embedding_count=database_embedding_count,
            embedding_model=artifact.embedding_model,
            data_version=artifact.data_version,
            index_version=artifact.index_version,
            dataset_content_hash=artifact.dataset_content_hash,
        )


class PgVectorSearchBackend:
    """Cosine search over one explicitly versioned pgvector product index."""

    source_name = "pgvector"

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
    ) -> None:
        if not data_version.strip() or not index_version.strip():
            raise ValueError("data_version and index_version must not be empty")
        configured_model = embedding_model or encoder.model_name
        if configured_model != encoder.model_name:
            raise ValueError(
                "embedding_model must exactly match the configured query encoder model_name"
            )
        self._session_factory = session_factory
        self.encoder = encoder
        self.embedding_model = configured_model
        self.data_version = data_version
        self.index_version = index_version
        configured_fingerprint = _require_sha256(encoder_fingerprint, "encoder_fingerprint")
        derived_fingerprint = embedding_encoder_fingerprint(encoder)
        if configured_fingerprint != derived_fingerprint:
            raise ValueError(
                "encoder_fingerprint does not match the configured query encoder contract"
            )
        self.encoder_fingerprint = configured_fingerprint
        self.dataset_content_hash = _require_sha256(dataset_content_hash, "dataset_content_hash")

    def encode_query(self, query: str) -> NDArray[np.float32] | None:
        """Encode and validate before a database connection or transaction is opened."""

        normalized_query = " ".join(query.split())
        if len(normalized_query) > _MAX_QUERY_CHARACTERS:
            raise ValueError(f"query must not exceed {_MAX_QUERY_CHARACTERS} characters")
        if not normalized_query:
            return None
        query_matrix = np.asarray(self.encoder.encode([normalized_query]), dtype=np.float32)
        if query_matrix.shape != (1, EMBEDDING_DIMENSION):
            raise ValueError(
                "query encoder returned incompatible shape: "
                f"expected (1, {EMBEDDING_DIMENSION}), got {query_matrix.shape}"
            )
        query_vector: NDArray[np.float32] = np.asarray(query_matrix[0], dtype=np.float32)
        if not np.isfinite(query_vector).all():
            raise ValueError("query encoder returned non-finite values")
        norm = float(np.linalg.norm(query_vector))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError("query encoder must return an L2-normalised vector")
        return query_vector

    def _search_vector_session(
        self,
        session: Session,
        query_vector: NDArray[np.float32],
        *,
        category: str,
        top_k: int = 50,
        candidate_ids: set[str] | frozenset[str] | None = None,
        filters: StructuredFilterSpec | None = None,
    ) -> list[SearchHit]:
        if top_k < 1 or candidate_ids is not None and not candidate_ids:
            return []
        if top_k > _MAX_DATABASE_TOP_K:
            raise ValueError(f"top_k must not exceed {_MAX_DATABASE_TOP_K}")

        category_key = normalize_postgres_category(category)
        distance = ProductEmbeddingRecord.embedding.cosine_distance(query_vector.tolist())
        similarity = (1.0 - distance).label("score")
        statement = (
            select(ProductEmbeddingRecord.product_id, similarity)
            .join(
                CanonicalProductRecord,
                (CanonicalProductRecord.product_id == ProductEmbeddingRecord.product_id)
                & (
                    CanonicalProductRecord.search_document_hash
                    == ProductEmbeddingRecord.content_hash
                ),
            )
            .where(
                ProductEmbeddingRecord.embedding_model == self.embedding_model,
                ProductEmbeddingRecord.data_version == self.data_version,
                ProductEmbeddingRecord.index_version == self.index_version,
                ProductEmbeddingRecord.encoder_fingerprint == self.encoder_fingerprint,
                ProductEmbeddingRecord.dataset_content_hash == self.dataset_content_hash,
                *postgres_structured_predicates(
                    category=category_key,
                    filters=filters,
                    candidate_ids=candidate_ids,
                ),
            )
            .order_by(distance, ProductEmbeddingRecord.product_id)
            .limit(top_k)
        )
        # pgvector applies ordinary WHERE filters after an approximate HNSW
        # scan. Iterative strict-order scanning expands that scan until the
        # category/version/stock predicates have enough eligible neighbors.
        session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
        rows = session.execute(statement).all()
        return [
            SearchHit(
                product_id=product_id,
                score=float(score),
                rank=rank,
                source=self.source_name,
            )
            for rank, (product_id, score) in enumerate(rows, start=1)
        ]

    def _search_session(
        self,
        session: Session,
        query: str,
        *,
        category: str,
        top_k: int = 50,
        candidate_ids: set[str] | frozenset[str] | None = None,
        filters: StructuredFilterSpec | None = None,
    ) -> list[SearchHit]:
        query_vector = self.encode_query(query)
        if query_vector is None:
            return []
        return self._search_vector_session(
            session,
            query_vector,
            category=category,
            top_k=top_k,
            candidate_ids=candidate_ids,
            filters=filters,
        )

    def search(
        self,
        query: str,
        *,
        category: str,
        top_k: int = 50,
        candidate_ids: set[str] | frozenset[str] | None = None,
        filters: StructuredFilterSpec | None = None,
    ) -> list[SearchHit]:
        query_vector = self.encode_query(query)
        if query_vector is None:
            return []
        with self._session_factory() as session:
            return self._search_vector_session(
                session,
                query_vector,
                category=category,
                top_k=top_k,
                candidate_ids=candidate_ids,
                filters=filters,
            )
