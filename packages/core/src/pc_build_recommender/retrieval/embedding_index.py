"""Batch-build reproducible product embedding artifacts from normalized JSONL.

Run with::

    python -m pc_build_recommender.retrieval.embedding_index \
      --input data/processed/products.jsonl \
      --output-dir artifacts/retrieval/product-embeddings \
      --data-version 2026-07-22 --device auto

The output contract is ``embeddings.npy`` (float32), ``ids.jsonl`` (row-to-ID
mapping and per-product content hashes), and ``manifest.json`` (versions,
encoder/device/batch metadata, content hash, and artifact checksums).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .vector import (
    EmbeddingEncoder,
    SentenceTransformerEmbeddingEncoder,
    StableHashEmbeddingEncoder,
)

MANIFEST_SCHEMA_VERSION = "product-embedding-index-manifest-v1"
TEXT_BUILDER_VERSION = "product-embedding-text-v1"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILENAME = "embeddings.npy"
ID_MAP_FILENAME = "ids.jsonl"
MANIFEST_FILENAME = "manifest.json"


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding backend cannot encode an otherwise valid batch."""


@dataclass(frozen=True, slots=True)
class EmbeddedProductText:
    product_id: str
    category: str
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class EmbeddingIndexResult:
    output_dir: Path
    embeddings_path: Path
    id_map_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    encoded_count: int
    reused_count: int
    skipped_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "embeddings_path": str(self.embeddings_path),
            "id_map_path": str(self.id_map_path),
            "manifest_path": str(self.manifest_path),
            "encoded_count": self.encoded_count,
            "reused_count": self.reused_count,
            "skipped_unchanged": self.skipped_unchanged,
            "content_hash": self.manifest["content_hash"],
            "encoder": self.manifest["encoder"],
            "matrix": self.manifest["matrix"],
        }


@dataclass(frozen=True, slots=True)
class _ReusableIndex:
    manifest: Mapping[str, Any]
    vectors_by_id: Mapping[str, tuple[str, NDArray[np.float32]]]
    dimension: int


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _product_data(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return canonical product fields from direct or normalized-record envelopes."""

    data = record.get("data")
    schema_version = str(record.get("schema_version", ""))
    is_normalized_envelope = (
        record.get("record_type") is not None
        or schema_version.startswith("pc-build-recommender.normalised-record.")
    )
    if is_normalized_envelope and isinstance(data, Mapping):
        return data
    return record


def _flatten_values(value: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        result: list[tuple[str, str]] = []
        for key in sorted(value, key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_flatten_values(value[key], prefix=name))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = sorted({_clean_scalar(item) for item in value if _clean_scalar(item)})
        return [(prefix, ", ".join(items))] if items else []
    cleaned = _clean_scalar(value)
    return [(prefix, cleaned)] if cleaned else []


def _tag_text(record: Mapping[str, Any], *field_names: str) -> str | None:
    values: list[str] = []
    for field_name in field_names:
        value = record.get(field_name)
        if value is None:
            continue
        if isinstance(value, Mapping):
            values.extend(f"{key} {item}" for key, item in _flatten_values(value))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values.extend(_clean_scalar(item) for item in value)
        else:
            values.append(_clean_scalar(value))
    cleaned = sorted({value for value in values if value})
    return ", ".join(cleaned) if cleaned else None


def build_product_embedding_text(record: Mapping[str, Any]) -> str:
    """Build stable, labelled text from a normalized canonical-product row."""

    record = _product_data(record)
    sections: list[str] = []
    scalar_fields = (
        ("category", "category"),
        ("brand", "brand"),
        ("model", "model"),
        ("canonical_name", "name"),
        ("manufacturer_part_number", "manufacturer part number"),
    )
    for field_name, label in scalar_fields:
        if value := _clean_scalar(record.get(field_name, "")):
            sections.append(f"{label}: {value}")

    attribute_values: dict[str, Any] = {}
    for field_name in ("common_attributes", "category_attributes", "attributes"):
        attributes = record.get(field_name)
        if isinstance(attributes, Mapping):
            attribute_values.update(attributes)
    specifications = [
        f"{name.replace('_', ' ')} {value}"
        for name, value in _flatten_values(attribute_values)
    ]
    if specifications:
        sections.append("specifications: " + "; ".join(specifications))

    tag_groups = (
        (("supported_workloads", "workload_tags", "workloads"), "workloads"),
        (("benchmark_tags", "benchmark_names"), "benchmarks"),
        (("compatibility_tags",), "compatibility"),
        (("review_aspects", "review_tags"), "review aspects"),
    )
    for field_names, label in tag_groups:
        if tag_value := _tag_text(record, *field_names):
            sections.append(f"{label}: {tag_value}")

    if search_text := _clean_scalar(record.get("search_text", "")):
        sections.append(f"search text: {search_text}")
    if not sections:
        raise ValueError("product row contains no embeddable fields")
    return ". ".join(sections) + "."


def _parse_timestamp(record: Mapping[str, Any]) -> float:
    for field_name in ("updated_at", "last_seen_at", "observed_at", "release_date"):
        value = record.get(field_name)
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.timestamp()
        except ValueError:
            continue
    return float("-inf")


def _input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not input_path.is_dir():
        raise ValueError(f"input is neither a file nor directory: {input_path}")
    files = sorted(path for path in input_path.rglob("*.jsonl") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no JSONL files found below {input_path}")
    return files


def load_normalized_product_jsonl(
    input_path: str | Path,
    *,
    limit: int | None = None,
    recent_first: bool = False,
) -> tuple[list[Mapping[str, Any]], tuple[Path, ...]]:
    """Load validated JSON objects from a file or directory of JSONL files."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    source_path = Path(input_path)
    files = _input_files(source_path)
    records: list[Mapping[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"expected a JSON object at {path}:{line_number}")
                product_data = _product_data(payload)
                product_id = product_data.get("product_id", product_data.get("id"))
                if not _clean_scalar(product_id):
                    raise ValueError(f"missing product_id at {path}:{line_number}")
                if not _clean_scalar(product_data.get("category")):
                    raise ValueError(f"missing category at {path}:{line_number}")
                records.append(dict(product_data))

    if not records:
        raise ValueError("normalized product input contains no records")
    ids = [_clean_scalar(record.get("product_id", record.get("id"))) for record in records]
    duplicates = sorted(product_id for product_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate product IDs in normalized input: {duplicates[:5]}")

    if recent_first:
        records.sort(
            key=lambda record: (
                -_parse_timestamp(record),
                _clean_scalar(record.get("product_id", record.get("id"))),
            )
        )
    if limit is not None:
        records = records[:limit]
    records.sort(key=lambda record: _clean_scalar(record.get("product_id", record.get("id"))))
    return records, tuple(files)


def _prepare_products(records: Sequence[Mapping[str, Any]]) -> tuple[EmbeddedProductText, ...]:
    products: list[EmbeddedProductText] = []
    for record in records:
        product_id = _clean_scalar(record.get("product_id", record.get("id")))
        category = _clean_scalar(record["category"]).casefold()
        text = build_product_embedding_text(record)
        content_payload = f"{TEXT_BUILDER_VERSION}\0{product_id}\0{text}".encode()
        products.append(
            EmbeddedProductText(
                product_id=product_id,
                category=category,
                text=text,
                content_hash=hashlib.sha256(content_payload).hexdigest(),
            )
        )
    return tuple(products)


def _dataset_content_hash(products: Sequence[EmbeddedProductText]) -> str:
    payload = "\n".join(
        f"{product.product_id}:{product.content_hash}" for product in products
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_encoder_fingerprint(encoder: EmbeddingEncoder) -> str:
    """Derive the immutable serving fingerprint from the encoder contract."""

    if isinstance(encoder, SentenceTransformerEmbeddingEncoder):
        kind = "sentence_transformer"
        declared_dimension: int | None = None
    elif isinstance(encoder, StableHashEmbeddingEncoder):
        kind = "deterministic_lexical_hash"
        declared_dimension = encoder.dimension
    else:
        kind = f"custom:{type(encoder).__module__}.{type(encoder).__qualname__}"
        declared_dimension = encoder.dimension
    fingerprint_payload = {
        "kind": kind,
        "model_name": encoder.model_name,
        "declared_dimension": declared_dimension,
        "text_builder_version": TEXT_BUILDER_VERSION,
    }
    model_revision = getattr(encoder, "revision", None)
    if model_revision is not None:
        fingerprint_payload["model_revision"] = str(model_revision)
    return hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _encoder_details(
    encoder: EmbeddingEncoder,
    *,
    requested_device: str,
    batch_size: int,
) -> dict[str, Any]:
    fingerprint = embedding_encoder_fingerprint(encoder)
    if isinstance(encoder, SentenceTransformerEmbeddingEncoder):
        kind = "sentence_transformer"
    elif isinstance(encoder, StableHashEmbeddingEncoder):
        kind = "deterministic_lexical_hash"
    else:
        kind = f"custom:{type(encoder).__module__}.{type(encoder).__qualname__}"
    resolved_device = str(getattr(encoder, "resolved_device", "cpu"))
    return {
        "kind": kind,
        "model_name": encoder.model_name,
        "fingerprint": fingerprint,
        "requested_device": str(getattr(encoder, "requested_device", requested_device)),
        "resolved_device": resolved_device,
        "batch_size": int(getattr(encoder, "batch_size", batch_size)),
        "model_revision": getattr(encoder, "revision", None),
    }


def _valid_artifact(path: Path, metadata: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and metadata.get("sha256") == _sha256_file(path)
        and metadata.get("bytes") == path.stat().st_size
    )


def _load_reusable_index(
    output_dir: Path,
    *,
    encoder_fingerprint: str,
) -> _ReusableIndex | None:
    embeddings_path = output_dir / EMBEDDINGS_FILENAME
    id_map_path = output_dir / ID_MAP_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    if not (embeddings_path.is_file() and id_map_path.is_file() and manifest_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            return None
        if manifest.get("encoder", {}).get("fingerprint") != encoder_fingerprint:
            return None
        artifacts = manifest["artifacts"]
        if not _valid_artifact(embeddings_path, artifacts["embeddings"]):
            return None
        if not _valid_artifact(id_map_path, artifacts["id_map"]):
            return None
        matrix = np.load(embeddings_path, allow_pickle=False)
        if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
            return None
        matrix_metadata = manifest.get("matrix", {})
        if matrix_metadata.get("dtype") != "float32":
            return None
        if matrix_metadata.get("shape") != list(matrix.shape):
            return None
        rows = [
            json.loads(line)
            for line in id_map_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != matrix.shape[0]:
            return None
        vectors: dict[str, tuple[str, NDArray[np.float32]]] = {}
        seen_row_indices: set[int] = set()
        for row in rows:
            row_index = int(row["row_index"])
            if not 0 <= row_index < matrix.shape[0] or row_index in seen_row_indices:
                return None
            seen_row_indices.add(row_index)
            product_id = str(row["product_id"])
            if product_id in vectors:
                return None
            vectors[product_id] = (str(row["content_hash"]), matrix[row_index].copy())
        if seen_row_indices != set(range(matrix.shape[0])):
            return None
        ordered_rows = sorted(rows, key=lambda row: int(row["row_index"]))
        stored_content_payload = "\n".join(
            f"{row['product_id']}:{row['content_hash']}" for row in ordered_rows
        ).encode()
        stored_content_hash = hashlib.sha256(stored_content_payload).hexdigest()
        if manifest.get("content_hash") != stored_content_hash:
            return None
        if manifest.get("product_count") != matrix.shape[0]:
            return None
        if manifest.get("encoder", {}).get("dimension") != matrix.shape[1]:
            return None
        return _ReusableIndex(
            manifest=manifest,
            vectors_by_id=vectors,
            dimension=int(matrix.shape[1]),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _normalise_matrix(values: Any, expected_rows: int) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] < 1:
        raise EmbeddingBackendError(
            f"encoder returned shape {matrix.shape}; expected ({expected_rows}, dimensions)"
        )
    if not np.isfinite(matrix).all():
        raise EmbeddingBackendError("encoder returned non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise EmbeddingBackendError("encoder returned a zero or invalid vector")
    return np.asarray(matrix / norms, dtype=np.float32)


def _encode(
    encoder: EmbeddingEncoder,
    texts: Sequence[str],
) -> NDArray[np.float32]:
    try:
        return _normalise_matrix(encoder.encode(texts), len(texts))
    except EmbeddingBackendError:
        raise
    except Exception as exc:
        raise EmbeddingBackendError(
            f"{type(encoder).__name__} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_matrix(path: Path, matrix: NDArray[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_artifacts(
    output_dir: Path,
    *,
    products: Sequence[EmbeddedProductText],
    matrix: NDArray[np.float32],
    manifest_base: Mapping[str, Any],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    embeddings_path = output_dir / EMBEDDINGS_FILENAME
    id_map_path = output_dir / ID_MAP_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    id_map_text = "".join(
        json.dumps(
            {
                "row_index": row_index,
                "product_id": product.product_id,
                "category": product.category,
                "content_hash": product.content_hash,
            },
            sort_keys=True,
        )
        + "\n"
        for row_index, product in enumerate(products)
    )
    _atomic_write_matrix(embeddings_path, matrix)
    _atomic_write_text(id_map_path, id_map_text)

    manifest = dict(manifest_base)
    manifest["artifacts"] = {
        "embeddings": {
            "path": EMBEDDINGS_FILENAME,
            "sha256": _sha256_file(embeddings_path),
            "bytes": embeddings_path.stat().st_size,
        },
        "id_map": {
            "path": ID_MAP_FILENAME,
            "sha256": _sha256_file(id_map_path),
            "bytes": id_map_path.stat().st_size,
        },
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return embeddings_path, id_map_path, manifest_path, manifest


def _build_with_encoder(
    products: Sequence[EmbeddedProductText],
    source_files: Sequence[Path],
    *,
    source_path: Path,
    output_dir: Path,
    encoder: EmbeddingEncoder,
    requested_device: str,
    batch_size: int,
    data_version: str,
    index_version: str,
    limit: int | None,
    recent_first: bool,
    fallback_reason: str | None,
) -> EmbeddingIndexResult:
    details = _encoder_details(
        encoder,
        requested_device=requested_device,
        batch_size=batch_size,
    )
    content_hash = _dataset_content_hash(products)
    reusable = _load_reusable_index(
        output_dir,
        encoder_fingerprint=str(details["fingerprint"]),
    )
    embeddings_path = output_dir / EMBEDDINGS_FILENAME
    id_map_path = output_dir / ID_MAP_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    if reusable is not None:
        previous = reusable.manifest
        unchanged_contract = (
            previous.get("content_hash") == content_hash
            and previous.get("data_version") == data_version
            and previous.get("index_version") == index_version
            and previous.get("text_builder_version") == TEXT_BUILDER_VERSION
            and previous.get("product_count") == len(products)
        )
        if unchanged_contract:
            return EmbeddingIndexResult(
                output_dir=output_dir,
                embeddings_path=embeddings_path,
                id_map_path=id_map_path,
                manifest_path=manifest_path,
                manifest=previous,
                encoded_count=0,
                reused_count=len(products),
                skipped_unchanged=True,
            )

    reusable_rows: dict[int, NDArray[np.float32]] = {}
    changed_indices: list[int] = []
    if reusable is not None:
        for index, product in enumerate(products):
            previous_row = reusable.vectors_by_id.get(product.product_id)
            if previous_row is not None and previous_row[0] == product.content_hash:
                reusable_rows[index] = previous_row[1]
            else:
                changed_indices.append(index)
    else:
        changed_indices = list(range(len(products)))

    encoded = (
        _encode(encoder, [products[index].text for index in changed_indices])
        if changed_indices
        else None
    )
    if encoded is not None:
        dimension = int(encoded.shape[1])
        if reusable is not None and reusable_rows and reusable.dimension != dimension:
            # An encoder serving the same model name changed its dimensionality;
            # the old rows are not safe to mix with the new model output.
            changed_indices = list(range(len(products)))
            reusable_rows.clear()
            encoded = _encode(encoder, [product.text for product in products])
            dimension = int(encoded.shape[1])
    elif reusable is not None:
        dimension = reusable.dimension
    else:
        raise RuntimeError("embedding build produced no rows")

    matrix = np.empty((len(products), dimension), dtype=np.float32)
    for index, vector in reusable_rows.items():
        matrix[index] = vector
    if encoded is not None:
        for encoded_index, product_index in enumerate(changed_indices):
            matrix[product_index] = encoded[encoded_index]
    if not np.isfinite(matrix).all():
        raise RuntimeError("assembled embedding matrix contains non-finite values")

    details["dimension"] = dimension
    details["normalised"] = True
    if fallback_reason is not None:
        details["fallback_reason"] = fallback_reason
    manifest_base: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "index_version": index_version,
        "data_version": data_version,
        "text_builder_version": TEXT_BUILDER_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "content_hash": content_hash,
        "product_count": len(products),
        "source": {
            "input_path": str(source_path.resolve()),
            "files": [
                {
                    "path": str(path.resolve()),
                    "relative_path": (
                        str(path.relative_to(source_path))
                        if source_path.is_dir()
                        else path.name
                    ),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in source_files
            ],
            "file_count": len(source_files),
            "limit": limit,
            "recent_first": recent_first,
        },
        "encoder": details,
        "matrix": {
            "dtype": "float32",
            "shape": [len(products), dimension],
            "l2_normalised": True,
        },
        "build": {
            "encoded_count": len(changed_indices),
            "reused_count": len(reusable_rows),
            "incremental": reusable is not None,
            "reuse_source": (
                {
                    "created_at_utc": reusable.manifest.get("created_at_utc"),
                    "resolved_device": reusable.manifest.get("encoder", {}).get(
                        "resolved_device"
                    ),
                    "batch_size": reusable.manifest.get("encoder", {}).get("batch_size"),
                }
                if reusable is not None and reusable_rows
                else None
            ),
        },
    }
    embeddings_path, id_map_path, manifest_path, manifest = _write_artifacts(
        output_dir,
        products=products,
        matrix=matrix,
        manifest_base=manifest_base,
    )
    return EmbeddingIndexResult(
        output_dir=output_dir,
        embeddings_path=embeddings_path,
        id_map_path=id_map_path,
        manifest_path=manifest_path,
        manifest=manifest,
        encoded_count=len(changed_indices),
        reused_count=len(reusable_rows),
        skipped_unchanged=False,
    )


def build_embedding_index(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    data_version: str,
    index_version: str = "product-embeddings-v1",
    encoder: EmbeddingEncoder | None = None,
    encoder_kind: str = "sentence-transformer",
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    model_revision: str | None = None,
    device: str = "auto",
    batch_size: int = 64,
    hash_dimension: int = 512,
    fallback_to_hash: bool = False,
    limit: int | None = None,
    recent_first: bool = False,
) -> EmbeddingIndexResult:
    """Build or incrementally refresh a disk-backed embedding matrix."""

    if not data_version.strip() or not index_version.strip():
        raise ValueError("data_version and index_version must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if encoder_kind not in {"sentence-transformer", "hash"}:
        raise ValueError("encoder_kind must be sentence-transformer or hash")
    records, source_files = load_normalized_product_jsonl(
        input_path,
        limit=limit,
        recent_first=recent_first,
    )
    products = _prepare_products(records)
    source = Path(input_path)
    target = Path(output_dir)

    selected_encoder = encoder
    try:
        if selected_encoder is None:
            if encoder_kind == "hash":
                selected_encoder = StableHashEmbeddingEncoder(hash_dimension)
            else:
                selected_encoder = SentenceTransformerEmbeddingEncoder(
                    model_name,
                    revision=model_revision,
                    device=device,
                    batch_size=batch_size,
                )
        return _build_with_encoder(
            products,
            source_files,
            source_path=source,
            output_dir=target,
            encoder=selected_encoder,
            requested_device=device,
            batch_size=batch_size,
            data_version=data_version,
            index_version=index_version,
            limit=limit,
            recent_first=recent_first,
            fallback_reason=None,
        )
    except (EmbeddingBackendError, ImportError, RuntimeError) as exc:
        if not fallback_to_hash or isinstance(selected_encoder, StableHashEmbeddingEncoder):
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        fallback = StableHashEmbeddingEncoder(hash_dimension)
        return _build_with_encoder(
            products,
            source_files,
            source_path=source,
            output_dir=target,
            encoder=fallback,
            requested_device=device,
            batch_size=batch_size,
            data_version=data_version,
            index_version=index_version,
            limit=limit,
            recent_first=recent_first,
            fallback_reason=fallback_reason,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Normalized product JSONL file/dir",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--index-version", default="product-embeddings-v1")
    parser.add_argument(
        "--encoder",
        dest="encoder_kind",
        choices=("sentence-transformer", "hash"),
        default="sentence-transformer",
    )
    parser.add_argument("--model", dest="model_name", default=DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hash-dimension", type=int, default=512)
    parser.add_argument("--fallback-to-hash", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recent-first", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_embedding_index(
        args.input,
        args.output_dir,
        data_version=args.data_version,
        index_version=args.index_version,
        encoder_kind=args.encoder_kind,
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        hash_dimension=args.hash_dimension,
        fallback_to_hash=args.fallback_to_hash,
        limit=args.limit,
        recent_first=args.recent_first,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
