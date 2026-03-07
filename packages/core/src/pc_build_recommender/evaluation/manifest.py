"""Content-addressed manifests for evaluation datasets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import DataUseDeclaration

MANIFEST_SCHEMA_VERSION = "pc-build-recommender.dataset-manifest.v1"


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON deterministically and reject non-standard floating-point values."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FileDigest:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute():
            raise ValueError("manifest file paths must be non-empty and relative")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: str
    dataset_name: str
    dataset_version: str
    row_count: int
    group_count: int
    files: tuple[FileDigest, ...]
    data_use: DataUseDeclaration
    metadata: dict[str, Any]
    content_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {self.schema_version!r}")
        if not self.dataset_name or not self.dataset_version:
            raise ValueError("dataset_name and dataset_version must not be empty")
        if self.row_count < 0 or self.group_count < 0:
            raise ValueError("dataset counts must be non-negative")
        if self.data_use.total_rows != self.row_count:
            raise ValueError("data-use total_rows must equal manifest row_count")
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a SHA-256 digest")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "row_count": self.row_count,
            "group_count": self.group_count,
            "files": [file_digest.to_dict() for file_digest in self.files],
            "synthetic_data": self.data_use.to_dict(),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "content_sha256": self.content_sha256}


def _relative_digest(root: Path, path: str | Path) -> FileDigest:
    candidate = Path(path)
    absolute = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"manifest file is outside dataset root: {absolute}") from exc
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return FileDigest(
        path=relative, size_bytes=absolute.stat().st_size, sha256=sha256_file(absolute)
    )


def build_dataset_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    root: str | Path,
    files: Sequence[str | Path],
    row_count: int,
    group_count: int,
    data_use: DataUseDeclaration,
    metadata: Mapping[str, object] | None = None,
) -> DatasetManifest:
    """Hash a dataset's files and semantic metadata into an immutable manifest."""

    dataset_root = Path(root).resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    digests = tuple(
        sorted((_relative_digest(dataset_root, path) for path in files), key=lambda item: item.path)
    )
    payload: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "row_count": row_count,
        "group_count": group_count,
        "files": [file_digest.to_dict() for file_digest in digests],
        "synthetic_data": data_use.to_dict(),
        "metadata": dict(metadata or {}),
    }
    content_sha256 = json_sha256(payload)
    return DatasetManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        row_count=row_count,
        group_count=group_count,
        files=digests,
        data_use=data_use,
        metadata=dict(metadata or {}),
        content_sha256=content_sha256,
    )


def _write_json_atomic(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialised)
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    return path


def write_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    """Write a deterministic manifest without silently changing its hash."""

    if json_sha256(manifest.content_payload()) != manifest.content_sha256:
        raise ValueError("manifest content hash does not match its payload")
    return _write_json_atomic(Path(path), manifest.to_dict())


def dataset_manifest_from_dict(payload: Mapping[str, Any]) -> DatasetManifest:
    synthetic = payload["synthetic_data"]
    if not isinstance(synthetic, Mapping):
        raise TypeError("synthetic_data must be an object")
    data_use = DataUseDeclaration(
        total_rows=int(synthetic["total_rows"]),
        evaluated_rows=int(synthetic["evaluated_rows"]),
        synthetic_rows=int(synthetic["synthetic_rows"]),
        synthetic_rows_excluded=bool(synthetic["synthetic_rows_excluded"]),
        synthetic_flags_declared=bool(synthetic["synthetic_flags_declared"]),
    )
    file_payloads = payload["files"]
    if not isinstance(file_payloads, list):
        raise TypeError("files must be a list")
    file_digests = tuple(
        FileDigest(
            path=str(item["path"]), size_bytes=int(item["size_bytes"]), sha256=str(item["sha256"])
        )
        for item in file_payloads
        if isinstance(item, Mapping)
    )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an object")
    return DatasetManifest(
        schema_version=str(payload["schema_version"]),
        dataset_name=str(payload["dataset_name"]),
        dataset_version=str(payload["dataset_version"]),
        row_count=int(payload["row_count"]),
        group_count=int(payload["group_count"]),
        files=file_digests,
        data_use=data_use,
        metadata=metadata,
        content_sha256=str(payload["content_sha256"]),
    )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("manifest root must be an object")
    return dataset_manifest_from_dict(payload)


def verify_dataset_manifest(manifest: DatasetManifest, *, root: str | Path) -> bool:
    """Verify both the manifest payload and every referenced file."""

    if json_sha256(manifest.content_payload()) != manifest.content_sha256:
        return False
    dataset_root = Path(root).resolve()
    for expected in manifest.files:
        candidate = (dataset_root / expected.path).resolve()
        try:
            candidate.relative_to(dataset_root)
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        if candidate.stat().st_size != expected.size_bytes:
            return False
        if sha256_file(candidate) != expected.sha256:
            return False
    return True
