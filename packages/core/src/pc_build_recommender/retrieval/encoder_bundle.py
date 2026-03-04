"""Fail-closed validation for locally mounted semantic-encoder bundles.

Production serving must never resolve a model name over the network.  This
module validates a bounded, immutable directory tree and derives a stable
content digest before SentenceTransformers is allowed to open it.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

ENCODER_BUNDLE_SCHEMA_VERSION = "pc-build-recommender.semantic-encoder-bundle.v1"
DEFAULT_MAX_ENCODER_BUNDLE_FILES = 4096
DEFAULT_MAX_ENCODER_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_ENCODER_BUNDLE_FILE_BYTES = 1024 * 1024 * 1024
_HASH_CHUNK_SIZE = 1024 * 1024


class EncoderBundleValidationError(ValueError):
    """Raised when a semantic-encoder directory is unsafe or does not match release metadata."""


@dataclass(frozen=True, slots=True)
class EncoderBundleFile:
    """One regular file included in a semantic-encoder bundle identity."""

    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ValidatedEncoderBundle:
    """Content identity returned only after every bundle file is checked."""

    path: Path
    sha256: str
    file_count: int
    size_bytes: int
    files: tuple[EncoderBundleFile, ...]


def _is_linklike(path: Path, metadata: os.stat_result | None = None) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if is_junction is not None and is_junction(path):
            return True
    except OSError as exc:
        raise EncoderBundleValidationError(
            f"cannot inspect semantic encoder bundle path: {path}"
        ) from exc
    details = metadata or path.lstat()
    if stat.S_ISLNK(details.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(details, "st_file_attributes", 0)
    return bool(reparse_flag and file_attributes & reparse_flag)


def _path_components(path: Path) -> tuple[Path, ...]:
    absolute = Path(os.path.abspath(path))
    components: list[Path] = []
    current = absolute
    while True:
        components.append(current)
        if current.parent == current:
            break
        current = current.parent
    components.reverse()
    return tuple(components)


def _assert_no_linklike_components(path: Path) -> os.stat_result:
    target_metadata: os.stat_result | None = None
    components = _path_components(path)
    for index, component in enumerate(components):
        try:
            metadata = os.lstat(component)
        except OSError as exc:
            raise EncoderBundleValidationError(
                f"semantic encoder bundle path is unavailable: {component}"
            ) from exc
        if _is_linklike(component, metadata):
            raise EncoderBundleValidationError(
                f"semantic encoder bundle paths must not contain symlinks or junctions: {component}"
            )
        if index < len(components) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise EncoderBundleValidationError(
                f"semantic encoder bundle parent is not a directory: {component}"
            )
        target_metadata = metadata
    assert target_metadata is not None
    return target_metadata


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _sha256_stable_file(path: Path, before: os.stat_result) -> str:
    path_metadata = _assert_no_linklike_components(path)
    expected_identity = _stat_identity(before)
    if _stat_identity(path_metadata) != expected_identity or not stat.S_ISREG(before.st_mode):
        raise EncoderBundleValidationError(
            f"semantic encoder bundle file changed before it was opened: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EncoderBundleValidationError(
            f"cannot open semantic encoder bundle file: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != expected_identity:
            raise EncoderBundleValidationError(
                f"semantic encoder bundle file changed while it was opened: {path}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _HASH_CHUNK_SIZE):
            digest.update(chunk)
        after_handle = os.fstat(descriptor)
    except OSError as exc:
        raise EncoderBundleValidationError(
            f"cannot read semantic encoder bundle file: {path}"
        ) from exc
    finally:
        os.close(descriptor)
    after_path = _assert_no_linklike_components(path)
    if (
        _stat_identity(after_handle) != expected_identity
        or _stat_identity(after_path) != expected_identity
    ):
        raise EncoderBundleValidationError(
            f"semantic encoder bundle changed while it was being validated: {path}"
        )
    return digest.hexdigest()


def _validate_digest(value: str, *, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise EncoderBundleValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_bundle_hash(files: tuple[EncoderBundleFile, ...]) -> str:
    payload = {
        "schema_version": ENCODER_BUNDLE_SCHEMA_VERSION,
        "files": [item.to_dict() for item in files],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_files(
    root: Path,
    *,
    max_files: int,
) -> list[tuple[str, Path, os.stat_result]]:
    pending = [root]
    found: list[tuple[str, Path, os.stat_result]] = []
    seen_casefolded_paths: set[str] = set()
    entry_count = 0
    max_entries = max_files * 2
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > max_entries:
                    raise EncoderBundleValidationError(
                        f"semantic encoder bundle exceeds the {max_entries}-entry limit"
                    )
                candidate = Path(entry.path)
                entry_metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or _is_linklike(candidate, entry_metadata):
                    raise EncoderBundleValidationError(
                        f"semantic encoder bundle must not contain links: {candidate}"
                    )
                metadata = _assert_no_linklike_components(candidate)
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(candidate)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise EncoderBundleValidationError(
                        f"semantic encoder bundle contains a non-regular entry: {candidate}"
                    )
                relative_path = candidate.relative_to(root).as_posix()
                folded_path = relative_path.casefold()
                if folded_path in seen_casefolded_paths:
                    raise EncoderBundleValidationError(
                        "semantic encoder bundle contains case-insensitive duplicate paths"
                    )
                seen_casefolded_paths.add(folded_path)
                found.append((relative_path, candidate, metadata))
                if len(found) > max_files:
                    raise EncoderBundleValidationError(
                        f"semantic encoder bundle exceeds the {max_files}-file limit"
                    )
    found.sort(key=lambda item: item[0])
    return found


def inspect_encoder_bundle(
    path: str | Path,
    *,
    max_files: int = DEFAULT_MAX_ENCODER_BUNDLE_FILES,
    max_total_bytes: int = DEFAULT_MAX_ENCODER_BUNDLE_BYTES,
    max_file_bytes: int = DEFAULT_MAX_ENCODER_BUNDLE_FILE_BYTES,
) -> ValidatedEncoderBundle:
    """Hash a bounded local bundle without following links or buffering model files."""

    if max_files < 1 or max_total_bytes < 1 or max_file_bytes < 1:
        raise ValueError("semantic encoder bundle limits must be positive")
    root = Path(os.path.abspath(path))
    root_metadata = _assert_no_linklike_components(root)
    if _is_linklike(root, root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
        raise EncoderBundleValidationError(
            f"semantic encoder bundle must be a direct regular directory: {root}"
        )
    candidates = _regular_files(root, max_files=max_files)
    if not candidates:
        raise EncoderBundleValidationError("semantic encoder bundle must not be empty")

    total_size = 0
    file_digests: list[EncoderBundleFile] = []
    for relative_path, candidate, before in candidates:
        if before.st_size > max_file_bytes:
            raise EncoderBundleValidationError(
                f"semantic encoder bundle file exceeds the size limit: {relative_path}"
            )
        total_size += before.st_size
        if total_size > max_total_bytes:
            raise EncoderBundleValidationError(
                f"semantic encoder bundle exceeds the {max_total_bytes}-byte limit"
            )
        file_digests.append(
            EncoderBundleFile(
                path=relative_path,
                size_bytes=before.st_size,
                sha256=_sha256_stable_file(candidate, before),
            )
        )
    if _stat_identity(_assert_no_linklike_components(root)) != _stat_identity(root_metadata):
        raise EncoderBundleValidationError(
            "semantic encoder bundle root changed while it was being validated"
        )
    files = tuple(file_digests)
    return ValidatedEncoderBundle(
        path=root,
        sha256=_canonical_bundle_hash(files),
        file_count=len(files),
        size_bytes=total_size,
        files=files,
    )


def validate_encoder_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_file_count: int | None = None,
    expected_size_bytes: int | None = None,
    require_content_addressed_path: bool = True,
) -> ValidatedEncoderBundle:
    """Validate a local bundle against its release identity and directory name."""

    expected_digest = _validate_digest(expected_sha256, field="expected_sha256")
    if expected_file_count is not None and expected_file_count < 1:
        raise EncoderBundleValidationError("expected_file_count must be positive")
    if expected_size_bytes is not None and expected_size_bytes < 1:
        raise EncoderBundleValidationError("expected_size_bytes must be positive")
    bundle = inspect_encoder_bundle(path)
    if bundle.sha256 != expected_digest:
        raise EncoderBundleValidationError(
            "semantic encoder bundle content hash does not match the release manifest"
        )
    if require_content_addressed_path and bundle.path.name != expected_digest:
        raise EncoderBundleValidationError(
            "semantic encoder bundle directory name must equal its content SHA-256"
        )
    if expected_file_count is not None and bundle.file_count != expected_file_count:
        raise EncoderBundleValidationError(
            "semantic encoder bundle file count does not match the release manifest"
        )
    if expected_size_bytes is not None and bundle.size_bytes != expected_size_bytes:
        raise EncoderBundleValidationError(
            "semantic encoder bundle size does not match the release manifest"
        )
    return bundle
