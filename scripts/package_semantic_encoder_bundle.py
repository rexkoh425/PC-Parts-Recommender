"""Publish a verified local Sentence-Transformers bundle for offline serving.

The embedding index records a logical Hugging Face model name and immutable revision, but a
production API must never resolve that name over the network. This command copies an already
verified local model snapshot into a content-addressed release directory, preserves its exact
source identity, and refuses to overwrite a bundle with different provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pc_build_recommender.evaluation.manifest import canonical_json_bytes, sha256_file
from pc_build_recommender.retrieval import (
    SentenceTransformerEmbeddingEncoder,
    embedding_encoder_fingerprint,
    inspect_encoder_bundle,
    validate_encoder_bundle,
)

BUNDLE_PROVENANCE_FILENAME = "bundle-provenance.json"
BUNDLE_PROVENANCE_SCHEMA_VERSION = "pc-build-recommender.semantic-encoder-bundle-provenance.v1"
EMBEDDING_MANIFEST_SCHEMA_VERSION = "product-embedding-index-manifest-v1"


@dataclass(frozen=True, slots=True)
class BundlePublication:
    """Immutable local encoder bundle publication result."""

    path: Path
    sha256: str
    file_count: int
    size_bytes: int
    reused_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "file_count": self.file_count,
            "size_bytes": self.size_bytes,
            "reused_existing": self.reused_existing,
        }


class EncoderBundlePublicationError(ValueError):
    """Raised when an encoder bundle cannot be published safely."""


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EncoderBundlePublicationError(f"{field} must be an object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EncoderBundlePublicationError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    result = _string(value, field=field)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise EncoderBundlePublicationError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _load_embedding_expectation(
    embedding_manifest: Path,
    *,
    model_name: str,
    model_revision: str,
) -> None:
    """Bind publication to the exact encoder contract used by an embedding index."""

    if not embedding_manifest.is_file():
        raise FileNotFoundError(embedding_manifest)
    try:
        payload = json.loads(embedding_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EncoderBundlePublicationError(
            f"embedding manifest is not valid JSON: {embedding_manifest}"
        ) from error
    manifest = _object(payload, field="embedding manifest")
    if manifest.get("schema_version") != EMBEDDING_MANIFEST_SCHEMA_VERSION:
        raise EncoderBundlePublicationError("unsupported embedding manifest schema")
    encoder = _object(manifest.get("encoder"), field="embedding manifest.encoder")
    if encoder.get("kind") != "sentence_transformer":
        raise EncoderBundlePublicationError(
            "embedding manifest must declare a sentence_transformer encoder"
        )
    declared_model_name = _string(
        encoder.get("model_name"), field="embedding manifest.encoder.model_name"
    )
    if declared_model_name != model_name:
        raise EncoderBundlePublicationError("embedding manifest model_name does not match bundle")
    if (
        _string(encoder.get("model_revision"), field="embedding manifest.encoder.model_revision")
        != model_revision
    ):
        raise EncoderBundlePublicationError(
            "embedding manifest model_revision does not match bundle"
        )
    dimension = encoder.get("dimension")
    if not isinstance(dimension, int) or dimension < 1:
        raise EncoderBundlePublicationError("embedding manifest encoder.dimension must be positive")
    if encoder.get("normalised") is not True:
        raise EncoderBundlePublicationError("embedding manifest must declare normalized embeddings")
    expected_fingerprint = _sha256(
        encoder.get("fingerprint"), field="embedding manifest.encoder.fingerprint"
    )
    derived = embedding_encoder_fingerprint(
        SentenceTransformerEmbeddingEncoder(model_name, revision=model_revision, device="cpu")
    )
    if derived != expected_fingerprint:
        raise EncoderBundlePublicationError(
            "embedding manifest encoder fingerprint does not match the local model contract"
        )


def _provenance_payload(
    *,
    source_sha256: str,
    source_file_count: int,
    source_size_bytes: int,
    source_files: Sequence[Mapping[str, object]],
    model_name: str,
    model_revision: str,
    licence: str,
    embedding_manifest: Path | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": BUNDLE_PROVENANCE_SCHEMA_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "licence": licence,
        "source_bundle": {
            "sha256": source_sha256,
            "file_count": source_file_count,
            "size_bytes": source_size_bytes,
            "files": list(source_files),
        },
    }
    if embedding_manifest is not None:
        payload["embedding_manifest"] = {
            "sha256": sha256_file(embedding_manifest),
            "filename": embedding_manifest.name,
        }
    return payload


def _read_provenance(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EncoderBundlePublicationError(f"bundle provenance is invalid: {path}") from error
    if not isinstance(value, dict):
        raise EncoderBundlePublicationError("bundle provenance must be a JSON object")
    return value


def _copy_verified_source(
    *,
    source_root: Path,
    destination_root: Path,
    source_files: Sequence[Mapping[str, object]],
) -> None:
    for entry in source_files:
        relative_path = _string(entry.get("path"), field="source file path")
        expected_sha256 = _sha256(entry.get("sha256"), field=f"source file {relative_path} SHA-256")
        source = source_root / relative_path
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != expected_sha256:
            raise EncoderBundlePublicationError(
                f"source file changed while it was copied: {relative_path}"
            )


def package_encoder_bundle(
    *,
    source: Path,
    output_root: Path,
    model_name: str,
    model_revision: str,
    licence: str,
    expected_source_sha256: str,
    embedding_manifest: Path | None = None,
) -> BundlePublication:
    """Copy one exact local encoder tree to an immutable content-addressed destination."""

    if not model_name.strip() or not model_revision.strip() or not licence.strip():
        raise EncoderBundlePublicationError("model name, revision, and licence must be non-empty")
    source_identity = inspect_encoder_bundle(source)
    if source_identity.sha256 != _sha256(
        expected_source_sha256, field="expected_source_sha256"
    ):
        raise EncoderBundlePublicationError(
            "local encoder source does not match the operator-pinned source SHA-256"
        )
    source_files = tuple(item.to_dict() for item in source_identity.files)
    if embedding_manifest is not None:
        _load_embedding_expectation(
            embedding_manifest,
            model_name=model_name,
            model_revision=model_revision,
        )
    provenance = _provenance_payload(
        source_sha256=source_identity.sha256,
        source_file_count=source_identity.file_count,
        source_size_bytes=source_identity.size_bytes,
        source_files=source_files,
        model_name=model_name,
        model_revision=model_revision,
        licence=licence.strip(),
        embedding_manifest=embedding_manifest,
    )

    root = Path(os.path.abspath(output_root))
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    temporary = Path(tempfile.mkdtemp(prefix=".encoder-bundle-", dir=root))
    try:
        _copy_verified_source(
            source_root=source_identity.path,
            destination_root=temporary,
            source_files=source_files,
        )
        current_source = inspect_encoder_bundle(source_identity.path)
        if current_source.sha256 != source_identity.sha256:
            raise EncoderBundlePublicationError("encoder source changed during publication")
        provenance_path = temporary / BUNDLE_PROVENANCE_FILENAME
        provenance_path.write_bytes(canonical_json_bytes(provenance) + b"\n")
        bundle = inspect_encoder_bundle(temporary)
        destination = root / bundle.sha256
        if destination.exists():
            validated = validate_encoder_bundle(
                destination,
                expected_sha256=bundle.sha256,
                expected_file_count=bundle.file_count,
                expected_size_bytes=bundle.size_bytes,
            )
            if _read_provenance(destination / BUNDLE_PROVENANCE_FILENAME) != provenance:
                raise EncoderBundlePublicationError(
                    "existing encoder bundle has different provenance and will not be replaced"
                )
            return BundlePublication(
                path=validated.path,
                sha256=validated.sha256,
                file_count=validated.file_count,
                size_bytes=validated.size_bytes,
                reused_existing=True,
            )
        try:
            os.replace(temporary, destination)
        except OSError as error:
            if not destination.exists():
                raise EncoderBundlePublicationError(
                    f"could not publish encoder bundle: {destination}"
                ) from error
            validated = validate_encoder_bundle(
                destination,
                expected_sha256=bundle.sha256,
                expected_file_count=bundle.file_count,
                expected_size_bytes=bundle.size_bytes,
            )
            if _read_provenance(destination / BUNDLE_PROVENANCE_FILENAME) != provenance:
                raise EncoderBundlePublicationError(
                    "concurrent encoder bundle publication has different provenance"
                ) from error
            return BundlePublication(
                path=validated.path,
                sha256=validated.sha256,
                file_count=validated.file_count,
                size_bytes=validated.size_bytes,
                reused_existing=True,
            )
        temporary = destination
        validated = validate_encoder_bundle(
            destination,
            expected_sha256=bundle.sha256,
            expected_file_count=bundle.file_count,
            expected_size_bytes=bundle.size_bytes,
        )
        if _read_provenance(destination / BUNDLE_PROVENANCE_FILENAME) != provenance:
            raise AssertionError("published encoder provenance did not round-trip")
        return BundlePublication(
            path=validated.path,
            sha256=validated.sha256,
            file_count=validated.file_count,
            size_bytes=validated.size_bytes,
            reused_existing=False,
        )
    finally:
        if temporary.exists() and temporary.parent == root and temporary.name.startswith(
            ".encoder-bundle-"
        ):
            shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="verified local model snapshot")
    parser.add_argument("--output-root", type=Path, required=True, help="parent of bundle digests")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--licence", required=True, help="declared model licence, for provenance")
    parser.add_argument(
        "--expected-source-sha256",
        required=True,
        help="content identity of the verified local source snapshot",
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        help="optional exact embedding-index manifest to bind to this encoder contract",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = package_encoder_bundle(
        source=args.source,
        output_root=args.output_root,
        model_name=args.model_name,
        model_revision=args.model_revision,
        licence=args.licence,
        expected_source_sha256=args.expected_source_sha256,
        embedding_manifest=args.embedding_manifest,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
