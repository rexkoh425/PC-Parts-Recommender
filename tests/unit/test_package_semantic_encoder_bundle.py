from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.package_semantic_encoder_bundle import (
    BUNDLE_PROVENANCE_FILENAME,
    BundlePublication,
    EncoderBundlePublicationError,
    package_encoder_bundle,
)

from pc_build_recommender.retrieval import (
    SentenceTransformerEmbeddingEncoder,
    embedding_encoder_fingerprint,
    inspect_encoder_bundle,
    validate_encoder_bundle,
)


def _source_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "1_Pooling").mkdir(parents=True)
    (source / "modules.json").write_text("[]\n", encoding="utf-8")
    (source / "config.json").write_text('{"hidden_size":3}\n', encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"model-bytes")
    (source / "1_Pooling" / "config.json").write_text(
        '{"word_embedding_dimension":3}\n', encoding="utf-8"
    )
    return source


def _embedding_manifest(tmp_path: Path, *, fingerprint: str | None = None) -> Path:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    expected = embedding_encoder_fingerprint(
        SentenceTransformerEmbeddingEncoder(model_name, revision=model_revision, device="cpu")
    )
    manifest = {
        "schema_version": "product-embedding-index-manifest-v1",
        "encoder": {
            "kind": "sentence_transformer",
            "model_name": model_name,
            "model_revision": model_revision,
            "fingerprint": fingerprint or expected,
            "dimension": 3,
            "normalised": True,
        },
    }
    path = tmp_path / "embedding-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _package(tmp_path: Path) -> BundlePublication:
    source = _source_bundle(tmp_path)
    return package_encoder_bundle(
        source=source,
        output_root=tmp_path / "published",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        licence="Apache-2.0",
        expected_source_sha256=inspect_encoder_bundle(source).sha256,
        embedding_manifest=_embedding_manifest(tmp_path),
    )


def test_package_encoder_bundle_publishes_and_validates_immutable_tree(tmp_path: Path) -> None:
    published = _package(tmp_path)

    assert not published.reused_existing
    assert published.path.name == published.sha256
    validated = validate_encoder_bundle(
        published.path,
        expected_sha256=published.sha256,
        expected_file_count=published.file_count,
        expected_size_bytes=published.size_bytes,
    )
    provenance = json.loads((published.path / BUNDLE_PROVENANCE_FILENAME).read_text("utf-8"))
    assert validated.sha256 == published.sha256
    assert provenance["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert provenance["model_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert provenance["licence"] == "Apache-2.0"
    assert provenance["embedding_manifest"]["filename"] == "embedding-manifest.json"
    assert "source" not in provenance


def test_package_encoder_bundle_adopts_only_the_same_publication_intent(tmp_path: Path) -> None:
    first = _package(tmp_path)
    second = package_encoder_bundle(
        source=tmp_path / "source",
        output_root=tmp_path / "published",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        licence="Apache-2.0",
        expected_source_sha256=inspect_encoder_bundle(tmp_path / "source").sha256,
        embedding_manifest=tmp_path / "embedding-manifest.json",
    )

    assert not first.reused_existing
    assert second.reused_existing
    assert second.path == first.path
    assert second.sha256 == first.sha256


def test_package_encoder_bundle_rejects_a_manifest_with_the_wrong_encoder_contract(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    manifest = _embedding_manifest(tmp_path, fingerprint="0" * 64)

    with pytest.raises(EncoderBundlePublicationError, match="fingerprint"):
        package_encoder_bundle(
            source=source,
            output_root=tmp_path / "published",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            licence="Apache-2.0",
            expected_source_sha256=inspect_encoder_bundle(source).sha256,
            embedding_manifest=manifest,
        )

    assert not (tmp_path / "published").exists()


def test_package_encoder_bundle_rejects_an_unpinned_or_wrong_source_snapshot(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)

    with pytest.raises(EncoderBundlePublicationError, match="operator-pinned source"):
        package_encoder_bundle(
            source=source,
            output_root=tmp_path / "published",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            licence="Apache-2.0",
            expected_source_sha256="0" * 64,
            embedding_manifest=_embedding_manifest(tmp_path),
        )
