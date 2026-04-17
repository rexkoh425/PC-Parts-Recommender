from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from pc_build_recommender.retrieval import (
    EncoderBundleValidationError,
    SentenceTransformerEmbeddingEncoder,
    inspect_encoder_bundle,
    validate_encoder_bundle,
)


def _content_addressed_bundle(tmp_path: Path) -> tuple[Path, str, int, int]:
    staging = tmp_path / "staging"
    (staging / "1_Pooling").mkdir(parents=True)
    (staging / "config.json").write_text('{"hidden_size":3}\n', encoding="utf-8")
    (staging / "modules.json").write_text("[]\n", encoding="utf-8")
    (staging / "1_Pooling" / "config.json").write_text(
        '{"word_embedding_dimension":3}\n',
        encoding="utf-8",
    )
    inspected = inspect_encoder_bundle(staging)
    published = tmp_path / inspected.sha256
    staging.rename(published)
    return published, inspected.sha256, inspected.file_count, inspected.size_bytes


def test_content_addressed_encoder_bundle_is_streamed_and_verified(tmp_path: Path) -> None:
    path, digest, file_count, size_bytes = _content_addressed_bundle(tmp_path)

    validated = validate_encoder_bundle(
        path,
        expected_sha256=digest,
        expected_file_count=file_count,
        expected_size_bytes=size_bytes,
    )

    assert validated.path == path.resolve()
    assert validated.sha256 == digest
    assert validated.file_count == 3
    assert validated.size_bytes == size_bytes
    assert tuple(file.path for file in validated.files) == (
        "1_Pooling/config.json",
        "config.json",
        "modules.json",
    )


def test_encoder_bundle_fails_closed_after_content_tampering(tmp_path: Path) -> None:
    path, digest, _, _ = _content_addressed_bundle(tmp_path)
    (path / "config.json").write_text('{"hidden_size":4}\n', encoding="utf-8")

    with pytest.raises(EncoderBundleValidationError, match="content hash"):
        validate_encoder_bundle(path, expected_sha256=digest)


def test_encoder_bundle_requires_content_addressed_directory_name(tmp_path: Path) -> None:
    staging = tmp_path / "not-a-digest"
    staging.mkdir()
    (staging / "modules.json").write_text("[]\n", encoding="utf-8")
    digest = inspect_encoder_bundle(staging).sha256

    with pytest.raises(EncoderBundleValidationError, match="directory name"):
        validate_encoder_bundle(staging, expected_sha256=digest)


def test_encoder_bundle_rejects_a_symlinked_parent_when_supported(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    bundle = real_parent / "bundle"
    bundle.mkdir()
    (bundle / "modules.json").write_text("[]\n", encoding="utf-8")
    linked_parent = tmp_path / "linked"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(EncoderBundleValidationError, match="symlinks or junctions"):
        inspect_encoder_bundle(linked_parent / "bundle")


def test_encoder_bundle_detects_file_replacement_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    target = bundle / "modules.json"
    target.write_text("[]\n", encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text('[{"changed":true}]\n', encoding="utf-8")
    real_open = os.open
    swapped = False

    def swapping_open(path: Path, flags: int) -> int:
        nonlocal swapped
        if not swapped and Path(path) == target:
            swapped = True
            replacement.replace(target)
        return int(real_open(path, flags))

    monkeypatch.setattr(os, "open", swapping_open)

    with pytest.raises(EncoderBundleValidationError, match="changed while it was opened"):
        inspect_encoder_bundle(bundle)


def test_offline_encoder_requires_an_existing_explicit_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit model_path"):
        SentenceTransformerEmbeddingEncoder("logical/model", device="cpu", local_files_only=True)

    with pytest.raises(FileNotFoundError):
        SentenceTransformerEmbeddingEncoder(
            "logical/model",
            device="cpu",
            model_path=tmp_path / "missing",
            local_files_only=True,
        )


def test_offline_encoder_loads_only_the_validated_local_path_and_warms_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(
            self,
            source: str,
            *,
            revision: str | None,
            device: str,
            local_files_only: bool,
        ) -> None:
            calls.update(
                source=source,
                revision=revision,
                device=device,
                local_files_only=local_files_only,
            )

        @staticmethod
        def get_embedding_dimension() -> int:
            return 3

        @staticmethod
        def encode(texts: list[str], **kwargs: object) -> NDArray[np.float32]:
            calls["texts"] = texts
            calls.update(kwargs)
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    encoder = SentenceTransformerEmbeddingEncoder(
        "logical/upstream-model",
        revision="pinned-revision",
        device="cpu",
        batch_size=8,
        model_path=model_path,
        local_files_only=True,
    )

    assert encoder.warmup(expected_dimension=3) == 3
    assert encoder.model_name == "logical/upstream-model"
    assert calls == {
        "source": str(model_path.resolve()),
        "revision": "pinned-revision",
        "device": "cpu",
        "local_files_only": True,
        "texts": ["BuildSignal semantic encoder readiness probe"],
        "batch_size": 8,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }


def test_encoder_warmup_rejects_wrong_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    class FakeSentenceTransformer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def get_embedding_dimension() -> int:
            return 2

        @staticmethod
        def encode(_texts: list[str], **_kwargs: object) -> NDArray[np.float32]:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    encoder = SentenceTransformerEmbeddingEncoder(
        "logical/upstream-model",
        device="cpu",
        model_path=model_path,
        local_files_only=True,
    )

    with pytest.raises(RuntimeError, match=r"expected \(1, 3\)"):
        encoder.warmup(expected_dimension=3)
