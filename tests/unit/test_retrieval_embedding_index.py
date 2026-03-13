from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from pc_build_recommender.retrieval.embedding_index import (
    MANIFEST_SCHEMA_VERSION,
    TEXT_BUILDER_VERSION,
    build_embedding_index,
    build_product_embedding_text,
    load_normalized_product_jsonl,
    main,
)
from pc_build_recommender.retrieval.vector import (
    SentenceTransformerEmbeddingEncoder,
    resolve_embedding_device,
)


def _write_products(path: Path, products: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )


def _products() -> list[dict[str, object]]:
    return [
        {
            "product_id": "gpu-4070s",
            "category": "gpu",
            "brand": "NVIDIA",
            "model": "RTX 4070 Super",
            "canonical_name": "NVIDIA GeForce RTX 4070 Super 12GB",
            "manufacturer_part_number": "4070S-12G",
            "category_attributes": {
                "vram_gb": 12,
                "board_power_w": 220,
                "power_connectors": ["12VHPWR"],
            },
            "supported_workloads": ["gaming_1440p", "local_ai"],
            "benchmark_tags": ["gaming", "inference"],
            "compatibility_tags": ["pcie_4", "atx_3_psu_recommended"],
            "review_aspects": ["noise", "thermals"],
            "updated_at": "2026-07-21T10:00:00Z",
        },
        {
            "product_id": "cpu-7950x",
            "category": "cpu",
            "brand": "AMD",
            "model": "Ryzen 9 7950X",
            "canonical_name": "AMD Ryzen 9 7950X",
            "category_attributes": {
                "socket": "AM5",
                "core_count": 16,
                "thread_count": 32,
            },
            "supported_workloads": ["compilation", "content_creation"],
            "updated_at": "2026-07-22T10:00:00Z",
        },
    ]


class CountingEncoder:
    model_name = "counting-test-v1"
    dimension = 32
    requested_device = "cpu"
    resolved_device = "cpu"
    batch_size = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for index, value in enumerate(digest):
                matrix[row, index] = value + 1
        return matrix


class FailingEncoder:
    model_name = "unavailable-semantic-model"
    dimension = 384
    requested_device = "cuda"
    resolved_device = "cuda"
    batch_size = 16

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        raise RuntimeError("model weights unavailable")


def test_product_embedding_text_is_stable_and_covers_retrieval_evidence() -> None:
    first = _products()[0]
    reordered = dict(reversed(list(first.items())))

    text = build_product_embedding_text(first)

    assert build_product_embedding_text(reordered) == text
    assert "category: gpu" in text
    assert "vram gb 12" in text
    assert "workloads: gaming_1440p, local_ai" in text
    assert "benchmarks: gaming, inference" in text
    assert "compatibility: atx_3_psu_recommended, pcie_4" in text
    assert "review aspects: noise, thermals" in text


def test_sentence_transformer_adapter_passes_resolved_device_and_batch(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, *, revision: str | None, device: str) -> None:
            calls["model_name"] = model_name
            calls["revision"] = revision
            calls["device"] = device

        @staticmethod
        def get_sentence_embedding_dimension() -> int:
            return 3

        @staticmethod
        def encode(texts: list[str], **kwargs: object) -> NDArray[np.float32]:
            calls["texts"] = texts
            calls.update(kwargs)
            return np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    encoder = SentenceTransformerEmbeddingEncoder("fake/model", device="cpu", batch_size=7)

    matrix = encoder.encode(["one", "two"])

    assert matrix.shape == (2, 3)
    assert encoder.requested_device == "cpu"
    assert encoder.resolved_device == "cpu"
    assert encoder.dimension == 3
    assert calls == {
        "model_name": "fake/model",
        "revision": None,
        "device": "cpu",
        "texts": ["one", "two"],
        "batch_size": 7,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    assert resolve_embedding_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="auto, cuda, cpu"):
        resolve_embedding_device("metal")


def test_load_jsonl_selects_bounded_recent_subset_then_stable_id_order(tmp_path) -> None:
    input_dir = tmp_path / "normalized"
    input_dir.mkdir()
    products = _products() + [
        {
            "product_id": "old-case",
            "category": "case",
            "canonical_name": "Old Case",
            "updated_at": "2025-01-01T00:00:00Z",
        }
    ]
    _write_products(input_dir / "products.jsonl", products)

    records, files = load_normalized_product_jsonl(
        input_dir,
        limit=2,
        recent_first=True,
    )

    assert [record["product_id"] for record in records] == ["cpu-7950x", "gpu-4070s"]
    assert files == (input_dir / "products.jsonl",)


def test_normalized_record_envelope_is_unwrapped_for_loading_and_text(tmp_path) -> None:
    source = tmp_path / "records.jsonl"
    inner_product = _products()[0]
    envelope = {
        "record_type": "canonical_product",
        "schema_version": "pc-build-recommender.normalised-record.v1",
        "source_record_id": "source-gpu-1",
        "data": inner_product,
        "training_eligible": True,
    }
    _write_products(source, [envelope])

    records, _ = load_normalized_product_jsonl(source)
    result = build_embedding_index(
        source,
        tmp_path / "index",
        data_version="envelope-v1",
        encoder_kind="hash",
        hash_dimension=64,
    )

    assert records == [inner_product]
    assert build_product_embedding_text(envelope) == build_product_embedding_text(inner_product)
    assert np.load(result.embeddings_path, allow_pickle=False).shape == (1, 64)
    id_row = json.loads(result.id_map_path.read_text(encoding="utf-8"))
    assert id_row["product_id"] == "gpu-4070s"


def test_none_is_not_accepted_as_a_product_id(tmp_path) -> None:
    source = tmp_path / "records.jsonl"
    _write_products(source, [{"product_id": None, "category": "gpu"}])

    with pytest.raises(ValueError, match="missing product_id"):
        load_normalized_product_jsonl(source)


def test_hash_index_persists_float32_matrix_id_map_and_manifest(tmp_path) -> None:
    source = tmp_path / "products.jsonl"
    output = tmp_path / "index"
    _write_products(source, _products())

    result = build_embedding_index(
        source,
        output,
        data_version="catalog-v1",
        encoder_kind="hash",
        hash_dimension=64,
        batch_size=17,
    )

    matrix = np.load(result.embeddings_path, allow_pickle=False)
    id_rows = [
        json.loads(line)
        for line in result.id_map_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 64)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(2), atol=1e-6)
    assert [row["product_id"] for row in id_rows] == ["cpu-7950x", "gpu-4070s"]
    assert all(len(row["content_hash"]) == 64 for row in id_rows)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["text_builder_version"] == TEXT_BUILDER_VERSION
    assert manifest["data_version"] == "catalog-v1"
    assert manifest["encoder"]["kind"] == "deterministic_lexical_hash"
    assert manifest["encoder"]["resolved_device"] == "cpu"
    assert manifest["encoder"]["batch_size"] == 17
    assert manifest["matrix"] == {
        "dtype": "float32",
        "l2_normalised": True,
        "shape": [2, 64],
    }
    assert manifest["artifacts"]["embeddings"]["bytes"] == result.embeddings_path.stat().st_size


def test_unchanged_content_skips_encoder_and_preserves_artifacts(tmp_path) -> None:
    source = tmp_path / "products.jsonl"
    output = tmp_path / "index"
    _write_products(source, _products())
    first_encoder = CountingEncoder()
    first = build_embedding_index(
        source,
        output,
        data_version="catalog-v1",
        encoder=first_encoder,
    )
    before_matrix = first.embeddings_path.read_bytes()
    second_encoder = CountingEncoder()

    second = build_embedding_index(
        source,
        output,
        data_version="catalog-v1",
        encoder=second_encoder,
    )

    assert first.encoded_count == 2
    assert second.encoded_count == 0
    assert second.reused_count == 2
    assert second.skipped_unchanged
    assert second_encoder.calls == []
    assert second.embeddings_path.read_bytes() == before_matrix


def test_changed_product_reencodes_one_row_and_reuses_the_other(tmp_path) -> None:
    source = tmp_path / "products.jsonl"
    output = tmp_path / "index"
    products = _products()
    _write_products(source, products)
    build_embedding_index(
        source,
        output,
        data_version="catalog-v1",
        encoder=CountingEncoder(),
    )
    products[0]["category_attributes"] = {"vram_gb": 16, "board_power_w": 225}
    _write_products(source, products)
    incremental_encoder = CountingEncoder()

    result = build_embedding_index(
        source,
        output,
        data_version="catalog-v2",
        encoder=incremental_encoder,
    )

    assert result.encoded_count == 1
    assert result.reused_count == 1
    assert not result.skipped_unchanged
    assert len(incremental_encoder.calls) == 1
    assert len(incremental_encoder.calls[0]) == 1
    assert "vram gb 16" in incremental_encoder.calls[0][0]
    assert result.manifest["build"] == {
        "encoded_count": 1,
        "reused_count": 1,
        "incremental": True,
        "reuse_source": {
            "created_at_utc": result.manifest["build"]["reuse_source"]["created_at_utc"],
            "resolved_device": "cpu",
            "batch_size": 8,
        },
    }


def test_backend_failure_can_fall_back_to_deterministic_hash(tmp_path) -> None:
    source = tmp_path / "products.jsonl"
    _write_products(source, _products())

    result = build_embedding_index(
        source,
        tmp_path / "index",
        data_version="catalog-v1",
        encoder=FailingEncoder(),
        fallback_to_hash=True,
        hash_dimension=96,
    )

    assert result.manifest["encoder"]["kind"] == "deterministic_lexical_hash"
    assert result.manifest["encoder"]["dimension"] == 96
    assert "FailingEncoder failed" in result.manifest["encoder"]["fallback_reason"]
    assert np.load(result.embeddings_path, allow_pickle=False).shape == (2, 96)


def test_cli_builds_hash_artifacts_without_model_download(tmp_path, capsys) -> None:
    source = tmp_path / "products.jsonl"
    output = tmp_path / "index"
    _write_products(source, _products())

    exit_code = main(
        [
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--data-version",
            "catalog-cli-v1",
            "--encoder",
            "hash",
            "--hash-dimension",
            "64",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed["matrix"]["shape"] == [2, 64]
    assert printed["encoded_count"] == 2
    assert (output / "manifest.json").is_file()


def test_duplicate_product_ids_are_rejected_before_encoding(tmp_path) -> None:
    source = tmp_path / "products.jsonl"
    products = _products()
    _write_products(source, [products[0], products[0]])

    with pytest.raises(ValueError, match="duplicate product IDs"):
        build_embedding_index(
            source,
            tmp_path / "index",
            data_version="catalog-v1",
            encoder_kind="hash",
        )


def test_corrupt_artifact_disables_incremental_reuse(tmp_path) -> None:
    source = tmp_path / "products.jsonl"
    output = tmp_path / "index"
    _write_products(source, _products())
    first = build_embedding_index(
        source,
        output,
        data_version="catalog-v1",
        encoder=CountingEncoder(),
    )
    first.id_map_path.write_text(
        first.id_map_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    encoder = CountingEncoder()

    rebuilt = build_embedding_index(
        source,
        output,
        data_version="catalog-v1",
        encoder=encoder,
    )

    assert rebuilt.encoded_count == 2
    assert rebuilt.reused_count == 0
    assert len(encoder.calls) == 1
