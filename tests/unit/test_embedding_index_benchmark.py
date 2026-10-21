from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_embedding_index.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_embedding_index", _SCRIPT)
assert _SPEC and _SPEC.loader
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def _records(count: int = 6) -> list[dict[str, object]]:
    return [
        {
            "product_id": f"gpu-{index:03}",
            "category": "gpu",
            "brand": "Example",
            "model": f"Model {index}",
            "category_attributes": {"vram_gb": 8},
        }
        for index in range(count)
    ]


def test_snapshot_mutations_are_deterministic_and_price_is_non_semantic() -> None:
    original = _records(100)
    semantic_a, selected_a = benchmark.make_snapshot(
        original, rate=0.02, kind="semantic", seed=7, scenario="2%"
    )
    semantic_b, selected_b = benchmark.make_snapshot(
        original, rate=0.02, kind="semantic", seed=7, scenario="2%"
    )
    priced, selected_price = benchmark.make_snapshot(
        original, rate=0, kind="price", seed=7, scenario="price-only"
    )
    assert selected_a == selected_b == [*selected_a]
    assert len(selected_a) == 2
    assert semantic_a == semantic_b
    assert selected_price == list(range(100))
    assert [benchmark.build_product_embedding_text(row) for row in priced] == [
        benchmark.build_product_embedding_text(row) for row in original
    ]


def test_output_writer_emits_parseable_evidence(tmp_path: Path) -> None:
    rows = [
        {
            "scenario": "0%",
            "full_seconds": 1.0,
            "incremental_seconds": 0.5,
            "speedup": 2.0,
            "full_encoded_count": 2,
            "incremental_encoded_count": 0,
            "incremental_reused_count": 2,
            "full_peak_ram_bytes": 10,
            "incremental_peak_ram_bytes": 11,
            "index_bytes": 12,
            "correctness": "passed",
            "run_order": "full/incremental",
        }
    ]
    summary = {
        "scenarios": [
            {
                "scenario": "0%",
                "median_full_seconds": 1.0,
                "median_incremental_seconds": 0.5,
                "median_speedup": 2.0,
                "recomputed": 0,
                "reused": 2,
                "peak_ram_bytes": 11,
                "index_bytes": 12,
                "correctness": "passed",
            }
        ]
    }
    benchmark._write_outputs(tmp_path, rows, summary)
    assert (
        list(csv.DictReader((tmp_path / "runs.csv").open(encoding="utf-8")))[0]["correctness"]
        == "passed"
    )
    assert (
        json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["scenarios"][0][
            "reused"
        ]
        == 2
    )
    assert "2.00x" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_harness_runs_actual_indexer_with_deterministic_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    class TestEncoder:
        requested_device = "cpu"
        resolved_device = "cpu"
        batch_size = 2
        model_name = "benchmark-test-encoder"
        revision = "test-revision"
        dimension = 384

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def warmup(self, *, expected_dimension: int) -> int:
            assert expected_dimension == self.dimension
            return self.dimension

        def encode(self, texts: list[str]) -> np.ndarray:
            matrix = np.ones((len(texts), self.dimension), dtype=np.float32)
            for index, text in enumerate(texts):
                column = int.from_bytes(hashlib.sha256(text.encode()).digest()[:2]) % self.dimension
                matrix[index, column] = float(len(text) + 1)
            return matrix

    input_path = tmp_path / "records.jsonl"
    benchmark._write_jsonl(input_path, _records())
    monkeypatch.setattr(benchmark, "SentenceTransformerEmbeddingEncoder", TestEncoder)
    summary = benchmark.run_benchmark(
        input_path=input_path,
        output_root=tmp_path / "evidence",
        encoder_kind="sentence-transformer",
        model_name="test/model",
        model_revision="test-revision",
        device="cpu",
        batch_size=2,
        limit=None,
        seed=7,
        repetitions=1,
    )
    assert len(summary["scenarios"]) == 6
    assert {row["correctness"] for row in summary["scenarios"]} == {"passed"}
