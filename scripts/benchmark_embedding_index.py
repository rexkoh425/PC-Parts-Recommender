"""Reproducibly benchmark full and incremental product embedding index builds.

The benchmark calls ``build_embedding_index`` directly.  It creates deterministic
snapshots, verifies each incremental result against an independent full rebuild,
and writes only measurement evidence below the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from pc_build_recommender.retrieval.embedding_index import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    build_embedding_index,
    build_product_embedding_text,
    load_normalized_product_jsonl,
)
from pc_build_recommender.retrieval.vector import SentenceTransformerEmbeddingEncoder

PINNED_MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SCENARIOS = (
    ("0%", 0.0, "semantic"),
    ("1%", 0.01, "semantic"),
    ("2%", 0.02, "semantic"),
    ("5%", 0.05, "semantic"),
    ("10%", 0.10, "semantic"),
    ("price-only", 0.0, "price"),
)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )


def _select_indices(count: int, rate: float, seed: int, scenario: str) -> list[int]:
    changed = 0 if rate == 0 else max(1, round(count * rate))
    chooser = random.Random(f"{seed}:{scenario}:{count}")
    return sorted(chooser.sample(range(count), changed))


def make_snapshot(
    records: Sequence[Mapping[str, Any]], *, rate: float, kind: str, seed: int, scenario: str
) -> tuple[list[dict[str, Any]], list[int]]:
    """Return a deterministic semantically changed or price-only catalogue."""

    snapshot = [dict(row) for row in records]
    indices = _select_indices(len(snapshot), rate, seed, scenario)
    if kind == "price":
        indices = list(range(len(snapshot)))
    before = [build_product_embedding_text(row) for row in snapshot]
    for index in indices:
        row = snapshot[index]
        if kind == "semantic":
            attributes = dict(row.get("category_attributes", {}))
            attributes["benchmark_semantic_revision"] = f"{scenario}-{index}"
            row["category_attributes"] = attributes
        elif kind == "price":
            # Price is intentionally not one of build_product_embedding_text's inputs.
            row["benchmark_price_sgd"] = round(100 + (index * 17 % 1900) / 10, 2)
        else:
            raise ValueError(f"unknown mutation kind: {kind}")
    after = [build_product_embedding_text(row) for row in snapshot]
    if kind == "semantic" and any(before[index] == after[index] for index in indices):
        raise AssertionError("semantic benchmark mutation did not change embedding text")
    if kind == "price" and before != after:
        raise AssertionError("price-only mutation unexpectedly changed embedding text")
    return snapshot, indices


def _artifact_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _working_set_bytes() -> int:
    """Read resident process memory when psutil is available in the project env."""

    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.Process().memory_info().rss)


class _RamSampler:
    def __init__(self) -> None:
        self.peak = _working_set_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, _working_set_bytes())

    def __enter__(self) -> _RamSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.peak = max(self.peak, _working_set_bytes())
        self._stop.set()
        self._thread.join()


def _run_build(
    *, input_path: Path, output_dir: Path, encoder: Any, data_version: str
) -> tuple[Any, float, int]:
    with _RamSampler() as sampler:
        started = time.perf_counter()
        result = build_embedding_index(
            input_path,
            output_dir,
            data_version=data_version,
            index_version="embedding-index-benchmark-v1",
            encoder=encoder,
            batch_size=int(encoder.batch_size),
            device=str(encoder.requested_device),
        )
        elapsed = time.perf_counter() - started
    return result, elapsed, sampler.peak


def _assert_equivalent(
    full: Any, incremental: Any, expected_encoded: int, product_count: int
) -> None:
    full_matrix = np.load(full.embeddings_path, allow_pickle=False)
    incremental_matrix = np.load(incremental.embeddings_path, allow_pickle=False)
    full_ids = full.id_map_path.read_text(encoding="utf-8")
    incremental_ids = incremental.id_map_path.read_text(encoding="utf-8")
    assert full_ids == incremental_ids, "index ID maps differ"
    assert full_matrix.shape == incremental_matrix.shape == (product_count, 384)
    np.testing.assert_allclose(full_matrix, incremental_matrix, rtol=1e-6, atol=1e-6)
    assert full.manifest["content_hash"] == incremental.manifest["content_hash"]
    assert incremental.encoded_count == expected_encoded
    assert incremental.reused_count == product_count - expected_encoded


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_benchmark(
    *,
    input_path: Path,
    output_root: Path,
    encoder_kind: str,
    model_name: str,
    model_revision: str | None,
    device: str,
    batch_size: int,
    limit: int | None,
    seed: int,
    repetitions: int,
    warmup: bool = True,
) -> dict[str, Any]:
    """Execute all scenarios and return the persisted benchmark summary."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    records, source_files = load_normalized_product_jsonl(input_path, limit=limit)
    output_root.mkdir(parents=True, exist_ok=False)
    work = output_root / ".work"
    work.mkdir()
    baseline_input = work / "original.jsonl"
    _write_jsonl(baseline_input, records)
    if encoder_kind != "sentence-transformer":
        raise ValueError("the performance benchmark requires encoder_kind=sentence-transformer")
    encoder = SentenceTransformerEmbeddingEncoder(
        model_name, revision=model_revision, device=device, batch_size=batch_size
    )
    warmup_seconds = 0.0
    if warmup:
        started = time.perf_counter()
        encoder.warmup(expected_dimension=384)
        warmup_seconds = time.perf_counter() - started
    baseline_dir = work / "baseline-index"
    build_embedding_index(
        baseline_input,
        baseline_dir,
        data_version="benchmark-baseline-v1",
        index_version="embedding-index-benchmark-v1",
        encoder=encoder,
        batch_size=batch_size,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    for scenario, rate, kind in SCENARIOS:
        snapshot, changed = make_snapshot(
            records, rate=rate, kind=kind, seed=seed, scenario=scenario
        )
        snapshot_path = work / f"{scenario}.jsonl"
        _write_jsonl(snapshot_path, snapshot)
        expected = len(changed) if kind == "semantic" else 0
        # Alternating order mitigates a systematic first-run/warm-cache advantage.
        for repetition in range(repetitions):
            full_dir = work / "full"
            incremental_dir = work / "incremental"
            for directory in (full_dir, incremental_dir):
                if directory.exists():
                    shutil.rmtree(directory)
            order = ("full", "incremental") if repetition % 2 == 0 else ("incremental", "full")
            results: dict[str, tuple[Any, float, int]] = {}
            for mode in order:
                if mode == "full":
                    results[mode] = _run_build(
                        input_path=snapshot_path,
                        output_dir=full_dir,
                        encoder=encoder,
                        data_version=f"benchmark-{scenario}-v1",
                    )
                else:
                    # Deliberately outside timing: refreshing starts from the same old index.
                    shutil.copytree(baseline_dir, incremental_dir)
                    results[mode] = _run_build(
                        input_path=snapshot_path,
                        output_dir=incremental_dir,
                        encoder=encoder,
                        data_version=f"benchmark-{scenario}-v1",
                    )
            full, full_seconds, full_ram = results["full"]
            incremental, incremental_seconds, incremental_ram = results["incremental"]
            _assert_equivalent(full, incremental, expected, len(records))
            rows.append(
                {
                    "scenario": scenario,
                    "change_rate": rate,
                    "repetition": repetition + 1,
                    "full_seconds": full_seconds,
                    "incremental_seconds": incremental_seconds,
                    "speedup": full_seconds / incremental_seconds,
                    "full_encoded_count": full.encoded_count,
                    "incremental_encoded_count": incremental.encoded_count,
                    "incremental_reused_count": incremental.reused_count,
                    "full_peak_ram_bytes": full_ram,
                    "incremental_peak_ram_bytes": incremental_ram,
                    "index_bytes": _artifact_size(full_dir),
                    "correctness": "passed",
                    "run_order": "/".join(order),
                }
            )
    summary_scenarios: list[dict[str, Any]] = []
    for scenario, rate, _ in SCENARIOS:
        grouped = [row for row in rows if row["scenario"] == scenario]
        summary_scenarios.append(
            {
                "scenario": scenario,
                "change_rate": rate,
                "median_full_seconds": median(row["full_seconds"] for row in grouped),
                "median_incremental_seconds": median(row["incremental_seconds"] for row in grouped),
                "median_speedup": median(row["full_seconds"] for row in grouped)
                / median(row["incremental_seconds"] for row in grouped),
                "recomputed": grouped[0]["incremental_encoded_count"],
                "reused": grouped[0]["incremental_reused_count"],
                "peak_ram_bytes": max(
                    max(row["full_peak_ram_bytes"], row["incremental_peak_ram_bytes"])
                    for row in grouped
                ),
                "index_bytes": grouped[0]["index_bytes"],
                "correctness": "passed",
            }
        )
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "input": {
            "path": str(input_path.resolve()),
            "source_files": [str(path.resolve()) for path in source_files],
            "product_count": len(records),
            "limit": limit,
        },
        "experiment": {
            "seed": seed,
            "repetitions": repetitions,
            "warm_start": True,
            "model_warmup_seconds_excluded": warmup_seconds,
            "order": "alternating full/incremental",
            "floating_point_tolerance": {"rtol": 1e-6, "atol": 1e-6},
        },
        "encoder": {
            "kind": encoder_kind,
            "model_name": model_name,
            "model_revision": model_revision,
            "device": encoder.resolved_device,
            "batch_size": batch_size,
            "dimension": encoder.dimension,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hardware": _hardware(),
        },
        "ram_metric": "peak current-process working set sampled every 20 ms",
        "scenarios": summary_scenarios,
    }
    _write_outputs(output_root, rows, summary)
    shutil.rmtree(work)
    return summary


def _hardware() -> dict[str, str]:
    hardware = {"cpu": platform.processor() or "unknown"}
    try:
        import torch

        hardware["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        hardware["torch"] = str(torch.__version__)
    except ImportError:
        hardware["gpu"] = "torch unavailable"
    return hardware


def _write_outputs(
    root: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> None:
    fields = list(rows[0])
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Embedding-index benchmark",
        "",
        "Warm-start timings exclude model loading and input/index copying.",
        "",
        "| Change rate | Full rebuild (s) | Incremental (s) | Speedup | "
        "Recomputed | Reused | Peak RAM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["scenarios"]:
        lines.append(
            f"| {item['scenario']} | {item['median_full_seconds']:.3f} | "
            f"{item['median_incremental_seconds']:.3f} | {item['median_speedup']:.2f}x | "
            f"{item['recomputed']} | {item['reused']} | "
            f"{item['peak_ram_bytes'] / 1024**2:.1f} MiB |"
        )
    lines.extend(
        ["", "All runs passed ID-order, shape, content-hash, and vector allclose verification.", ""]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluation/embedding-index-benchmark")
    )
    parser.add_argument(
        "--encoder", default="sentence-transformer", choices=("sentence-transformer",)
    )
    parser.add_argument("--model", default=DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    parser.add_argument("--model-revision", default=PINNED_MINILM_REVISION)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args(argv)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_dir / timestamp
    summary = run_benchmark(
        input_path=args.input,
        output_root=root,
        encoder_kind=args.encoder,
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.dataset_size,
        seed=args.seed,
        repetitions=args.repetitions,
    )
    print(json.dumps({"output_dir": str(root), "scenarios": summary["scenarios"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
