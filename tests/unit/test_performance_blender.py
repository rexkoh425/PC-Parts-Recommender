from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path
from typing import Any

import pytest
from training.prepare_blender_performance import (
    BenchmarkExecutionIdentity,
    _generation_for,
    _JsonlRecordStream,
    _leakage_family,
    estimate_blender_preparation_memory_mib,
    extract_hardware_family,
    prepare_blender_performance,
    source_promotion_blockers,
)
from training.prepare_blender_performance import (
    main as prepare_blender_main,
)


def test_cpu_refresh_architectures_share_one_generation_stratum() -> None:
    products = [
        {"category_attributes": {"architecture": "Raptor Lake"}},
        {"category_attributes": {"architecture": "Raptor Lake Refresh"}},
    ]

    assert _generation_for("cpu", "cpu:intel_core_i7_13700", products) == "raptor_lake"


def _catalog_cpu(
    product_id: str,
    name: str,
    *,
    architecture: str,
    cores: int,
    threads: int,
    base: float,
    boost: float,
    tdp: float,
) -> dict[str, Any]:
    return {
        "training_eligible": True,
        "published_claims_eligible": True,
        "data": {
            "product_id": product_id,
            "category": "cpu",
            "canonical_name": name,
            "category_attributes": {
                "architecture": architecture,
                "core_count": cores,
                "thread_count": threads,
                "base_clock_ghz": base,
                "boost_clock_ghz": boost,
                "tdp_watts": tdp,
            },
        },
    }


def _benchmark(
    identifier: str,
    name: str,
    score: float,
    *,
    cores: int,
    threads: int,
    scene: str = "bmw27",
    sockets: int = 1,
    unit: str = "seconds",
    higher_is_better: bool = False,
    score_source_field: str = "total_render_time",
    blender_build_hash: str = "test-build-hash",
    benchmark_script: str = "test-script-v1",
    scene_checksum: str = "test-scene-checksum",
) -> dict[str, Any]:
    return {
        "training_eligible": True,
        "published_claims_eligible": True,
        "source_record_id": identifier,
        "data": {
            "benchmark_version": "2.91.2",
            "preset": scene,
            "operating_system": "Windows",
            "score": score,
            "unit": unit,
            "higher_is_better": higher_is_better,
            "source_url": f"https://opendata.blender.org/benchmarks/{identifier}/",
        },
        "normalisation_metadata": {
            "device_type": "CPU",
            "hardware_name": name,
            "system_cpu_cores": cores,
            "cpu_threads": threads,
            "system_cpu_sockets": sockets,
            "score_source_field": score_source_field,
            "blender_build_hash": blender_build_hash,
            "benchmark_script": benchmark_script,
            "scene_checksum": scene_checksum,
        },
    }


def _fixture_catalog() -> list[dict[str, Any]]:
    return [
        _catalog_cpu(
            "cpu-3600",
            "AMD Ryzen 5 3600",
            architecture="Zen 2",
            cores=6,
            threads=12,
            base=3.6,
            boost=4.2,
            tdp=65,
        ),
        _catalog_cpu(
            "cpu-3600x",
            "AMD Ryzen 5 3600X",
            architecture="Zen 2",
            cores=6,
            threads=12,
            base=3.8,
            boost=4.4,
            tdp=95,
        ),
        _catalog_cpu(
            "cpu-5600x",
            "AMD Ryzen 5 5600X",
            architecture="Zen 3",
            cores=6,
            threads=12,
            base=3.7,
            boost=4.6,
            tdp=65,
        ),
        _catalog_cpu(
            "cpu-6700",
            "Intel Core i7 6700",
            architecture="Skylake",
            cores=4,
            threads=8,
            base=3.4,
            boost=4.0,
            tdp=65,
        ),
        _catalog_cpu(
            "cpu-6700k",
            "Intel Core i7 6700K",
            architecture="Skylake",
            cores=4,
            threads=8,
            base=4.0,
            boost=4.2,
            tdp=91,
        ),
    ]


def test_hardware_family_normalization_preserves_exact_variants() -> None:
    assert (
        extract_hardware_family("AMD Ryzen 7 2700 Eight-Core Processor", "cpu")
        == "cpu:amd_ryzen_7_2700"
    )
    assert (
        extract_hardware_family("Intel Core i7-6700 CPU @ 3.40GHz", "cpu")
        == "cpu:intel_core_i7_6700"
    )
    assert (
        extract_hardware_family("MSI GeForce RTX4070 SUPER 12GB", "gpu")
        == "gpu:nvidia_geforce_rtx_4070_super"
    )
    assert extract_hardware_family("GeForce RTX 4070 Ti", "gpu") != extract_hardware_family(
        "GeForce RTX 4070", "gpu"
    )


def test_leakage_family_keeps_distinct_workstation_cpu_lines() -> None:
    assert _leakage_family("cpu", "cpu:intel_core_i9_9900k") == "cpu:intel_core_i9_9900"
    assert _leakage_family("cpu", "cpu:intel_core_i9_9900x") == "cpu:intel_core_i9_9900x"
    assert _leakage_family("cpu", "cpu:amd_ryzen_9_7950x") == "cpu:amd_ryzen_9_7950"
    assert (
        _leakage_family("cpu", "cpu:amd_ryzen_threadripper_pro_5995wx")
        == "cpu:amd_ryzen_threadripper_pro_5995wx"
    )


def test_preparation_keeps_intel_x_series_out_of_desktop_generation_group() -> None:
    catalog = [
        _catalog_cpu(
            "cpu-9900",
            "Intel Core i9 9900",
            architecture="Coffee Lake",
            cores=8,
            threads=16,
            base=3.1,
            boost=5.0,
            tdp=65,
        ),
        _catalog_cpu(
            "cpu-9900x",
            "Intel Core i9 9900X",
            architecture="Skylake-X",
            cores=10,
            threads=20,
            base=3.5,
            boost=4.4,
            tdp=165,
        ),
        _catalog_cpu(
            "cpu-9700",
            "Intel Core i7 9700",
            architecture="Coffee Lake",
            cores=8,
            threads=8,
            base=3.0,
            boost=4.7,
            tdp=65,
        ),
    ]
    benchmarks = [
        _benchmark("9900", "Intel Core i9-9900 CPU @ 3.10GHz", 12.0, cores=8, threads=16),
        _benchmark("9900x", "Intel Core i9-9900X CPU @ 3.50GHz", 10.0, cores=10, threads=20),
        _benchmark("9700", "Intel Core i7-9700 CPU @ 3.00GHz", 11.0, cores=8, threads=8),
    ]

    rows, _manifest = prepare_blender_performance(
        benchmarks,
        catalog,
        category_filter="cpu",
        minimum_pilot_products=3,
        minimum_credible_products=3,
    )

    generations = {row["product_family"]: row["hardware_generation"] for row in rows}
    assert set(generations) == {
        "cpu:intel_core_i9_9900",
        "cpu:intel_core_i9_9900x",
        "cpu:intel_core_i7_9700",
    }
    assert generations["cpu:intel_core_i9_9900"] != generations["cpu:intel_core_i9_9900x"]


def test_preparation_converts_render_seconds_to_observed_throughput_target() -> None:
    benchmarks = [
        _benchmark("a", "AMD Ryzen 5 3600 6-Core Processor", 10.0, cores=6, threads=12),
        _benchmark("b", "AMD Ryzen 5 3600 6-Core Processor", 14.0, cores=6, threads=12),
        _benchmark("c", "AMD Ryzen 5 3600X 6-Core Processor", 9.0, cores=6, threads=12),
        _benchmark("d", "AMD Ryzen 5 5600X 6-Core Processor", 7.0, cores=6, threads=12),
        _benchmark("e", "Intel Core i7-6700 CPU @ 3.40GHz", 18.0, cores=4, threads=8),
        _benchmark("f", "Intel Core i7-6700K CPU @ 4.00GHz", 15.0, cores=4, threads=8),
        _benchmark(
            "other",
            "AMD Ryzen 5 3600 6-Core Processor",
            50.0,
            cores=6,
            threads=12,
            scene="classroom",
        ),
    ]

    rows, manifest = prepare_blender_performance(
        benchmarks,
        _fixture_catalog(),
        category_filter="cpu",
        minimum_pilot_products=3,
        minimum_credible_products=10,
    )

    assert len(rows) == 5
    assert manifest["selected_cohort"]["scene"] == "bmw27"
    assert manifest["selected_cohort"]["joined_observations"] == 6
    assert manifest["selected_cohort"]["leakage_groups"] == 3
    assert manifest["selected_cohort"]["blender_build_hash"] == "test-build-hash"
    assert manifest["selected_cohort"]["benchmark_script"] == "test-script-v1"
    assert manifest["selected_cohort"]["scene_checksum"] == "test-scene-checksum"
    assert manifest["promotion"]["eligible"] is False
    assert all(row["is_synthetic"] is False for row in rows)
    assert all(row["eligible_for_external_claims"] is False for row in rows)
    ryzen_3600 = next(
        row
        for row in rows
        if row["benchmark_hardware_names_json"] == '["AMD Ryzen 5 3600 6-Core Processor"]'
    )
    assert ryzen_3600["observed_source_value_median"] == pytest.approx(12.0)
    assert ryzen_3600["observed_source_unit"] == "seconds"
    assert ryzen_3600["observed_source_higher_is_better"] is False
    assert ryzen_3600["observed_source_score_field"] == "total_render_time"
    assert ryzen_3600["blender_build_hash"] == "test-build-hash"
    assert ryzen_3600["benchmark_script"] == "test-script-v1"
    assert ryzen_3600["scene_checksum"] == "test-scene-checksum"
    assert ryzen_3600["target_score"] == pytest.approx(1000.0 / 12.0)
    assert ryzen_3600["benchmark_observation_count"] == 2
    assert manifest["target"]["source_unit"] == "seconds"
    assert manifest["target"]["source_higher_is_better"] is False


def test_preparation_preserves_samples_per_minute_without_inverting_target() -> None:
    def throughput(
        identifier: str,
        name: str,
        score: float,
        *,
        cores: int,
        threads: int,
    ) -> dict[str, Any]:
        return _benchmark(
            identifier,
            name,
            score,
            cores=cores,
            threads=threads,
            unit="samples/minute",
            higher_is_better=True,
            score_source_field="samples_per_minute",
        )

    benchmarks = [
        throughput("a", "AMD Ryzen 5 3600 6-Core Processor", 80.0, cores=6, threads=12),
        throughput("b", "AMD Ryzen 5 3600 6-Core Processor", 100.0, cores=6, threads=12),
        throughput("c", "AMD Ryzen 5 3600X 6-Core Processor", 95.0, cores=6, threads=12),
        throughput("d", "AMD Ryzen 5 5600X 6-Core Processor", 130.0, cores=6, threads=12),
        throughput("e", "Intel Core i7-6700 CPU @ 3.40GHz", 60.0, cores=4, threads=8),
        throughput("f", "Intel Core i7-6700K CPU @ 4.00GHz", 70.0, cores=4, threads=8),
    ]

    rows, manifest = prepare_blender_performance(
        benchmarks,
        _fixture_catalog(),
        category_filter="cpu",
        minimum_pilot_products=3,
        minimum_credible_products=10,
        workload="content_creation",
    )

    ryzen_3600 = next(
        row
        for row in rows
        if row["benchmark_hardware_names_json"] == '["AMD Ryzen 5 3600 6-Core Processor"]'
    )
    assert ryzen_3600["observed_source_value_median"] == pytest.approx(90.0)
    assert ryzen_3600["target_score"] == pytest.approx(90.0)
    assert ryzen_3600["target_score"] != pytest.approx(1000.0 / 90.0)
    assert ryzen_3600["target_transform"] == "median(samples_per_minute)"
    assert ryzen_3600["workload"] == "content_creation"
    assert manifest["target"] == {
        "column": "target_score",
        "formula": "median(samples_per_minute)",
        "target_scale": None,
        "higher_is_better": True,
        "unit": "samples/minute",
        "source_value_kind": "observed",
        "source_unit": "samples/minute",
        "source_higher_is_better": True,
        "source_score_field": "samples_per_minute",
        "aggregation": (
            "median within exact benchmark, execution, score-semantics cohort and hardware family"
        ),
    }


def test_exact_execution_identity_prevents_cross_build_aggregation() -> None:
    cohort = ("2.91.2", "bmw27", "CPU", "Windows")
    identities = {
        "a": BenchmarkExecutionIdentity("build-a", "script-a", "checksum-a"),
        "b": BenchmarkExecutionIdentity("build-b", "script-b", "checksum-b"),
    }
    hardware = (
        ("AMD Ryzen 5 3600 6-Core Processor", 6, 12),
        ("AMD Ryzen 5 5600X 6-Core Processor", 6, 12),
        ("Intel Core i7-6700 CPU @ 3.40GHz", 4, 8),
    )
    benchmarks = [
        _benchmark(
            f"{identity_key}-{index}",
            name,
            score * multiplier,
            cores=cores,
            threads=threads,
            blender_build_hash=identity.blender_build_hash,
            benchmark_script=identity.benchmark_script,
            scene_checksum=identity.scene_checksum,
        )
        for identity_key, identity, multiplier in (
            ("a", identities["a"], 1.0),
            ("b", identities["b"], 10.0),
        )
        for index, (name, cores, threads) in enumerate(hardware)
        for score in (10.0 + index,)
    ]

    with pytest.raises(ValueError, match="multiple execution or score contracts"):
        prepare_blender_performance(
            benchmarks,
            _fixture_catalog(),
            category_filter="cpu",
            pinned_cohort=cohort,
            minimum_pilot_products=3,
            minimum_credible_products=10,
        )

    rows_a, manifest_a = prepare_blender_performance(
        benchmarks,
        _fixture_catalog(),
        category_filter="cpu",
        pinned_cohort=cohort,
        pinned_execution_identity=identities["a"],
        minimum_pilot_products=3,
        minimum_credible_products=10,
    )
    rows_b, manifest_b = prepare_blender_performance(
        benchmarks,
        _fixture_catalog(),
        category_filter="cpu",
        pinned_cohort=cohort,
        pinned_execution_identity=identities["b"],
        minimum_pilot_products=3,
        minimum_credible_products=10,
    )

    assert manifest_a["schema_version"] == "pc-build-recommender.blender-performance-dataset.v3"
    assert identities["a"].as_dict().items() <= manifest_a["selected_cohort"].items()
    assert identities["b"].as_dict().items() <= manifest_b["selected_cohort"].items()
    assert all(row["benchmark_observation_count"] == 1 for row in rows_a + rows_b)
    assert {row["product_id"] for row in rows_a}.isdisjoint({row["product_id"] for row in rows_b})
    scores_a = {row["target_score"] for row in rows_a}
    scores_b = {row["target_score"] for row in rows_b}
    assert scores_a.isdisjoint(scores_b)


def test_preparation_rejects_mislabeled_samples_per_minute_direction() -> None:
    benchmarks = [
        _benchmark(
            identifier,
            name,
            score,
            cores=cores,
            threads=threads,
            unit="samples/minute",
            higher_is_better=False,
            score_source_field="samples_per_minute",
        )
        for identifier, name, score, cores, threads in (
            ("a", "AMD Ryzen 5 3600 6-Core Processor", 80.0, 6, 12),
            ("b", "AMD Ryzen 5 5600X 6-Core Processor", 100.0, 6, 12),
            ("c", "Intel Core i7-6700 CPU @ 3.40GHz", 60.0, 4, 8),
        )
    ]

    with pytest.raises(ValueError, match="no Blender observations joined"):
        prepare_blender_performance(
            benchmarks,
            _fixture_catalog(),
            category_filter="cpu",
            minimum_pilot_products=3,
            minimum_credible_products=10,
        )


def test_cpu_topology_mismatch_is_rejected_before_aggregation() -> None:
    benchmarks = [
        _benchmark("a", "AMD Ryzen 5 3600 6-Core Processor", 10.0, cores=6, threads=12),
        _benchmark(
            "bad-sockets",
            "AMD Ryzen 5 3600 6-Core Processor",
            2.0,
            cores=6,
            threads=12,
            sockets=2,
        ),
        _benchmark("b", "AMD Ryzen 5 5600X 6-Core Processor", 7.0, cores=6, threads=12),
        _benchmark("c", "Intel Core i7-6700 CPU @ 3.40GHz", 18.0, cores=4, threads=8),
    ]

    rows, manifest = prepare_blender_performance(
        benchmarks,
        _fixture_catalog(),
        minimum_pilot_products=3,
        minimum_credible_products=10,
    )

    ryzen_3600 = next(
        row
        for row in rows
        if row["benchmark_hardware_names_json"] == '["AMD Ryzen 5 3600 6-Core Processor"]'
    )
    assert ryzen_3600["benchmark_observation_count"] == 1
    assert manifest["matching"]["counts"]["join_rejected_cpu_not_single_socket"] == 1


def test_source_promotion_blockers_accept_complete_hash_sample_and_full_catalog(
    tmp_path: Path,
) -> None:
    blender_dir = tmp_path / "blender"
    catalog_dir = tmp_path / "catalog"
    blender_dir.mkdir()
    catalog_dir.mkdir()
    blender_records = blender_dir / "records.jsonl"
    catalog_records = catalog_dir / "records.jsonl"
    blender_records.write_text("{}\n", encoding="utf-8")
    catalog_records.write_text("{}\n", encoding="utf-8")
    (blender_dir / "manifest.json").write_text(
        json.dumps(
            {
                "statistics": {
                    "selection": "hash_sample",
                    "scan_complete": True,
                    "sample_population": 1_000_000,
                }
            }
        ),
        encoding="utf-8",
    )
    (catalog_dir / "manifest.json").write_text(
        json.dumps({"statistics": {"selected_records": 25, "available_records": 25}}),
        encoding="utf-8",
    )

    assert source_promotion_blockers(blender_records, catalog_records) == []


def test_jsonl_record_stream_is_single_pass_and_counts_nonblank_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"record":1}\n\n{"record":2}\n', encoding="utf-8")

    stream = _JsonlRecordStream(path)

    assert list(stream) == [{"record": 1}, {"record": 2}]
    assert stream.records_read == 2
    with pytest.raises(RuntimeError, match="already consumed"):
        list(stream)


def test_jsonl_record_stream_refuses_an_oversized_source_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"record":"this is deliberately too long"}\n')

    stream = _JsonlRecordStream(path, maximum_record_bytes=12)

    with pytest.raises(MemoryError, match="record exceeds the 12-byte limit"):
        list(stream)


def test_preparation_memory_estimate_excludes_streamed_blender_file(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog-records.jsonl"
    catalog_path.write_bytes(b"x" * (1024 * 1024))

    estimate = estimate_blender_preparation_memory_mib(
        catalog_path,
        maximum_record_bytes=1024 * 1024,
        catalog_memory_expansion_factor=2,
        runtime_memory_mb=10,
    )

    # Two MiB reserves the bounded raw/decoded record buffer, 8 MiB reserves
    # SQLite's configured cache, and only the catalogue receives the 2x
    # expansion factor.
    assert estimate == pytest.approx(22.0)


def test_preparation_cli_enforces_the_host_memory_cap(tmp_path: Path) -> None:
    blender_path = tmp_path / "blender-records.jsonl"
    catalog_path = tmp_path / "catalog-records.jsonl"
    blender_path.write_text("{}\n", encoding="utf-8")
    catalog_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(MemoryError, match="projected used memory"):
        prepare_blender_main(
            [
                "--blender-records",
                str(blender_path),
                "--buildcores-records",
                str(catalog_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--max-host-used-gb",
                "0.001",
                "--minimum-free-memory-mb",
                "0",
            ]
        )


def test_preparation_releases_joined_raw_envelopes_while_streaming() -> None:
    """Large inputs must not retain every raw benchmark envelope in Python memory."""

    class TrackedData(dict[str, Any]):
        pass

    hardware = (
        ("AMD Ryzen 5 3600 6-Core Processor", 6, 12),
        ("AMD Ryzen 5 3600X 6-Core Processor", 6, 12),
        ("AMD Ryzen 5 5600X 6-Core Processor", 6, 12),
        ("Intel Core i7-6700 CPU @ 3.40GHz", 4, 8),
        ("Intel Core i7-6700K CPU @ 4.00GHz", 4, 8),
    )

    class StreamingBenchmarks:
        def __init__(self, count: int) -> None:
            self._count = count
            self._index = 0
            self._data_references: list[weakref.ReferenceType[TrackedData]] = []

        def __iter__(self) -> StreamingBenchmarks:
            return self

        def __next__(self) -> dict[str, Any]:
            if self._index >= self._count:
                raise StopIteration
            # The currently yielded envelope can still be referenced by the
            # consumer.  Earlier input data must be collectible before the
            # next source row arrives; retaining it would scale with raw JSONL
            # size rather than the selected output cohort.
            if len(self._data_references) > 1:
                gc.collect()
                retained = [reference for reference in self._data_references[:-1] if reference()]
                assert not retained
            name, cores, threads = hardware[self._index % len(hardware)]
            envelope = _benchmark(
                str(self._index),
                name,
                10.0 + (self._index % 5),
                cores=cores,
                threads=threads,
            )
            tracked_data = TrackedData(envelope["data"])
            envelope["data"] = tracked_data
            self._data_references.append(weakref.ref(tracked_data))
            self._index += 1
            return envelope

    benchmarks = StreamingBenchmarks(count=25)

    rows, manifest = prepare_blender_performance(
        benchmarks,
        _fixture_catalog(),
        category_filter="cpu",
        minimum_pilot_products=3,
        minimum_credible_products=10,
    )

    assert len(rows) == 5
    assert manifest["selected_cohort"]["joined_observations"] == 25


def test_preparation_cli_streams_sources_and_records_exact_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blender_path = tmp_path / "blender-records.jsonl"
    catalog_path = tmp_path / "catalog-records.jsonl"
    output_dir = tmp_path / "prepared"
    benchmarks = [
        _benchmark("a", "AMD Ryzen 5 3600 6-Core Processor", 10.0, cores=6, threads=12),
        _benchmark("b", "AMD Ryzen 5 3600X 6-Core Processor", 9.0, cores=6, threads=12),
        _benchmark("c", "AMD Ryzen 5 5600X 6-Core Processor", 7.0, cores=6, threads=12),
        _benchmark("d", "Intel Core i7-6700 CPU @ 3.40GHz", 18.0, cores=4, threads=8),
        _benchmark("e", "Intel Core i7-6700K CPU @ 4.00GHz", 15.0, cores=4, threads=8),
    ]
    blender_path.write_text(
        "\n".join(json.dumps(row) for row in benchmarks) + "\n",
        encoding="utf-8",
    )
    catalog_path.write_text(
        "\n".join(json.dumps(row) for row in _fixture_catalog()) + "\n",
        encoding="utf-8",
    )

    assert (
        prepare_blender_main(
            [
                "--blender-records",
                str(blender_path),
                "--buildcores-records",
                str(catalog_path),
                "--output-dir",
                str(output_dir),
                "--category",
                "cpu",
                "--minimum-pilot-products",
                "3",
                "--minimum-credible-products",
                "10",
                "--max-host-used-gb",
                "1024",
                "--minimum-free-memory-mb",
                "0",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"]["blender_records"]["rows"] == len(benchmarks)
    assert manifest["sources"]["buildcores_records"]["rows"] == len(_fixture_catalog())
    assert manifest["output"]["rows"] == 5
    assert manifest["sources"]["blender_records"]["path"] == "<external>/blender-records.jsonl"
    assert manifest["sources"]["buildcores_records"]["path"] == "<external>/catalog-records.jsonl"
    assert manifest["output"]["csv"] == "<external>/blender_performance.csv"
    assert manifest["bounded_memory"]["maximum_record_bytes"] == 1_000_000
    assert manifest["bounded_memory"]["sqlite_cache_kib"] == 8 * 1024
