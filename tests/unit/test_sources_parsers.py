from __future__ import annotations

import csv
import json
import zipfile
from datetime import UTC, date, datetime

import pytest
from pipelines.sources.blender import BlenderOpenDataAdapter
from pipelines.sources.buildcores import BuildCoresOpenDBAdapter
from pipelines.sources.mlperf import MLPerfInferenceAdapter
from pipelines.sources.retailer_csv import ConsentedRetailerCSVAdapter, RetailerFeedPolicy
from pipelines.sources.rights import DataUse, DataUseRights, require_data_use


def _retailer_rights(
    *,
    may_cache: bool = True,
    may_train: bool = False,
) -> DataUseRights:
    return DataUseRights(
        contract_reference="agreement-2026-01",
        contract_version_url="contract://fixture/agreement-2026-01",
        consent_effective_on=date(2026, 1, 1),
        consent_expires_on=None,
        retention_days=365,
        deletion_required_on_termination=True,
        deletion_sla_days=30,
        territories=("SG",),
        may_display=True,
        may_cache=may_cache,
        may_store_history=True,
        may_redistribute=False,
        may_embed=False,
        may_train=may_train,
        may_derive=True,
    )


def test_buildcores_cpu_is_normalised_with_stable_provenance(tmp_path) -> None:
    archive_path = tmp_path / "buildcores.zip"
    payload = {
        "opendb_id": "11111111-1111-4111-8111-111111111111",
        "cores": {"total": 8, "threads": 16},
        "clocks": {"performance": {"base": 4.2, "boost": 5.0}},
        "specifications": {
            "integratedGraphics": {"model": "None"},
            "memory": {"maxSupport": 192},
            "tdp": 65,
            "ppt": 88,
            "includesCooler": False,
        },
        "socket": "AM5",
        "series": "Ryzen 7000",
        "microarchitecture": "Zen 4",
        "metadata": {
            "name": "AMD Ryzen 7 Fixture",
            "manufacturer": "AMD",
            "variant": "Fixture",
            "part_numbers": ["AMD Ryzen 7 Fixture", "100-FIXTURE"],
            "releaseYear": 2024,
        },
        "general_product_information": {"manufacturer_url": "https://example.test/cpu"},
    }
    member = "buildcores-open-db-fixture/open-db/CPU/11111111-1111-4111-8111-111111111111.json"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, json.dumps(payload))

    adapter = BuildCoresOpenDBAdapter(raw_root=tmp_path / "raw")
    snapshot = adapter.fetch(archive_path=archive_path)
    batch = adapter.parse(snapshot, categories=("CPU",), per_category_limit=1)

    assert batch.accepted_count == 1
    assert batch.rejected_count == 0
    record = batch.records[0]
    product = record["data"]
    assert product["product_id"] == "prod_buildcores_11111111-1111-4111-8111-111111111111"
    assert product["category"] == "cpu"
    assert product["manufacturer_part_number"] == "100-FIXTURE"
    assert product["category_attributes"]["core_count"] == 8
    assert product["category_attributes"]["integrated_graphics"] is None
    assert product["provenance"][0]["raw_content_hash"] != snapshot.content_sha256


def test_blender_parser_rejects_ambiguous_multi_device_observation(tmp_path) -> None:
    archive_path = tmp_path / "blender.zip"
    valid = {
        "benchmark_script": {"label": "fixture-script"},
        "blender_version": {"version": "4.3.0", "build_hash": "abc"},
        "device_info": {
            "device_type": "CPU",
            "compute_devices": [{"name": "Fixture CPU", "type": "CPU"}],
            "num_cpu_threads": 16,
        },
        "scene": {"label": "monster", "checksum": "scene"},
        "stats": {"samples_per_minute": 123.4},
        "system_info": {"system": "Linux", "num_cpu_cores": 8, "num_cpu_sockets": 1},
        "timestamp": "2026-01-02T03:04:05+00:00",
    }
    ambiguous = {
        **valid,
        "device_info": {
            "device_type": "CUDA",
            "compute_devices": [
                {"name": "GPU A", "type": "CUDA"},
                {"name": "GPU B", "type": "CUDA"},
            ],
        },
    }
    submission = {
        "id": "submission-fixture",
        "created_at": "2026-01-02T03:04:05+00:00",
        "data": [valid, ambiguous],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("LICENSE.txt", "CC0")
        archive.writestr("fixture.jsonl", json.dumps(submission) + "\n")

    adapter = BlenderOpenDataAdapter(raw_root=tmp_path / "raw")
    snapshot = adapter.fetch(archive_path=archive_path)
    batch = adapter.parse(snapshot, max_observations=2, max_submissions_scan=1)

    assert batch.accepted_count == 1
    assert batch.rejected_count == 1
    assert batch.records[0]["normalisation_metadata"]["hardware_name"] == "Fixture CPU"
    assert batch.records[0]["data"]["score"] == 123.4
    assert batch.records[0]["value_kind"] == "observed"


def test_blender_hash_sample_is_deterministic_and_scans_the_population(tmp_path) -> None:
    archive_path = tmp_path / "blender-sample.zip"
    submissions = []
    for index in range(12):
        submissions.append(
            {
                "id": f"submission-{index}",
                "created_at": "2026-01-02T03:04:05+00:00",
                "data": [
                    {
                        "benchmark_script": {"label": "fixture-script"},
                        "blender_version": {"version": "4.3.0", "build_hash": "abc"},
                        "device_info": {
                            "device_type": "CPU",
                            "compute_devices": [{"name": f"Fixture CPU {index}", "type": "CPU"}],
                            "num_cpu_threads": 16,
                        },
                        "scene": {"label": "monster", "checksum": "scene"},
                        "stats": {"samples_per_minute": float(index + 1)},
                        "system_info": {
                            "system": "Linux",
                            "num_cpu_cores": 8,
                            "num_cpu_sockets": 1,
                        },
                        "timestamp": "2026-01-02T03:04:05+00:00",
                    }
                ],
            }
        )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("LICENSE.txt", "CC0")
        archive.writestr(
            "fixture.jsonl",
            "\n".join(json.dumps(submission) for submission in submissions) + "\n",
        )

    adapter = BlenderOpenDataAdapter(raw_root=tmp_path / "raw")
    snapshot = adapter.fetch(archive_path=archive_path)
    first = adapter.parse(
        snapshot,
        max_observations=4,
        selection="hash_sample",
        sample_seed="fixture-seed",
    )
    second = adapter.parse(
        snapshot,
        max_observations=4,
        selection="hash_sample",
        sample_seed="fixture-seed",
    )

    first_ids = [record["source_record_id"] for record in first.records]
    second_ids = [record["source_record_id"] for record in second.records]
    assert first_ids == second_ids
    assert len(first_ids) == 4
    assert first.statistics["submissions_scanned"] == 12
    assert first.statistics["valid_observations_seen"] == 12
    assert first.statistics["sample_population"] == 12
    assert first.statistics["selection"] == "hash_sample"


def test_blender_parser_reports_uncapped_rejections_by_input_scope(tmp_path) -> None:
    archive_path = tmp_path / "blender-rejections.zip"
    valid_observation = {
        "benchmark_script": {"label": "fixture-script"},
        "blender_version": {"version": "4.3.0", "build_hash": "abc"},
        "device_info": {
            "device_type": "CPU",
            "compute_devices": [{"name": "Fixture CPU", "type": "CPU"}],
            "num_cpu_threads": 16,
        },
        "scene": {"label": "monster", "checksum": "scene"},
        "stats": {"samples_per_minute": 123.4},
        "system_info": {"system": "Linux", "num_cpu_cores": 8},
        "timestamp": "2026-01-02T03:04:05+00:00",
    }
    submissions = [
        "{invalid-json",
        json.dumps([]),
        json.dumps(
            {
                "id": "missing-observations",
                "created_at": "2026-01-02T03:04:05+00:00",
                "data": {},
            }
        ),
        json.dumps(
            {
                "id": "mixed-observations",
                "created_at": "2026-01-02T03:04:05+00:00",
                "data": [None, {}, valid_observation],
            }
        ),
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("LICENSE.txt", "CC0")
        archive.writestr("fixture.jsonl", "\n".join(submissions) + "\n")

    adapter = BlenderOpenDataAdapter(raw_root=tmp_path / "raw")
    snapshot = adapter.fetch(archive_path=archive_path)
    batch = adapter.parse(
        snapshot,
        max_observations=10,
        maximum_recorded_rejections=2,
    )

    assert batch.accepted_count == 1
    assert batch.rejected_count == 2
    assert batch.statistics["submissions_scanned"] == 4
    assert batch.statistics["submission_rejection_denominator"] == 4
    assert batch.statistics["submission_rejection_count"] == 3
    assert batch.statistics["submission_rejection_rate"] == pytest.approx(3 / 4)
    assert batch.statistics["submission_rejection_counts"] == {
        "invalid_json": 1,
        "missing_observation_list": 1,
        "submission_not_object": 1,
    }
    assert batch.statistics["observations_seen"] == 3
    assert batch.statistics["observation_rejection_denominator"] == 3
    assert batch.statistics["observation_rejection_count"] == 2
    assert batch.statistics["observation_rejection_rate"] == pytest.approx(2 / 3)
    assert batch.statistics["observation_rejection_counts"] == {
        "invalid_benchmark_observation": 1,
        "observation_not_object": 1,
    }
    assert batch.statistics["total_rejections"] == 5
    assert batch.statistics["recorded_rejections"] == 2
    assert batch.statistics["recorded_rejections_truncated"] == 3
    assert batch.statistics["rejections_truncated"] is True
    assert sum(batch.statistics["rejection_counts"].values()) == 5


def test_mlperf_marks_only_single_accelerator_result_component_eligible(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"

    def result(result_id: str, accelerator_count: int) -> dict[str, object]:
        return {
            "ID": result_id,
            "Submitter": "Fixture",
            "Availability": "available",
            "Category": "closed",
            "Suite": "edge",
            "System": f"Fixture system {accelerator_count}",
            "Platform": "fixture",
            "UsedModel": "llama-fixture",
            "Scenario": "Offline",
            "Nodes": 1,
            "Processor": "Fixture CPU",
            "Accelerator": "Fixture GPU",
            "Total Accelerators": accelerator_count,
            "Location": f"./{result_id}",
            "Software": "Fixture runtime",
            "operating_system": "Linux",
            "errors": 0,
            "version": "v6.0",
            "Details": "https://example.test/result",
            "Performance_Result": 42.0,
            "Performance_Units": "Tokens/s",
        }

    summary_path.write_text(
        json.dumps([result("fixture-one", 1), result("fixture-eight", 8)]),
        encoding="utf-8",
    )
    adapter = MLPerfInferenceAdapter(raw_root=tmp_path / "raw")
    snapshot = adapter.fetch(summary_path=summary_path)
    batch = adapter.parse(snapshot)

    assert batch.accepted_count == 2
    assert batch.rejected_count == 0
    assert batch.statistics["component_model_eligible_rows"] == 1
    eligibility = [
        row["normalisation_metadata"]["component_model_eligible"] for row in batch.records
    ]
    assert eligibility == [True, False]


def test_consented_retailer_csv_requires_valid_new_listing(tmp_path) -> None:
    csv_path = tmp_path / "retailer.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_listing_id",
                "title",
                "currency",
                "base_price",
                "shipping_price",
                "stock_status",
                "condition",
                "listing_url",
                "observed_at",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_listing_id": "valid-1",
                "title": "Fixture GPU",
                "currency": "SGD",
                "base_price": "599.90",
                "shipping_price": "0",
                "stock_status": "in_stock",
                "condition": "new",
                "listing_url": "https://example.test/valid-1",
                "observed_at": datetime.now(UTC).isoformat(),
            }
        )
        writer.writerow(
            {
                "source_listing_id": "used-1",
                "title": "Used Fixture GPU",
                "currency": "SGD",
                "base_price": "300",
                "stock_status": "in_stock",
                "condition": "used",
                "listing_url": "https://example.test/used-1",
            }
        )
    policy = RetailerFeedPolicy(
        retailer="Fixture Retailer",
        feed_id="fixture",
        source_url="controlled://fixture-feed",
        licence_or_access_note="Fixture consent permits application display only.",
        rights=_retailer_rights(),
    )
    adapter = ConsentedRetailerCSVAdapter(raw_root=tmp_path / "raw", policy=policy)
    batch = adapter.parse(adapter.fetch(csv_path=csv_path))

    assert batch.accepted_count == 1
    assert batch.rejected_count == 1
    assert batch.records[0]["training_eligible"] is False
    assert batch.records[0]["data_use_rights"]["may_display"] is True
    assert batch.records[0]["data_use_rights"]["may_embed"] is False
    assert batch.records[0]["data"]["listing"]["stock_status"] == "in_stock"
    assert batch.records[0]["data"]["listing"]["base_price"] == "599.90"
    require_data_use(batch.records[0], DataUse.DISPLAY)
    with pytest.raises(PermissionError, match="does not permit embed"):
        require_data_use(batch.records[0], DataUse.EMBED)


def test_retailer_policy_fails_closed_without_operational_rights() -> None:
    with pytest.raises(PermissionError, match="does not permit cache"):
        RetailerFeedPolicy(
            retailer="Fixture Retailer",
            feed_id="fixture",
            source_url="controlled://fixture-feed",
            licence_or_access_note="Fixture policy.",
            rights=_retailer_rights(may_cache=False),
        )

    with pytest.raises(PermissionError, match="does not permit train"):
        RetailerFeedPolicy(
            retailer="Fixture Retailer",
            feed_id="fixture",
            source_url="controlled://fixture-feed",
            licence_or_access_note="Fixture policy.",
            rights=_retailer_rights(),
            training_eligible=True,
        )


def test_retailer_policy_mapping_requires_every_right_and_lifecycle_term() -> None:
    rights_payload = _retailer_rights().to_dict()
    payload = {
        "retailer": "Fixture Retailer",
        "feed_id": "fixture",
        "source_url": "controlled://fixture-feed",
        "licence_or_access_note": "Fixture policy.",
        "rights": rights_payload,
    }
    policy = RetailerFeedPolicy.from_mapping(payload)
    assert policy.consent_reference == "agreement-2026-01"

    del rights_payload["deletion_sla_days"]
    with pytest.raises(ValueError, match="missing fields"):
        RetailerFeedPolicy.from_mapping(payload)
