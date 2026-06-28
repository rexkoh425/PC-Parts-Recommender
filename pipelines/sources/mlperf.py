"""Official MLPerf Inference v6.0 summary-results adapter."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pc_build_recommender.domain.enums import WorkloadLabel
from pc_build_recommender.domain.models import BenchmarkResult
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION, stable_identifier
from pipelines.sources.base import (
    ParsedBatch,
    RawSnapshot,
    fetch_http_snapshot,
    rejected_record,
    sha256_bytes,
    snapshot_local_file,
)

MLPERF_RESULTS_COMMIT = "4d3916ac9cf474b679cdfcf492d43a0559418ad1"
MLPERF_SUMMARY_URL = (
    "https://raw.githubusercontent.com/mlcommons/inference_results_v6.0/"
    f"{MLPERF_RESULTS_COMMIT}/summary_results.json"
)
MLPERF_LICENSE_NOTE = (
    "MLCommons inference_results_v6.0 repository is Apache-2.0. Results are system-level; "
    "only available one-node, one-accelerator rows are eligible for component attribution."
)
MLPERF_PARSER_VERSION = "mlperf-inference-v6-summary-v1"
MLPERF_RELEASED_AT = datetime(2026, 4, 3, 22, 36, 55, tzinfo=UTC)


class MLPerfInferenceAdapter:
    """Fetch and normalise the pinned official v6.0 result summary."""

    def __init__(self, *, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def fetch(self, *, summary_path: str | Path | None = None) -> RawSnapshot:
        if summary_path is not None:
            return snapshot_local_file(
                source_name="mlperf_inference_v6",
                source_url=MLPERF_SUMMARY_URL,
                source_type="benchmark",
                source_path=summary_path,
                raw_root=self.raw_root,
                parser_version=MLPERF_PARSER_VERSION,
                licence_or_access_note=MLPERF_LICENSE_NOTE,
                suffix=".json",
                media_type="application/json",
            )
        return fetch_http_snapshot(
            source_name="mlperf_inference_v6",
            source_url=MLPERF_SUMMARY_URL,
            source_type="benchmark",
            raw_root=self.raw_root,
            parser_version=MLPERF_PARSER_VERSION,
            licence_or_access_note=MLPERF_LICENSE_NOTE,
            suffix=".json",
            maximum_bytes=10 * 1024 * 1024,
        )

    def parse(
        self,
        snapshot: RawSnapshot,
        *,
        available_only: bool = False,
        single_accelerator_only: bool = False,
        max_records: int | None = None,
    ) -> ParsedBatch:
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive or None")
        with snapshot.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError("MLPerf summary root must be a list")

        batch = ParsedBatch(
            source_name=snapshot.source_name,
            snapshot_sha256=snapshot.content_sha256,
        )
        eligible_component_rows = 0
        availability_counts: dict[str, int] = {}
        accelerator_count_buckets: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                batch.rejected.append(
                    rejected_record(str(index), "result_not_object")
                )
                continue
            availability = str(row.get("Availability", "unknown")).strip().lower()
            total_accelerators = self._nonnegative_int(row.get("Total Accelerators"))
            if available_only and availability != "available":
                continue
            if single_accelerator_only and total_accelerators != 1:
                continue
            try:
                normalised = self._normalise_result(row=row, index=index, snapshot=snapshot)
            except (TypeError, ValueError) as exc:
                batch.rejected.append(
                    rejected_record(
                        str(row.get("ID", index)),
                        "invalid_mlperf_result",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            batch.records.append(normalised)
            metadata = normalised["normalisation_metadata"]
            if metadata["component_model_eligible"]:
                eligible_component_rows += 1
            availability_counts[availability] = availability_counts.get(availability, 0) + 1
            bucket = str(total_accelerators if total_accelerators is not None else "unknown")
            accelerator_count_buckets[bucket] = accelerator_count_buckets.get(bucket, 0) + 1
            model = str(normalised["data"]["preset"])
            model_counts[model] = model_counts.get(model, 0) + 1
            if max_records is not None and batch.accepted_count >= max_records:
                break

        batch.statistics = {
            "repository_commit": MLPERF_RESULTS_COMMIT,
            "source_rows": len(payload),
            "available_only": available_only,
            "single_accelerator_only": single_accelerator_only,
            "component_model_eligible_rows": eligible_component_rows,
            "availability_counts": dict(sorted(availability_counts.items())),
            "total_accelerator_counts": dict(sorted(accelerator_count_buckets.items())),
            "model_counts": dict(sorted(model_counts.items())),
        }
        return batch

    @classmethod
    def _normalise_result(
        cls, *, row: dict[str, Any], index: int, snapshot: RawSnapshot
    ) -> dict[str, Any]:
        result_id = str(row.get("ID", "")).strip()
        if not result_id:
            raise ValueError("missing ID")
        score = cls._positive_float(row.get("Performance_Result"))
        if score is None:
            raise ValueError("missing positive Performance_Result")
        units = str(row.get("Performance_Units", "")).strip()
        if not units:
            raise ValueError("missing Performance_Units")
        model = str(row.get("UsedModel") or row.get("Model") or "").strip()
        scenario = str(row.get("Scenario", "")).strip()
        if not model or not scenario:
            raise ValueError("model and scenario are required")
        total_accelerators = cls._nonnegative_int(row.get("Total Accelerators"))
        nodes = cls._positive_int(row.get("Nodes")) or 1
        accelerator = cls._clean_hardware_name(row.get("Accelerator"))
        processor = cls._clean_hardware_name(row.get("Processor"))
        if total_accelerators == 0:
            hardware_name = processor
            hardware_role = "cpu"
        else:
            hardware_name = accelerator
            hardware_role = "accelerator"
        if hardware_name is None:
            raise ValueError("missing attributable hardware name")
        availability = str(row.get("Availability", "unknown")).strip().lower()
        error_count = cls._nonnegative_int(row.get("errors")) or 0
        component_model_eligible = (
            availability == "available"
            and nodes == 1
            and total_accelerators == 1
            and error_count == 0
        )
        location = str(row.get("Location", "")).strip()
        benchmark_id = stable_identifier(
            "bench_mlperf",
            result_id,
            location,
            model,
            scenario,
            units,
        )
        external_product_id = stable_identifier(
            "ext_mlperf_hardware", hardware_role, hardware_name
        )
        source_url = str(row.get("Details", "")).strip() or (
            "https://github.com/mlcommons/inference_results_v6.0"
        )
        benchmark = BenchmarkResult(
            benchmark_id=benchmark_id,
            product_id=external_product_id,
            workload=WorkloadLabel.LOCAL_AI,
            benchmark_name="MLPerf Inference",
            benchmark_version=str(row.get("version") or "v6.0").strip(),
            score=score,
            unit=units,
            higher_is_better=(
                "latency" not in units.casefold() and not units.casefold().endswith("ms")
            ),
            preset=model,
            operating_system=str(row.get("operating_system", "")).strip() or None,
            source_url=source_url,
            observed_at=MLPERF_RELEASED_AT,
        )
        raw_record_bytes = json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return {
            "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
            "record_type": "benchmark_observation",
            "source_record_id": f"{result_id}:{index}",
            "archive_snapshot_sha256": snapshot.content_sha256,
            "raw_record_sha256": sha256_bytes(raw_record_bytes),
            "training_eligible": True,
            "published_claims_eligible": True,
            "value_kind": "observed",
            "provenance": {
                "source_name": snapshot.source_name,
                "source_url": source_url,
                "source_type": "benchmark",
                "retrieved_at": snapshot.retrieved_at.isoformat(),
                "parser_version": snapshot.parser_version,
                "licence_or_access_note": snapshot.licence_or_access_note,
                "extraction_confidence": 0.95,
            },
            "normalisation_metadata": {
                "result_id": result_id,
                "submitter": row.get("Submitter"),
                "availability": availability,
                "division": row.get("Category"),
                "suite": row.get("Suite"),
                "scenario": scenario,
                "system": row.get("System"),
                "platform": row.get("Platform"),
                "hardware_name": hardware_name,
                "hardware_role": hardware_role,
                "processor": processor,
                "accelerator": accelerator,
                "nodes": nodes,
                "total_accelerators": total_accelerators,
                "component_model_eligible": component_model_eligible,
                "component_attribution_reason": (
                    "available single-node single-accelerator result"
                    if component_model_eligible
                    else (
                        "system-level result; preserve configuration and do not "
                        "attribute to one component"
                    )
                ),
                "software": row.get("Software"),
                "weight_data_types": row.get("weight_data_types"),
                "accuracy": row.get("Accuracy"),
                "has_power": row.get("has_power"),
                "observed_at_basis": "pinned results repository commit timestamp",
            },
            "data": benchmark.model_dump(mode="json"),
        }

    @staticmethod
    def _positive_float(value: object) -> float | None:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _positive_int(cls, value: object) -> int | None:
        parsed = cls._positive_float(value)
        return int(parsed) if parsed is not None and parsed.is_integer() else None

    @staticmethod
    def _nonnegative_int(value: object) -> int | None:
        if not isinstance(value, (str, bytes, bytearray, int, float)):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    def _clean_hardware_name(value: object) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        text = re.sub(r"\s*\(x\d+\)\s*$", "", text, flags=re.IGNORECASE)
        return text or None
