"""Blender Open Data benchmark snapshot adapter."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pc_build_recommender.domain.enums import WorkloadLabel
from pc_build_recommender.domain.models import BenchmarkResult
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION, stable_identifier
from pipelines.sources.base import (
    ParseResult,
    FetchedSnapshot,
    fetch_http_snapshot,
    rejected_record,
    sha256_bytes,
    snapshot_local_file,
)

BLENDER_SNAPSHOT_URL = "https://opendata.blender.org/snapshots/opendata-latest.zip"
BLENDER_LICENSE_NOTE = (
    "Blender Open Data benchmark results are dedicated to the public domain under CC0 1.0; "
    "the snapshot embeds LICENSE.txt. Preserve benchmark version, scene, device backend, and OS."
)
BLENDER_PARSER_VERSION = "blender-open-data-v1"
_SUPPORTED_DEVICE_TYPES = {"CPU", "CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"}
BlenderSelection = Literal["head", "hash_sample"]
BlenderRejectionScope = Literal["submission", "observation"]


class BlenderOpenDataAdapter:
    """Fetch and stream benchmark observations without expanding the 1.8 GB JSONL file."""

    def __init__(self, *, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def fetch(self, *, archive_path: str | Path | None = None) -> FetchedSnapshot:
        if archive_path is not None:
            return snapshot_local_file(
                source_name="blender_open_data",
                source_url=BLENDER_SNAPSHOT_URL,
                source_type="benchmark",
                source_path=archive_path,
                raw_root=self.raw_root,
                parser_version=BLENDER_PARSER_VERSION,
                licence_or_access_note=BLENDER_LICENSE_NOTE,
                suffix=".zip",
                media_type="application/zip",
            )
        return fetch_http_snapshot(
            source_name="blender_open_data",
            source_url=BLENDER_SNAPSHOT_URL,
            source_type="benchmark",
            raw_root=self.raw_root,
            parser_version=BLENDER_PARSER_VERSION,
            licence_or_access_note=BLENDER_LICENSE_NOTE,
            suffix=".zip",
            maximum_bytes=250 * 1024 * 1024,
            timeout_seconds=300,
        )

    def parse(
        self,
        snapshot: FetchedSnapshot,
        *,
        max_observations: int = 3_000,
        max_submissions_scan: int | None = None,
        minimum_created_at: datetime | None = None,
        maximum_recorded_rejections: int = 1_000,
        selection: BlenderSelection = "head",
        sample_seed: str = "buildsignal-blender-v1",
    ) -> ParseResult:
        if max_observations <= 0:
            raise ValueError("max_observations must be positive")
        if max_submissions_scan is not None and max_submissions_scan <= 0:
            raise ValueError("max_submissions_scan must be positive or None")
        if minimum_created_at is not None and minimum_created_at.tzinfo is None:
            raise ValueError("minimum_created_at must be timezone aware")
        if maximum_recorded_rejections < 0:
            raise ValueError("maximum_recorded_rejections must not be negative")
        if selection not in {"head", "hash_sample"}:
            raise ValueError("selection must be 'head' or 'hash_sample'")
        if not sample_seed.strip():
            raise ValueError("sample_seed must not be blank")

        batch = ParseResult(
            source_name=snapshot.source_name,
            snapshot_sha256=snapshot.content_sha256,
        )
        rejection_counts_by_scope: dict[BlenderRejectionScope, dict[str, int]] = {
            "submission": {},
            "observation": {},
        }
        submissions_scanned = 0
        observations_seen = 0
        valid_observations = 0
        sample_heap: list[tuple[int, str, dict[str, Any]]] = []
        with zipfile.ZipFile(snapshot.path) as archive:
            jsonl_entries = [
                entry for entry in archive.infolist() if entry.filename.lower().endswith(".jsonl")
            ]
            if len(jsonl_entries) != 1:
                raise ValueError(
                    f"expected exactly one Blender JSONL member, found {len(jsonl_entries)}"
                )
            with archive.open(jsonl_entries[0]) as binary_stream:
                import io

                with io.TextIOWrapper(binary_stream, encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if (
                            max_submissions_scan is not None
                            and submissions_scanned >= max_submissions_scan
                        ):
                            break
                        submissions_scanned += 1
                        try:
                            submission = json.loads(line)
                        except json.JSONDecodeError as exc:
                            self._record_rejection(
                                batch,
                                rejection_counts_by_scope,
                                maximum_recorded_rejections,
                                str(line_number),
                                "invalid_json",
                                scope="submission",
                                error=str(exc),
                            )
                            continue
                        if not isinstance(submission, dict):
                            self._record_rejection(
                                batch,
                                rejection_counts_by_scope,
                                maximum_recorded_rejections,
                                str(line_number),
                                "submission_not_object",
                                scope="submission",
                            )
                            continue
                        created_at = self._parse_datetime(submission.get("created_at"))
                        if minimum_created_at is not None and (
                            created_at is None or created_at < minimum_created_at
                        ):
                            continue
                        data = submission.get("data")
                        if not isinstance(data, list):
                            self._record_rejection(
                                batch,
                                rejection_counts_by_scope,
                                maximum_recorded_rejections,
                                str(submission.get("id", line_number)),
                                "missing_observation_list",
                                scope="submission",
                            )
                            continue
                        for observation_index, observation in enumerate(data):
                            observations_seen += 1
                            record_id = f"{submission.get('id', line_number)}:{observation_index}"
                            if not isinstance(observation, dict):
                                self._record_rejection(
                                    batch,
                                    rejection_counts_by_scope,
                                    maximum_recorded_rejections,
                                    record_id,
                                    "observation_not_object",
                                    scope="observation",
                                )
                                continue
                            try:
                                normalised = self._normalise_observation(
                                    submission=submission,
                                    observation=observation,
                                    observation_index=observation_index,
                                    snapshot=snapshot,
                                )
                            except (TypeError, ValueError) as exc:
                                self._record_rejection(
                                    batch,
                                    rejection_counts_by_scope,
                                    maximum_recorded_rejections,
                                    record_id,
                                    "invalid_benchmark_observation",
                                    scope="observation",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                continue
                            valid_observations += 1
                            if selection == "head":
                                batch.records.append(normalised)
                            else:
                                priority = self._sample_priority(
                                    sample_seed=sample_seed,
                                    source_record_id=str(normalised["source_record_id"]),
                                )
                                heap_item = (
                                    -priority,
                                    str(normalised["source_record_id"]),
                                    normalised,
                                )
                                if len(sample_heap) < max_observations:
                                    heapq.heappush(sample_heap, heap_item)
                                elif priority < -sample_heap[0][0]:
                                    heapq.heapreplace(sample_heap, heap_item)
                            if selection == "head" and batch.accepted_count >= max_observations:
                                break
                        if selection == "head" and batch.accepted_count >= max_observations:
                            break

        if selection == "hash_sample":
            batch.records = [
                item[2] for item in sorted(sample_heap, key=lambda item: (-item[0], item[1]))
            ]

        accepted_by_device: dict[str, int] = {}
        versions: dict[str, int] = {}
        for record in batch.records:
            metadata = record["normalisation_metadata"]
            device_type = str(metadata["device_type"])
            version = str(record["data"]["benchmark_version"])
            accepted_by_device[device_type] = accepted_by_device.get(device_type, 0) + 1
            versions[version] = versions.get(version, 0) + 1

        submission_rejection_counts = rejection_counts_by_scope["submission"]
        observation_rejection_counts = rejection_counts_by_scope["observation"]
        submission_rejection_count = sum(submission_rejection_counts.values())
        observation_rejection_count = sum(observation_rejection_counts.values())
        total_rejections = submission_rejection_count + observation_rejection_count
        rejection_counts = {
            reason: submission_rejection_counts.get(reason, 0)
            + observation_rejection_counts.get(reason, 0)
            for reason in submission_rejection_counts.keys() | observation_rejection_counts.keys()
        }
        unrecorded_rejections = total_rejections - batch.rejected_count

        batch.statistics = {
            "submissions_scanned": submissions_scanned,
            "observations_seen": observations_seen,
            "valid_observations_seen": valid_observations,
            "accepted_by_device": dict(sorted(accepted_by_device.items())),
            "benchmark_versions": dict(sorted(versions.items())),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "submission_rejection_counts": dict(sorted(submission_rejection_counts.items())),
            "observation_rejection_counts": dict(sorted(observation_rejection_counts.items())),
            "submission_rejection_count": submission_rejection_count,
            "submission_rejection_denominator": submissions_scanned,
            "submission_rejection_rate": (
                submission_rejection_count / submissions_scanned if submissions_scanned else 0.0
            ),
            "observation_rejection_count": observation_rejection_count,
            "observation_rejection_denominator": observations_seen,
            "observation_rejection_rate": (
                observation_rejection_count / observations_seen if observations_seen else 0.0
            ),
            "total_rejections": total_rejections,
            "recorded_rejections": batch.rejected_count,
            "recorded_rejections_truncated": unrecorded_rejections,
            "rejections_truncated": unrecorded_rejections > 0,
            "selection": selection,
            "sample_seed": sample_seed if selection == "hash_sample" else None,
            "sample_population": valid_observations,
            "scan_complete": max_submissions_scan is None,
        }
        return batch

    @staticmethod
    def _sample_priority(*, sample_seed: str, source_record_id: str) -> int:
        """Return a stable content-independent priority for bounded uniform sampling."""

        digest = hashlib.sha256(f"{sample_seed}\0{source_record_id}".encode()).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _record_rejection(
        batch: ParseResult,
        counts_by_scope: dict[BlenderRejectionScope, dict[str, int]],
        maximum_recorded: int,
        record_id: str,
        reason: str,
        *,
        scope: BlenderRejectionScope,
        **details: object,
    ) -> None:
        counts = counts_by_scope[scope]
        counts[reason] = counts.get(reason, 0) + 1
        if batch.rejected_count < maximum_recorded:
            batch.rejected.append(rejected_record(record_id, reason, **details))

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else None

    @classmethod
    def _normalise_observation(
        cls,
        *,
        submission: dict[str, Any],
        observation: dict[str, Any],
        observation_index: int,
        snapshot: FetchedSnapshot,
    ) -> dict[str, Any]:
        submission_id = str(submission.get("id", "")).strip()
        if not submission_id:
            raise ValueError("missing submission id")
        device_info = cls._mapping(observation.get("device_info"))
        device_type = str(device_info.get("device_type", "")).strip().upper()
        if device_type not in _SUPPORTED_DEVICE_TYPES:
            raise ValueError(f"unsupported device type: {device_type or 'missing'}")
        compute_devices = device_info.get("compute_devices")
        if not isinstance(compute_devices, list):
            raise ValueError("missing compute_devices")
        names = []
        for device in compute_devices:
            item = cls._mapping(device)
            item_type = str(item.get("type", "")).strip().upper()
            if device_type == "CPU" and item_type != "CPU":
                continue
            if device_type != "CPU" and item_type == "CPU":
                continue
            name = cls._normalise_hardware_name(item.get("name"))
            if name:
                names.append(name)
        unique_names = list(dict.fromkeys(names))
        if len(unique_names) != 1:
            raise ValueError(f"ambiguous compute device names: {unique_names}")
        hardware_name = unique_names[0]

        stats = cls._mapping(observation.get("stats"))
        score, unit, higher_is_better, score_field = cls._score(stats)
        scene = cls._mapping(observation.get("scene"))
        scene_label = str(scene.get("label", "unknown")).strip() or "unknown"
        blender_version = cls._mapping(observation.get("blender_version"))
        version = str(
            blender_version.get("version") or blender_version.get("label") or "unknown"
        ).strip()
        observed_at = cls._parse_datetime(observation.get("timestamp"))
        if observed_at is None:
            observed_at = cls._parse_datetime(submission.get("created_at"))
        if observed_at is None:
            raise ValueError("missing observation timestamp")
        system_info = cls._mapping(observation.get("system_info"))
        benchmark_id = stable_identifier(
            "bench_blender",
            submission_id,
            observation_index,
            device_type,
            scene_label,
            observed_at.isoformat(),
        )
        external_product_id = stable_identifier("ext_blender_hardware", device_type, hardware_name)
        benchmark = BenchmarkResult(
            benchmark_id=benchmark_id,
            product_id=external_product_id,
            workload=WorkloadLabel.CONTENT_CREATION,
            benchmark_name="Blender Open Data",
            benchmark_version=version,
            score=score,
            unit=unit,
            higher_is_better=higher_is_better,
            preset=scene_label,
            operating_system=str(system_info.get("system", "")).strip() or None,
            source_url=f"https://opendata.blender.org/benchmarks/{submission_id}/",
            observed_at=observed_at,
        )
        raw_record_bytes = json.dumps(
            observation,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return {
            "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
            "record_type": "benchmark_observation",
            "source_record_id": f"{submission_id}:{observation_index}",
            "archive_snapshot_sha256": snapshot.content_sha256,
            "raw_record_sha256": sha256_bytes(raw_record_bytes),
            "training_eligible": True,
            "published_claims_eligible": True,
            "value_kind": "observed",
            "provenance": {
                "source_name": snapshot.source_name,
                "source_url": benchmark.source_url,
                "source_type": "benchmark",
                "retrieved_at": snapshot.retrieved_at.isoformat(),
                "parser_version": snapshot.parser_version,
                "licence_or_access_note": snapshot.licence_or_access_note,
                "extraction_confidence": 0.90,
            },
            "normalisation_metadata": {
                "hardware_name": hardware_name,
                "device_type": device_type,
                "score_source_field": score_field,
                "scene_checksum": scene.get("checksum"),
                "blender_build_hash": blender_version.get("build_hash"),
                "benchmark_script": cls._mapping(observation.get("benchmark_script")).get("label"),
                "cpu_threads": cls._positive_int(device_info.get("num_cpu_threads")),
                "system_cpu_cores": cls._positive_int(system_info.get("num_cpu_cores")),
                "system_cpu_sockets": cls._positive_int(system_info.get("num_cpu_sockets")),
            },
            "data": benchmark.model_dump(mode="json"),
        }

    @staticmethod
    def _mapping(value: object) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

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

    @classmethod
    def _score(cls, stats: dict[str, Any]) -> tuple[float, str, bool, str]:
        samples_per_minute = cls._positive_float(stats.get("samples_per_minute"))
        if samples_per_minute is not None:
            return samples_per_minute, "samples/minute", True, "samples_per_minute"
        total_render_time = cls._positive_float(stats.get("total_render_time"))
        if total_render_time is not None:
            return total_render_time, "seconds", False, "total_render_time"
        render_time = cls._positive_float(stats.get("render_time_no_sync"))
        if render_time is not None:
            return render_time, "seconds", False, "render_time_no_sync"
        raise ValueError("no positive benchmark score")

    @staticmethod
    def _normalise_hardware_name(value: object) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text or None
        print("DEBUG", locals())  # noqa
