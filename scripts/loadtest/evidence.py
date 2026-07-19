"""Build a bounded, content-addressed evidence record from one Locust CSV run.

The development load profile deliberately remains non-promotable.  This module
preserves the exact profile, endpoint-level Locust statistics, API release
metadata, host capacity snapshot, and raw-output digests so a later reviewer
can distinguish a reproducible local measurement from a production claim.
"""

from __future__ import annotations

import csv
import ctypes
import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pc_build_recommender.evaluation.manifest import canonical_json_bytes, sha256_file, sha256_json
from scripts.loadtest.profile import LoadProfile, LoadProfileError, normalise_target_url

LOAD_EVIDENCE_SCHEMA_VERSION = "pc-build-recommender.locust-load-evidence.v1"
MAXIMUM_CSV_BYTES = 64 * 1024 * 1024
MAXIMUM_CSV_ROWS = 10_000
MAXIMUM_API_METADATA_BYTES = 1024 * 1024
MAXIMUM_API_TIMEOUT_SECONDS = 60.0
_REQUIRED_STATS_COLUMNS = {
    "Type",
    "Name",
    "Request Count",
    "Failure Count",
    "Min Response Time",
    "Max Response Time",
    "Requests/s",
    "Failures/s",
    "50%",
    "95%",
    "99%",
}
_LOCUST_OUTPUT_SUFFIXES = (
    "_stats.csv",
    "_stats_history.csv",
    "_failures.csv",
    "_exceptions.csv",
)
_ALLOWED_CACHE_STATES = frozenset({"cold", "warm", "unknown"})
_ALLOWED_DATABASE_STATES = frozenset({"in_memory_demo", "postgres", "unknown"})


class LoadEvidenceError(ValueError):
    """Raised when Locust outputs cannot support a bounded evidence record."""


@dataclass(frozen=True, slots=True)
class EndpointLoadMetrics:
    """The endpoint-level metric fields retained from Locust's stable CSV contract."""

    method: str
    path: str
    request_count: int
    failure_count: int
    min_response_time_ms: float
    max_response_time_ms: float
    requests_per_second: float
    failures_per_second: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "path": self.path,
            "request_count": self.request_count,
            "failure_count": self.failure_count,
            "failure_rate": self.failure_count / self.request_count,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "requests_per_second": self.requests_per_second,
            "failures_per_second": self.failures_per_second,
            "p50_response_time_ms": self.p50_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
        }


def _require_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = MAXIMUM_CSV_BYTES,
) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise LoadEvidenceError(f"{label} must be a regular file")
    if resolved.stat().st_size > maximum_bytes:
        raise LoadEvidenceError(f"{label} exceeds the {maximum_bytes} byte safety limit")
    return resolved


def _parse_nonnegative_int(value: object, *, field: str, row_label: str) -> int:
    if not isinstance(value, str) or not value.strip():
        raise LoadEvidenceError(f"{row_label} has no {field!r} value")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise LoadEvidenceError(f"{row_label} has an invalid integer {field!r}") from exc
    if parsed < 0:
        raise LoadEvidenceError(f"{row_label} has a negative {field!r}")
    return parsed


def _parse_nonnegative_float(value: object, *, field: str, row_label: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise LoadEvidenceError(f"{row_label} has no {field!r} value")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LoadEvidenceError(f"{row_label} has an invalid number {field!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise LoadEvidenceError(f"{row_label} has an invalid non-negative {field!r}")
    return parsed


def _stats_rows(path: Path) -> list[dict[str, str]]:
    resolved = _require_regular_file(path, label="Locust stats CSV")
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LoadEvidenceError("Locust stats CSV has no header")
        missing = sorted(_REQUIRED_STATS_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise LoadEvidenceError(
                "Locust stats CSV is missing required columns: " + ", ".join(missing)
            )
        rows: list[dict[str, str]] = []
        for row_index, raw in enumerate(reader, start=2):
            if row_index > MAXIMUM_CSV_ROWS + 1:
                raise LoadEvidenceError(
                    f"Locust stats CSV exceeds the {MAXIMUM_CSV_ROWS} row safety limit"
                )
            row = {str(key): str(value or "") for key, value in raw.items() if key is not None}
            if not any(row.values()):
                continue
            rows.append(row)
    return rows


def read_endpoint_metrics(
    *,
    profile: LoadProfile,
    stats_csv_path: Path,
) -> dict[str, EndpointLoadMetrics]:
    """Read exactly one Locust stats row for every endpoint in the reviewed profile."""

    expected = {
        (profile.search.method, profile.search.path): "search",
        (profile.build.method, profile.build.path): "build",
    }
    matched: dict[str, EndpointLoadMetrics] = {}
    for raw in _stats_rows(stats_csv_path):
        key = (raw.get("Type", "").strip().upper(), raw.get("Name", "").strip())
        endpoint_name = expected.get(key)
        if endpoint_name is None:
            continue
        if endpoint_name in matched:
            raise LoadEvidenceError(f"Locust stats CSV has duplicate {endpoint_name} endpoint rows")
        row_label = f"Locust {key[0]} {key[1]} row"
        request_count = _parse_nonnegative_int(
            raw.get("Request Count"), field="Request Count", row_label=row_label
        )
        if request_count == 0:
            raise LoadEvidenceError(f"{row_label} has zero requests")
        failure_count = _parse_nonnegative_int(
            raw.get("Failure Count"), field="Failure Count", row_label=row_label
        )
        if failure_count > request_count:
            raise LoadEvidenceError(f"{row_label} has more failures than requests")
        matched[endpoint_name] = EndpointLoadMetrics(
            method=key[0],
            path=key[1],
            request_count=request_count,
            failure_count=failure_count,
            min_response_time_ms=_parse_nonnegative_float(
                raw.get("Min Response Time"), field="Min Response Time", row_label=row_label
            ),
            max_response_time_ms=_parse_nonnegative_float(
                raw.get("Max Response Time"), field="Max Response Time", row_label=row_label
            ),
            requests_per_second=_parse_nonnegative_float(
                raw.get("Requests/s"), field="Requests/s", row_label=row_label
            ),
            failures_per_second=_parse_nonnegative_float(
                raw.get("Failures/s"), field="Failures/s", row_label=row_label
            ),
            p50_response_time_ms=_parse_nonnegative_float(
                raw.get("50%"), field="50%", row_label=row_label
            ),
            p95_response_time_ms=_parse_nonnegative_float(
                raw.get("95%"), field="95%", row_label=row_label
            ),
            p99_response_time_ms=_parse_nonnegative_float(
                raw.get("99%"), field="99%", row_label=row_label
            ),
        )
    missing = sorted(set(expected.values()).difference(matched))
    if missing:
        raise LoadEvidenceError(
            "Locust stats CSV lacks profile endpoint rows: " + ", ".join(missing)
        )
    return matched


def _locust_output_digests(csv_prefix: Path) -> dict[str, dict[str, object]]:
    prefix = csv_prefix.resolve()
    if not prefix.name:
        raise LoadEvidenceError("Locust CSV prefix must name a file prefix")
    result: dict[str, dict[str, object]] = {}
    for suffix in _LOCUST_OUTPUT_SUFFIXES:
        path = prefix.parent / f"{prefix.name}{suffix}"
        resolved = _require_regular_file(path, label=f"Locust output {path.name}")
        result[path.name] = {
            "file_name": path.name,
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    return result


def _json_from_url(*, origin: str, path: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        f"{origin}{path}",
        headers={"Accept": "application/json", "User-Agent": "BuildSignalLoadEvidence/1.0"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - origin is validated
            if response.status != 200:
                raise LoadEvidenceError(f"{path} returned HTTP {response.status}")
            payload = response.read(MAXIMUM_API_METADATA_BYTES + 1)
    except HTTPError as exc:
        raise LoadEvidenceError(f"{path} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise LoadEvidenceError(f"unable to read {path}: {exc.reason}") from exc
    if len(payload) > MAXIMUM_API_METADATA_BYTES:
        raise LoadEvidenceError(f"{path} response exceeds the metadata size limit")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LoadEvidenceError(f"{path} did not return valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LoadEvidenceError(f"{path} response must be a JSON object")
    return {str(key): value for key, value in parsed.items()}


def _required_text(payload: Mapping[str, Any], *, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise LoadEvidenceError(f"API metadata {name!r} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(payload: Mapping[str, Any], *, name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise LoadEvidenceError(f"API metadata {name!r} must be a non-negative integer")
    return value


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LoadEvidenceError(f"{name} must be an object")
    return {str(key): nested for key, nested in value.items()}


def _validated_api_metadata(value: Mapping[str, object]) -> dict[str, object]:
    payload = _mapping(value, name="api_metadata")
    target = payload.get("target_origin")
    if not isinstance(target, str):
        raise LoadEvidenceError("api_metadata must include a target_origin")
    try:
        target_origin = normalise_target_url(
            target,
            confirmation=os.environ.get("PCBR_LOAD_CONFIRM"),
        )
    except LoadProfileError as exc:
        raise LoadEvidenceError(str(exc)) from exc
    ready = _mapping(payload.get("ready"), name="api_metadata.ready")
    freshness = _mapping(payload.get("freshness"), name="api_metadata.freshness")
    data_version = _required_text(ready, name="data_version")
    if data_version != _required_text(freshness, name="data_version"):
        raise LoadEvidenceError("API ready and freshness endpoints disagree on data_version")
    production_ready = freshness.get("production_ready")
    if type(production_ready) is not bool:
        raise LoadEvidenceError("API metadata 'production_ready' must be a boolean")
    return {
        "target_origin": target_origin,
        "ready": {
            "data_version": data_version,
            "ranking_model": _required_text(ready, name="ranking_model"),
            "rule_version": _required_text(ready, name="rule_version"),
            "solver_version": _required_text(ready, name="solver_version"),
        },
        "freshness": {
            "data_version": data_version,
            "product_count": _required_nonnegative_int(freshness, name="product_count"),
            "listing_count": _required_nonnegative_int(freshness, name="listing_count"),
            "production_ready": production_ready,
            "release_artifact_verification": _required_text(
                freshness, name="release_artifact_verification"
            ),
        },
    }


def collect_api_metadata(*, target_origin: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    """Capture only bounded release/freshness fields from the declared API origin."""

    if not 0 < timeout_seconds <= MAXIMUM_API_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be between zero and {MAXIMUM_API_TIMEOUT_SECONDS}")
    try:
        origin = normalise_target_url(
            target_origin,
            confirmation=os.environ.get("PCBR_LOAD_CONFIRM"),
        )
    except LoadProfileError as exc:
        raise LoadEvidenceError(str(exc)) from exc
    ready = _json_from_url(origin=origin, path="/health/ready", timeout_seconds=timeout_seconds)
    freshness = _json_from_url(
        origin=origin,
        path="/v1/system/freshness",
        timeout_seconds=timeout_seconds,
    )
    if ready.get("status") != "ready":
        raise LoadEvidenceError("API ready endpoint did not report ready")
    return _validated_api_metadata(
        {"target_origin": origin, "ready": ready, "freshness": freshness}
    )


def _total_memory_bytes() -> int | None:
    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory = _MemoryStatus()
        memory.dwLength = ctypes.sizeof(_MemoryStatus)
        windll = getattr(ctypes, "windll", None)
        if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return int(memory.ullTotalPhys)
        return None
    try:
        sysconf = getattr(os, "sysconf", None)
        if sysconf is None:
            return None
        return int(sysconf("SC_PAGE_SIZE")) * int(sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def collect_host_metadata() -> dict[str, object]:
    """Record capacity and OS facts without hostnames, users, paths, or credentials."""

    return {
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu_logical_count": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
        "container_memory_limit_bytes": None,
    }


def _validated_host_metadata(value: Mapping[str, object]) -> dict[str, object]:
    payload = _mapping(value, name="host_metadata")
    operating_system = _mapping(
        payload.get("operating_system"),
        name="host_metadata.operating_system",
    )
    for name in ("system", "release", "machine"):
        _required_text(operating_system, name=name)
    cpu_logical_count = payload.get("cpu_logical_count")
    if cpu_logical_count is not None and (
        type(cpu_logical_count) is not int or cpu_logical_count < 1
    ):
        raise LoadEvidenceError(
            "host_metadata.cpu_logical_count must be a positive integer or null"
        )
    normalized: dict[str, object] = {
        "operating_system": {
            name: _required_text(operating_system, name=name)
            for name in ("system", "release", "machine")
        },
        "cpu_logical_count": cpu_logical_count,
    }
    for name in ("total_memory_bytes", "container_memory_limit_bytes"):
        memory_value = payload.get(name)
        if memory_value is not None and (type(memory_value) is not int or memory_value < 1):
            raise LoadEvidenceError(f"host_metadata.{name} must be a positive integer or null")
        normalized[name] = memory_value
    return normalized


def _finite_bounded_number(value: float, *, name: str, minimum: float, maximum: float) -> float:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise LoadEvidenceError(f"{name} must be finite and between {minimum} and {maximum}")
    return value


def build_load_evidence(
    *,
    profile: LoadProfile,
    csv_prefix: Path,
    api_metadata: Mapping[str, object],
    host_metadata: Mapping[str, object],
    users: int,
    spawn_rate_per_second: float,
    run_time_seconds: float,
    warmup_seconds: float,
    cache_state: str,
    database_state: str,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Bind a reviewed profile and Locust outputs into a conservative evidence artifact."""

    if type(users) is not int or not 1 <= users <= 10_000:
        raise LoadEvidenceError("users must be an integer between 1 and 10000")
    _finite_bounded_number(
        spawn_rate_per_second,
        name="spawn_rate_per_second",
        minimum=0.01,
        maximum=10_000.0,
    )
    _finite_bounded_number(run_time_seconds, name="run_time_seconds", minimum=1.0, maximum=86_400.0)
    _finite_bounded_number(warmup_seconds, name="warmup_seconds", minimum=0.0, maximum=86_400.0)
    if cache_state not in _ALLOWED_CACHE_STATES:
        raise LoadEvidenceError(f"cache_state must be one of {sorted(_ALLOWED_CACHE_STATES)}")
    if database_state not in _ALLOWED_DATABASE_STATES:
        raise LoadEvidenceError(f"database_state must be one of {sorted(_ALLOWED_DATABASE_STATES)}")
    if not isinstance(api_metadata, Mapping) or not isinstance(host_metadata, Mapping):
        raise TypeError("api_metadata and host_metadata must be mappings")
    validated_api_metadata = _validated_api_metadata(api_metadata)
    validated_host_metadata = _validated_host_metadata(host_metadata)
    target_origin = str(validated_api_metadata["target_origin"])
    endpoint_metrics = read_endpoint_metrics(
        profile=profile,
        stats_csv_path=csv_prefix.parent / f"{csv_prefix.name}_stats.csv",
    )
    output_digests = _locust_output_digests(csv_prefix)
    created = (created_at or datetime.now(UTC)).astimezone(UTC)

    assessments = {
        "search_p95_target_ms": 500.0,
        "build_p95_target_ms": 2500.0,
        "search_p95_within_target": (
            endpoint_metrics["search"].p95_response_time_ms <= 500.0
        ),
        "build_p95_within_target": endpoint_metrics["build"].p95_response_time_ms <= 2500.0,
        "claim_status": (
            "development_only_not_a_production_latency_claim"
            if profile.reportability == "development_only"
            else "production_claim_requires_release_and_operator_review"
        ),
    }
    payload: dict[str, object] = {
        "schema_version": LOAD_EVIDENCE_SCHEMA_VERSION,
        "created_at": created.isoformat(),
        "profile": {
            "profile_name": profile.profile_name,
            "profile_sha256": profile.sha256,
            "reportability": profile.reportability,
        },
        "target": {"origin": target_origin},
        "api_metadata": validated_api_metadata,
        "host_metadata": validated_host_metadata,
        "load_configuration": {
            "users": users,
            "spawn_rate_per_second": spawn_rate_per_second,
            "run_time_seconds": run_time_seconds,
            "warmup_seconds": warmup_seconds,
            "cache_state": cache_state,
            "database_state": database_state,
        },
        "raw_locust_outputs": output_digests,
        "endpoint_metrics": {
            name: metric.to_dict() for name, metric in sorted(endpoint_metrics.items())
        },
        "threshold_assessment": assessments,
    }
    return {**payload, "content_sha256": sha256_json(payload)}


def write_load_evidence(*, evidence: Mapping[str, object], output_path: Path) -> Path:
    """Atomically write one no-overwrite evidence record after validating its content digest."""

    output = output_path.resolve()
    if output.exists():
        raise LoadEvidenceError(f"output already exists and will not be overwritten: {output}")
    content_sha256 = evidence.get("content_sha256")
    if not isinstance(content_sha256, str):
        raise LoadEvidenceError("evidence must include content_sha256")
    payload = {key: value for key, value in evidence.items() if key != "content_sha256"}
    if sha256_json(payload) != content_sha256:
        raise LoadEvidenceError("evidence content_sha256 does not match its payload")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(canonical_json_bytes(evidence) + b"\n")
            temporary_name = handle.name
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
    return output


__all__ = [
    "LOAD_EVIDENCE_SCHEMA_VERSION",
    "EndpointLoadMetrics",
    "LoadEvidenceError",
    "build_load_evidence",
    "collect_api_metadata",
    "collect_host_metadata",
    "read_endpoint_metrics",
    "write_load_evidence",
]
