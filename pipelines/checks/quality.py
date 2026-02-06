"""Source-independent ingestion data-quality checks and audit reports."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pipelines.sources.base import ParseResult

DATA_QUALITY_SCHEMA_VERSION = "pc-build-recommender.data-quality.v1"
_SNAPSHOT_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_QUALITY_REPORT_FIELDS = {
    "schema_version",
    "source_name",
    "snapshot_sha256",
    "status",
    "accepted_count",
    "rejected_count",
    "rejection_rate",
    "checks",
    "record_type_counts",
    "category_counts",
    "eligibility_counts",
    "source_statistics",
}


@dataclass(frozen=True, slots=True)
class QualityRegressionPolicy:
    """Fail-closed limits for a comparable, last-known-good source batch."""

    minimum_baseline_accepted_count: int = 10
    minimum_accepted_count_fraction: float = 0.70
    minimum_retailer_listing_count_fraction: float = 0.70
    minimum_category_count_fraction: float = 0.60
    maximum_rejection_rate_increase: float = 0.15

    def __post_init__(self) -> None:
        if self.minimum_baseline_accepted_count < 1:
            raise ValueError("minimum_baseline_accepted_count must be positive")
        for name, value in (
            ("minimum_accepted_count_fraction", self.minimum_accepted_count_fraction),
            (
                "minimum_retailer_listing_count_fraction",
                self.minimum_retailer_listing_count_fraction,
            ),
            ("minimum_category_count_fraction", self.minimum_category_count_fraction),
            ("maximum_rejection_rate_increase", self.maximum_rejection_rate_increase),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class QualityBaseline:
    """Only the aggregate fields needed to compare one prior successful batch."""

    source_name: str
    snapshot_sha256: str
    accepted_count: int
    rejected_count: int
    rejection_rate: float
    record_type_counts: dict[str, int]
    category_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    schema_version: str
    source_name: str
    snapshot_sha256: str
    status: str
    accepted_count: int
    rejected_count: int
    rejection_rate: float
    checks: tuple[dict[str, Any], ...]
    record_type_counts: dict[str, int]
    category_counts: dict[str, int]
    eligibility_counts: dict[str, int]
    source_statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_name": self.source_name,
            "snapshot_sha256": self.snapshot_sha256,
            "status": self.status,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "rejection_rate": self.rejection_rate,
            "checks": list(self.checks),
            "record_type_counts": self.record_type_counts,
            "category_counts": self.category_counts,
            "eligibility_counts": self.eligibility_counts,
            "source_statistics": self.source_statistics,
        }


def _check(name: str, severity: str, count: int, message: str) -> dict[str, Any]:
    return {"name": name, "severity": severity, "count": count, "message": message}


def _is_positive_number(value: object) -> bool:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 0


def _record_data(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    return dict(data) if isinstance(data, dict) else {}


def _is_linklike(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction(path))


def _count_mapping(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            return None
        result[key] = count
    return result


def _quality_baseline_from_payload(payload: object) -> QualityBaseline | None:
    if not isinstance(payload, dict) or set(payload) != _QUALITY_REPORT_FIELDS:
        return None
    if payload.get("schema_version") != DATA_QUALITY_SCHEMA_VERSION:
        return None
    source_name = payload.get("source_name")
    snapshot_sha256 = payload.get("snapshot_sha256")
    accepted_count = payload.get("accepted_count")
    rejected_count = payload.get("rejected_count")
    rejection_rate = payload.get("rejection_rate")
    record_type_counts = _count_mapping(payload.get("record_type_counts"))
    category_counts = _count_mapping(payload.get("category_counts"))
    if (
        not isinstance(source_name, str)
        or not isinstance(snapshot_sha256, str)
        or not _SNAPSHOT_SHA256_PATTERN.fullmatch(snapshot_sha256)
        or type(accepted_count) is not int
        or accepted_count < 0
        or type(rejected_count) is not int
        or rejected_count < 0
        or not isinstance(rejection_rate, (int, float))
        or not math.isfinite(float(rejection_rate))
        or not 0 <= float(rejection_rate) <= 1
        or record_type_counts is None
        or category_counts is None
    ):
        return None
    return QualityBaseline(
        source_name=source_name,
        snapshot_sha256=snapshot_sha256,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        rejection_rate=float(rejection_rate),
        record_type_counts=record_type_counts,
        category_counts=category_counts,
    )


def load_previous_quality_baseline(
    *,
    processed_root: str | Path,
    source_name: str,
    current_snapshot_sha256: str,
    variant: str | None,
    maximum_candidates: int = 1_000,
) -> QualityBaseline | None:
    """Return the newest valid, passing comparable report without reading raw records.

    Comparisons are deliberately variant-specific. A bounded development sample must not be
    judged against a full source import, and an invalid or warning-only prior report is never
    treated as a trusted baseline.
    """

    if not _SNAPSHOT_SHA256_PATTERN.fullmatch(current_snapshot_sha256):
        raise ValueError("current_snapshot_sha256 must be a SHA-256 digest")
    if _SOURCE_NAME_PATTERN.fullmatch(source_name) is None:
        raise ValueError("source_name must be a lowercase slug")
    if variant is not None and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", variant) is None:
        raise ValueError("variant must be a lowercase slug")
    if not 1 <= maximum_candidates <= 10_000:
        raise ValueError("maximum_candidates must be between 1 and 10000")
    source_root = Path(processed_root) / source_name
    if not source_root.exists() or _is_linklike(source_root) or not source_root.is_dir():
        return None
    candidates: list[tuple[int, Path, Path]] = []
    scanned_snapshots = 0
    try:
        children = sorted(source_root.iterdir(), key=lambda item: item.name)
    except OSError:
        return None
    for snapshot_directory in children:
        if (
            _is_linklike(snapshot_directory)
            or not snapshot_directory.is_dir()
            or not _SNAPSHOT_SHA256_PATTERN.fullmatch(snapshot_directory.name)
        ):
            continue
        scanned_snapshots += 1
        if scanned_snapshots > maximum_candidates:
            break
        run_directory = snapshot_directory / variant if variant is not None else snapshot_directory
        quality_path = run_directory / "data-quality.json"
        manifest_path = run_directory / "manifest.json"
        try:
            invalid_paths = (
                _is_linklike(run_directory)
                or not run_directory.is_dir()
                or _is_linklike(quality_path)
                or _is_linklike(manifest_path)
                or not quality_path.is_file()
                or not manifest_path.is_file()
            )
            quality_size = quality_path.stat().st_size
            manifest_size = manifest_path.stat().st_size
        except OSError:
            continue
        if invalid_paths or quality_size > 1024 * 1024 or manifest_size > 1024 * 1024:
            continue
        try:
            candidates.append((quality_path.stat().st_mtime_ns, quality_path, manifest_path))
        except OSError:
            continue
    ordered_candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    for _, quality_path, manifest_path in ordered_candidates:
        try:
            payload = json.loads(quality_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        baseline = _quality_baseline_from_payload(payload)
        if baseline is None or baseline.source_name != source_name:
            continue
        if payload.get("status") != "pass" or not isinstance(manifest, dict):
            continue
        metadata = manifest.get("metadata")
        manifest_snapshot = manifest.get("source_snapshot_sha256")
        if isinstance(metadata, dict):
            manifest_snapshot = metadata.get("raw_snapshot_sha256", manifest_snapshot)
        if (
            baseline.snapshot_sha256 == current_snapshot_sha256
            or manifest.get("source_name") != source_name
            or manifest_snapshot != baseline.snapshot_sha256
            or manifest.get("accepted_count") != baseline.accepted_count
            or manifest.get("rejected_count") != baseline.rejected_count
        ):
            continue
        return baseline
    return None


def quality_regression_checks(
    *,
    baseline: QualityBaseline | None,
    accepted_count: int,
    rejection_rate: float,
    record_type_counts: dict[str, int],
    category_counts: dict[str, int],
    policy: QualityRegressionPolicy,
) -> list[dict[str, Any]]:
    if baseline is None or baseline.accepted_count < policy.minimum_baseline_accepted_count:
        return []
    minimum_accepted = math.ceil(
        baseline.accepted_count * policy.minimum_accepted_count_fraction
    )
    accepted_shortfall = max(0, minimum_accepted - accepted_count)
    checks = [
        _check(
            "accepted_count_regression",
            "error",
            accepted_shortfall,
            f"Previous passing batch accepted {baseline.accepted_count}; minimum is "
            f"{minimum_accepted} ({policy.minimum_accepted_count_fraction:.0%}), observed "
            f"{accepted_count}.",
        )
    ]
    previous_listings = baseline.record_type_counts.get("retailer_listing", 0)
    if previous_listings >= policy.minimum_baseline_accepted_count:
        current_listings = record_type_counts.get("retailer_listing", 0)
        minimum_listings = math.ceil(
            previous_listings * policy.minimum_retailer_listing_count_fraction
        )
        checks.append(
            _check(
                "retailer_listing_count_regression",
                "error",
                max(0, minimum_listings - current_listings),
                f"Previous passing batch retained {previous_listings} retailer listings; minimum "
                f"is {minimum_listings} ({policy.minimum_retailer_listing_count_fraction:.0%}), "
                f"observed {current_listings}.",
            )
        )
    missing_record_types = sum(
        1
        for record_type, previous_count in baseline.record_type_counts.items()
        if previous_count >= policy.minimum_baseline_accepted_count
        and record_type_counts.get(record_type, 0) == 0
    )
    checks.append(
        _check(
            "record_type_regression",
            "error",
            missing_record_types,
            "A record type with sufficient prior coverage disappeared from the current batch.",
        )
    )
    category_shortfall = 0
    affected_categories: list[str] = []
    for category, previous_count in baseline.category_counts.items():
        if previous_count < policy.minimum_baseline_accepted_count:
            continue
        minimum_category = math.ceil(previous_count * policy.minimum_category_count_fraction)
        current_category = category_counts.get(category, 0)
        shortfall = max(0, minimum_category - current_category)
        if shortfall:
            category_shortfall += shortfall
            affected_categories.append(category)
    checks.append(
        _check(
            "category_count_regression",
            "error",
            category_shortfall,
            "Category coverage regressed below the comparable baseline for: "
            + (", ".join(sorted(affected_categories)) if affected_categories else "none")
            + ".",
        )
    )
    rejection_rate_delta = rejection_rate - baseline.rejection_rate
    checks.append(
        _check(
            "rejection_rate_regression",
            "error",
            int(rejection_rate_delta > policy.maximum_rejection_rate_increase),
            f"Previous rejection rate {baseline.rejection_rate:.4f}; "
            f"observed {rejection_rate:.4f}; "
            f"maximum allowed increase is {policy.maximum_rejection_rate_increase:.4f}.",
        )
    )
    return checks


def evaluate_batch_quality(
    batch: ParseResult,
    *,
    maximum_rejection_rate: float = 0.20,
    baseline: QualityBaseline | None = None,
    regression_policy: QualityRegressionPolicy | None = None,
) -> DataQualityReport:
    """Evaluate structural, numeric, data-use, and comparable-source regression invariants."""

    if not 0 <= maximum_rejection_rate <= 1:
        raise ValueError("maximum_rejection_rate must be between zero and one")
    total_rejections = int(batch.statistics.get("total_rejections", batch.rejected_count))
    total_seen = batch.accepted_count + total_rejections
    rejection_rate = total_rejections / total_seen if total_seen else 0.0
    checks: list[dict[str, Any]] = []

    source_ids = [str(record.get("source_record_id", "")) for record in batch.records]
    missing_source_ids = sum(not value for value in source_ids)
    duplicate_source_ids = len(source_ids) - len(set(source_ids))
    missing_envelope = sum(
        not record.get("schema_version")
        or not record.get("record_type")
        or not isinstance(record.get("data"), dict)
        for record in batch.records
    )
    checks.extend(
        [
            _check(
                "accepted_records_present",
                "error",
                int(batch.accepted_count == 0),
                "At least one accepted record is required for a successful ingestion.",
            ),
            _check(
                "source_record_ids_present",
                "error",
                missing_source_ids,
                "Every accepted row must retain its source record identifier.",
            ),
            _check(
                "source_record_ids_unique",
                "error",
                duplicate_source_ids,
                "Duplicate source IDs would make idempotent upserts ambiguous.",
            ),
            _check(
                "normalised_envelope_complete",
                "error",
                missing_envelope,
                "Normalised records require schema, type, and data fields.",
            ),
            _check(
                "rejection_rate_within_threshold",
                "error",
                int(rejection_rate > maximum_rejection_rate),
                f"Observed rejection rate {rejection_rate:.4f}; "
                f"threshold is {maximum_rejection_rate:.4f}.",
            ),
        ]
    )

    invalid_products = 0
    invalid_benchmarks = 0
    invalid_listings = 0
    invalid_controlled_use = 0
    record_type_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    eligibility_counts = {
        "training_eligible": 0,
        "training_ineligible": 0,
        "published_claims_eligible": 0,
        "published_claims_ineligible": 0,
    }
    for record in batch.records:
        record_type = str(record.get("record_type", "unknown"))
        record_type_counts[record_type] = record_type_counts.get(record_type, 0) + 1
        training_eligible = record.get("training_eligible") is True
        published_eligible = record.get("published_claims_eligible") is True
        eligibility_counts["training_eligible" if training_eligible else "training_ineligible"] += 1
        eligibility_counts[
            "published_claims_eligible" if published_eligible else "published_claims_ineligible"
        ] += 1
        data = _record_data(record)
        if record_type == "canonical_product":
            required = ("product_id", "category", "brand", "model", "canonical_name")
            if any(not data.get(field_name) for field_name in required):
                invalid_products += 1
            category = str(data.get("category", "unknown"))
            category_counts[category] = category_counts.get(category, 0) + 1
        elif record_type == "benchmark_observation":
            if (
                not data.get("benchmark_id")
                or not data.get("product_id")
                or not data.get("unit")
                or not _is_positive_number(data.get("score"))
            ):
                invalid_benchmarks += 1
        elif record_type == "retailer_listing":
            listing = data.get("listing")
            if not isinstance(listing, dict):
                invalid_listings += 1
            else:
                try:
                    price = Decimal(str(listing.get("base_price")))
                except (InvalidOperation, TypeError):
                    price = Decimal("-1")
                currency = str(listing.get("currency", ""))
                if (
                    price <= 0
                    or re.fullmatch(r"[A-Z]{3}", currency) is None
                    or not listing.get("source_listing_id")
                ):
                    invalid_listings += 1
        if batch.source_name in {"bizgram_controlled_pdf", "dynacore_controlled_pdf"} and (
            training_eligible or published_eligible
        ):
            invalid_controlled_use += 1

    checks.extend(
        [
            _check(
                "canonical_products_valid",
                "error",
                invalid_products,
                "Canonical products require stable IDs and core identity fields.",
            ),
            _check(
                "benchmark_values_valid",
                "error",
                invalid_benchmarks,
                "Benchmarks require a product key, finite positive score, and unit.",
            ),
            _check(
                "retailer_prices_valid",
                "error",
                invalid_listings,
                "Retailer listings require a positive base price and ISO currency.",
            ),
            _check(
                "controlled_import_use_restricted",
                "error",
                invalid_controlled_use,
                "Controlled retailer imports must be ineligible for training and claims.",
            ),
        ]
    )

    policy = regression_policy or QualityRegressionPolicy()
    checks.extend(
        quality_regression_checks(
            baseline=baseline,
            accepted_count=batch.accepted_count,
            rejection_rate=rejection_rate,
            record_type_counts=record_type_counts,
            category_counts=category_counts,
            policy=policy,
        )
    )
    error_count = sum(int(check["count"]) for check in checks if check["severity"] == "error")
    warning_count = sum(int(check["count"]) for check in checks if check["severity"] == "warning")
    status = "fail" if error_count else "warning" if warning_count else "pass"
    return DataQualityReport(
        schema_version=DATA_QUALITY_SCHEMA_VERSION,
        source_name=batch.source_name,
        snapshot_sha256=batch.snapshot_sha256,
        status=status,
        accepted_count=batch.accepted_count,
        rejected_count=total_rejections,
        rejection_rate=rejection_rate,
        checks=tuple(checks),
        record_type_counts=dict(sorted(record_type_counts.items())),
        category_counts=dict(sorted(category_counts.items())),
        eligibility_counts=eligibility_counts,
        source_statistics=batch.statistics,
    )


def evaluate_batch_quality_against_previous(
    batch: ParseResult,
    *,
    processed_root: str | Path,
    variant: str | None,
    maximum_rejection_rate: float = 0.20,
    regression_policy: QualityRegressionPolicy | None = None,
) -> DataQualityReport:
    """Evaluate a batch against the latest trusted comparable output when one exists."""

    baseline = load_previous_quality_baseline(
        processed_root=processed_root,
        source_name=batch.source_name,
        current_snapshot_sha256=batch.snapshot_sha256,
        variant=variant,
    )
    return evaluate_batch_quality(
        batch,
        maximum_rejection_rate=maximum_rejection_rate,
        baseline=baseline,
        regression_policy=regression_policy,
    )


def write_quality_report(report: DataQualityReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                report.to_dict(),
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path
