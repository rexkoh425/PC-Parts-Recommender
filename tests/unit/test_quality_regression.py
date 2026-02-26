from __future__ import annotations

from pathlib import Path

import pytest
from pipelines.checks.quality import (
    evaluate_batch_quality_against_previous,
    load_previous_quality_baseline,
    write_quality_report,
)
from pipelines.parsing.writer import write_parsed_batch
from pipelines.sources.base import ParseResult


def _product_records(*, count: int, category: str = "cpu") -> list[dict[str, object]]:
    return [
        {
            "schema_version": "pc-build-recommender.normalised-record.v1",
            "record_type": "canonical_product",
            "source_record_id": f"{category}-{index}",
            "training_eligible": True,
            "published_claims_eligible": True,
            "data": {
                "product_id": f"{category}-{index}",
                "category": category,
                "brand": "Fixture",
                "model": f"Model {index}",
                "canonical_name": f"Fixture {category} {index}",
            },
        }
        for index in range(count)
    ]


def _persist_passing_baseline(
    root: Path,
    *,
    snapshot: str = "a" * 64,
    variant: str | None = None,
) -> None:
    batch = ParseResult(
        source_name="fixture_quality",
        snapshot_sha256=snapshot,
        records=_product_records(count=10),
    )
    artifacts = write_parsed_batch(
        batch,
        processed_root=root,
        prefer_parquet=False,
        variant=variant,
    )
    report = evaluate_batch_quality_against_previous(
        batch,
        processed_root=root,
        variant=variant,
    )
    assert report.status == "pass"
    write_quality_report(report, artifacts.output_directory / "data-quality.json")


def test_accepted_count_regression_is_a_promotion_blocker(tmp_path: Path) -> None:
    _persist_passing_baseline(tmp_path)
    current = ParseResult(
        source_name="fixture_quality",
        snapshot_sha256="b" * 64,
        records=_product_records(count=6),
    )

    report = evaluate_batch_quality_against_previous(
        current,
        processed_root=tmp_path,
        variant=None,
    )

    accepted_check = next(
        check for check in report.checks if check["name"] == "accepted_count_regression"
    )
    assert accepted_check["severity"] == "error"
    assert accepted_check["count"] == 1
    assert report.status == "fail"


def test_category_regression_is_detected_even_when_total_count_is_stable(tmp_path: Path) -> None:
    _persist_passing_baseline(tmp_path)
    current = ParseResult(
        source_name="fixture_quality",
        snapshot_sha256="c" * 64,
        records=[
            *_product_records(count=5, category="cpu"),
            *_product_records(count=5, category="gpu"),
        ],
    )

    report = evaluate_batch_quality_against_previous(
        current,
        processed_root=tmp_path,
        variant=None,
    )

    category_check = next(
        check for check in report.checks if check["name"] == "category_count_regression"
    )
    assert category_check["severity"] == "error"
    assert category_check["count"] == 1
    assert report.status == "fail"


def test_baselines_are_variant_specific_and_require_a_passing_prior_report(tmp_path: Path) -> None:
    _persist_passing_baseline(tmp_path, variant="portfolio")
    current = ParseResult(
        source_name="fixture_quality",
        snapshot_sha256="d" * 64,
        records=_product_records(count=1),
    )

    unrelated_variant = evaluate_batch_quality_against_previous(
        current,
        processed_root=tmp_path,
        variant="fast",
    )

    assert load_previous_quality_baseline(
        processed_root=tmp_path,
        source_name="fixture_quality",
        current_snapshot_sha256=current.snapshot_sha256,
        variant="fast",
    ) is None
    assert not any(
        check["name"] == "accepted_count_regression" for check in unrelated_variant.checks
    )
    assert unrelated_variant.status == "pass"


def test_baseline_lookup_rejects_an_unsafe_source_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_name"):
        load_previous_quality_baseline(
            processed_root=tmp_path,
            source_name="../escape",
            current_snapshot_sha256="e" * 64,
            variant=None,
        )
