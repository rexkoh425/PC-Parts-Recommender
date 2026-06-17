from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import scripts.import_processed_catalog as catalog_import
from scripts.import_processed_catalog import main

from pc_build_recommender.catalog import (
    CanonicalIdentityImportError,
    CanonicalIdentityMember,
    CanonicalIdentityPreflightReport,
)


def _production_arguments(tmp_path: Path) -> list[str]:
    return [
        "import_processed_catalog.py",
        "--buildcores",
        str(tmp_path / "catalog.jsonl"),
        "--offers",
        str(tmp_path / "offers.jsonl"),
        "--reviewed-mappings",
        str(tmp_path / "reviewed.json"),
        "--review-evidence",
        str(tmp_path / "review-evidence.jsonl"),
        "--serving-manifest",
        str(tmp_path / "serving-manifest.json"),
        "--serving-manifest-sha256",
        "a" * 64,
        "--source-registry",
        str(tmp_path / "source-registry.yaml"),
        "--source-trust-root-sha256",
        "b" * 64,
        "--require-production-ready",
    ]


@pytest.mark.parametrize(
    "caller_authority",
    [
        ["--entity-resolution-model", "caller-model"],
        ["--entity-resolution-evaluation", "caller-evaluation.json"],
        ["--allow-unpromoted-entity-resolution-shadow"],
    ],
)
def test_production_import_rejects_caller_supplied_er_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caller_authority: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", _production_arguments(tmp_path) + caller_authority)

    with pytest.raises(
        ValueError,
        match="authority must come only from the serving manifest",
    ):
        main()


def test_production_import_requires_operator_pinned_serving_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _production_arguments(tmp_path)
    manifest_index = arguments.index("--serving-manifest")
    del arguments[manifest_index : manifest_index + 2]
    digest_index = arguments.index("--serving-manifest-sha256")
    del arguments[digest_index : digest_index + 2]
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(ValueError, match="requires a pinned serving manifest"):
        main()


def test_pinned_serving_manifest_requires_independent_source_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _production_arguments(tmp_path)
    registry_index = arguments.index("--source-registry")
    del arguments[registry_index : registry_index + 2]
    trust_index = arguments.index("--source-trust-root-sha256")
    del arguments[trust_index : trust_index + 2]
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(ValueError, match="requires independent source registry"):
        main()


def test_identity_preflight_failure_writes_machine_readable_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_output = tmp_path / "identity-report.json"
    member = CanonicalIdentityMember(
        product_id="prod_missing_mpn",
        category="gpu",
        brand="Example",
        manufacturer_part_number=None,
    )
    preflight = CanonicalIdentityPreflightReport(
        record_count=1,
        complete_identity_count=0,
        missing_brand_products=(),
        missing_mpn_products=(member,),
        duplicate_identity_groups=(),
        duplicate_product_id_groups=(),
    )

    def fail_preflight(*_args: object, **_kwargs: object) -> object:
        raise CanonicalIdentityImportError(preflight)

    monkeypatch.setattr(catalog_import, "stream_processed_catalog", fail_preflight)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_processed_catalog.py",
            "--buildcores",
            str(tmp_path / "catalog.jsonl"),
            "--offers",
            str(tmp_path / "offers.jsonl"),
            "--report-output",
            str(report_output),
        ],
    )

    assert main() == 2
    payload = json.loads(report_output.read_text(encoding="utf-8"))
    assert payload["database_upserted"] is False
    assert payload["canonical_identity_gate_failed"] is True
    diagnostics = payload["canonical_identity_preflight"]
    assert diagnostics["schema_version"].endswith("canonical-identity-preflight.v1")
    assert diagnostics["missing_mpn_count"] == 1
    assert diagnostics["production_ready"] is False


def test_standalone_import_refuses_a_production_database_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _production_arguments(tmp_path)
        + ["--database-url", "postgresql+psycopg://release.invalid/pcbr"],
    )

    with pytest.raises(
        ValueError,
        match="production-ready database writes require scripts/import_catalog_release.py",
    ):
        main()
