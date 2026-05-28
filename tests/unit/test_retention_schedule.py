from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.assets import retention as retention_asset
from pipelines.retention.registry import (
    MAXIMUM_REGISTRY_BYTES,
    RetentionRegistryError,
    load_governed_web_retention_sources,
    load_governed_web_source_admissions,
    load_restricted_web_sources,
)
from pipelines.retention.web import WebRetentionError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _registry(*, source_name: str = "reviewed_web", interval: str = "60") -> str:
    return f"""\
schema_version: pc-build-recommender.source-registry.v1
sources:
  {source_name}:
    kind: exact_url_schema_org_product_offer_crawl
    template: governed_web_product
    retention_maintenance:
      engine: governed_web_receipts_v2
      required: true
      maximum_interval_minutes: {interval}
"""


def _write_registry(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "source_registry.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _report(*, mismatch: bool = False) -> SimpleNamespace:
    raw = SimpleNamespace(
        receipts_scanned=3,
        expired_receipts_eligible=1,
        expired_receipts_removed=0 if mismatch else 1,
        expired_bodies_eligible=1,
        expired_bodies_removed=1,
        orphan_bodies_eligible=0,
        orphan_bodies_removed=0,
        crash_leftovers_eligible=0,
        crash_leftovers_removed=0,
        cache_files_eligible=0,
        cache_files_removed=0,
        orphan_bodies_in_grace=2,
        crash_leftovers_in_grace=1,
        unrelated_files_preserved=4,
    )
    processed = SimpleNamespace(
        runs_scanned=2,
        expired_runs_eligible=1,
        expired_runs_removed=1,
        unrelated_files_preserved=5,
        publication_operations_scanned=0,
        publication_operations_eligible=0,
        publication_operations_removed=0,
        publication_operations_in_grace=0,
        published_residues_detected=0,
        published_residues_removed=0,
    )
    return SimpleNamespace(source_name="reviewed_web", raw=raw, processed=processed)


def test_pipeline_definitions_import_without_yaml_or_dagster() -> None:
    script = f"""
import builtins
import sys

sys.path[:0] = [{str(REPOSITORY_ROOT)!r}, {str(REPOSITORY_ROOT / "packages/core/src")!r}]
real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.partition('.')[0]
    if root in {{'dagster', 'yaml'}}:
        raise ModuleNotFoundError(f'No module named {{root!r}}', name=root)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
import pipelines.definitions as definitions
assert definitions.defs is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_loader_reads_the_registry_of_record() -> None:
    assert load_governed_web_retention_sources(
        REPOSITORY_ROOT / "data" / "source_registry.yaml"
    ) == ("dynacore_web_research",)


def test_loader_reads_governed_web_source_admission() -> None:
    admissions = load_governed_web_source_admissions(
        REPOSITORY_ROOT / "data" / "source_registry.yaml"
    )

    assert [(item.source_name, item.allowed_hosts, item.usage_scope) for item in admissions] == [
        ("dynacore_web_research", ("dynacoretech.com",), "internal_research")
    ]


def test_loader_reads_reviewed_restricted_web_sources() -> None:
    restrictions = load_restricted_web_sources(REPOSITORY_ROOT / "data" / "source_registry.yaml")

    assert [(item.source_name, item.hosts) for item in restrictions] == [
        ("pchardware_org", ("pchardware.org",))
    ]


def test_restricted_source_loader_rejects_incomplete_host_restriction(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        """\
schema_version: pc-build-recommender.source-registry.v1
sources: {}
blocked_or_restricted_sources:
  unsafe_source:
    reason: Fixture terms prohibit crawling.
    hosts:
      - unsafe.example.test
""",
    )

    with pytest.raises(RetentionRegistryError, match="fields are incomplete or unknown"):
        load_restricted_web_sources(path)


def test_governed_web_admission_rejects_a_source_without_reviewed_hosts(tmp_path: Path) -> None:
    with pytest.raises(RetentionRegistryError, match="allowed_hosts"):
        load_governed_web_source_admissions(_write_registry(tmp_path, _registry()))


@pytest.mark.parametrize("interval", ["true", "0", "59", "61", "1.5", "'60'"])
def test_loader_rejects_non_integer_or_out_of_bounds_interval(
    tmp_path: Path, interval: str
) -> None:
    path = _write_registry(tmp_path, _registry(interval=interval))

    with pytest.raises(RetentionRegistryError, match="invalid maintenance interval"):
        load_governed_web_retention_sources(path)


def test_loader_rejects_duplicate_keys_and_aliases(tmp_path: Path) -> None:
    duplicate = _write_registry(
        tmp_path,
        _registry() + "  reviewed_web:\n    kind: canonical_component_catalogue\n",
    )
    with pytest.raises(RetentionRegistryError, match="duplicate key"):
        load_governed_web_retention_sources(duplicate)

    alias = _write_registry(
        tmp_path,
        """\
schema_version: pc-build-recommender.source-registry.v1
sources:
  reviewed_web: &source
    kind: exact_url_schema_org_product_offer_crawl
    template: governed_web_product
    retention_maintenance:
      engine: governed_web_receipts_v2
      required: true
      maximum_interval_minutes: 60
  copied_web: *source
""",
    )
    with pytest.raises(RetentionRegistryError, match="aliases are not permitted"):
        load_governed_web_retention_sources(alias)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (_registry(source_name="../escape"), "unsafe governed-web source name"),
        (
            _registry().replace("schema_version:", "unknown: value\nschema_version:"),
            "root fields are incomplete or unknown",
        ),
        (
            "schema_version: pc-build-recommender.source-registry.v1\nsources: {}\n",
            "no managed governed-web source",
        ),
    ],
)
def test_loader_rejects_unsafe_or_incomplete_registry(
    tmp_path: Path, content: str, message: str
) -> None:
    with pytest.raises(RetentionRegistryError, match=message):
        load_governed_web_retention_sources(_write_registry(tmp_path, content))


def test_loader_bounds_bytes_before_yaml_parsing(tmp_path: Path) -> None:
    path = tmp_path / "source_registry.yaml"
    path.write_bytes(b"x" * (MAXIMUM_REGISTRY_BYTES + 1))

    with pytest.raises(RetentionRegistryError, match="exceeds"):
        load_governed_web_retention_sources(path)


def test_execute_retention_uses_registry_sources_and_destructive_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        retention_asset,
        "load_governed_web_retention_sources",
        lambda path: ("reviewed_web",),
    )

    def maintain(**kwargs: Any) -> tuple[SimpleNamespace, ...]:
        calls.update(kwargs)
        return (_report(),)

    monkeypatch.setattr(retention_asset, "maintain_web_retention", maintain)
    result = retention_asset.execute_governed_web_retention(
        registry_path=tmp_path / "registry.yaml",
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        orphan_grace_hours=12,
        maximum_entries=321,
    )

    assert calls["source_names"] == ("reviewed_web",)
    assert calls["dry_run"] is False
    assert calls["orphan_grace"].total_seconds() == 12 * 60 * 60
    assert calls["maximum_entries"] == 321
    assert result == {
        "status": "ok",
        "source_names": ["reviewed_web"],
        "source_count": 1,
        "raw_receipts_scanned": 3,
        "raw_receipts_removed": 1,
        "raw_bodies_removed": 1,
        "processed_runs_scanned": 2,
        "processed_runs_removed": 1,
        "publication_operations_scanned": 0,
        "publication_operations_removed": 0,
        "published_residues_removed": 0,
        "preserved_unknown_files": 9,
        "grace_leftovers": 3,
    }


@pytest.mark.parametrize(
    ("orphan_grace_hours", "maximum_entries"),
    [(0, 100), (721, 100), (True, 100), (24, 0), (24, 1_000_001), (24, True)],
)
def test_execute_retention_rejects_unbounded_configuration(
    tmp_path: Path,
    orphan_grace_hours: Any,
    maximum_entries: Any,
) -> None:
    with pytest.raises(ValueError):
        retention_asset.execute_governed_web_retention(
            registry_path=tmp_path / "registry.yaml",
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
            orphan_grace_hours=orphan_grace_hours,
            maximum_entries=maximum_entries,
        )


def test_execute_retention_fails_on_removal_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        retention_asset,
        "load_governed_web_retention_sources",
        lambda path: ("reviewed_web",),
    )
    monkeypatch.setattr(
        retention_asset,
        "maintain_web_retention",
        lambda **kwargs: (_report(mismatch=True),),
    )

    with pytest.raises(WebRetentionError, match="removal count mismatch"):
        retention_asset.execute_governed_web_retention(
            registry_path=tmp_path / "registry.yaml",
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
        )


def test_execute_wdc_retention_uses_its_separate_destructive_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    report = SimpleNamespace(
        raw_pairs_eligible=2,
        raw_pairs_removed=2,
        category_index_eligible=True,
        category_index_removed=True,
        sealed_runs_eligible=1,
        sealed_runs_removed=1,
        working_runs_eligible=1,
        working_runs_removed=1,
        to_dict=lambda: {
            "raw_pairs_eligible": 2,
            "raw_pairs_removed": 2,
            "category_index_eligible": True,
            "category_index_removed": True,
            "sealed_runs_eligible": 1,
            "sealed_runs_removed": 1,
            "working_runs_eligible": 1,
            "working_runs_removed": 1,
            "unrelated_entries_preserved": 3,
        },
    )

    def maintain(**kwargs: Any) -> SimpleNamespace:
        calls.update(kwargs)
        return report

    monkeypatch.setattr(retention_asset, "maintain_wdc_research_retention", maintain)
    result = retention_asset.execute_wdc_research_retention(
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "quarantine",
        category_index=tmp_path / "quarantine" / "wdc.sqlite3",
        maximum_entries=321,
    )

    assert calls["dry_run"] is False
    assert calls["maximum_entries"] == 321
    assert result == {
        "status": "ok",
        "raw_pairs_eligible": 2,
        "raw_pairs_removed": 2,
        "category_index_eligible": True,
        "category_index_removed": True,
        "sealed_runs_eligible": 1,
        "sealed_runs_removed": 1,
        "working_runs_eligible": 1,
        "working_runs_removed": 1,
        "unrelated_entries_preserved": 3,
    }


def test_execute_wdc_retention_rejects_a_removal_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = SimpleNamespace(
        raw_pairs_eligible=1,
        raw_pairs_removed=0,
        category_index_eligible=False,
        category_index_removed=False,
        sealed_runs_eligible=0,
        sealed_runs_removed=0,
        working_runs_eligible=0,
        working_runs_removed=0,
    )
    monkeypatch.setattr(
        retention_asset,
        "maintain_wdc_research_retention",
        lambda **kwargs: report,
    )

    with pytest.raises(retention_asset.WDCRetentionError, match="removal count mismatch"):
        retention_asset.execute_wdc_research_retention(
            raw_root=tmp_path / "raw",
            output_root=tmp_path / "quarantine",
            category_index=tmp_path / "quarantine" / "wdc.sqlite3",
        )


def test_dagster_definitions_enable_hourly_retention_by_default() -> None:
    dagster = pytest.importorskip("dagster")
    from pipelines import definitions

    assert definitions.defs is not None
    schedule = definitions.defs.resolve_schedule_def("governed_web_retention_hourly")
    assert schedule.cron_schedule == "0 * * * *"
    assert schedule.execution_timezone == "Asia/Singapore"
    assert schedule.default_status is dagster.DefaultScheduleStatus.RUNNING
    assert definitions.defs.resolve_job_def("governed_web_retention") is not None
    wdc_schedule = definitions.defs.resolve_schedule_def("wdc_research_retention_daily")
    assert wdc_schedule.cron_schedule == "15 0 * * *"
    assert wdc_schedule.execution_timezone == "Asia/Singapore"
    assert wdc_schedule.default_status is dagster.DefaultScheduleStatus.RUNNING
    assert definitions.defs.resolve_job_def("wdc_research_retention") is not None


def test_dagster_retention_asset_uses_bounded_environment_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dagster = pytest.importorskip("dagster")
    registry_path = _write_registry(tmp_path, _registry())
    monkeypatch.setenv("SOURCE_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("PROCESSED_DATA_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("WEB_RETENTION_ORPHAN_GRACE_HOURS", "12")
    monkeypatch.setenv("WEB_RETENTION_MAXIMUM_ENTRIES", "321")

    result = dagster.materialize(
        [retention_asset.governed_web_retention_maintenance],
        raise_on_error=False,
    )

    assert result.success
    output = result.output_for_node("governed_web_retention_maintenance")
    assert output["status"] == "ok"
    assert output["source_names"] == ["reviewed_web"]
    assert output["source_count"] == 1


def test_dagster_wdc_retention_asset_uses_bounded_environment_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dagster = pytest.importorskip("dagster")
    monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("WDC_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    monkeypatch.setenv(
        "WDC_CATEGORY_INDEX_PATH",
        str(tmp_path / "quarantine" / "wdc-products-category-index.sqlite3"),
    )
    monkeypatch.setenv("WDC_RETENTION_MAXIMUM_ENTRIES", "321")

    result = dagster.materialize(
        [retention_asset.wdc_research_retention_maintenance],
        raise_on_error=False,
    )

    assert result.success
    output = result.output_for_node("wdc_research_retention_maintenance")
    assert output["status"] == "ok"
    assert output["raw_pairs_eligible"] == 0


def test_dagster_container_carries_registry_and_daemon_wiring() -> None:
    dockerfile = (REPOSITORY_ROOT / "infra" / "dagster.Dockerfile").read_text(encoding="utf-8")
    development_compose = (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    production_compose = (REPOSITORY_ROOT / "docker-compose.production.yml").read_text(
        encoding="utf-8"
    )

    assert "--extra pipeline" in dockerfile
    assert "data/source_registry.yaml ./config/source_registry.yaml" in dockerfile
    assert "SOURCE_REGISTRY_PATH: /app/config/source_registry.yaml" in development_compose
    assert "SOURCE_REGISTRY_PATH: /app/config/source_registry.yaml" in production_compose
    assert "dagster-daemon:" in production_compose
    assert 'command: ["dagster-daemon", "run"' in production_compose
