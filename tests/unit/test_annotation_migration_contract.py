from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    import_paths = (str(_ROOT / "packages" / "core" / "src"), str(_ROOT))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        (*import_paths, existing) if existing else import_paths
    )
    return environment


def test_alembic_metadata_import_does_not_load_model_stack() -> None:
    code = """
import sys
from pc_build_recommender.annotation import orm as annotation_orm
from pc_build_recommender.catalog.orm import Base

assert annotation_orm.AnnotationProjectRecord.metadata is Base.metadata
for forbidden in (
    "lightgbm",
    "pc_build_recommender.entity_resolution",
    "pc_build_recommender.ranking",
    "pc_build_recommender.retrieval",
):
    assert forbidden not in sys.modules, forbidden
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_postgresql_migration_blocks_truncate_on_every_append_only_table() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(_ROOT / "db" / "alembic.ini"),
            "upgrade",
            "20260723_0006",
            "--sql",
        ],
        cwd=_ROOT,
        env=_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    ddl = result.stdout
    for table_name in (
        "annotation_judgments",
        "annotation_adjudications",
        "annotation_audit_events",
    ):
        assert f"BEFORE TRUNCATE ON {table_name}" in ddl
