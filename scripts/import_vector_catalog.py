"""Validate and idempotently import a normalized catalog plus pgvector artifact."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from pc_build_recommender.retrieval import (
    PostgresVectorCatalogRepository,
    validate_embedding_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://pcbr:pcbr_local_only@localhost:5432/pc_build_recommender"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Require the caller to have already run Alembic to head",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate all disk hashes and row contracts without connecting to PostgreSQL",
    )
    return parser


def _upgrade_database(database_url: str) -> None:
    config = Config(str(REPOSITORY_ROOT / "db" / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, "head")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = validate_embedding_artifact(args.catalog, args.artifact_dir)
    evidence: dict[str, object] = {
        "validation": "passed",
        "product_count": artifact.product_count,
        "matrix_shape": list(artifact.vectors.shape),
        "embedding_model": artifact.embedding_model,
        "data_version": artifact.data_version,
        "index_version": artifact.index_version,
        "dataset_content_hash": artifact.dataset_content_hash,
        "embeddings_artifact_sha256": artifact.embeddings_artifact_sha256,
        "id_map_artifact_sha256": artifact.id_map_artifact_sha256,
        "encoder": artifact.manifest["encoder"],
    }
    if args.verify_only:
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    if not args.database_url.startswith("postgresql"):
        raise ValueError("--database-url must use PostgreSQL")
    if not args.skip_migrations:
        _upgrade_database(args.database_url)
    engine = create_engine(args.database_url, pool_pre_ping=True)
    try:
        result = PostgresVectorCatalogRepository(engine).import_artifact(
            artifact,
            batch_size=args.batch_size,
        )
    finally:
        engine.dispose()
    evidence["database_import"] = result.to_dict()
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
