"""Import one pinned production catalogue release and verify its exact durable identity.

The release is deliberately non-destructive. Existing canonical products or retailer
listings outside the pinned release stop the job before any write. Stale provenance
inside the vector import also fails the vector transaction instead of being deleted.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from services.api.durability import SqlAlchemyDurableStore
from services.api.serving_release import (
    ProductionCatalogRelease,
    load_production_catalog_release,
    production_catalog_policy_from_entity_resolution,
)
from sqlalchemy import Connection
from sqlalchemy.orm import Session, sessionmaker

from pc_build_recommender.application import ServingConfigurationError
from pc_build_recommender.catalog import (
    ProcessedCatalogData,
    create_db_engine,
    load_processed_catalog,
    seed_processed_catalog,
    session_scope,
    validate_production_readiness,
)
from pc_build_recommender.evaluation.manifest import sha256_file
from pc_build_recommender.retrieval import (
    PostgresVectorCatalogRepository,
    ValidatedEmbeddingArtifact,
    validate_embedding_artifact,
)

_EMBEDDING_FIELDS = frozenset(
    {
        "artifact_manifest",
        "data_version",
        "index_version",
        "embedding_model",
        "encoder_revision",
        "encoder_fingerprint",
        "dataset_content_hash",
        "manifest_schema_version",
        "device",
        "batch_size",
        "rrf_k",
        "encoder_bundle",
    }
)
_ARTIFACT_REFERENCE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildcores", type=Path, required=True)
    parser.add_argument("--offers", type=Path, required=True)
    parser.add_argument("--reviewed-mappings", type=Path, required=True)
    parser.add_argument("--review-evidence", type=Path, required=True)
    parser.add_argument("--serving-manifest", type=Path, required=True)
    parser.add_argument("--serving-manifest-sha256", required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--source-trust-root-sha256", required=True)
    parser.add_argument("--readiness-artifact", type=Path, required=True)
    parser.add_argument("--expected-data-version", required=True)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL SQLAlchemy URL. Production entrypoints provide DATABASE_URL.",
    )
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-line-bytes", type=int, default=8 * 1024 * 1024)
    return parser


def _exact_mapping(
    value: object,
    *,
    field: str,
    expected_fields: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ServingConfigurationError(f"{field} must be an object")
    actual_fields = set(value)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ServingConfigurationError(
            f"{field} fields do not match the serving contract; "
            f"missing={missing}, extra={extra}"
        )
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServingConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _required_sha256(value: object, *, field: str) -> str:
    digest = _required_text(value, field=field)
    if len(digest) != 64 or any(character not in _SHA256_CHARACTERS for character in digest):
        raise ServingConfigurationError(f"{field} must be a lowercase SHA-256")
    return digest


def _pinned_embedding_artifact(
    release: ProductionCatalogRelease,
) -> ValidatedEmbeddingArtifact:
    """Resolve and validate only the embedding artifact pinned by the serving manifest."""

    embedding = _exact_mapping(
        release.manifest.get("embedding"),
        field="manifest.embedding",
        expected_fields=_EMBEDDING_FIELDS,
    )
    reference = _exact_mapping(
        embedding["artifact_manifest"],
        field="manifest.embedding.artifact_manifest",
        expected_fields=_ARTIFACT_REFERENCE_FIELDS,
    )
    relative_path = Path(
        _required_text(reference["path"], field="manifest.embedding.artifact_manifest.path")
    )
    if relative_path.is_absolute() or any(part == ".." for part in relative_path.parts):
        raise ServingConfigurationError(
            "manifest.embedding.artifact_manifest.path must be relative and confined"
        )
    release_root = release.manifest_path.parent.resolve()
    artifact_manifest = (release_root / relative_path).resolve()
    try:
        artifact_manifest.relative_to(release_root)
    except ValueError as error:
        raise ServingConfigurationError(
            "manifest.embedding.artifact_manifest.path escapes the serving release"
        ) from error
    if artifact_manifest.name != "manifest.json" or not artifact_manifest.is_file():
        raise ServingConfigurationError(
            "manifest.embedding.artifact_manifest must resolve to an existing manifest.json"
        )
    declared_size = reference["size_bytes"]
    if type(declared_size) is not int or declared_size < 0:
        raise ServingConfigurationError(
            "manifest.embedding.artifact_manifest.size_bytes must be non-negative"
        )
    if artifact_manifest.stat().st_size != declared_size:
        raise ServingConfigurationError(
            "embedding artifact manifest size does not match the serving manifest"
        )
    declared_sha256 = _required_sha256(
        reference["sha256"],
        field="manifest.embedding.artifact_manifest.sha256",
    )
    if sha256_file(artifact_manifest) != declared_sha256:
        raise ServingConfigurationError(
            "embedding artifact manifest SHA-256 does not match the serving manifest"
        )

    artifact = validate_embedding_artifact(
        release.catalog_path,
        artifact_manifest.parent,
    )
    encoder_manifest = artifact.manifest.get("encoder")
    encoder_revision = (
        encoder_manifest.get("model_revision")
        if isinstance(encoder_manifest, Mapping)
        else None
    )
    expected_identity = {
        "data_version": artifact.data_version,
        "index_version": artifact.index_version,
        "embedding_model": artifact.embedding_model,
        "encoder_fingerprint": artifact.encoder_fingerprint,
        "dataset_content_hash": artifact.dataset_content_hash,
        "manifest_schema_version": artifact.manifest.get("schema_version"),
        "encoder_revision": encoder_revision,
    }
    mismatches = sorted(
        field_name
        for field_name, actual_value in expected_identity.items()
        if embedding[field_name] != actual_value
    )
    if mismatches:
        raise ServingConfigurationError(
            "embedding artifact identity does not match the serving manifest: "
            + ", ".join(mismatches)
        )
    return artifact


def _release_report(data: ProcessedCatalogData) -> dict[str, Any]:
    if data.readiness is None:
        raise ValueError("processed catalogue has no measured readiness report")
    return {
        **data.stats.to_dict(),
        "database_upserted": False,
        "review_evidence_count": len(data.review_evidence),
        "readiness": data.readiness.to_dict(),
    }


def _verify_readiness_artifact(
    data: ProcessedCatalogData,
    *,
    readiness_artifact: Path,
    expected_data_version: str,
) -> None:
    try:
        expected = json.loads(readiness_artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"catalogue readiness artifact is invalid: {error}") from error
    if not isinstance(expected, dict):
        raise ValueError("catalogue readiness artifact must be a JSON object")
    if data.stats.data_version != expected_data_version:
        raise ValueError(
            "recomputed catalogue data version does not match the configured API data version"
        )
    if expected != _release_report(data):
        raise ValueError(
            "catalogue readiness artifact does not exactly match the recomputed pinned release"
        )


def _load_release_data(
    release: ProductionCatalogRelease,
    *,
    max_line_bytes: int,
) -> ProcessedCatalogData:
    policy = production_catalog_policy_from_entity_resolution(
        release.entity_resolution_policy
    )
    data = load_processed_catalog(
        release.catalog_path,
        offer_path=release.offers_path,
        reviewed_mapping_path=release.reviewed_mappings_path,
        review_evidence_path=release.review_evidence_path,
        entity_resolution_evaluation_path=release.entity_resolution_evaluation_path,
        entity_resolution_runtime=release.entity_resolution_runtime,
        entity_resolution_policy=release.entity_resolution_policy,
        entity_resolution_binding_sha256=release.entity_resolution.identity.binding_sha256,
        require_production_entity_resolution=True,
        production_policy=policy,
        max_line_bytes=max_line_bytes,
    )
    if data.readiness is None:
        raise ValueError("processed catalogue has no measured readiness report")
    validate_production_readiness(data.readiness, policy)
    return data


def _release_session_factory(connection: Connection) -> sessionmaker[Session]:
    """Create sessions that cannot commit past the outer release transaction.

    The processed importer and vector importer deliberately own their normal unit-of-work
    boundaries. Binding both to one already-open connection in rollback-only join mode keeps
    their inner commits from propagating to the outer release transaction. A later vector or
    exact-identity failure therefore rolls back the earlier catalogue upsert as well.
    """

    return sessionmaker(
        bind=connection,
        class_=Session,
        expire_on_commit=False,
        autoflush=False,
        join_transaction_mode="rollback_only",
    )


def run_catalog_release(
    *,
    buildcores: Path,
    offers: Path,
    reviewed_mappings: Path,
    review_evidence: Path,
    serving_manifest: Path,
    serving_manifest_sha256: str,
    source_registry: Path,
    source_trust_root_sha256: str,
    readiness_artifact: Path,
    expected_data_version: str,
    database_url: str,
    batch_size: int,
    max_line_bytes: int,
) -> dict[str, Any]:
    """Run the fail-closed production catalogue release sequence."""

    if not database_url.startswith(("postgresql://", "postgresql+")):
        raise ValueError("production catalogue release requires a PostgreSQL database URL")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_line_bytes < 1:
        raise ValueError("max_line_bytes must be positive")

    release = load_production_catalog_release(
        serving_manifest,
        catalog_path=buildcores,
        offers_path=offers,
        reviewed_mappings_path=reviewed_mappings,
        review_evidence_path=review_evidence,
        current_source_registry_path=source_registry,
        expected_source_trust_root_sha256=source_trust_root_sha256,
        expected_catalog_data_version=expected_data_version,
        expected_manifest_sha256=serving_manifest_sha256,
    )
    embedding_artifact = _pinned_embedding_artifact(release)
    data = _load_release_data(release, max_line_bytes=max_line_bytes)
    _verify_readiness_artifact(
        data,
        readiness_artifact=readiness_artifact,
        expected_data_version=expected_data_version,
    )
    if {product.product_id: product for product in embedding_artifact.products} != {
        product.product_id: product for product in data.products
    }:
        raise ServingConfigurationError(
            "embedding artifact canonical products do not exactly match the processed release"
        )

    engine = create_db_engine(database_url)
    try:
        store = SqlAlchemyDurableStore(engine)
        store.verify_schema()
        expected_product_ids = tuple(product.product_id for product in data.products)
        expected_listing_ids = tuple(listing.listing_id for listing in data.listings)
        store.verify_no_unexpected_catalog_ids(
            product_ids=expected_product_ids,
            listing_ids=expected_listing_ids,
        )

        # Keep catalogue rows, release-bound search documents/vectors, and the final exact
        # identity check inside one outer transaction. The generic catalogue upsert correctly
        # invalidates ``search_document_hash``; the pinned vector import must restore it before
        # this transaction is allowed to commit.
        with engine.begin() as connection:
            release_factory = _release_session_factory(connection)
            release_store = SqlAlchemyDurableStore(engine, release_factory)
            with session_scope(release_factory) as session:
                seed_processed_catalog(session, data)

            vector_result = PostgresVectorCatalogRepository(connection).import_artifact(
                embedding_artifact,
                batch_size=batch_size,
                reconcile_stale_provenance=False,
            )
            release_store.verify_catalog_identity(
                product_ids=expected_product_ids,
                listing_ids=expected_listing_ids,
                canonical_products=data.products,
                retailer_listings=data.listings,
            )
    finally:
        engine.dispose()

    return {
        "schema_version": "pc-build-recommender.catalog-release-import.v1",
        "status": "passed",
        "serving_manifest_sha256": release.manifest_sha256,
        "data_version": data.stats.data_version,
        "canonical_products": len(data.products),
        "retailer_listings": len(data.listings),
        "authorized_source_release": {
            "manifest_sha256": release.source_release.manifest_sha256,
            "source_name": release.source_release.source_name,
            "raw_snapshot_sha256": release.source_release.raw_snapshot_sha256,
            "processed_run_sha256": release.source_release.processed_run_sha256,
            "authority_expires_at": release.source_release.authority_expires_at.isoformat(),
            "accepted_count": release.source_release.accepted_count,
            "rejected_count": release.source_release.rejected_count,
        },
        "embedding": {
            "data_version": embedding_artifact.data_version,
            "index_version": embedding_artifact.index_version,
            "embedding_model": embedding_artifact.embedding_model,
            "dataset_content_hash": embedding_artifact.dataset_content_hash,
            "product_count": embedding_artifact.product_count,
            "database_import": vector_result.to_dict(),
        },
        "identity_verification": "exact",
        "atomic_release_transaction": True,
        "stale_row_policy": "fail_closed",
        "destructive_reconciliation_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.database_url:
        raise ValueError("production catalogue release requires DATABASE_URL")
    report = run_catalog_release(
        buildcores=args.buildcores,
        offers=args.offers,
        reviewed_mappings=args.reviewed_mappings,
        review_evidence=args.review_evidence,
        serving_manifest=args.serving_manifest,
        serving_manifest_sha256=args.serving_manifest_sha256,
        source_registry=args.source_registry,
        source_trust_root_sha256=args.source_trust_root_sha256,
        readiness_artifact=args.readiness_artifact,
        expected_data_version=args.expected_data_version,
        database_url=args.database_url,
        batch_size=args.batch_size,
        max_line_bytes=args.max_line_bytes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
