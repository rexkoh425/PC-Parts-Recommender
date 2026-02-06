"""Inspect or persist normalized BuildCores and governed retailer artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from services.api.durability import SqlAlchemyDurableStore
from services.api.serving_release import (
    load_production_catalog_release,
    production_catalog_policy_from_entity_resolution,
)

from pc_build_recommender.catalog import (
    ProductionCatalogReadinessError,
    StreamedCatalogImportResult,
    build_db_engine,
    create_session_factory,
    init_database,
    session_scope,
    stream_processed_catalog,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the normalized BuildCores catalog, conservatively map governed offers, "
            "and optionally upsert the accepted records into SQLAlchemy storage."
        )
    )
    parser.add_argument("--buildcores", type=Path, required=True)
    offers = parser.add_mutually_exclusive_group(required=True)
    offers.add_argument(
        "--offers",
        type=Path,
        help="Governed retailer-offer JSONL artifact.",
    )
    offers.add_argument(
        "--dynacore",
        dest="legacy_dynacore_offers",
        type=Path,
        help="Deprecated alias for --offers.",
    )
    parser.add_argument("--reviewed-mappings", type=Path)
    parser.add_argument(
        "--review-evidence",
        type=Path,
        help=(
            "Optional bounded JSONL review-evidence artifact. Production imports require the "
            "exact manifest-pinned artifact, including an intentionally empty artifact when "
            "no permitted evidence is available."
        ),
    )
    parser.add_argument("--entity-resolution-evaluation", type=Path)
    parser.add_argument(
        "--entity-resolution-model",
        type=Path,
        help=(
            "Content-addressed human-trained LightGBM artifact directory. Production imports "
            "bind it to --entity-resolution-evaluation before automatic ML mappings are allowed."
        ),
    )
    parser.add_argument(
        "--allow-unpromoted-entity-resolution-shadow",
        action="store_true",
        help=(
            "Load a non-promoted human diagnostic in shadow mode; it can queue reviews but can "
            "never persist automatic ML mappings."
        ),
    )
    parser.add_argument(
        "--serving-manifest",
        type=Path,
        help=(
            "Operator-reviewed serving release manifest. Production imports load entity-"
            "resolution model, calibration, policy, evaluation, and rights authority only from "
            "this content-addressed release."
        ),
    )
    parser.add_argument(
        "--serving-manifest-sha256",
        help="Lowercase SHA-256 pinned by the deployment operator for --serving-manifest.",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Optional SQLAlchemy URL. If omitted, the command is read-only and prints a report.",
    )
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-line-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--require-production-ready",
        action="store_true",
        help="Fail closed and roll back database writes when the production policy is unmet.",
    )
    parser.add_argument(
        "--require-migrated-schema",
        action="store_true",
        help="Require Alembic-created tables instead of creating development tables.",
    )
    parser.add_argument(
        "--readiness-artifact",
        type=Path,
        help="Frozen read-only import report that the recomputed release must match exactly.",
    )
    parser.add_argument(
        "--expected-data-version",
        help="Immutable serving data version that the recomputed catalogue must match.",
    )
    parser.add_argument(
        "--verify-durable-identity",
        action="store_true",
        help="Verify every imported product and listing ID exists after the transaction commits.",
    )
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--decisions-output",
        type=Path,
        help="Optional JSON mapping-decision queue/audit output.",
    )
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def verify_release_evidence(
    result: StreamedCatalogImportResult,
    *,
    readiness_artifact: Path,
    expected_data_version: str,
) -> None:
    """Fail closed unless the live import exactly reproduces the reviewed release evidence."""

    expected = _load_json_object(
        readiness_artifact,
        description="catalogue readiness artifact",
    )
    actual = result.to_dict()
    expected_version = expected.get("data_version")
    readiness = expected.get("readiness")
    readiness_version = readiness.get("data_version") if isinstance(readiness, dict) else None
    if result.stats.data_version != expected_data_version:
        raise ValueError(
            "recomputed catalogue data version does not match the configured API data version: "
            f"{result.stats.data_version!r} != {expected_data_version!r}"
        )
    if expected_version != expected_data_version or readiness_version != expected_data_version:
        raise ValueError(
            "catalogue readiness artifact data version does not match the configured API data "
            "version"
        )
    if not isinstance(readiness, dict) or readiness.get("production_ready") is not True:
        raise ValueError("catalogue readiness artifact is not production ready")
    if expected.get("database_upserted") is not False:
        raise ValueError("catalogue readiness artifact must be a read-only import report")

    # The reviewed artifact is produced by the read-only form of this command. Database
    # persistence is the sole expected difference during the release job.
    actual["database_upserted"] = False
    if actual != expected:
        raise ValueError(
            "catalogue readiness artifact does not exactly match the recomputed products, "
            "offers, reviewed mappings, entity-resolution evidence, and rights evaluation"
        )


def main() -> int:
    args = _parser().parse_args()
    offer_path = args.offers or args.legacy_dynacore_offers
    assert offer_path is not None
    if (args.readiness_artifact is None) != (args.expected_data_version is None):
        raise ValueError(
            "--readiness-artifact and --expected-data-version must be provided together"
        )
    if (args.require_migrated_schema or args.verify_durable_identity) and not args.database_url:
        raise ValueError(
            "--require-migrated-schema and --verify-durable-identity require a database URL"
        )
    if args.readiness_artifact is not None and not args.require_production_ready:
        raise ValueError("release evidence verification requires --require-production-ready")
    if (args.serving_manifest is None) != (args.serving_manifest_sha256 is None):
        raise ValueError(
            "--serving-manifest and --serving-manifest-sha256 must be provided together"
        )
    if args.require_production_ready and args.serving_manifest is None:
        raise ValueError("production catalogue import requires a pinned serving manifest")
    if args.require_production_ready and (
        args.entity_resolution_model is not None
        or args.entity_resolution_evaluation is not None
        or args.allow_unpromoted_entity_resolution_shadow
    ):
        raise ValueError(
            "production entity-resolution authority must come only from the serving manifest"
        )

    catalog_release = None
    if args.serving_manifest is not None:
        catalog_release = load_production_catalog_release(
            args.serving_manifest,
            catalog_path=args.buildcores,
            offers_path=offer_path,
            reviewed_mappings_path=args.reviewed_mappings,
            review_evidence_path=args.review_evidence,
            expected_catalog_data_version=args.expected_data_version,
            expected_manifest_sha256=args.serving_manifest_sha256,
        )
    entity_resolution_runtime = (
        catalog_release.entity_resolution_runtime if catalog_release is not None else None
    )
    entity_resolution_evaluation_path = (
        catalog_release.entity_resolution_evaluation_path
        if catalog_release is not None
        else args.entity_resolution_evaluation
    )
    entity_resolution_policy = (
        catalog_release.entity_resolution_policy if catalog_release is not None else None
    )
    production_policy = (
        production_catalog_policy_from_entity_resolution(entity_resolution_policy)
        if entity_resolution_policy is not None
        else None
    )
    try:
        if args.database_url:
            engine = build_db_engine(args.database_url)
            try:
                durable_store = SqlAlchemyDurableStore(engine)
                if args.require_migrated_schema:
                    durable_store.verify_schema()
                else:
                    init_database(engine)
                factory = create_session_factory(engine)
                with session_scope(factory) as session:
                    result = stream_processed_catalog(
                        args.buildcores,
                        offer_path=offer_path,
                        session=session,
                        reviewed_mapping_path=args.reviewed_mappings,
                        review_evidence_path=args.review_evidence,
                        entity_resolution_evaluation_path=(entity_resolution_evaluation_path),
                        entity_resolution_model_path=(
                            args.entity_resolution_model if catalog_release is None else None
                        ),
                        entity_resolution_runtime=entity_resolution_runtime,
                        entity_resolution_policy=entity_resolution_policy,
                        entity_resolution_binding_sha256=(
                            catalog_release.entity_resolution.identity.binding_sha256
                            if catalog_release is not None
                            else None
                        ),
                        allow_unpromoted_entity_resolution=(
                            args.allow_unpromoted_entity_resolution_shadow
                        ),
                        batch_size=args.batch_size,
                        max_line_bytes=args.max_line_bytes,
                        require_production_ready=args.require_production_ready,
                        production_policy=production_policy,
                    )
                    if args.readiness_artifact is not None:
                        verify_release_evidence(
                            result,
                            readiness_artifact=args.readiness_artifact,
                            expected_data_version=args.expected_data_version,
                        )
                if args.verify_durable_identity:
                    durable_store.verify_catalog_identity(
                        product_ids=result.product_ids,
                        listing_ids=result.listing_ids,
                    )
            finally:
                engine.dispose()
        else:
            result = stream_processed_catalog(
                args.buildcores,
                offer_path=offer_path,
                reviewed_mapping_path=args.reviewed_mappings,
                review_evidence_path=args.review_evidence,
                entity_resolution_evaluation_path=entity_resolution_evaluation_path,
                entity_resolution_model_path=(
                    args.entity_resolution_model if catalog_release is None else None
                ),
                entity_resolution_runtime=entity_resolution_runtime,
                entity_resolution_policy=entity_resolution_policy,
                entity_resolution_binding_sha256=(
                    catalog_release.entity_resolution.identity.binding_sha256
                    if catalog_release is not None
                    else None
                ),
                allow_unpromoted_entity_resolution=(args.allow_unpromoted_entity_resolution_shadow),
                batch_size=args.batch_size,
                max_line_bytes=args.max_line_bytes,
                require_production_ready=args.require_production_ready,
                production_policy=production_policy,
            )
    except ProductionCatalogReadinessError as error:
        report = {
            "database_upserted": False,
            "production_gate_failed": True,
            "readiness": error.report.to_dict(),
        }
        if args.report_output:
            _atomic_json(args.report_output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    report = result.to_dict()
    if args.decisions_output:
        _atomic_json(
            args.decisions_output,
            {
                "data_version": result.stats.data_version,
                "decisions": [decision.to_dict() for decision in result.mapping_decisions],
            },
        )
    if args.report_output:
        _atomic_json(args.report_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
