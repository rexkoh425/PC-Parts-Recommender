"""Fetch licensed sources and controlled imports into reproducible local datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = REPOSITORY_ROOT / "packages" / "core" / "src"
for import_root in (REPOSITORY_ROOT, CORE_SOURCE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pipelines.checks.quality import (  # noqa: E402
    evaluate_batch_quality_against_previous,
    write_quality_report,
)
from pipelines.parsing.writer import ProcessedArtifacts, write_parsed_batch  # noqa: E402
from pipelines.retention.publication import (  # noqa: E402
    begin_web_processed_publication,
    publish_web_processed_publication,
    seal_web_processed_publication,
)
from pipelines.retention.registry import (  # noqa: E402
    load_governed_web_source_admissions,
    load_restricted_web_sources,
)
from pipelines.retention.web import (  # noqa: E402
    WEB_PROCESSED_RETENTION_RECEIPT,
    WEB_PROCESSED_RETENTION_SCHEMA_VERSION,
    WebRetentionError,
    write_web_processed_retention_receipt,
)
from pipelines.source_release import (  # noqa: E402
    AuthorizedBatchAuthorityArtifacts,
    AuthorizedBatchReleaseArtifacts,
    VerifiedAuthorizedBatchRelease,
    publish_awin_production_batch_release,
)
from pipelines.sources.awin_feed import AwinLocalFeedAdapter  # noqa: E402
from pipelines.sources.base import ParsedBatch, RawSnapshot, sha256_file  # noqa: E402
from pipelines.sources.bizgram_pdf import BizgramControlledPDFAdapter  # noqa: E402
from pipelines.sources.blender import BlenderOpenDataAdapter  # noqa: E402
from pipelines.sources.buildcores import BuildCoresOpenDBAdapter  # noqa: E402
from pipelines.sources.dynacore_pdf import DynacoreControlledPDFAdapter  # noqa: E402
from pipelines.sources.mlperf import MLPerfInferenceAdapter  # noqa: E402
from pipelines.sources.pci_ids import (  # noqa: E402
    DEFAULT_RECORD_LIMIT as DEFAULT_PCI_IDS_RECORD_LIMIT,
)
from pipelines.sources.pci_ids import PCIIDRepositoryAdapter  # noqa: E402
from pipelines.sources.retailer_csv import (  # noqa: E402
    ConsentedRetailerCSVAdapter,
    RetailerFeedPolicy,
)
from pipelines.sources.signed_policy import verify_signed_policy  # noqa: E402
from pipelines.sources.web_product import (  # noqa: E402
    WebProductCrawlerAdapter,
    WebSourcePolicy,
)
from pipelines.sources.wikidata import (  # noqa: E402
    DEFAULT_MAX_RECORDS as DEFAULT_WIKIDATA_RECORD_LIMIT,
)
from pipelines.sources.wikidata import (  # noqa: E402
    WikidataEnrichmentAdapter,
    load_wikidata_candidates,
)

from pc_build_recommender.domain.enums import ComponentCategory  # noqa: E402

PORTFOLIO_BUILDCORES_LIMITS = {
    "CPU": 250,
    "GPU": 350,
    "Motherboard": 650,
    "RAM": 450,
    "Storage": 450,
    "PSU": 350,
    "CPUCooler": 250,
    "PCCase": 250,
}

WebProcessedRetentionError = WebRetentionError
__all__ = [
    "WEB_PROCESSED_RETENTION_RECEIPT",
    "WEB_PROCESSED_RETENTION_SCHEMA_VERSION",
    "WebProcessedRetentionError",
    "main",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=(
            "all",
            "buildcores",
            "blender",
            "mlperf",
            "pci_ids",
            "wikidata",
            "bizgram",
            "dynacore",
            "awin_feed",
            "retailer_csv",
            "web_product",
        ),
        help="Source to process; repeat to select several. Defaults to open sources only.",
    )
    parser.add_argument("--raw-root", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument(
        "--processed-root", type=Path, default=REPOSITORY_ROOT / "data" / "processed"
    )
    parser.add_argument("--buildcores-archive", type=Path)
    parser.add_argument(
        "--buildcores-profile",
        choices=("fast", "portfolio", "full"),
        default="portfolio",
    )
    parser.add_argument("--blender-archive", type=Path)
    parser.add_argument("--blender-limit", type=int, default=3_000)
    parser.add_argument("--blender-scan-limit", type=int, default=10_000)
    parser.add_argument(
        "--blender-scan-all",
        action="store_true",
        help="Scan every submission in the pinned Blender snapshot.",
    )
    parser.add_argument(
        "--blender-selection",
        choices=("head", "hash_sample"),
        default="head",
        help="Choose the stream head or a deterministic bounded sample.",
    )
    parser.add_argument("--blender-sample-seed", default="buildsignal-blender-v1")
    parser.add_argument("--mlperf-summary", type=Path)
    parser.add_argument("--pci-ids-snapshot", type=Path)
    parser.add_argument(
        "--pci-ids-format",
        choices=("gzip", "plain"),
        default="gzip",
        help="Use the official compressed snapshot by default; also supports pci.ids.",
    )
    parser.add_argument(
        "--pci-ids-sha256",
        help="Optional expected SHA-256 for an exact remote or local snapshot pin.",
    )
    parser.add_argument(
        "--pci-ids-record-limit",
        type=int,
        default=DEFAULT_PCI_IDS_RECORD_LIMIT,
        help="Maximum vendor, device, and subsystem alias records retained in memory.",
    )
    parser.add_argument(
        "--wikidata-candidates",
        type=Path,
        help="Bounded JSONL canonical-product candidates for Wikidata identity enrichment.",
    )
    parser.add_argument(
        "--wikidata-response-json",
        type=Path,
        help="Controlled raw API-response fixture for deterministic reproduction without network.",
    )
    parser.add_argument(
        "--wikidata-limit",
        type=int,
        default=DEFAULT_WIKIDATA_RECORD_LIMIT,
        help="Maximum candidate products to search and enrichment records to retain.",
    )
    parser.add_argument(
        "--wikidata-category",
        action="append",
        choices=tuple(category.value for category in ComponentCategory),
        help=(
            "Optional canonical-product category to collect; repeat for several categories. "
            "The filter is applied before any official API request."
        ),
    )
    parser.add_argument("--wikidata-language", default="en")
    parser.add_argument("--wikidata-search-limit", type=int, default=3)
    parser.add_argument("--wikidata-timeout", type=float, default=30.0)
    parser.add_argument(
        "--wikidata-request-delay",
        type=float,
        default=0.2,
        help="Delay after successful API requests; the minimum accepted value is 0.2 seconds.",
    )
    parser.add_argument("--dynacore-pdf", type=Path)
    parser.add_argument(
        "--include-controlled-dynacore",
        action="store_true",
        help="Include the development-only Dynacore source when --source all is used.",
    )
    parser.add_argument("--bizgram-pdf", type=Path)
    parser.add_argument(
        "--include-controlled-bizgram",
        action="store_true",
        help="Include the quarantined local Bizgram source when --source all is used.",
    )
    parser.add_argument("--retailer-csv", type=Path)
    parser.add_argument(
        "--awin-feed",
        type=Path,
        help="Local Awin CSV or gzip feed already acquired by an authorised operator.",
    )
    parser.add_argument(
        "--awin-policy-json",
        type=Path,
        help="Exact signed Awin policy envelope; never a feed download URL.",
    )
    parser.add_argument(
        "--awin-policy-signature",
        type=Path,
        help="Detached Ed25519 signature document for --awin-policy-json.",
    )
    parser.add_argument(
        "--awin-trust-root",
        type=Path,
        help="Local Ed25519 policy trust-root document.",
    )
    parser.add_argument(
        "--awin-trust-root-sha256",
        help="Externally pinned SHA-256 of the exact Awin policy trust-root bytes.",
    )
    parser.add_argument(
        "--retailer-policy-json",
        type=Path,
        help=(
            "Path to a complete retailer policy including machine-readable rights, "
            "consent, retention, and deletion terms."
        ),
    )
    parser.add_argument(
        "--web-policy-json",
        type=Path,
        help=(
            "Explicit web-source policy containing URL/category allowlists, acquisition "
            "authority, exact terms hash, rights, and crawl limits."
        ),
    )
    parser.add_argument(
        "--web-url",
        action="append",
        help="Approved product URL to crawl; repeat for multiple policy-mapped pages.",
    )
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "source_registry.yaml",
        help=(
            "Reviewed source registry used to reject hosts whose current terms prohibit "
            "automated extraction before any governed-web request is sent."
        ),
    )
    parser.add_argument(
        "--authorized-release-root",
        type=Path,
        help=(
            "Content-addressed control bundles for signed production source batches. "
            "Defaults to <processed-root>/_authorized-source-releases."
        ),
    )
    parser.add_argument("--no-parquet", action="store_true")
    return parser


def _write_and_report(
    *,
    batch: ParsedBatch,
    snapshot: RawSnapshot,
    processed_root: Path,
    maximum_rejection_rate: float,
    variant: str | None,
    prefer_parquet: bool,
) -> dict[str, Any]:
    artifacts = write_parsed_batch(
        batch,
        processed_root=processed_root,
        prefer_parquet=prefer_parquet,
        variant=variant,
    )
    quality = evaluate_batch_quality_against_previous(
        batch,
        processed_root=processed_root,
        variant=variant,
        maximum_rejection_rate=maximum_rejection_rate,
    )
    quality_path = write_quality_report(quality, artifacts.output_directory / "data-quality.json")
    return _summary(
        snapshot=snapshot,
        batch=batch,
        artifacts=artifacts,
        quality_status=quality.status,
        quality_path=quality_path,
    )


def _summary(
    *,
    snapshot: RawSnapshot,
    batch: ParsedBatch,
    artifacts: ProcessedArtifacts,
    quality_status: str,
    quality_path: Path,
) -> dict[str, Any]:
    return {
        "source_name": snapshot.source_name,
        "source_url": snapshot.source_url,
        "raw_path": str(snapshot.path),
        "raw_sha256": snapshot.content_sha256,
        "raw_byte_count": snapshot.byte_count,
        "raw_snapshot_reused": snapshot.reused,
        "accepted_count": batch.accepted_count,
        "recorded_rejection_count": batch.rejected_count,
        "statistics": batch.statistics,
        "records_path": str(artifacts.records_jsonl),
        "records_sha256": sha256_file(artifacts.records_jsonl),
        "rejections_path": str(artifacts.rejections_jsonl),
        "manifest_path": str(artifacts.manifest_json),
        "parquet_path": str(artifacts.parquet_path) if artifacts.parquet_path else None,
        "data_quality_status": quality_status,
        "data_quality_path": str(quality_path),
    }


def _run_buildcores(args: argparse.Namespace) -> dict[str, Any]:
    adapter = BuildCoresOpenDBAdapter(raw_root=args.raw_root)
    snapshot = adapter.fetch(archive_path=args.buildcores_archive)
    if args.buildcores_profile == "fast":
        batch = adapter.parse(snapshot, per_category_limit=100)
        variant = "fast-800"
    elif args.buildcores_profile == "portfolio":
        batch = adapter.parse(snapshot, per_category_limits=PORTFOLIO_BUILDCORES_LIMITS)
        variant = "portfolio-3000"
    else:
        batch = adapter.parse(snapshot, per_category_limit=None)
        variant = "full"
    return _write_and_report(
        batch=batch,
        snapshot=snapshot,
        processed_root=args.processed_root,
        maximum_rejection_rate=0.05,
        variant=variant,
        prefer_parquet=not args.no_parquet,
    )


def _run_blender(args: argparse.Namespace) -> dict[str, Any]:
    adapter = BlenderOpenDataAdapter(raw_root=args.raw_root)
    snapshot = adapter.fetch(archive_path=args.blender_archive)
    batch = adapter.parse(
        snapshot,
        max_observations=args.blender_limit,
        max_submissions_scan=None if args.blender_scan_all else args.blender_scan_limit,
        selection=args.blender_selection,
        sample_seed=args.blender_sample_seed,
    )
    scan_label = "all" if args.blender_scan_all else str(args.blender_scan_limit)
    return _write_and_report(
        batch=batch,
        snapshot=snapshot,
        processed_root=args.processed_root,
        maximum_rejection_rate=0.10,
        variant=f"{args.blender_selection}-{args.blender_limit}-scan-{scan_label}",
        prefer_parquet=not args.no_parquet,
    )


def _run_mlperf(args: argparse.Namespace) -> dict[str, Any]:
    adapter = MLPerfInferenceAdapter(raw_root=args.raw_root)
    snapshot = adapter.fetch(summary_path=args.mlperf_summary)
    batch = adapter.parse(snapshot)
    return _write_and_report(
        batch=batch,
        snapshot=snapshot,
        processed_root=args.processed_root,
        maximum_rejection_rate=0.05,
        variant=None,
        prefer_parquet=not args.no_parquet,
    )


def _run_pci_ids(args: argparse.Namespace) -> dict[str, Any]:
    adapter = PCIIDRepositoryAdapter(raw_root=args.raw_root)
    snapshot = adapter.fetch(
        snapshot_path=args.pci_ids_snapshot,
        snapshot_format=args.pci_ids_format,
        expected_sha256=args.pci_ids_sha256,
    )
    batch = adapter.parse(snapshot, max_records=args.pci_ids_record_limit)
    return _write_and_report(
        batch=batch,
        snapshot=snapshot,
        processed_root=args.processed_root,
        maximum_rejection_rate=0.01,
        variant=f"limit_{args.pci_ids_record_limit}",
        prefer_parquet=not args.no_parquet,
    )


def _run_wikidata(args: argparse.Namespace) -> dict[str, Any]:
    if args.wikidata_response_json is None and args.wikidata_candidates is None:
        raise ValueError(
            "--wikidata-candidates is required unless --wikidata-response-json is supplied"
        )
    adapter = WikidataEnrichmentAdapter(
        raw_root=args.raw_root,
        request_delay_seconds=args.wikidata_request_delay,
    )
    if args.wikidata_response_json is not None:
        snapshot = adapter.fetch(
            response_path=args.wikidata_response_json,
            max_records=args.wikidata_limit,
            search_limit=args.wikidata_search_limit,
            language=args.wikidata_language,
            timeout_seconds=args.wikidata_timeout,
        )
    else:
        candidates = load_wikidata_candidates(
            args.wikidata_candidates,
            max_records=args.wikidata_limit,
            categories=args.wikidata_category,
        )
        snapshot = adapter.fetch(
            candidates,
            max_records=args.wikidata_limit,
            search_limit=args.wikidata_search_limit,
            language=args.wikidata_language,
            timeout_seconds=args.wikidata_timeout,
        )
    batch = adapter.parse(snapshot, max_records=args.wikidata_limit)
    category_label = "all" if not args.wikidata_category else "-".join(args.wikidata_category)
    return _write_and_report(
        batch=batch,
        snapshot=snapshot,
        processed_root=args.processed_root,
        maximum_rejection_rate=0.80,
        variant=f"{batch.statistics['language']}-{category_label}-limit-{args.wikidata_limit}",
        prefer_parquet=not args.no_parquet,
    )


def _run_dynacore(args: argparse.Namespace) -> dict[str, Any]:
    if args.dynacore_pdf is None:
        raise ValueError("--dynacore-pdf is required for the controlled Dynacore import")
    adapter = DynacoreControlledPDFAdapter(raw_root=args.raw_root)
    snapshot = adapter.fetch(pdf_path=args.dynacore_pdf)
    batch = adapter.parse(snapshot)
    return _write_and_report(
        batch=batch,
        snapshot=snapshot,
        processed_root=args.processed_root,
        maximum_rejection_rate=0.30,
        variant=None,
        prefer_parquet=not args.no_parquet,
    )


def _run_bizgram(args: argparse.Namespace) -> dict[str, Any]:
    if args.bizgram_pdf is None:
        raise ValueError("--bizgram-pdf is required for the controlled Bizgram import")
    adapter = BizgramControlledPDFAdapter(raw_root=args.raw_root)
    snapshot = adapter.fetch(pdf_path=args.bizgram_pdf)
    batch = adapter.parse(snapshot)
    return _write_and_report(
        batch=batch,
        snapshot=snapshot,
        processed_root=args.processed_root,
        # Almost all rows in this broad office/IT price list are intentionally
        # quarantined as out-of-scope or ambiguous.  A high threshold is not a
        # production-quality claim; it allows the explicit review queue to be written.
        maximum_rejection_rate=0.98,
        variant=None,
        prefer_parquet=not args.no_parquet,
    )


def _run_retailer_csv(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "--retailer-csv": args.retailer_csv,
        "--retailer-policy-json": args.retailer_policy_json,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"retailer CSV import is missing arguments: {', '.join(missing)}")
    try:
        policy_payload = json.loads(args.retailer_policy_json.read_text(encoding="utf-8"))
        if not isinstance(policy_payload, dict):
            raise TypeError("retailer policy root must be an object")
        policy = RetailerFeedPolicy.from_mapping(policy_payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid retailer policy: {exc}") from exc
    adapter = ConsentedRetailerCSVAdapter(raw_root=args.raw_root, policy=policy)
    snapshot = adapter.fetch(csv_path=args.retailer_csv)
    batch = adapter.parse(snapshot)
    return _write_and_report(
        batch=batch,
        snapshot=snapshot,
        processed_root=args.processed_root,
        maximum_rejection_rate=0.10,
        variant=None,
        prefer_parquet=not args.no_parquet,
    )


def _run_awin_feed(args: argparse.Namespace) -> dict[str, Any]:
    """Verify authority and materialise one operator-supplied local Awin feed.

    This command deliberately accepts no network URL, API key, or download
    credential. Its JSON result identifies inputs by content hash and safe
    ``awin://`` provenance, never by the operator's local input paths.
    """

    required = {
        "--awin-feed": args.awin_feed,
        "--awin-policy-json": args.awin_policy_json,
        "--awin-policy-signature": args.awin_policy_signature,
        "--awin-trust-root": args.awin_trust_root,
        "--awin-trust-root-sha256": args.awin_trust_root_sha256,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Awin feed import is missing arguments: {', '.join(missing)}")

    try:
        verified_policy = verify_signed_policy(
            policy_path=args.awin_policy_json,
            signature_path=args.awin_policy_signature,
            trust_root_path=args.awin_trust_root,
            expected_trust_root_sha256=args.awin_trust_root_sha256,
        )
    except (OSError, TypeError, ValueError) as exc:
        # Verification exceptions can include local paths. Keep the CLI boundary
        # useful but credential-safe by returning only a stable failure class.
        raise ValueError(f"Awin signed policy verification failed ({type(exc).__name__})") from None

    try:
        adapter = AwinLocalFeedAdapter(
            raw_root=args.raw_root,
            verified_policy=verified_policy,
        )
        snapshot = adapter.fetch(feed_path=args.awin_feed)
        artifacts = adapter.materialize(snapshot, processed_root=args.processed_root)
        quality_payload = json.loads(artifacts.quality_json.read_text(encoding="utf-8"))
        if not isinstance(quality_payload, dict) or quality_payload.get("status") not in {
            "pass",
            "warning",
            "fail",
        }:
            raise ValueError("published Awin quality report is invalid")
        policy = adapter.policy
        production_release: VerifiedAuthorizedBatchRelease | None = None
        if policy.production_catalog_eligible:
            release_root = (
                args.authorized_release_root
                if args.authorized_release_root is not None
                else args.processed_root / "_authorized-source-releases"
            )
            production_release = publish_awin_production_batch_release(
                release_root=release_root,
                artifacts=AuthorizedBatchReleaseArtifacts(
                    raw_snapshot=snapshot.raw.path,
                    raw_metadata=snapshot.raw.metadata_path,
                    authorization_receipt=snapshot.authorization_receipt_path,
                    records=artifacts.records_jsonl,
                    rejections=artifacts.rejections_jsonl,
                    processed_manifest=artifacts.manifest_json,
                    quality_report=artifacts.quality_json,
                ),
                authority=AuthorizedBatchAuthorityArtifacts(
                    policy=args.awin_policy_json,
                    policy_signature=args.awin_policy_signature,
                    trust_root=args.awin_trust_root,
                    source_registry=args.source_registry,
                ),
                expected_trust_root_sha256=args.awin_trust_root_sha256,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        # Feed/parser errors may include the operator's local filename. Do not
        # reflect it into structured logs or command output.
        raise ValueError(f"Awin feed ingestion failed ({type(exc).__name__})") from None

    return {
        "source_name": policy.source_name,
        "source_uri": policy.source_uri,
        "raw_snapshot": {
            "sha256": snapshot.raw.content_sha256,
            "byte_count": snapshot.raw.byte_count,
            "reused": snapshot.raw.reused,
            "artifact_path": str(snapshot.raw.path),
        },
        "authorization_receipt": {
            "sha256": sha256_file(snapshot.authorization_receipt_path),
            "artifact_path": str(snapshot.authorization_receipt_path),
        },
        "policy_authority": {
            "policy_id": verified_policy.policy_id,
            "policy_sha256": verified_policy.policy_sha256,
            "signature_sha256": verified_policy.signature_sha256,
            "trust_root_sha256": verified_policy.trust_root_sha256,
            "signer_key_id": verified_policy.key_id,
        },
        "accepted_count": artifacts.accepted_count,
        "recorded_rejection_count": artifacts.rejected_count,
        "processed_run_sha256": artifacts.output_directory.name,
        "processed_run_reused": artifacts.reused,
        "records": {
            "sha256": sha256_file(artifacts.records_jsonl),
            "artifact_path": str(artifacts.records_jsonl),
        },
        "rejections": {
            "sha256": sha256_file(artifacts.rejections_jsonl),
            "artifact_path": str(artifacts.rejections_jsonl),
        },
        "manifest": {
            "sha256": sha256_file(artifacts.manifest_json),
            "artifact_path": str(artifacts.manifest_json),
        },
        "data_quality_status": str(quality_payload["status"]),
        "data_quality": {
            "sha256": sha256_file(artifacts.quality_json),
            "artifact_path": str(artifacts.quality_json),
        },
        "production_release": (
            {
                "manifest_sha256": production_release.manifest_sha256,
                "content_sha256": production_release.content_sha256,
                "artifact_path": str(production_release.manifest_path),
                "source_registry_sha256": production_release.source_registry_sha256,
                "reused": production_release.reused,
            }
            if production_release is not None
            else None
        ),
    }


def _run_web_product(args: argparse.Namespace) -> dict[str, Any]:
    if args.web_policy_json is None or not args.web_url:
        raise ValueError("web product crawl requires --web-policy-json and at least one --web-url")
    try:
        policy_payload = json.loads(args.web_policy_json.read_text(encoding="utf-8"))
        if not isinstance(policy_payload, dict):
            raise TypeError("web source policy root must be an object")
        policy = WebSourcePolicy.from_mapping(policy_payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid web source policy: {exc}") from exc
    registry_path = Path(
        getattr(args, "source_registry", REPOSITORY_ROOT / "data" / "source_registry.yaml")
    )
    admissions = {
        admission.source_name: admission
        for admission in load_governed_web_source_admissions(registry_path)
    }
    admission = admissions.get(policy.source_name)
    if admission is None:
        raise ValueError(
            "web source policy is not a registered governed-web source; add a reviewed source "
            "registry entry before crawling"
        )
    if policy.allowed_hosts != admission.allowed_hosts:
        raise ValueError(
            "web source policy allowed_hosts do not match the reviewed source registry entry"
        )
    if policy.usage_scope.value != admission.usage_scope:
        raise ValueError(
            "web source policy usage_scope does not match the reviewed source registry entry"
        )
    for restricted in load_restricted_web_sources(registry_path):
        matched_hosts = sorted(
            host
            for host in policy.allowed_hosts
            if any(
                host == restricted_host or host.endswith(f".{restricted_host}")
                for restricted_host in restricted.hosts
            )
        )
        if matched_hosts:
            raise ValueError(
                "web source policy host is restricted by source registry entry "
                f"{restricted.source_name!r}; obtain written consent and complete a new "
                "terms review before removing that restriction "
                f"(matched: {', '.join(matched_hosts)})"
            )
    result = WebProductCrawlerAdapter(raw_root=args.raw_root, policy=policy).crawl(args.web_url)
    if not result.pages:
        raise WebProcessedRetentionError(
            "governed web crawl produced no page retrieval time for processed retention"
        )
    publication = begin_web_processed_publication(
        processed_root=args.processed_root,
        source_name=policy.source_name,
        run_sha256=result.batch.snapshot_sha256,
    )
    artifacts = write_parsed_batch(
        result.batch,
        processed_root=publication.workspace_processed_root,
        prefer_parquet=not args.no_parquet,
    )
    if artifacts.output_directory.resolve() != publication.staged_run_directory.resolve():
        raise WebProcessedRetentionError(
            "governed web writer returned an unexpected processed run directory"
        )
    quality = evaluate_batch_quality_against_previous(
        result.batch,
        processed_root=args.processed_root,
        variant=None,
        maximum_rejection_rate=0.25,
    )
    quality_path = write_quality_report(quality, artifacts.output_directory / "data-quality.json")
    write_web_processed_retention_receipt(
        processed_root=publication.workspace_processed_root,
        output_directory=artifacts.output_directory,
        policy=policy,
        retrieval_started_at=result.retrieval_started_at,
        retrieval_completed_at=result.retrieval_completed_at,
    )
    seal_web_processed_publication(publication)
    final_directory = publish_web_processed_publication(publication)
    artifacts = ProcessedArtifacts(
        output_directory=final_directory,
        records_jsonl=final_directory / artifacts.records_jsonl.name,
        rejections_jsonl=final_directory / artifacts.rejections_jsonl.name,
        manifest_json=final_directory / artifacts.manifest_json.name,
        parquet_path=(
            final_directory / artifacts.parquet_path.name if artifacts.parquet_path else None
        ),
        accepted_count=artifacts.accepted_count,
        rejected_count=artifacts.rejected_count,
    )
    quality_path = final_directory / quality_path.name
    receipt_path = final_directory / WEB_PROCESSED_RETENTION_RECEIPT
    return {
        "source_name": result.batch.source_name,
        "crawl_run_sha256": result.batch.snapshot_sha256,
        "policy_fingerprint": result.policy_fingerprint,
        "development_only": policy.development_only,
        "retention_receipt_path": str(receipt_path),
        "retention_receipt_sha256": sha256_file(receipt_path),
        "page_snapshots": [
            {
                "requested_url": page.requested_url,
                "final_url": page.final_url,
                "raw_path": str(page.snapshot.path),
                "raw_sha256": page.snapshot.content_sha256,
                "raw_byte_count": page.snapshot.byte_count,
                "receipt_path": str(page.snapshot.metadata_path),
                "receipt_sha256": sha256_file(page.snapshot.metadata_path),
                "retrieved_at": page.snapshot.retrieved_at.isoformat(),
                "not_modified": page.not_modified,
            }
            for page in result.pages
        ],
        "robots_sha256_by_host": dict(result.robots_sha256_by_host),
        "terms_snapshot_sha256": result.terms_snapshot_sha256,
        "terms_post_snapshot_sha256": result.terms_post_snapshot_sha256,
        "terms_canonical_sha256": result.terms_canonical_sha256,
        "accepted_count": result.batch.accepted_count,
        "recorded_rejection_count": result.batch.rejected_count,
        "statistics": result.batch.statistics,
        "records_path": str(artifacts.records_jsonl),
        "records_sha256": sha256_file(artifacts.records_jsonl),
        "rejections_path": str(artifacts.rejections_jsonl),
        "manifest_path": str(artifacts.manifest_json),
        "parquet_path": str(artifacts.parquet_path) if artifacts.parquet_path else None,
        "data_quality_status": quality.status,
        "data_quality_path": str(quality_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = set(args.source or ("buildcores", "blender", "mlperf"))
    if "all" in selected:
        selected = {"buildcores", "blender", "mlperf", "pci_ids"}
        if args.include_controlled_dynacore:
            selected.add("dynacore")
        if args.include_controlled_bizgram:
            selected.add("bizgram")
    runners = {
        "buildcores": _run_buildcores,
        "blender": _run_blender,
        "mlperf": _run_mlperf,
        "pci_ids": _run_pci_ids,
        "wikidata": _run_wikidata,
        "bizgram": _run_bizgram,
        "dynacore": _run_dynacore,
        "awin_feed": _run_awin_feed,
        "retailer_csv": _run_retailer_csv,
        "web_product": _run_web_product,
    }
    summaries: list[dict[str, Any]] = []
    try:
        for source_name in sorted(selected):
            summaries.append(runners[source_name](args))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    quality_failed = any(item["data_quality_status"] == "fail" for item in summaries)
    status = "quality_failed" if quality_failed else "ok"
    print(json.dumps({"status": status, "sources": summaries}, indent=2, sort_keys=True))
    return 1 if quality_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
