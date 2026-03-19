"""Create and adjudicate the durable retailer mapping review queue."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pc_build_recommender.catalog import (
    MappingOutcome,
    ReviewStatus,
    stream_processed_catalog,
    upsert_mapping_review,
    validate_review_target,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    queue = commands.add_parser("queue", help="Generate an unresolved mapping queue.")
    queue.add_argument("--buildcores", type=Path, required=True)
    queue.add_argument("--offers", "--dynacore", dest="offers", type=Path, required=True)
    queue.add_argument("--reviewed-mappings", type=Path)
    queue.add_argument("--output", type=Path, required=True)
    queue.add_argument("--max-line-bytes", type=int, default=8 * 1024 * 1024)

    approve = commands.add_parser("approve", help="Approve a listing-to-product mapping.")
    approve.add_argument("--manifest", type=Path, required=True)
    approve.add_argument("--buildcores", type=Path, required=True)
    approve.add_argument("--offers", "--dynacore", dest="offers", type=Path, required=True)
    approve.add_argument("--listing-id", required=True)
    approve.add_argument("--product-id", required=True)
    approve.add_argument("--reviewed-by", required=True)
    approve.add_argument("--evidence", required=True)
    approve.add_argument("--max-line-bytes", type=int, default=8 * 1024 * 1024)

    reject = commands.add_parser("reject", help="Persist a reviewed no-match decision.")
    reject.add_argument("--manifest", type=Path, required=True)
    reject.add_argument("--offers", "--dynacore", dest="offers", type=Path, required=True)
    reject.add_argument("--listing-id", required=True)
    reject.add_argument("--reviewed-by", required=True)
    reject.add_argument("--evidence", required=True)
    reject.add_argument("--max-line-bytes", type=int, default=8 * 1024 * 1024)
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


def _queue(args: argparse.Namespace) -> int:
    result = stream_processed_catalog(
        args.buildcores,
        offer_path=args.offers,
        reviewed_mapping_path=args.reviewed_mappings,
        max_line_bytes=args.max_line_bytes,
    )
    resolved = {
        MappingOutcome.AUTO_MATCHED,
        MappingOutcome.REVIEWED_MATCHED,
        MappingOutcome.REVIEW_REJECTED,
    }
    decisions = [
        decision.to_dict()
        for decision in result.mapping_decisions
        if decision.outcome not in resolved
    ]
    payload = {
        "schema_version": "pc-build-recommender.mapping-review-queue.v1",
        "data_version": result.stats.data_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "unresolved_count": len(decisions),
        "decisions": decisions,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _approve(args: argparse.Namespace) -> int:
    validate_review_target(
        offer_path=args.offers,
        listing_id=args.listing_id,
        buildcores_path=args.buildcores,
        product_id=args.product_id,
        max_line_bytes=args.max_line_bytes,
    )
    review = upsert_mapping_review(
        args.manifest,
        listing_id=args.listing_id,
        product_id=args.product_id,
        status=ReviewStatus.APPROVED,
        reviewed_by=args.reviewed_by,
        evidence=args.evidence,
    )
    print(json.dumps(review.to_dict(), indent=2, sort_keys=True))
    return 0


def _reject(args: argparse.Namespace) -> int:
    validate_review_target(
        offer_path=args.offers,
        listing_id=args.listing_id,
        max_line_bytes=args.max_line_bytes,
    )
    review = upsert_mapping_review(
        args.manifest,
        listing_id=args.listing_id,
        status=ReviewStatus.REJECTED,
        reviewed_by=args.reviewed_by,
        evidence=args.evidence,
    )
    print(json.dumps(review.to_dict(), indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "queue":
        return _queue(args)
    if args.command == "approve":
        return _approve(args)
    if args.command == "reject":
        return _reject(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
