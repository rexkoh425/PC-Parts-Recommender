"""Safely enforce governed-web raw and processed retention receipts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pipelines.retention.legacy_web import (  # noqa: E402
    plan_legacy_web_retention_migration,
)
from pipelines.retention.web import WebRetentionError, maintain_web_retention  # noqa: E402


def _aware_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed.astimezone(UTC)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-name",
        action="append",
        required=True,
        help="Exact governed-web source root to maintain; repeat for multiple sources.",
    )
    parser.add_argument("--raw-root", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument(
        "--processed-root", type=Path, default=REPOSITORY_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--orphan-grace-hours",
        type=int,
        default=24,
        help="Minimum age before deleting an unreceipted raw body or recognized temp file.",
    )
    parser.add_argument(
        "--maximum-entries",
        "--maximum-entries-per-source",
        dest="maximum_entries",
        type=int,
        default=100_000,
        help="Global fail-closed work bound across every configured source and run tree.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report only.")
    parser.add_argument(
        "--plan-legacy-migration",
        action="store_true",
        help=(
            "Read-only validation of legacy v1 receipts. Plans zero writes and reports the "
            "exact authority, rights, or timestamp evidence required before migration."
        ),
    )
    parser.add_argument(
        "--now",
        type=_aware_timestamp,
        help="Auditable evaluation timestamp; defaults to the current UTC time.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.plan_legacy_migration:
            legacy_reports = plan_legacy_web_retention_migration(
                raw_root=args.raw_root,
                processed_root=args.processed_root,
                source_names=tuple(args.source_name),
                now=args.now,
                maximum_entries=args.maximum_entries,
            )
            blockers = [
                blocker
                for report in legacy_reports
                for blocker in report.blockers
            ]
            print(
                json.dumps(
                    {
                        "status": "blocked" if blockers else "ok",
                        "mode": "legacy-migration-plan",
                        "dry_run": True,
                        "write_actions_planned": 0,
                        "sources": [report.to_dict() for report in legacy_reports],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2 if blockers else 0
        reports = maintain_web_retention(
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            source_names=tuple(args.source_name),
            now=args.now,
            orphan_grace=timedelta(hours=args.orphan_grace_hours),
            maximum_entries=args.maximum_entries,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, WebRetentionError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "dry_run": args.dry_run,
                "sources": [report.to_dict() for report in reports],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
