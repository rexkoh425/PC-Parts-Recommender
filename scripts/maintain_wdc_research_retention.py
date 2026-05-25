"""Safely delete expired WDC research-quarantine artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pipelines.retention.wdc import WDCRetentionError, maintain_wdc_research_retention  # noqa: E402


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
    parser.add_argument("--raw-root", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "quarantine",
    )
    parser.add_argument(
        "--category-index",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "quarantine" / "wdc-products-category-index.sqlite3",
    )
    parser.add_argument(
        "--maximum-entries",
        type=int,
        default=100_000,
        help="Global fail-closed bound across raw receipts, index files, and run trees.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report only.")
    parser.add_argument(
        "--now",
        type=_aware_timestamp,
        help="Auditable evaluation timestamp; defaults to the current UTC time.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = maintain_wdc_research_retention(
            raw_root=args.raw_root,
            output_root=args.output_root,
            category_index=args.category_index,
            now=args.now,
            dry_run=args.dry_run,
            maximum_entries=args.maximum_entries,
        )
    except (OSError, ValueError, WDCRetentionError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "ok", **report.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
