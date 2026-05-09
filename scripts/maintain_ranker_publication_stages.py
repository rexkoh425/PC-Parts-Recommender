"""Report or remove old crash-orphaned stages for one exact ranker bundle."""

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

from pc_build_recommender.ranking import (  # noqa: E402
    DEFAULT_MAXIMUM_PARENT_ENTRIES,
    RankerPublicationMaintenanceError,
    maintain_ranker_publication_stages,
)


def _aware_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed.astimezone(UTC)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        required=True,
        type=Path,
        help="Absolute directory containing the one exact final bundle and its hidden stages.",
    )
    parser.add_argument(
        "--bundle-name",
        required=True,
        help="Exact direct bundle directory name, for example ranker-artifact.",
    )
    parser.add_argument(
        "--minimum-age-hours",
        type=float,
        default=24.0,
        help="Preserve stages younger than this strictly positive threshold.",
    )
    parser.add_argument(
        "--maximum-entries",
        type=int,
        default=DEFAULT_MAXIMUM_PARENT_ENTRIES,
        help="Fail-closed bound on direct entries inspected under the explicit parent.",
    )
    parser.add_argument(
        "--now",
        type=_aware_timestamp,
        help="Auditable evaluation time; defaults to current UTC time.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform eligible removals. Without this flag the command is always dry-run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = maintain_ranker_publication_stages(
            args.parent,
            bundle_name=args.bundle_name,
            minimum_age=timedelta(hours=args.minimum_age_hours),
            dry_run=not args.apply,
            now=args.now,
            maximum_entries=args.maximum_entries,
        )
    except (FileNotFoundError, OSError, ValueError, RankerPublicationMaintenanceError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "blocked" if report.blocked_count else "ok",
                **report.to_dict(),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if report.blocked_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
