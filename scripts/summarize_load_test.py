"""Create a content-addressed evidence artifact from one bounded Locust run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = REPOSITORY_ROOT / "packages" / "core" / "src"
for import_root in (REPOSITORY_ROOT, CORE_SOURCE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.loadtest.evidence import (  # noqa: E402
    LoadEvidenceError,
    build_load_evidence,
    collect_api_metadata,
    collect_host_metadata,
    write_load_evidence,
)
from scripts.loadtest.profile import LoadProfileError, load_profile  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--csv-prefix", type=Path, required=True)
    parser.add_argument("--target-origin", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--spawn-rate", type=float, required=True)
    parser.add_argument("--run-time-seconds", type=float, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--cache-state", choices=("cold", "warm", "unknown"), required=True)
    parser.add_argument(
        "--database-state",
        choices=("in_memory_demo", "postgres", "unknown"),
        required=True,
    )
    parser.add_argument("--api-timeout-seconds", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile = load_profile(args.profile)
        api_metadata = collect_api_metadata(
            target_origin=args.target_origin,
            timeout_seconds=args.api_timeout_seconds,
        )
        evidence = build_load_evidence(
            profile=profile,
            csv_prefix=args.csv_prefix,
            api_metadata=api_metadata,
            host_metadata=collect_host_metadata(),
            users=args.users,
            spawn_rate_per_second=args.spawn_rate,
            run_time_seconds=args.run_time_seconds,
            warmup_seconds=args.warmup_seconds,
            cache_state=args.cache_state,
            database_state=args.database_state,
        )
        output = write_load_evidence(evidence=evidence, output_path=args.output)
    except (LoadEvidenceError, LoadProfileError, OSError, ValueError) as error:
        print(
            json.dumps({"status": "error", "message": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    assessment = evidence.get("threshold_assessment")
    if not isinstance(assessment, dict) or not isinstance(assessment.get("claim_status"), str):
        raise AssertionError("load evidence lacks a valid threshold assessment")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output),
                "content_sha256": evidence["content_sha256"],
                "claim_status": assessment["claim_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
