"""Independently verify one signed production source-batch release."""

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

from pipelines.source_release import (  # noqa: E402
    AuthorizedBatchReleaseError,
    verify_awin_production_batch_release,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--trust-root-sha256", required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--raw-snapshot", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        release = verify_awin_production_batch_release(
            manifest_path=args.manifest,
            expected_manifest_sha256=args.manifest_sha256,
            expected_trust_root_sha256=args.trust_root_sha256,
            current_source_registry=args.source_registry,
            raw_snapshot=args.raw_snapshot,
            records=args.records,
            rejections=args.rejections,
        )
    except (AuthorizedBatchReleaseError, OSError, TypeError, ValueError) as error:
        # Local source and policy filenames may contain confidential operator
        # details.  Keep the machine-readable boundary useful without echoing
        # paths or arbitrary exception messages.
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"authorized source release verification failed ({type(error).__name__})"
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    summary: dict[str, Any] = {
        "status": "ok",
        "schema_version": "pc-build-recommender.authorized-source-release-verification.v1",
        "manifest_sha256": release.manifest_sha256,
        "content_sha256": release.content_sha256,
        "source_name": release.source_name,
        "raw_snapshot_sha256": release.raw_snapshot_sha256,
        "processed_run_sha256": release.processed_run_sha256,
        "accepted_count": release.accepted_count,
        "rejected_count": release.rejected_count,
        "policy_sha256": release.policy_sha256,
        "source_registry_sha256": release.source_registry_sha256,
        "authority_expires_at": release.authority_expires_at.isoformat(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
