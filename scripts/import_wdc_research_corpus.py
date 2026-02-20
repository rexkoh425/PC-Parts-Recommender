"""Import the historical WDC Products corpus into a research-only quarantine."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pipelines.sources.wdc_products import (
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_MAX_CATEGORY_DECOMPRESSED_BYTES,
    DEFAULT_MAX_CORPUS_DECOMPRESSED_BYTES,
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_SELECTED_RECORDS,
    WDCProductsResearchSource,
    build_wdc_category_index,
    import_wdc_research_candidates,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Local PDC2020-C JSONL or JSONL.GZ file. It is copied to raw storage first.",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        help="Local WDC majority-voted category JSONL or JSONL.GZ file.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Explicitly allow downloading both official WDC files. The corpus is over 5 GB; "
            "this is never enabled implicitly."
        ),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/quarantine"),
        help="Research-only output root; artifacts are never loaded by the production catalog.",
    )
    parser.add_argument(
        "--category-index",
        type=Path,
        default=Path("data/quarantine/wdc-products-category-index.sqlite3"),
    )
    parser.add_argument(
        "--category-record-budget",
        type=_positive_integer,
        help="Optional new-row budget for a deliberately paused category-index run.",
    )
    parser.add_argument(
        "--corpus-record-budget",
        type=_positive_integer,
        help="Optional new-row budget for a deliberately paused corpus run.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=_positive_integer,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
    parser.add_argument("--max-line-bytes", type=_positive_integer, default=DEFAULT_MAX_LINE_BYTES)
    parser.add_argument(
        "--max-category-decompressed-bytes",
        type=_positive_integer,
        default=DEFAULT_MAX_CATEGORY_DECOMPRESSED_BYTES,
    )
    parser.add_argument(
        "--max-corpus-decompressed-bytes",
        type=_positive_integer,
        default=DEFAULT_MAX_CORPUS_DECOMPRESSED_BYTES,
    )
    parser.add_argument(
        "--maximum-selected-records",
        type=_positive_integer,
        default=DEFAULT_MAX_SELECTED_RECORDS,
    )
    parser.add_argument(
        "--maximum-output-bytes",
        type=_positive_integer,
        default=DEFAULT_MAX_OUTPUT_BYTES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.download and (args.corpus is not None or args.categories is not None):
        raise ValueError("--download cannot be combined with local --corpus or --categories")
    if not args.download and (args.corpus is None or args.categories is None):
        raise ValueError("provide both local files, or explicitly pass --download")

    source = WDCProductsResearchSource(args.raw_root)
    corpus_snapshot = source.fetch_corpus(corpus_path=args.corpus)
    category_snapshot = source.fetch_categories(category_path=args.categories)
    category_result = build_wdc_category_index(
        category_snapshot,
        index_path=args.category_index,
        record_budget=args.category_record_budget,
        checkpoint_interval=args.checkpoint_interval,
        max_line_bytes=args.max_line_bytes,
        max_decompressed_bytes=args.max_category_decompressed_bytes,
    )
    report: dict[str, object] = {
        "research_only": True,
        "production_eligible": False,
        "category_index": category_result.to_dict(),
    }
    if not category_result.complete:
        report["status"] = "category_index_paused"
        report["next_action"] = "repeat the identical command to resume"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 3

    corpus_result = import_wdc_research_candidates(
        corpus_snapshot,
        category_index_path=args.category_index,
        output_root=args.output_root,
        record_budget=args.corpus_record_budget,
        checkpoint_interval=args.checkpoint_interval,
        max_line_bytes=args.max_line_bytes,
        max_decompressed_bytes=args.max_corpus_decompressed_bytes,
        maximum_selected_records=args.maximum_selected_records,
        maximum_output_bytes=args.maximum_output_bytes,
    )
    report["corpus_import"] = corpus_result.to_dict()
    report["status"] = "complete" if corpus_result.complete else "corpus_import_paused"
    if not corpus_result.complete:
        report["next_action"] = "repeat the identical command to resume"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if corpus_result.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
