"""Operate the PC-domain entity-resolution review and human-label workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pc_build_recommender.entity_resolution import (
    CanonicalProductRecord,
    DeterministicHardNegativeSampler,
    ListingRow,
    PCDomainCandidateBlocker,
    ReviewQueue,
    SourceUsePolicy,
    load_controlled_pc_workflow_inputs,
    sample_active_learning,
)
from training._common import (
    read_json_lines,
    sha256_file,
    write_json,
    write_json_lines,
)


def _source_policy(path: Path) -> SourceUsePolicy:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("source policy must be a JSON object")
    return SourceUsePolicy.from_dict(payload)


def _create_queue(args: argparse.Namespace) -> dict[str, object]:
    policy = _source_policy(args.source_policy)
    listings = tuple(ListingRow.from_dict(row) for row in read_json_lines(args.listings))
    products = tuple(
        CanonicalProductRecord.from_dict(row) for row in read_json_lines(args.products)
    )
    blocker = PCDomainCandidateBlocker(
        max_candidates=args.max_candidates,
        minimum_text_score=args.minimum_text_score,
    )
    candidates = blocker.generate(listings, products)
    if args.hard_negatives_only:
        sampled = DeterministicHardNegativeSampler(
            max_per_listing=args.max_hard_negatives_per_listing
        ).sample(candidates)
        candidates = tuple(item.candidate for item in sampled)
    queue = ReviewQueue.from_candidates(
        candidates,
        source_policy=policy,
        created_at=args.created_at,
    )
    queue.export_jsonl(args.output)
    return {
        "operation": "create_queue",
        "output": str(args.output.resolve()),
        "queue_id": queue.queue_id,
        "listing_count": len(listings),
        "product_count": len(products),
        "candidate_count": len(candidates),
        "label_count": 0,
        "supervision_status": "UNLABELED",
        "source_policy": policy.to_dict(),
    }


def _create_controlled_queue(args: argparse.Namespace) -> dict[str, object]:
    inputs = load_controlled_pc_workflow_inputs(args.catalogue, args.listings)
    blocker = PCDomainCandidateBlocker(
        max_candidates=args.max_candidates,
        minimum_text_score=args.minimum_text_score,
    )
    candidates = blocker.generate(inputs.listings, inputs.products)
    if args.hard_negatives_only:
        sampled = DeterministicHardNegativeSampler(
            max_per_listing=args.max_hard_negatives_per_listing
        ).sample(candidates)
        candidates = tuple(item.candidate for item in sampled)
    queue = ReviewQueue.from_candidates(
        candidates,
        source_policy=inputs.source_policy,
        created_at=args.created_at,
    )
    queue.export_jsonl(args.output)
    return {
        "operation": "create_controlled_queue",
        "output": str(args.output.resolve()),
        "queue_id": queue.queue_id,
        "listing_count": len(inputs.listings),
        "product_count": len(inputs.products),
        "candidate_count": len(candidates),
        "label_count": 0,
        "supervision_status": "UNLABELED",
        "source_policy": inputs.source_policy.to_dict(),
    }


def _export_sheet(args: argparse.Namespace) -> dict[str, object]:
    queue = ReviewQueue.import_jsonl(args.queue)
    queue.export_label_sheet(args.output)
    return {
        "operation": "export_label_sheet",
        "queue_id": queue.queue_id,
        "output": str(args.output.resolve()),
        "item_count": len(queue.items),
    }


def _import_sheet(args: argparse.Namespace) -> dict[str, object]:
    queue = ReviewQueue.import_jsonl(args.queue)
    reviewed = queue.import_label_sheet(args.sheet)
    reviewed.export_jsonl(args.output)
    return {
        "operation": "import_label_sheet",
        "queue_id": reviewed.queue_id,
        "output": str(args.output.resolve()),
        "state_counts": reviewed.manifest()["state_counts"],
        "label_provenance": "attributable human reviews only",
    }


def _export_training(args: argparse.Namespace) -> dict[str, object]:
    queue = ReviewQueue.import_jsonl(args.queue)
    examples = queue.human_labelled_examples()
    write_json_lines(args.output, (example.to_dict() for example in examples))
    evidence = [
        {
            "queue_item_id": item.queue_item_id,
            "snapshot_sha256": item.snapshot_sha256,
            "human_label": item.human_label.value if item.human_label else None,
            "reviewer_id": item.reviewer_id,
            "reviewed_at": item.reviewed_at,
        }
        for item in queue.items
        if item.is_binary_human_label
    ]
    manifest = {
        "schema_version": "pc-build-recommender.er-human-label-export.v1",
        "label_source": "attributable_human_reviews",
        "queue_id": queue.queue_id,
        "queue_path": str(args.queue.resolve()),
        "queue_sha256": sha256_file(args.queue),
        "source_policy": queue.source_policy.to_dict(),
        "training_eligible": queue.source_policy.training_eligible,
        "row_count": len(examples),
        "positive_count": sum(example.label for example in examples),
        "uncertain_or_nonlabelled_items_excluded": len(queue.items) - len(examples),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "label_evidence": evidence,
        "claim_guard": "No candidate-generation heuristic was converted into a label.",
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    return {"operation": "export_training", **manifest, "manifest": str(manifest_path.resolve())}


def _sample_active(args: argparse.Namespace) -> dict[str, object]:
    queue = ReviewQueue.import_jsonl(args.queue)
    score_rows = read_json_lines(args.scores)
    probabilities = {str(row["queue_item_id"]): float(row["probability"]) for row in score_rows}
    if len(probabilities) != len(score_rows):
        raise ValueError("score file contains duplicate queue_item_id values")
    batch = sample_active_learning(
        queue.items,
        probabilities,
        limit=args.limit,
        model_version=args.model_version,
        data_version=queue.source_policy.data_version,
        operating_threshold=args.operating_threshold,
        manual_review_threshold=args.manual_review_threshold,
        max_per_listing=args.max_per_listing,
        seed=args.seed,
    )
    payload = batch.to_dict()
    write_json(args.output, payload)
    return {"operation": "sample_active_learning", "output": str(args.output.resolve()), **payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-queue")
    create.add_argument("--listings", type=Path, required=True)
    create.add_argument("--products", type=Path, required=True)
    create.add_argument("--source-policy", type=Path, required=True)
    create.add_argument("--created-at", required=True, help="timezone-aware ISO-8601 timestamp")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--max-candidates", type=int, default=50)
    create.add_argument("--minimum-text-score", type=float, default=0.12)
    create.add_argument("--hard-negatives-only", action="store_true")
    create.add_argument("--max-hard-negatives-per-listing", type=int, default=5)
    create.set_defaults(handler=_create_queue)

    controlled = subparsers.add_parser("create-controlled-queue")
    controlled.add_argument("--catalogue", type=Path, required=True)
    controlled.add_argument("--listings", type=Path, required=True)
    controlled.add_argument("--created-at", required=True, help="timezone-aware ISO-8601 timestamp")
    controlled.add_argument("--output", type=Path, required=True)
    controlled.add_argument("--max-candidates", type=int, default=50)
    controlled.add_argument("--minimum-text-score", type=float, default=0.12)
    controlled.add_argument("--hard-negatives-only", action="store_true")
    controlled.add_argument("--max-hard-negatives-per-listing", type=int, default=5)
    controlled.set_defaults(handler=_create_controlled_queue)

    export_sheet = subparsers.add_parser("export-sheet")
    export_sheet.add_argument("--queue", type=Path, required=True)
    export_sheet.add_argument("--output", type=Path, required=True)
    export_sheet.set_defaults(handler=_export_sheet)

    import_sheet = subparsers.add_parser("import-sheet")
    import_sheet.add_argument("--queue", type=Path, required=True)
    import_sheet.add_argument("--sheet", type=Path, required=True)
    import_sheet.add_argument("--output", type=Path, required=True)
    import_sheet.set_defaults(handler=_import_sheet)

    export_training = subparsers.add_parser("export-training")
    export_training.add_argument("--queue", type=Path, required=True)
    export_training.add_argument("--output", type=Path, required=True)
    export_training.add_argument("--manifest", type=Path)
    export_training.set_defaults(handler=_export_training)

    sample = subparsers.add_parser("sample-active")
    sample.add_argument("--queue", type=Path, required=True)
    sample.add_argument("--scores", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--model-version", required=True)
    sample.add_argument("--limit", type=int, default=100)
    sample.add_argument("--operating-threshold", type=float, default=0.98)
    sample.add_argument("--manual-review-threshold", type=float, default=0.80)
    sample.add_argument("--max-per-listing", type=int, default=2)
    sample.add_argument("--seed", type=int, default=20260722)
    sample.set_defaults(handler=_sample_active)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
