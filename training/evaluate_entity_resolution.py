"""Evaluate a persisted entity resolver on a labelled external JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pc_build_recommender.entity_resolution import (
    load_entity_resolver,
    pair_example_from_dict,
)
from training._common import (
    print_json,
    read_json_lines,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-synthetic-diagnostics",
        action="store_true",
        help="include synthetic pairs but mark the evaluation non-promotable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    pairs = tuple(pair_example_from_dict(row) for row in read_json_lines(args.input))
    synthetic_count = sum(pair.is_synthetic for pair in pairs)
    resolver = load_entity_resolver(args.artifact_dir)
    training_overlap_count: int | None = None
    training_overlap_fraction: float | None = None
    source_reused: bool | None = None
    evidence_path = args.artifact_dir / "training_evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict) or evidence.get("leakage_unit") != "listing_id":
            raise ValueError("entity artifact contains invalid training evidence")
        training_hashes = {str(value) for value in evidence["training_group_hashes"]}
        evaluation_hashes = {
            sha256_text(pair.listing.listing_id) for pair in pairs
        }
        training_overlap_count = len(training_hashes & evaluation_hashes)
        training_overlap_fraction = training_overlap_count / max(1, len(evaluation_hashes))
        source_reused = str(evidence["source_sha256"]) == sha256_file(args.input)
    evaluation = resolver.evaluate(
        pairs,
        include_synthetic=args.include_synthetic_diagnostics,
        classification_threshold=args.classification_threshold,
    )
    target_metrics_met = (
        evaluation.precision >= 0.99 and evaluation.recall >= 0.94 and evaluation.f1 >= 0.96
    )
    promotion_block_reasons: list[str] = []
    if reason := evaluation.non_promotable_reason:
        promotion_block_reasons.append(reason)
    if not target_metrics_met:
        promotion_block_reasons.append("precision, recall, or F1 target was not met")
    if training_overlap_count is None:
        promotion_block_reasons.append("artifact has no training-group leakage evidence")
    elif training_overlap_count:
        promotion_block_reasons.append(
            f"evaluation overlaps {training_overlap_count} training listing groups"
        )
    report = {
        "created_at": utc_now_iso(),
        "task": "entity_resolution_external_evaluation",
        "model_type": resolver.model_type,
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "rows": len(pairs),
            "synthetic_rows": synthetic_count,
            "source_reused": source_reused,
            "training_group_overlap_count": training_overlap_count,
            "training_group_overlap_fraction": training_overlap_fraction,
        },
        "metrics": evaluation.to_dict(),
        "targets": {"precision": 0.99, "recall": 0.94, "f1": 0.96},
        "target_metrics_met": target_metrics_met,
        "promotion_eligible": not promotion_block_reasons,
        "promotion_block_reasons": promotion_block_reasons,
    }
    report_path = args.report or args.artifact_dir / "external_evaluation.json"
    write_json(report_path, report)
    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
