from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pc_build_recommender.entity_resolution import (
    ActiveLearningBatch,
    BlockingScoreComponent,
    CanonicalProductRecord,
    HumanMatchLabel,
    ListingRow,
    PairFeatureExtractor,
    PCBlockingCandidate,
    ReviewConflictError,
    ReviewQueue,
    ReviewState,
    SourceUsePolicy,
    evaluate_grouped_human_reviews,
    evaluate_grouped_review_queue,
    load_controlled_pc_workflow_inputs,
    sample_active_learning,
    select_precision_first_threshold,
    split_human_review_queue,
)

CREATED_AT = "2026-07-22T10:00:00+08:00"
REVIEWED_AT = "2026-07-22T11:00:00+08:00"


def _candidate(
    index: int,
    *,
    category: str = "memory",
    listing_capacity: int = 32,
    product_capacity: int = 32,
) -> PCBlockingCandidate:
    listing = ListingRow(
        listing_id=f"test-listing-{index}",
        title=f"Aster Velocity {listing_capacity}GB DDR5 kit {index}",
        category=category,
        brand="Aster",
        manufacturer_part_number=f"LIST-{index}",
        attributes={"capacity_gb": listing_capacity, "module_count": 2},
    )
    product = CanonicalProductRecord(
        product_id=f"test-product-{index}",
        category="memory" if category == "RAM" else category,
        brand="Aster",
        model=f"Velocity {product_capacity}",
        canonical_name=f"Aster Velocity {product_capacity}GB DDR5 kit {index}",
        manufacturer_part_number=f"PRODUCT-{index}",
        attributes={"capacity_gb": product_capacity, "module_count": 2},
    )
    from pc_build_recommender.entity_resolution import find_numeric_conflicts

    conflicts = find_numeric_conflicts(listing, product)
    return PCBlockingCandidate(
        listing=listing,
        product=product,
        blocking_score=0.5 + min(index, 4) * 0.05,
        score_components=(BlockingScoreComponent("test_similarity", 0.5),),
        conflicts=conflicts,
    )


def _policy(*, training_eligible: bool = True) -> SourceUsePolicy:
    return SourceUsePolicy(
        listing_source="test_controlled_offer_fixture",
        catalogue_source="test_licensed_catalogue_fixture",
        data_version="test-data-v1",
        training_eligible=training_eligible,
        published_metrics_eligible=training_eligible,
        scope_note="Structural unit-test fixture only; never reported as measured human data.",
    )


def _queue(count: int = 3, *, training_eligible: bool = True) -> ReviewQueue:
    return ReviewQueue.from_candidates(
        [_candidate(index) for index in range(count)],
        source_policy=_policy(training_eligible=training_eligible),
        created_at=CREATED_AT,
    )


def _write_review_sheet(path: Path, *, labels: list[str]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = tuple(rows[0])
    for row, label in zip(rows, labels, strict=True):
        row["state"] = ReviewState.LABELLED.value
        row["human_label"] = label
        row["reviewer_id"] = "unit-test-reviewer"
        row["reviewed_at"] = REVIEWED_AT
        row["reviewer_note"] = "structural test decision"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _label_queue(queue: ReviewQueue, labels: list[HumanMatchLabel]) -> ReviewQueue:
    result = queue
    for item, label in zip(queue.items, labels, strict=True):
        result = result.replace_item(
            item.with_review(
                state=ReviewState.LABELLED,
                reviewer_id="unit-test-reviewer",
                reviewed_at=REVIEWED_AT,
                human_label=label,
                reviewer_note="structural test decision",
            )
        )
    return result


def test_review_queue_round_trip_and_human_label_import_are_idempotent(tmp_path: Path) -> None:
    queue = _queue(count=2)
    queue_path = queue.export_jsonl(tmp_path / "queue.jsonl")
    restored = ReviewQueue.import_jsonl(queue_path)
    sheet = restored.export_label_sheet(tmp_path / "labels.csv")
    _write_review_sheet(
        sheet,
        labels=[HumanMatchLabel.MATCH.value, HumanMatchLabel.UNCERTAIN.value],
    )

    reviewed = restored.import_label_sheet(sheet)
    repeated = reviewed.import_label_sheet(sheet)

    assert repeated == reviewed
    assert len(reviewed.human_labelled_examples()) == 1
    assert reviewed.human_labelled_examples()[0].label == 1
    assert reviewed.items[1].human_label is HumanMatchLabel.UNCERTAIN

    _write_review_sheet(
        sheet,
        labels=[HumanMatchLabel.NON_MATCH.value, HumanMatchLabel.UNCERTAIN.value],
    )
    with pytest.raises(ReviewConflictError):
        reviewed.import_label_sheet(sheet)


def test_review_queue_import_rejects_manifest_and_snapshot_tampering(tmp_path: Path) -> None:
    path = _queue(count=1).export_jsonl(tmp_path / "queue.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["source_policy"]["scope_note"] = "tampered"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identifier does not match"):
        ReviewQueue.import_jsonl(path)


def test_controlled_source_policy_blocks_training_export() -> None:
    queue = _label_queue(
        _queue(count=2, training_eligible=False),
        [HumanMatchLabel.MATCH, HumanMatchLabel.NON_MATCH],
    )

    with pytest.raises(PermissionError, match="forbids model training"):
        queue.human_labelled_examples()
    evaluation = evaluate_grouped_review_queue(
        queue,
        [0.9, 0.1],
        threshold=0.5,
        n_resamples=20,
    )
    assert not evaluation.reportable


def test_active_learning_is_deterministic_unlabelled_and_conflict_aware() -> None:
    queue = ReviewQueue.from_candidates(
        [
            _candidate(0, category="RAM", product_capacity=64),
            _candidate(1),
            _candidate(2, category="storage"),
        ],
        source_policy=_policy(),
        created_at=CREATED_AT,
    )
    probabilities = {
        queue.items[0].queue_item_id: 0.90,
        queue.items[1].queue_item_id: 0.50,
        queue.items[2].queue_item_id: 0.97,
    }

    first = sample_active_learning(
        queue.items,
        probabilities,
        limit=2,
        model_version="test-model-v1",
        data_version="test-data-v1",
    )
    second = sample_active_learning(
        queue.items,
        probabilities,
        limit=2,
        model_version="test-model-v1",
        data_version="test-data-v1",
    )

    assert isinstance(first, ActiveLearningBatch)
    assert first == second
    assert all(item.state is ReviewState.PENDING for item in queue.items)
    assert first.to_dict()["label_policy"] == "unlabelled review selection; no labels are inferred"
    conflict_selection = next(
        selection
        for selection in first.selections
        if selection.queue_item_id == queue.items[0].queue_item_id
    )
    assert "conflict_model_disagreement" in conflict_selection.reasons


def test_precision_first_threshold_maximises_recall_and_fails_closed() -> None:
    labels = [1, 1, 1, 0, 0]
    scores = [0.99, 0.95, 0.90, 0.91, 0.10]

    selected = select_precision_first_threshold(
        labels,
        scores,
        minimum_precision=0.99,
        minimum_predicted_matches=2,
    )
    insufficient = select_precision_first_threshold(
        labels,
        scores,
        minimum_precision=0.99,
        minimum_predicted_matches=4,
    )
    confidence_gated = select_precision_first_threshold(
        labels,
        scores,
        minimum_precision=0.99,
        minimum_predicted_matches=2,
        require_precision_ci_lower_bound=True,
    )

    assert selected.threshold == 0.95
    assert selected.precision == 1.0
    assert selected.recall == pytest.approx(2 / 3)
    assert selected.deployment_eligible
    assert insufficient.threshold is None
    assert not insufficient.deployment_eligible
    assert confidence_gated.threshold is None


def test_group_split_and_evaluation_use_listing_as_the_leakage_unit() -> None:
    queue = _queue(count=8)
    labels = [
        HumanMatchLabel.MATCH if index % 2 == 0 else HumanMatchLabel.NON_MATCH for index in range(8)
    ]
    reviewed = _label_queue(queue, labels)

    splits = split_human_review_queue(
        reviewed,
        weights={"train": 0.5, "test": 0.5},
        seed=7,
    )
    train_groups = {item.listing.listing_id for item in splits["train"]}
    test_groups = {item.listing.listing_id for item in splits["test"]}
    evaluation = evaluate_grouped_human_reviews(
        reviewed.items,
        [0.9 if label is HumanMatchLabel.MATCH else 0.1 for label in labels],
        threshold=0.5,
        source_policy_reportable=True,
        evidence_scope="unit_test_only",
        n_resamples=20,
        seed=7,
    )

    assert not train_groups & test_groups
    assert evaluation.listing_group_count == 8
    assert evaluation.reportable
    assert evaluation.evaluation.metadata["bootstrap_unit"] == "listing_id"
    assert evaluation.to_dict()["label_source"] == "attributable_human_reviews"


def test_conflict_aware_features_cover_aliases_identifiers_and_variant_severity() -> None:
    candidate = _candidate(
        0,
        category="RAM",
        listing_capacity=32,
        product_capacity=64,
    )

    features = PairFeatureExtractor().extract(candidate.listing, candidate.product)

    assert features.numeric_conflict == 1.0
    assert features.numeric_conflict_count == 1.0
    assert features.capacity_conflict == 1.0
    assert features.numeric_conflict_severity == 0.5
    assert features.mpn_mismatch == 1.0


def test_conflict_parser_does_not_invent_space_grouped_numeric_semantics() -> None:
    listing = ListingRow(
        listing_id="ambiguous-weight-like-value",
        title="Aster memory listing",
        category="memory",
        attributes={"capacity_gb": "1 206"},
    )
    product = CanonicalProductRecord(
        product_id="capacity-1206",
        category="memory",
        brand="Aster",
        model="Capacity",
        canonical_name="Aster memory product",
        attributes={"capacity_gb": 1206},
    )

    features = PairFeatureExtractor().extract(listing, product)

    assert features.numeric_conflict == 0.0
    assert features.numeric_conflict_count == 0.0


def test_controlled_workflow_adapter_propagates_source_rights_without_guessing_brand(
    tmp_path: Path,
) -> None:
    catalogue = tmp_path / "catalogue.jsonl"
    listings = tmp_path / "listings.jsonl"
    catalogue.write_text(
        json.dumps(
            {
                "archive_snapshot_sha256": "catalogue-hash",
                "record_type": "canonical_product",
                "training_eligible": True,
                "published_claims_eligible": True,
                "data": {
                    "product_id": "product-1",
                    "category": "memory",
                    "brand": "Aster",
                    "model": "Velocity",
                    "canonical_name": "Aster Velocity 32GB",
                    "manufacturer_part_number": "MEM-32",
                    "gtin": None,
                    "common_attributes": {"msrp_sgd": 150},
                    "category_attributes": {"capacity_gb": 32},
                    "provenance": [{"source_name": "buildcores_open_db"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    listings.write_text(
        json.dumps(
            {
                "archive_snapshot_sha256": "listing-hash",
                "record_type": "retailer_listing",
                "training_eligible": False,
                "published_claims_eligible": False,
                "source_record_id": "row-1",
                "provenance": {"source_name": "dynacore_controlled_pdf"},
                "normalisation_metadata": {
                    "category": "RAM",
                    "section": "memory",
                    "variant": None,
                    "confidence_flags": ["controlled"],
                },
                "data": {
                    "listing": {
                        "listing_id": "listing-1",
                        "title": "Aster Velocity 32GB",
                        "currency": "SGD",
                        "base_price": "145.00",
                        "retailer": "Dynacore",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = load_controlled_pc_workflow_inputs(catalogue, listings)

    assert inputs.products[0].brand == "Aster"
    assert inputs.listings[0].brand == ""
    assert inputs.listings[0].category == "memory"
    assert not inputs.source_policy.training_eligible
    assert not inputs.source_policy.published_metrics_eligible
    assert inputs.source_policy.listing_source == "dynacore_controlled_pdf"
