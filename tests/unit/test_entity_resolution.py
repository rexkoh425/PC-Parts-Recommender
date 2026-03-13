from __future__ import annotations

import json

import numpy as np
import pytest

from pc_build_recommender.entity_resolution import (
    FEATURE_NAMES,
    CandidateBlocker,
    CanonicalProductRecord,
    ListingRecord,
    MatchOutcome,
    MatchThresholds,
    LabelledPair,
    PairFeatureExtractor,
    PlattCalibrator,
    evaluate_binary_predictions,
    extract_numeric_facts,
    find_numeric_conflicts,
    normalize_identifier,
    normalize_text,
    pair_example_from_dict,
    synthetic_catalog,
)


def _memory_pair(*, listing_capacity: int = 32, product_capacity: int = 32) -> LabelledPair:
    product = CanonicalProductRecord(
        product_id=f"product-{product_capacity}",
        category="memory",
        brand="Aster",
        model="Velocity M1",
        canonical_name=f"Aster Velocity M1 {product_capacity}GB DDR5-6000",
        manufacturer_part_number="MEM-001",
        attributes={"capacity_gb": product_capacity, "module_count": 2},
        price_sgd=150.0,
        embedding=(1.0, 0.0),
    )
    listing = ListingRecord(
        listing_id=f"listing-{listing_capacity}",
        title=f"ASTER VELOCITY M1 {listing_capacity} GB DDR5 6000",
        category="memory",
        brand="aster",
        manufacturer_part_number="MEM 001",
        attributes={"capacity_gb": listing_capacity, "module_count": 2},
        current_price_sgd=155.0,
        embedding=(1.0, 0.0),
    )
    return LabelledPair(
        pair_id=f"pair-{listing_capacity}-{product_capacity}",
        listing=listing,
        product=product,
        label=int(listing_capacity == product_capacity),
    )


def test_text_and_identifier_normalisation_is_deterministic() -> None:
    assert normalize_text("  MÖBIUS—RTX４０７０  ") == "mobius rtx4070"
    assert normalize_identifier("AB-12 / 34") == "ab1234"
    assert normalize_text(None) == ""


def test_numeric_fact_extraction_uses_canonical_units() -> None:
    facts = extract_numeric_facts("2x16GB kit, 1 TB SSD, 0.75 kW PSU, 36 cm radiator")
    values = {(fact.kind, fact.value, fact.unit) for fact in facts}
    assert ("total_capacity", 32.0, "gb") in values
    assert ("capacity", 1024.0, "gb") in values
    assert ("power", 750.0, "w") in values
    assert ("length", 360.0, "mm") in values


def test_pair_records_round_trip_nested_and_flat() -> None:
    pair = _memory_pair()
    nested = pair_example_from_dict(json.loads(json.dumps(pair.to_dict())))
    flat = LabelledPair.from_flat_dict(pair.to_flat_dict())

    assert nested.to_dict() == pair.to_dict()
    assert flat.to_dict() == pair.to_dict()
    with pytest.raises(TypeError):
        pair.listing.attributes["capacity_gb"] = 64  # type: ignore[index]


def test_numeric_variant_conflict_is_a_hard_gate_even_with_exact_mpn() -> None:
    pair = _memory_pair(listing_capacity=32, product_capacity=64)
    conflicts = find_numeric_conflicts(pair.listing, pair.product)

    assert [conflict.field for conflict in conflicts] == ["capacity_gb"]
    assert CandidateBlocker().candidates(pair.listing, [pair.product]) == ()


def test_candidate_blocker_is_deterministic_and_ranks_exact_identifier_first() -> None:
    exact = _memory_pair().product
    similar = CanonicalProductRecord(
        product_id="product-similar",
        category="memory",
        brand="Aster",
        model="Velocity M1 Plus",
        canonical_name="Aster Velocity M1 Plus 32GB DDR5-6000",
        manufacturer_part_number="OTHER-002",
        attributes={"capacity_gb": 32, "module_count": 2},
    )
    listing = _memory_pair().listing
    candidates = CandidateBlocker().candidates(listing, [similar, exact])

    assert [candidate.product.product_id for candidate in candidates][0] == exact.product_id
    assert "exact_manufacturer_part_number" in candidates[0].reasons
    assert candidates == CandidateBlocker().candidates(listing, [similar, exact])


def test_pair_feature_contract_and_numeric_conflict_column() -> None:
    extractor = PairFeatureExtractor()
    positive = _memory_pair()
    conflict = _memory_pair(listing_capacity=32, product_capacity=64)
    matrix = extractor.transform([positive, conflict])

    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()
    assert matrix[0, FEATURE_NAMES.index("numeric_conflict")] == 0.0
    assert matrix[1, FEATURE_NAMES.index("numeric_conflict")] == 1.0
    assert matrix[0, FEATURE_NAMES.index("exact_mpn_match")] == 1.0


def test_threshold_boundaries_and_conflict_override() -> None:
    thresholds = MatchThresholds(auto_match=0.98, manual_review=0.80)

    assert thresholds.decide(0.98).outcome is MatchOutcome.AUTO_MATCH
    assert thresholds.decide(0.80).outcome is MatchOutcome.MANUAL_REVIEW
    assert thresholds.decide(0.79).outcome is MatchOutcome.REJECT
    conflict = thresholds.decide(1.0, hard_conflict=True)
    assert conflict.outcome is MatchOutcome.REJECT
    assert conflict.reason == "numeric_variant_conflict"


def test_platt_calibrator_is_monotonic_and_json_round_trips() -> None:
    calibrator = PlattCalibrator().fit([0.05, 0.2, 0.8, 0.95], [0, 0, 1, 1])
    restored = PlattCalibrator.from_dict(calibrator.to_dict())
    result = restored.predict_proba([0.1, 0.9])

    assert 0.0 < result[0] < result[1] < 1.0


def test_synthetic_data_is_deterministic_and_explicitly_non_promotable() -> None:
    first = synthetic_catalog(seed=17, product_count=12)
    second = synthetic_catalog(seed=17, product_count=12)

    assert first.to_dict() == second.to_dict()
    assert all(pair.is_synthetic for pair in first.pairs)
    probabilities = np.asarray([0.9 if pair.label else 0.1 for pair in first.pairs])
    evaluation = evaluate_binary_predictions(
        [pair.label for pair in first.pairs],
        probabilities,
        is_synthetic=[pair.is_synthetic for pair in first.pairs],
        include_synthetic=True,
    )
    assert evaluation.f1 == 1.0
    assert not evaluation.eligible_for_promotion
    assert evaluation.non_promotable_reason == "synthetic rows were included in evaluation metrics"

    with pytest.raises(ValueError, match="no evaluable non-synthetic rows"):
        evaluate_binary_predictions(
            [pair.label for pair in first.pairs],
            probabilities,
            is_synthetic=[True] * len(first.pairs),
        )
