from __future__ import annotations

from dataclasses import fields

from pc_build_recommender.entity_resolution import (
    CandidateBlocker,
    CandidateSupervision,
    CanonicalProductRecord,
    DeterministicHardNegativeSampler,
    ListingRow,
    PCDomainCandidateBlocker,
    UnlabeledHardNegativeCandidate,
    canonical_pc_category,
)


def _product(
    product_id: str,
    *,
    capacity_gb: int,
    mpn: str,
) -> CanonicalProductRecord:
    return CanonicalProductRecord(
        product_id=product_id,
        category="memory",
        brand="Aster",
        model="Velocity M1",
        canonical_name=f"Aster Velocity M1 {capacity_gb}GB DDR5-6000",
        manufacturer_part_number=mpn,
        attributes={"capacity_gb": capacity_gb, "module_count": 2},
    )


def _listing(*, mpn: str = "MEM-32") -> ListingRow:
    return ListingRow(
        listing_id="listing-1",
        title="Aster Velocity M1 32GB DDR5-6000 2x16GB",
        category="RAM",
        brand="Aster",
        manufacturer_part_number=mpn,
        attributes={"capacity_gb": 32, "module_count": 2},
    )


def test_pc_blocker_retains_numeric_conflict_as_unlabeled_audit_evidence() -> None:
    listing = _listing(mpn="SAME-MPN")
    conflicting = _product("memory-64", capacity_gb=64, mpn="SAME-MPN")

    candidate = PCDomainCandidateBlocker().candidates(listing, [conflicting])[0]

    assert candidate.supervision_status is CandidateSupervision.UNLABELED
    assert candidate.has_hard_conflict
    assert [conflict.field for conflict in candidate.conflicts] == ["capacity_gb"]
    assert "numeric_conflict:capacity_gb" in candidate.reasons
    assert "numeric_conflict_review" in candidate.reasons
    # The production auto-match blocker continues to fail closed on this pair.
    assert CandidateBlocker().candidates(listing, [conflicting]) == ()


def test_hard_negative_sampler_is_deterministic_and_never_emits_labels() -> None:
    listing = _listing()
    exact = _product("memory-32", capacity_gb=32, mpn="MEM-32")
    close_variant = _product("memory-64", capacity_gb=64, mpn="MEM-64")
    other_variant = _product("memory-48", capacity_gb=48, mpn="MEM-48")
    blocker = PCDomainCandidateBlocker()
    forward = blocker.generate([listing], [exact, close_variant, other_variant])
    reverse = blocker.generate([listing], [other_variant, close_variant, exact])
    sampler = DeterministicHardNegativeSampler(max_per_listing=2)

    first = sampler.sample(forward)
    second = sampler.sample(reverse)

    assert [item.product.product_id for item in first] == [
        item.product.product_id for item in second
    ]
    assert len(first) == 2
    assert all(item.supervision_status is CandidateSupervision.UNLABELED for item in first)
    assert all("hard_numeric_conflict:capacity_gb" in item.selection_reasons for item in first)
    assert "label" not in {item.name for item in fields(UnlabeledHardNegativeCandidate)}
    assert all(item.to_metadata()["supervision_status"] == "UNLABELED" for item in first)
    assert all("label" not in item.to_metadata() for item in first)
    assert exact.product_id not in {item.product.product_id for item in first}


def test_category_aliases_and_identifier_mismatch_are_auditable() -> None:
    listing = _listing(mpn="LISTING-MPN")
    same_variant = _product("memory-alt", capacity_gb=32, mpn="CANONICAL-MPN")

    candidate = PCDomainCandidateBlocker().candidates(listing, [same_variant])[0]
    sampled = DeterministicHardNegativeSampler(max_per_listing=1).sample([candidate])[0]

    assert canonical_pc_category("RAM") == canonical_pc_category("memory") == "memory"
    assert "manufacturer_part_number_mismatch" in candidate.reasons
    assert sampled.selection_reasons == (
        "manufacturer_part_number_mismatch",
        "near_model_variant",
    )
    assert not candidate.has_hard_conflict
