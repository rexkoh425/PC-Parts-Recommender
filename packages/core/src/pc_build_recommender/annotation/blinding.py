"""Shared safeguards for human-annotation evidence.

The annotation service and batch-preparation tooling must apply exactly the same
reviewer-blinding rules.  Keeping these checks in one small dependency-free module
avoids a future compiler accepting payloads which the service rejects, or worse,
allowing model-derived hints into a human label collection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

FORBIDDEN_REVIEWER_BIAS_KEYS = frozenset(
    {
        "blocking_score",
        "bm25_score",
        "embedding",
        "entity_resolution_probability",
        "entity_resolution_score",
        "grade",
        "human_label",
        "label",
        "model_prediction",
        "model_probability",
        "model_score",
        "predicted_label",
        "rank_position",
        "recommended_label",
        "relevance_grade",
        "relevance_label",
        "relevance_score",
        "rrf_score",
        "silver_label",
        "synthetic_label",
        "vector_similarity",
    }
)


def validate_blinded_annotation_payload(value: object, *, path: str = "evidence") -> None:
    """Reject model-derived hints and non-finite values from reviewer evidence.

    This is intentionally recursive because a score can otherwise be hidden in a
    nested source, feature, or product object.  It permits observed product
    specifications and benchmark values; only ranking/label/model signals are
    withheld from reviewers.
    """

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalised = str(key).strip().casefold()
            if normalised in FORBIDDEN_REVIEWER_BIAS_KEYS:
                raise ValueError(f"{path}.{key} exposes model-derived reviewer bias")
            validate_blinded_annotation_payload(nested, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            validate_blinded_annotation_payload(nested, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
