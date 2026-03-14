"""Catalogue-facing aliases for strict ER production evaluation evidence."""

from pc_build_recommender.entity_resolution.release_contracts import (
    ER_EVALUATION_SCHEMA_VERSION,
    ER_EVALUATION_SCHEMA_VERSION_V2,
    EntityResolutionProductionEvaluation,
    load_entity_resolution_evaluation,
)

# Preserve the catalogue API name while the core contract distinguishes this persisted
# production report from the lower-level binary-classification metrics record.
EntityResolutionEvaluation = EntityResolutionProductionEvaluation

__all__ = [
    "ER_EVALUATION_SCHEMA_VERSION",
    "ER_EVALUATION_SCHEMA_VERSION_V2",
    "EntityResolutionEvaluation",
    "load_entity_resolution_evaluation",
]
