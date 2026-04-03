"""Greedy build selection.

Picks the best-scoring component per category independently, which cannot honour
a global budget or cross-component constraints. Replaced by the CP-SAT engine.
"""

from collections.abc import Sequence


def build_greedy(candidates_by_category: dict[str, Sequence[dict]]) -> dict[str, dict]:
    chosen: dict[str, dict] = {}
    for category, candidates in candidates_by_category.items():
        ranked = sorted(candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True)
        if ranked:
            chosen[category] = ranked[0]
    return chosen
