from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.build_llm_judge_queue import explicit_unknown_price_requirement
from scripts.generate_llm_ranking_queries import exclude_previous, generate, intent_signature


def test_new_queries_keep_matching_constraints_in_same_group() -> None:
    queries = generate()
    assert len(queries) == 320
    assert len({q["query_id"] for q in queries}) == 320
    assert len({q["query_group_id"] for q in queries}) >= 300
    assert set(Counter(q["category"] for q in queries).values()) == {40}
    groups_by_constraints: dict[str, str] = {}
    for query in queries:
        signature = json.dumps(
            [query["category"], sorted(query["must"], key=lambda c: c["field"])],
            sort_keys=True,
        )
        assert (
            groups_by_constraints.setdefault(signature, query["query_group_id"])
            == query["query_group_id"]
        )
    previous = json.loads(
        Path("evals/retrieval/real_catalog_silver_queries.v1.json").read_text(encoding="utf-8")
    )["queries"]
    assert {q["query_text"].casefold() for q in previous}.isdisjoint(
        {q["query_text"].casefold() for q in queries}
    )
    filtered = exclude_previous(queries, previous)
    assert len(filtered) == 316
    assert {intent_signature(q) for q in previous}.isdisjoint(
        {intent_signature(q) for q in filtered}
    )


def test_explicit_budget_is_recorded_as_unknown_price_constraint():
    req = explicit_unknown_price_requirement("graphics card under $1,000")
    assert req is not None and req["field"] == "known_price" and req["value"] == 1000
    assert explicit_unknown_price_requirement("graphics card under 200 W") is None
