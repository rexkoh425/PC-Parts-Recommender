from scripts.select_diverse_llm_intents import canonicalize_proposal, valid_proposal


def test_proposal_constraints_must_be_supported_and_present_in_query():
    proposal = {
        "query_text": "Need an AM5 CPU with at least 8 cores",
        "intent_family": "am5_upgrade",
        "must": [
            {"field": "category_attributes.core_count", "operator": "ge", "value": 8},
            {"field": "category_attributes.socket", "operator": "eq", "value": "AM5"},
        ],
    }
    assert valid_proposal("cpu", proposal)
    proposal["must"][0]["value"] = 16
    assert not valid_proposal("cpu", proposal)
    proposal["must"][0]["value"] = 80
    assert not valid_proposal("cpu", proposal)
    proposal["must"][0]["value"] = 8
    proposal["must"][0]["field"] = "category_attributes.benchmark_score"
    assert not valid_proposal("cpu", proposal)


def test_canonicalizes_catalogue_alias_and_rejects_unknown_value():
    row = {
        "query_text": "NVMe storage",
        "must": [{"field": "category_attributes.interface", "operator": "eq", "value": "nvme"}],
    }
    vocab = {("storage", "interface"): {"nvmepcie": {"nvme_pcie"}}}
    canonical = canonicalize_proposal("storage", row, vocab)
    assert canonical["must"][0]["value"] == "nvme_pcie"
    row["must"][0]["value"] = "unknown"
    assert canonicalize_proposal("storage", row, vocab) is None


def test_strict_under_limit_remains_strict():
    row = {
        "query_text": "air cooler under 70 mm",
        "must": [{"field": "category_attributes.height_mm", "operator": "le", "value": 70}],
    }
    assert canonicalize_proposal("cooler", row, {})["must"][0]["operator"] == "lt"
