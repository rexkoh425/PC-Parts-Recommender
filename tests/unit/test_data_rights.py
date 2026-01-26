from __future__ import annotations

from datetime import date

import pytest

from pc_build_recommender.data_rights import DataUseRights, production_catalog_rights_are_valid


def _rights() -> dict[str, object]:
    return {
        "contract_reference": "fixture-v1",
        "contract_version_url": "contract://fixture/v1",
        "consent_effective_on": "2026-01-01",
        "consent_expires_on": None,
        "retention_days": 365,
        "deletion_required_on_termination": True,
        "deletion_sla_days": 30,
        "territories": ["SG"],
        "may_display": True,
        "may_cache": True,
        "may_store_history": True,
        "may_redistribute": False,
        "may_embed": False,
        "may_train": False,
        "may_derive": True,
    }


def test_production_catalog_rights_require_grants_active_consent_and_territory() -> None:
    evaluation_date = date(2026, 7, 22)
    assert production_catalog_rights_are_valid(_rights(), territory="SG", on_date=evaluation_date)

    denied = _rights()
    denied["may_cache"] = False
    assert not production_catalog_rights_are_valid(denied, territory="SG", on_date=evaluation_date)

    expired = _rights()
    expired["consent_expires_on"] = "2026-07-21"
    assert not production_catalog_rights_are_valid(expired, territory="SG", on_date=evaluation_date)

    wrong_territory = _rights()
    wrong_territory["territories"] = ["MY"]
    assert not production_catalog_rights_are_valid(
        wrong_territory, territory="SG", on_date=evaluation_date
    )


@pytest.mark.parametrize("field", ["retention_days", "deletion_sla_days"])
@pytest.mark.parametrize("invalid_value", [True, False, 30.0, "30"])
def test_data_use_rights_rejects_coerced_lifecycle_integers(
    field: str,
    invalid_value: object,
) -> None:
    payload = _rights()
    payload[field] = invalid_value

    with pytest.raises(TypeError, match=rf"{field} must be an integer or null"):
        DataUseRights.from_mapping(payload)
