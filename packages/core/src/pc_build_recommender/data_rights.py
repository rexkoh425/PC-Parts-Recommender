"""Machine-readable source-use rights shared by ingestion and serving gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any


class DataUse(StrEnum):
    DISPLAY = "display"
    CACHE = "cache"
    STORE_HISTORY = "store_history"
    REDISTRIBUTE = "redistribute"
    EMBED = "embed"
    TRAIN = "train"
    DERIVE = "derive"

    @property
    def field_name(self) -> str:
        return f"may_{self.value}"


PRODUCTION_CATALOG_USES = (
    DataUse.DISPLAY,
    DataUse.CACHE,
    DataUse.STORE_HISTORY,
    DataUse.DERIVE,
)


@dataclass(frozen=True, slots=True)
class DataUseRights:
    """One source contract's explicit grants and lifecycle obligations."""

    contract_reference: str
    contract_version_url: str
    consent_effective_on: date
    consent_expires_on: date | None
    retention_days: int | None
    deletion_required_on_termination: bool
    deletion_sla_days: int | None
    territories: tuple[str, ...]
    may_display: bool
    may_cache: bool
    may_store_history: bool
    may_redistribute: bool
    may_embed: bool
    may_train: bool
    may_derive: bool

    def __post_init__(self) -> None:
        if not isinstance(self.contract_reference, str) or not self.contract_reference.strip():
            raise ValueError("contract_reference is required")
        if not isinstance(self.contract_version_url, str) or not self.contract_version_url.strip():
            raise ValueError("contract_version_url is required")
        if not isinstance(self.consent_effective_on, date):
            raise TypeError("consent_effective_on must be a date")
        if self.consent_expires_on is not None and not isinstance(self.consent_expires_on, date):
            raise TypeError("consent_expires_on must be a date or null")
        if self.consent_expires_on is not None and (
            self.consent_expires_on < self.consent_effective_on
        ):
            raise ValueError("consent expiry cannot precede its effective date")
        if self.retention_days is not None and type(self.retention_days) is not int:
            raise TypeError("retention_days must be an integer or null")
        if self.retention_days is not None and self.retention_days < 1:
            raise ValueError("retention_days must be positive or null for contract-defined")
        if type(self.deletion_required_on_termination) is not bool:
            raise TypeError("deletion_required_on_termination must be a boolean")
        if self.deletion_sla_days is not None and type(self.deletion_sla_days) is not int:
            raise TypeError("deletion_sla_days must be an integer or null")
        if self.deletion_required_on_termination:
            if self.deletion_sla_days is None or self.deletion_sla_days < 1:
                raise ValueError("a deletion SLA is required when termination deletion applies")
        elif self.deletion_sla_days is not None:
            raise ValueError("deletion_sla_days requires deletion_required_on_termination")
        territories = tuple(
            sorted(
                {territory.strip().upper() for territory in self.territories if territory.strip()}
            )
        )
        if not territories:
            raise ValueError("at least one allowed territory is required")
        object.__setattr__(self, "territories", territories)
        for use in DataUse:
            if type(getattr(self, use.field_name)) is not bool:
                raise TypeError(f"{use.field_name} must be a boolean")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DataUseRights:
        required = {
            "contract_reference",
            "contract_version_url",
            "consent_effective_on",
            "consent_expires_on",
            "retention_days",
            "deletion_required_on_termination",
            "deletion_sla_days",
            "territories",
            *(use.field_name for use in DataUse),
        }
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        if missing:
            raise ValueError(f"data-use rights missing fields: {missing}")
        if extra:
            raise ValueError(f"data-use rights contain unknown fields: {extra}")
        territories = payload["territories"]
        if not isinstance(territories, list | tuple):
            raise TypeError("territories must be an array")
        effective = date.fromisoformat(str(payload["consent_effective_on"]))
        raw_expiry = payload["consent_expires_on"]
        expiry = date.fromisoformat(str(raw_expiry)) if raw_expiry is not None else None
        return cls(
            contract_reference=str(payload["contract_reference"]),
            contract_version_url=str(payload["contract_version_url"]),
            consent_effective_on=effective,
            consent_expires_on=expiry,
            retention_days=payload["retention_days"],
            deletion_required_on_termination=payload["deletion_required_on_termination"],
            deletion_sla_days=payload["deletion_sla_days"],
            territories=tuple(str(item) for item in territories),
            **{use.field_name: payload[use.field_name] for use in DataUse},
        )

    def assert_consent_active(self, *, on_date: date | None = None) -> None:
        today = on_date or date.today()
        if today < self.consent_effective_on:
            raise PermissionError("source consent is not yet effective")
        if self.consent_expires_on is not None and today > self.consent_expires_on:
            raise PermissionError("source consent has expired")

    def assert_allowed(self, use: DataUse) -> None:
        if not getattr(self, use.field_name):
            raise PermissionError(f"contract {self.contract_reference} does not permit {use.value}")

    def assert_catalog_serving_allowed(
        self,
        *,
        territory: str,
        on_date: date | None = None,
    ) -> None:
        self.assert_consent_active(on_date=on_date)
        required_territory = territory.strip().upper()
        if not required_territory or required_territory not in self.territories:
            raise PermissionError(
                f"contract {self.contract_reference} does not permit territory "
                f"{required_territory or '<empty>'}"
            )
        for use in PRODUCTION_CATALOG_USES:
            self.assert_allowed(use)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_reference": self.contract_reference,
            "contract_version_url": self.contract_version_url,
            "consent_effective_on": self.consent_effective_on.isoformat(),
            "consent_expires_on": (
                self.consent_expires_on.isoformat() if self.consent_expires_on is not None else None
            ),
            "retention_days": self.retention_days,
            "deletion_required_on_termination": self.deletion_required_on_termination,
            "deletion_sla_days": self.deletion_sla_days,
            "territories": list(self.territories),
            **{use.field_name: getattr(self, use.field_name) for use in DataUse},
        }


def require_data_use(record: Mapping[str, Any], use: DataUse) -> None:
    """Fail closed when a normalized record is missing or denies a requested use."""

    raw_rights = record.get("data_use_rights")
    if not isinstance(raw_rights, Mapping):
        raise PermissionError("record has no machine-readable data-use rights")
    if raw_rights.get(use.field_name) is not True:
        raise PermissionError(f"record does not permit {use.value}")
    rights = DataUseRights.from_mapping(raw_rights)
    rights.assert_consent_active()
    rights.assert_allowed(use)


def production_catalog_rights_are_valid(
    rights: object,
    *,
    territory: str = "SG",
    on_date: date | None = None,
) -> bool:
    if not isinstance(rights, Mapping):
        return False
    try:
        parsed = DataUseRights.from_mapping(rights)
        parsed.assert_catalog_serving_allowed(territory=territory, on_date=on_date)
    except (PermissionError, TypeError, ValueError):
        return False
    return True
