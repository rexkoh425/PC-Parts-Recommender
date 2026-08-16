"""Static contract checks for bounded, fail-closed freshness monitoring."""

from __future__ import annotations

from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _alerts_by_name() -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(
        (REPOSITORY_ROOT / "infra" / "monitoring" / "alerts.yml").read_text(
            encoding="utf-8"
        )
    )
    return {
        str(rule["alert"]): rule
        for group in payload["groups"]
        for rule in group["rules"]
        if "alert" in rule
    }


def test_freshness_alerts_fail_closed_for_stale_unknown_probe_and_scrape_loss() -> None:
    alerts = _alerts_by_name()

    stale_expression = str(alerts["PcbrCatalogueDataStale"]["expr"])
    assert 'pcbr_data_stale{kind=~"catalogue|prices"}' in stale_expression
    assert alerts["PcbrCatalogueDataStale"]["for"] == "5m"

    unavailable_expression = str(alerts["PcbrFreshnessTimestampUnavailable"]["expr"])
    assert (
        'pcbr_data_timestamp_available{kind=~"catalogue|prices"}'
        in unavailable_expression
    )
    assert alerts["PcbrFreshnessTimestampUnavailable"]["for"] == "5m"

    probe_expression = str(alerts["PcbrCatalogueFreshnessProbeFailed"]["expr"])
    assert "pcbr_catalogue_freshness_probe_success == 0" in probe_expression
    assert "absent(pcbr_catalogue_freshness_probe_success)" in probe_expression

    missing_series_expression = str(alerts["PcbrFreshnessMetricSeriesMissing"]["expr"])
    assert missing_series_expression.count('up{job="pcbr-api"} == 1') == 4
    assert missing_series_expression.count("unless on(job, instance)") == 4
    for metric in ("pcbr_data_stale", "pcbr_data_timestamp_available"):
        for kind in ("catalogue", "prices"):
            assert f'{metric}{{kind="{kind}"}}' in missing_series_expression

    target_expression = str(alerts["PcbrApiMetricsTargetDown"]["expr"])
    assert 'up{job="pcbr-api"} != 1' in target_expression
    assert 'absent(up{job="pcbr-api"})' in target_expression
    assert alerts["PcbrApiMetricsTargetDown"]["for"] == "2m"


def test_freshness_alerts_use_only_closed_or_static_labels() -> None:
    expressions = "\n".join(
        str(_alerts_by_name()[name]["expr"])
        for name in (
            "PcbrCatalogueDataStale",
            "PcbrFreshnessTimestampUnavailable",
            "PcbrCatalogueFreshnessProbeFailed",
            "PcbrFreshnessMetricSeriesMissing",
            "PcbrApiMetricsTargetDown",
        )
    )

    for unbounded_label in (
        "product_id",
        "listing_id",
        "retailer",
        "source_url",
        "data_version",
        "model_version",
    ):
        assert unbounded_label not in expressions
