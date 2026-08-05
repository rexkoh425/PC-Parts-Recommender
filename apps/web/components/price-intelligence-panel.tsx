import { formatEvidenceTimestamp, humanizeAttributeKey } from "../lib/catalogue";
import { formatSgd } from "../lib/format";
import type { PriceIntelligenceSummary } from "../lib/types";

function price(value: number | null): string {
  return typeof value === "number" ? formatSgd(value) : "Not available";
}

function percentage(value: number | null, suffix: string): string {
  return typeof value === "number" ? `${value.toFixed(1)}% ${suffix}` : "Insufficient history";
}

function percentile(value: number | null): string {
  return typeof value === "number"
    ? `${value.toFixed(1)} / 100 percentile rank (lower is cheaper)`
    : "Insufficient history";
}

function trend(value: PriceIntelligenceSummary["stock_trend"]): string {
  return value === "insufficient_history" ? "Insufficient history" : humanizeAttributeKey(value);
}

export function PriceIntelligencePanel({
  intelligence,
}: {
  intelligence: PriceIntelligenceSummary;
}) {
  return (
    <div className="price-intelligence" aria-label="Descriptive price history summary">
      <div className="price-intelligence__boundary" role="note">
        <strong>Past prices</strong>
        <p>
          Calculated only from stored observations through {formatEvidenceTimestamp(intelligence.as_of)}.

        </p>
      </div>
      <dl className="price-intelligence__metrics">
        <div><dt>Current observed total</dt><dd>{price(intelligence.current_delivered_price_sgd)}</dd></div>
        <div><dt>30-day median</dt><dd>{price(intelligence.median_30d_sgd)}</dd></div>
        <div><dt>90-day median</dt><dd>{price(intelligence.median_90d_sgd)}</dd></div>
        <div><dt>Recent observed low</dt><dd>{price(intelligence.recent_low_90d_sgd)}</dd></div>
        <div><dt>90-day price position</dt><dd>{percentile(intelligence.percentile_90d)}</dd></div>
        <div><dt>90-day volatility</dt><dd>{percentage(intelligence.volatility_90d_pct, "coefficient of variation")}</dd></div>
        <div><dt>Current observed sellers</dt><dd>{intelligence.current_seller_count.toLocaleString("en-SG")}</dd></div>
        <div><dt>Observed stock trend</dt><dd>{trend(intelligence.stock_trend)}</dd></div>
        <div><dt>Observed seller trend</dt><dd>{trend(intelligence.seller_trend)}</dd></div>
        <div><dt>History coverage</dt><dd>{intelligence.history_days_90d} observed days / 90</dd></div>
      </dl>
      <div className="price-intelligence__labels" aria-label="Descriptive price labels">
        {intelligence.labels.map((label) => <span key={label}>{label}</span>)}
      </div>
      {intelligence.analysis_truncated && (
        <p className="price-intelligence__notice">
          The calculation used the newest {intelligence.observations_analyzed.toLocaleString("en-SG")} observations under the service safety limit.
        </p>
      )}
      {intelligence.anomalies.length > 0 && (
        <details className="price-intelligence__anomalies">
          <summary>{intelligence.anomalies.length} robust price outlier {intelligence.anomalies.length === 1 ? "flag" : "flags"}</summary>
          <ul>
            {intelligence.anomalies.map((anomaly) => (
              <li key={`${anomaly.listing_id}-${anomaly.observed_at}`}>
                <span>{humanizeAttributeKey(anomaly.direction)} observation: {formatSgd(anomaly.delivered_price_sgd)}</span>
                <small>{formatEvidenceTimestamp(anomaly.observed_at)}</small>
                {anomaly.source_url && <a href={anomaly.source_url} target="_blank" rel="noreferrer">Stored source</a>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
