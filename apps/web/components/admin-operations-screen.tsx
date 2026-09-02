"use client";

import { useRef, useState } from "react";
import { getAdminOperations, USING_DEMO_DATA } from "@/lib/api";
import { categoryLabels, formatFreshness } from "@/lib/format";
import type { AdminOperationsResponse } from "@/lib/types";

function count(value: number): string {
  return value.toLocaleString("en-SG");
}

function MappingQueue({ data }: { data: NonNullable<AdminOperationsResponse["mapping_queue"]> }) {
  const items = [
    ["Offers inspected", data.offer_count],
    ["Mapped", data.matched_count],
    ["Unmatched", data.unmatched_count],
    ["Manual review", data.manual_review_count],
    ["Conflicting", data.rejected_conflict_count],
    ["Model rejected", data.model_rejected_count],
  ];
  return (
    <section className="operations-card" aria-labelledby="mapping-queue-heading">
      <p className="eyebrow">Entity resolution</p>
      <h2 id="mapping-queue-heading">Mapping queue</h2>
      <dl className="operations-metrics">
        {items.map(([label, value]) => (
          <div key={label as string}>
            <dt>{label}</dt>
            <dd>{count(value as number)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function PriceFreshness({ data }: { data: NonNullable<AdminOperationsResponse["price_freshness"]> }) {
  return (
    <section className="operations-card" aria-labelledby="price-freshness-heading">
      <p className="eyebrow">Observed price snapshots</p>
      <h2 id="price-freshness-heading">Freshness</h2>
      <dl className="operations-metrics operations-metrics--three">
        <div><dt>Snapshots</dt><dd>{count(data.snapshot_count)}</dd></div>
        <div><dt>Stale snapshots</dt><dd>{data.stale_snapshot_count === null || data.stale_snapshot_count === undefined ? "Not reported" : count(data.stale_snapshot_count)}</dd></div>
        <div><dt>Newest observation</dt><dd>{data.newest_observed_at ? formatFreshness(data.newest_observed_at).replace("Updated ", "") : "Not reported"}</dd></div>
      </dl>
      <p className="operations-note">A price snapshot is considered stale after {data.stale_after_hours} hours. Counts are observations, not unique offers.</p>
    </section>
  );
}

function PipelineOperations({
  data,
}: {
  data: NonNullable<AdminOperationsResponse["pipeline_operations"]>;
}) {
  return (
    <section className="operations-card" aria-labelledby="pipeline-operations-heading">
      <p className="eyebrow">Instrumented pipeline</p>
      <h2 id="pipeline-operations-heading">Operation receipts</h2>
      <dl className="operations-metrics operations-metrics--three">
        <div><dt>Events</dt><dd>{count(data.event_count)}</dd></div>
        <div><dt>Failures</dt><dd>{count(data.failed_count)}</dd></div>
        <div>
          <dt>Latest failure</dt>
          <dd>{data.latest_failure_at ? formatFreshness(data.latest_failure_at).replace("Updated ", "") : "None in window"}</dd>
        </div>
      </dl>
      <p className="operations-note">
        Last {data.event_window_hours} hours. {data.invalid_receipt_count > 0
          ? `${count(data.invalid_receipt_count)} invalid receipt${data.invalid_receipt_count === 1 ? "" : "s"} excluded. `
          : ""}{data.truncated ? "Older entries were dropped." : ""}
      </p>
    </section>
  );
}

function OperationsData({ data }: { data: AdminOperationsResponse }) {
  return (
    <div className="operations-data" aria-live="polite">
      <div className="operations-provenance">
        <span>Serving mode <strong>{data.mode.replaceAll("_", " ")}</strong></span>
        <span>Data <strong>{data.data_version}</strong></span>
        <span>Measured <strong>{formatFreshness(data.generated_at).replace("Updated ", "")}</strong></span>
      </div>

      <div className="operations-grid">
        {data.mapping_queue && <MappingQueue data={data.mapping_queue} />}
        {data.price_freshness && <PriceFreshness data={data.price_freshness} />}
        {data.pipeline_operations && <PipelineOperations data={data.pipeline_operations} />}
        <section className="operations-card operations-card--blockers" aria-labelledby="release-blockers-heading">
          <p className="eyebrow">Release gate</p>
          <h2 id="release-blockers-heading">Compatibility and release blockers</h2>
          {data.release_blockers.length ? (
            <ul className="operations-list operations-list--alert">
              {data.release_blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}
            </ul>
          ) : <p className="operations-ok">No current release blockers were reported.</p>}
        </section>
      </div>

      <section className="operations-table-card" aria-labelledby="missing-fields-heading">
        <div className="section-heading">
          <div><p className="eyebrow">Catalogue quality</p><h2 id="missing-fields-heading">Missing critical fields</h2></div>
          <p>Only aggregate counts are exposed.</p>
        </div>
        {data.missing_critical_fields.length ? (
          <div className="operations-table-wrap">
            <table className="operations-table">
              <thead><tr><th>Category</th><th>Field group</th><th>Missing products</th><th>Products checked</th></tr></thead>
              <tbody>
                {data.missing_critical_fields.map((item) => (
                  <tr key={`${item.category}-${item.field_group}`}>
                    <th scope="row">{categoryLabels[item.category]}</th>
                    <td data-label="Field group">{item.field_group}</td>
                    <td data-label="Missing products">{count(item.missing_product_count)}</td>
                    <td data-label="Products checked">{count(item.product_count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="operations-ok">No missing critical field count is currently reported.</p>}
      </section>

      <section className="operations-disclosure" aria-labelledby="operations-evidence-heading">
        <h2 id="operations-evidence-heading">Operational evidence boundary</h2>
        <p>
          {data.pipeline_failure_events_available
            ? "This response includes instrumented pipeline-operation outcomes. Scheduler, queue, and worker-control failures remain in Dagster."
            : "Pipeline-operation receipts are unavailable for this serving response; inspect the authenticated pipeline run store separately."}
        </p>
        {data.notes.length > 0 && <ul>{data.notes.map((note) => <li key={note}>{note}</li>)}</ul>}
      </section>
    </div>
  );
}

export function AdminOperationsScreen() {
  const [token, setToken] = useState("");
  const [data, setData] = useState<AdminOperationsResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const errorRef = useRef<HTMLDivElement>(null);

  async function loadOperations(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const suppliedToken = token.trim();
    if (!suppliedToken) {
      setError("Enter an administrator token to load restricted operational counters.");
      window.requestAnimationFrame(() => errorRef.current?.focus());
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await getAdminOperations(suppliedToken);
      setData(response);
    } catch (requestError) {
      setData(null);
      setError(requestError instanceof Error ? requestError.message : "The operations service could not be loaded.");
      window.requestAnimationFrame(() => errorRef.current?.focus());
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell operations-page">
      <header className="operations-header">
        <div>
          <p className="eyebrow">Restricted operator workspace</p>
          <h1>Catalogue operations</h1>
          <p className="lede">Review aggregated catalogue health and release blockers without exposing raw retailer records or source payloads.</p>
        </div>
        <div className="operations-header__mark" aria-hidden="true"><span>READ ONLY</span><strong>OPS</strong><small>Aggregate evidence</small></div>
      </header>

      {USING_DEMO_DATA ? (
        <section className="operations-unavailable" role="status">
          <strong>Not connected in this public portfolio demo.</strong>
          <p>Protected operations counters are only available when a serving API and administrator token are configured. No token can unlock demo data.</p>
        </section>
      ) : (
        <>
          <form className="operations-access" onSubmit={loadOperations} noValidate>
            <div>
              <label htmlFor="admin-token">Administrator token</label>
              <input id="admin-token" type="password" autoComplete="off" spellCheck="false" maxLength={512} value={token} onChange={(event) => setToken(event.target.value)} aria-describedby="admin-token-help" />
              <small id="admin-token-help">Held only in this page&apos;s memory for the request. It is never saved to browser storage.</small>
            </div>
            <button className="button button--primary" type="submit" disabled={loading}>
              {loading && <span className="button-spinner" aria-hidden="true" />}
              {loading ? "Loading counters" : "Load operations"}
            </button>
          </form>
          {error && <div className="error-summary" role="alert" tabIndex={-1} ref={errorRef}><strong>Operations unavailable</strong><p>{error}</p></div>}
          {data && <OperationsData data={data} />}
        </>
      )}
    </main>
  );
}
