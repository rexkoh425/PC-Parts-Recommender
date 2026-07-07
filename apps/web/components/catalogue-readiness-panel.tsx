"use client";

import { useEffect, useState } from "react";
import { getFreshness, USING_DEMO_DATA } from "@/lib/api";
import { categoryLabels, humanizeToken } from "@/lib/format";
import type { ComponentKind, FreshnessSummary } from "@/lib/types";

const requiredCategories: ComponentKind[] = [
  "cpu",
  "gpu",
  "motherboard",
  "memory",
  "storage",
  "psu",
  "cooler",
  "case",
];

function countCoveredCategories(values: Record<string, number>): number {
  return requiredCategories.filter((category) => (values[category] ?? 0) > 0).length;
}

function labelForCategory(category: ComponentKind): string {
  return categoryLabels[category] ?? humanizeToken(category);
}

function LoadingState() {
  return (
    <section className="catalogue-readiness shell" id="data-status" aria-labelledby="data-status-title">
      <div className="catalogue-readiness__heading">
        <div>
          <p className="eyebrow">Catalogue release gate</p>
          <h2 id="data-status-title">Checking catalogue readiness.</h2>
        </div>
        <p>Fetching the current coverage and release policy summary.</p>
      </div>
    </section>
  );
}

export function CatalogueReadinessPanel() {
  const [freshness, setFreshness] = useState<FreshnessSummary | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    getFreshness({ signal: controller.signal })
      .then((result) => {
        if (active) setFreshness(result);
      })
      .catch(() => {
        if (active) setUnavailable(true);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  if (!freshness && !unavailable) return <LoadingState />;

  const report = freshness?.catalogue_readiness;
  const isDemo = USING_DEMO_DATA;
  const productionReady = Boolean(freshness?.production_ready && report?.production_ready);
  const blockers = report?.production_blockers ?? freshness?.readiness_blockers ?? [];
  const pricedCategories = report
    ? countCoveredCategories(report.matched_listings_by_category)
    : 0;
  const inStockCategories = report
    ? countCoveredCategories(report.in_stock_listings_by_category)
    : 0;

  return (
    <section className="catalogue-readiness shell" id="data-status" aria-labelledby="data-status-title">
      <div className="catalogue-readiness__heading">
        <div>
          <p className="eyebrow">{isDemo ? "Public demo boundary" : "Catalogue release gate"}</p>
          <h2 id="data-status-title">
            {isDemo
              ? "The demo is interactive; the market-data release is not connected."
              : productionReady
                ? "The catalogue has passed its release gate."
                : "The catalogue is blocked from production recommendations."}
          </h2>
        </div>
        <p>
          {isDemo
            ? "Sample components support the portfolio flow only. They do not represent current retailers, stock, source rights, or a production recommendation release."
            : "This gate combines category coverage, compatibility fields, listing provenance, data-use rights, entity-resolution controls, and stock coverage."}
        </p>
      </div>

      <div className={"catalogue-readiness__status catalogue-readiness__status--" + (productionReady ? "ready" : "blocked")}>
        <span className="catalogue-readiness__status-dot" aria-hidden="true" />
        <div>
          <strong>
            {isDemo
              ? "Demo-only data"
              : productionReady
                ? "Production recommendation traffic eligible"
                : unavailable
                  ? "Readiness service unavailable"
                  : "Production recommendation traffic blocked"}
          </strong>
          <p>
            {isDemo
              ? "Illustrative prices and curated parts are explicitly separated from a measured release."
              : productionReady
                ? "All configured hard release checks are passing for the current data version."
                : blockers[0] ?? "No release-policy summary was returned."}
          </p>
        </div>
      </div>

      {report ? (
        <>
          <div className="catalogue-readiness__metrics" aria-label="Catalogue readiness metrics">
            <article>
              <strong>{freshness?.product_count.toLocaleString("en-SG")}</strong>
              <span>verified canonical products</span>
            </article>
            <article>
              <strong>{freshness?.listing_count.toLocaleString("en-SG")}</strong>
              <span>matched retailer listings</span>
            </article>
            <article>
              <strong>{Math.round(report.mapping_rate * 100)}%</strong>
              <span>offer-to-product mapping rate</span>
            </article>
            <article>
              <strong>{inStockCategories}/{requiredCategories.length}</strong>
              <span>categories with known in-stock coverage</span>
            </article>
          </div>

          <div className="catalogue-readiness__coverage">
            <div>
              <h3>Coverage by component category</h3>
              <p>
                Price coverage: {pricedCategories}/{requiredCategories.length} required categories
                {" · "}In-stock coverage: {inStockCategories}/{requiredCategories.length}
              </p>
            </div>
            <ul>
              {requiredCategories.map((category) => {
                const products = report.products_by_category[category] ?? 0;
                const ready = report.compatibility_ready_products_by_category[category] ?? 0;
                const matched = report.matched_listings_by_category[category] ?? 0;
                const inStock = report.in_stock_listings_by_category[category] ?? 0;
                const covered = products > 0 && matched > 0 && inStock > 0;
                return (
                  <li key={category} className={covered ? "is-covered" : "is-missing"}>
                    <span className="catalogue-readiness__coverage-dot" aria-hidden="true" />
                    <strong>{labelForCategory(category)}</strong>
                    <small>
                      {products} products · {ready} compatibility-ready · {matched} priced · {inStock} in stock
                    </small>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="catalogue-readiness__provenance">
            <span>Data rights territory: {report.rights_territory}</span>
            <span>
              Valid production offer grants: {report.offer_rights_production_valid_count.toLocaleString("en-SG")}
            </span>
            <span>
              Entity-resolution model: {report.entity_resolution_model_version ?? "not configured"}
            </span>
          </div>
        </>
      ) : (
        <div className="catalogue-readiness__empty">
          <strong>{isDemo ? "A measured release report is intentionally unavailable in this demo." : "No measured release report is available."}</strong>
          <p>
            {isDemo
              ? "Use the interactive builder to inspect the product flow, then read the evidence section for the current project's verified boundaries."
              : "The service will continue to fail closed until the data pipeline publishes a versioned readiness report."}
          </p>
        </div>
      )}

      {!isDemo && blockers.length > 1 && (
        <details className="catalogue-readiness__blockers">
          <summary>Show all {blockers.length} release blockers</summary>
          <ul>
            {blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}
          </ul>
        </details>
      )}
    </section>
  );
}
