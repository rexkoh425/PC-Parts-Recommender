"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  getProduct,
  getProductBenchmarks,
  getProductPrices,
  getProductReviews,
  getSessionId,
  readProductImpression,
  trackInteraction,
  USING_DEMO_DATA,
} from "@/lib/api";
import {
  confidencePresentation,
  formatAttributeValue,
  formatEvidenceTimestamp,
  humanizeAttributeKey,
  observedStockLabel,
  priceObservationPresentation,
  stockTone,
} from "@/lib/catalogue";
import { categoryLabels, formatSgd } from "@/lib/format";
import type {
  ProductBenchmarksResponse,
  ProductDetail,
  ProductPricesResponse,
  ProductReviewsResponse,
} from "@/lib/types";
import { PriceIntelligencePanel } from "./price-intelligence-panel";

export interface ProductEvidenceState {
  product: ProductDetail;
  prices?: ProductPricesResponse;
  pricesError?: string;
  benchmarks?: ProductBenchmarksResponse;
  benchmarksError?: string;
  reviews?: ProductReviewsResponse;
  reviewsError?: string;
}

// Read the impression context straight off the URL rather than through
// useSearchParams(). The value only ever feeds analytics, never markup, and
// useSearchParams() opts its subtree out of static prerendering — which would
// push these records back to client-only rendering, the exact problem the
// server-side load above exists to fix.
function readImpressionContext(): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("impression");
}

function reasonFor(result: PromiseSettledResult<unknown>): string | undefined {
  if (result.status === "fulfilled") return undefined;
  return result.reason instanceof Error ? result.reason.message : "This evidence is unavailable.";
}

function EvidencePanelError({ title, message }: { title: string; message?: string }) {
  return (
    <div className="evidence-panel-error" role="status">
      <span aria-hidden="true">!</span>
      <div><strong>{title}</strong><p>{message ?? "This evidence is unavailable."}</p></div>
    </div>
  );
}

export function ProductDetailScreen({
  productId,
  initialState = null,
}: {
  productId: string;
  /**
   * Evidence resolved on the server. When present the screen renders complete
   * markup on the first pass, which is what a crawler and a slow connection
   * see; the fetch effect below then has nothing left to do.
   */
  initialState?: ProductEvidenceState | null;
}) {
  const [state, setState] = useState<ProductEvidenceState | null>(initialState);
  const [error, setError] = useState("");
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    // Already served from the server, and not a retry: nothing to fetch.
    if (initialState && retryKey === 0) return;
    let active = true;
    const controller = new AbortController();
    const requestOptions = { signal: controller.signal };
    Promise.allSettled([
      getProduct(productId, requestOptions),
      getProductPrices(productId, requestOptions),
      getProductBenchmarks(productId, requestOptions),
      getProductReviews(productId, requestOptions),
    ]).then(([productResult, pricesResult, benchmarksResult, reviewsResult]) => {
      if (!active) return;
      if (productResult.status === "rejected") {
        setError(
          productResult.reason instanceof Error
            ? productResult.reason.message
            : "The product could not be loaded.",
        );
        return;
      }
      const next: ProductEvidenceState = {
        product: productResult.value,
        prices: pricesResult.status === "fulfilled" ? pricesResult.value : undefined,
        pricesError: reasonFor(pricesResult),
        benchmarks:
          benchmarksResult.status === "fulfilled" ? benchmarksResult.value : undefined,
        benchmarksError: reasonFor(benchmarksResult),
        reviews: reviewsResult.status === "fulfilled" ? reviewsResult.value : undefined,
        reviewsError: reasonFor(reviewsResult),
      };
      setState(next);
    });
    return () => {
      active = false;
      controller.abort();
    };
  }, [initialState, productId, retryKey]);

  const viewedProductId = state?.product.product_id;
  const viewedCategory = state?.product.category;
  useEffect(() => {
    if (!viewedProductId || !viewedCategory) return;
    void trackInteraction({
      event_type: "component_viewed",
      session_id: getSessionId(),
      impression_token: readProductImpression(viewedProductId, readImpressionContext()),
      metadata: { category: viewedCategory, surface: "catalogue_detail" },
    });
  }, [viewedProductId, viewedCategory]);

  useEffect(() => {
    if (!state || typeof window === "undefined") return;
    const evidenceTarget = window.location.hash.slice(1);
    if (!["prices-heading", "benchmarks-heading", "reviews-heading"].includes(evidenceTarget)) {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      const heading = document.getElementById(evidenceTarget);
      heading?.scrollIntoView({ block: "start" });
      heading?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [state]);

  if (error) {
    return (
      <main className="shell state-page">
        <div className="state-card" role="alert">
          <span className="state-card__icon" aria-hidden="true">?</span>
          <p className="eyebrow">Product evidence unavailable</p>
          <h1>We could not load this catalogue product.</h1>
          <p>{error}</p>
          <div className="button-row">
            <button
              className="button button--primary"
              type="button"
              onClick={() => {
                setError("");
                setState(null);
                setRetryKey((key) => key + 1);
              }}
            >Try again</button>
            <Link className="button button--secondary" href="/catalogue">Back to catalogue</Link>
          </div>
        </div>
      </main>
    );
  }

  if (!state) {
    return (
      <main className="shell product-page" aria-busy="true">
        <div className="skeleton skeleton--eyebrow" />
        <div className="skeleton skeleton--title" />
        <div className="product-layout">
          <div className="skeleton-card skeleton-card--tall" />
          <div className="skeleton-card" />
        </div>
        <p className="sr-only" role="status">Loading product evidence.</p>
      </main>
    );
  }

  const { product, prices, benchmarks, reviews } = state;
  const currentPrice = prices?.current_lowest_price_sgd ?? product.lowest_price_sgd;
  /*
   * Live records populate roughly half their attribute slots, so listing every
   * key gave a table where seven of fourteen rows read "Not reported" - noise
   * that buries the specs that do exist. An absent value tells the reader
   * nothing, so it is left out rather than printed.
   */
  const attributes = Object.entries(product.attributes)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .filter(([, value]) => !(Array.isArray(value) && value.length === 0))
    .sort(([left], [right]) => left.localeCompare(right));

  return (
    <main className="shell product-page">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link href="/catalogue">Catalogue</Link>
        <span aria-hidden="true">/</span>
        <span aria-current="page">{product.canonical_name}</span>
      </nav>

      <header className="product-header">
        <div className="product-header__identity">
          <div className="product-header__topline">
            <span className={`category-chip category-chip--${product.category}`}>
              {categoryLabels[product.category]}
            </span>
            <span className={`observed-stock observed-stock--${stockTone(product.stock_status)}`}>
              <span aria-hidden="true" />
              {observedStockLabel(product.stock_status)}
            </span>
          </div>
          <p className="eyebrow">{product.brand ?? "Brand not reported"}</p>
          <h1>{product.canonical_name}</h1>
          <p className="lede">
            Specifications, price history, benchmarks and reviews for this part.
          </p>
          <Link className="product-header__compare" href={`/compare?products=${encodeURIComponent(product.product_id)}`}>
            Compare similar parts
            <span aria-hidden="true">→</span>
          </Link>
        </div>
        <div className="product-price-block">
          <small>{USING_DEMO_DATA ? "Price" : "Lowest observed listing"}</small>
          <strong>
            {typeof currentPrice === "number" ? formatSgd(currentPrice) : "Price unavailable"}
          </strong>
          <span>
            {USING_DEMO_DATA
              ? "Recorded August 2026. Not a live retailer quote."
              : "Verify price and availability with the retailer before purchase."}
          </span>
        </div>
      </header>

      <div className="product-layout">
        <div className="product-main">
          <section className="detail-section product-section" aria-labelledby="specification-heading">
            <div className="section-heading">
              <div><h2 id="specification-heading">Specifications</h2></div>
              <p>Updated {formatEvidenceTimestamp(product.updated_at)}</p>
            </div>
            {attributes.length ? (
              <dl className="specification-grid">
                {attributes.map(([key, value]) => (
                  <div key={key}><dt>{humanizeAttributeKey(key)}</dt><dd>{formatAttributeValue(value)}</dd></div>
                ))}
              </dl>
            ) : (
              <div className="empty-evidence"><strong>No structured specifications are available.</strong><p>Missing attributes are not inferred from the product name.</p></div>
            )}
          </section>

          <section className="detail-section product-section" aria-labelledby="prices-heading">
            <div className="section-heading">
              <div><h2 id="prices-heading" tabIndex={-1}>Price history</h2></div>
              <p>{prices ? "" : "Price history is unavailable."}</p>
            </div>
            {prices?.price_intelligence && (
              <PriceIntelligencePanel intelligence={prices.price_intelligence} />
            )}
            {state.pricesError ? <EvidencePanelError title="Price evidence could not be loaded" message={state.pricesError} /> : prices?.observations.length ? (
              <div className="observation-list">
                {prices.observations.slice(0, 12).map((observation) => {
                  const delivered = observation.base_price_sgd + observation.shipping_price_sgd;
                  const offer = priceObservationPresentation(observation);
                  return (
                    <article key={`${observation.listing_id}-${observation.observed_at}`}>
                      <div><strong>{observation.retailer}</strong><span>{offer.conditionLabel}</span></div>
                      <div><strong>{formatSgd(delivered)}</strong><small>{observation.shipping_price_sgd ? `${formatSgd(observation.shipping_price_sgd)} shipping included` : "No shipping charge recorded"}</small></div>
                      <div><span>{observedStockLabel(observation.stock_status)}</span><small>{offer.eligibilityLabel}</small></div>
                      <div><span>{formatEvidenceTimestamp(observation.observed_at)}</span>{observation.listing_url && <a href={observation.listing_url} target="_blank" rel="noreferrer" aria-label={`Open ${observation.retailer} retailer observation`}>Open retailer ↗</a>}</div>
                    </article>
                  );
                })}
              </div>
            ) : (
              <div className="empty-evidence"><strong>No price recorded</strong><p>Prices come from a separate retailer dataset that does not cover this part.</p></div>
            )}
          </section>

          <section className="detail-section product-section" aria-labelledby="benchmarks-heading">
            <div className="section-heading">
              <div><p className="eyebrow">Observed versus modelled</p><h2 id="benchmarks-heading" tabIndex={-1}>Benchmark evidence</h2></div>
              <p>{benchmarks ? "" : "Benchmarks are unavailable."}</p>
            </div>
            {state.benchmarksError ? <EvidencePanelError title="Benchmark evidence could not be loaded" message={state.benchmarksError} /> : benchmarks?.benchmarks.length ? (
              <div className="benchmark-grid">
                {benchmarks.benchmarks.map((benchmark, index) => (
                  <article key={`${benchmark.benchmark_name}-${benchmark.workload}-${index}`}>
                    <div className="benchmark-card__topline"><span className={`evidence-badge evidence-badge--${benchmark.basis}`}>{benchmark.basis === "observed" ? "Observed" : "Predicted"}</span><small>{humanizeAttributeKey(benchmark.workload)}</small></div>
                    <h3>{benchmark.benchmark_name}</h3>
                    <strong className="benchmark-card__score">{benchmark.score.toLocaleString("en-SG")} <small>{benchmark.unit}</small></strong>
                    <dl><div><dt>Direction</dt><dd>{benchmark.higher_is_better ? "Higher is better" : "Lower is better"}</dd></div></dl>
                    <footer><span>{formatEvidenceTimestamp(benchmark.observed_at)}</span>{benchmark.source_url && <a href={benchmark.source_url} target="_blank" rel="noreferrer" aria-label={`Open source for ${benchmark.benchmark_name}`}>Source ↗</a>}</footer>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty-evidence"><strong>No benchmarks for this part</strong><p>We would rather show nothing than estimate a score we cannot support.</p></div>
            )}
          </section>

          <section className="detail-section product-section" aria-labelledby="reviews-heading">
            <div className="section-heading">
              <div><p className="eyebrow">Permitted cited sources</p><h2 id="reviews-heading" tabIndex={-1}>Review evidence</h2></div>
              <p>{reviews ? "" : "Reviews are unavailable."}</p>
            </div>
            {state.reviewsError ? <EvidencePanelError title="Review evidence could not be loaded" message={state.reviewsError} /> : reviews?.evidence.length ? (
              <div className="review-evidence-list">
                {reviews.evidence.map((evidence, index) => {
                  const evidenceConfidence = confidencePresentation(evidence.confidence);
                  return (
                    <article key={`${evidence.aspect}-${index}`}>
                      <div><span className={`sentiment-chip sentiment-chip--${evidence.sentiment}`}>{evidence.sentiment}</span><strong>{humanizeAttributeKey(evidence.aspect)}</strong></div>
                      <p>{evidence.evidence_text}</p>
                      <footer><span>{evidenceConfidence.label}</span><span>{formatEvidenceTimestamp(evidence.published_at)}</span>{evidence.source_url && <a href={evidence.source_url} target="_blank" rel="noreferrer" aria-label={`Open ${humanizeAttributeKey(evidence.aspect)} review evidence source`}>Evidence source ↗</a>}</footer>
                    </article>
                  );
                })}
              </div>
            ) : (
              <div className="empty-evidence"><strong>No reviews for this part</strong><p>We only summarise reviews we can cite.</p></div>
            )}
          </section>
        </div>

        <aside className="product-evidence-rail" aria-label="Product evidence summary">
          {(product.source_attributions?.length ?? 0) > 0 && (
            <section className="evidence-rail-card evidence-rail-card--attribution" aria-labelledby="source-attribution-heading">
              <p className="profile-kicker">Source and licence</p>
              <h2 id="source-attribution-heading">Attribution</h2>
              <ul>
                {product.source_attributions?.map((attribution) => (
                  <li key={`${attribution.source_name}-${attribution.source_url}`}>
                    <a href={attribution.source_url} target="_blank" rel="noreferrer">
                      {attribution.source_name}
                    </a>
                    <p>{attribution.attribution_notice ?? attribution.licence_or_access_note}</p>
                    {attribution.licence_url && (
                      <a
                        className="source-attribution__licence"
                        href={attribution.licence_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        View licence
                      </a>
                    )}
                    <small>Retrieved {formatEvidenceTimestamp(attribution.retrieved_at)}</small>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <div className="compatibility-context-card">
            <span aria-hidden="true">◇</span>
            <p className="profile-kicker">Depends on the build</p>
            <h2>Whether this fits depends on your other parts.</h2>
            <p>A socket, connector, dimension, or interface only becomes compatible relative to the other selected parts. Generating a build runs the full compatibility check.</p>
            <Link className="button button--primary" href="/">Use in a complete build</Link>
          </div>

          {product.manufacturer_part_number && (
            <p className="record-mpn">
              <span>Manufacturer part number</span>
              <code>{product.manufacturer_part_number}</code>
            </p>
          )}

          {product.source_url && <a className="source-record-link" href={product.source_url} target="_blank" rel="noreferrer">Open primary product source ↗</a>}
        </aside>
      </div>
    </main>
  );
}
