"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  productImpressionContext,
  rememberProductImpression,
  searchProducts,
  USING_DEMO_DATA,
} from "@/lib/api";
import {
  formatEvidenceTimestamp,
  observedStockLabel,
  stockTone,
} from "@/lib/catalogue";
import { categoryLabels, formatSgd } from "@/lib/format";
import type {
  ComponentCategory,
  ProductSearchItem,
  ProductSearchResponse,
} from "@/lib/types";
import { StatusPill } from "./status-pill";

const categoryOptions = Object.entries(categoryLabels) as Array<[ComponentCategory, string]>;

interface CatalogueScreenProps {
  initialQuery: string;
  initialCategory?: ComponentCategory;
  initialBrand: string;
  initialInStockOnly: boolean;
  initialPage: number;
  initialPageSize: number;
  initialCursor?: string;
}

interface CatalogueLocation {
  query: string;
  category?: ComponentCategory;
  brand: string;
  inStockOnly: boolean;
  page?: number;
  pageSize: number;
  cursor?: string;
}

function catalogueHref(location: CatalogueLocation): string {
  const params = new URLSearchParams();
  if (location.query) params.set("q", location.query);
  if (location.category) params.set("category", location.category);
  if (location.brand) params.set("brand", location.brand);
  if (location.inStockOnly) params.set("in_stock", "1");
  if ((location.page ?? 1) > 1) params.set("page", String(location.page));
  if (location.pageSize !== 24) params.set("page_size", String(location.pageSize));
  if (location.cursor) params.set("cursor", location.cursor);
  return params.size ? `/catalogue?${params.toString()}` : "/catalogue";
}

function CatalogueSkeleton() {
  return (
    <div className="catalogue-grid" aria-busy="true" aria-label="Loading catalogue products">
      {[0, 1, 2, 3, 4, 5].map((index) => (
        <div className="skeleton-card catalogue-skeleton" key={index}>
          <div className="skeleton skeleton--line" />
          <div className="skeleton skeleton--price" />
          <div className="skeleton skeleton--block" />
        </div>
      ))}
      <p className="sr-only" role="status">Searching the catalogue.</p>
    </div>
  );
}

function ProductResultCard({
  product,
  queryId,
  rankPosition,
}: {
  product: ProductSearchItem;
  queryId: string;
  rankPosition: number;
}) {
  const priceKnown = typeof product.lowest_price_sgd === "number";
  const impressionContext = productImpressionContext({
    surface: "catalogue_result",
    sourceId: queryId,
    productId: product.product_id,
    rankPosition,
  });
  const productHref = `/products/${encodeURIComponent(product.product_id)}?impression=${encodeURIComponent(impressionContext)}`;
  return (
    <article className="catalogue-card">
      <div className="catalogue-card__topline">
        <span className={`category-chip category-chip--${product.category}`}>
          {categoryLabels[product.category]}
        </span>
        <span className={`observed-stock observed-stock--${stockTone(product.stock_status)}`}>
          <span aria-hidden="true" />
          {observedStockLabel(product.stock_status)}
        </span>
      </div>
      <div className="catalogue-card__identity">
        <small>{product.brand ?? "Brand not reported"}</small>
        <h2>{product.canonical_name}</h2>
        {product.model && <p>{product.model}</p>}
      </div>
      <div className="catalogue-card__price">
        <small>{USING_DEMO_DATA ? "Illustrative demo price" : "Lowest observed listing"}</small>
        <strong>{priceKnown ? formatSgd(product.lowest_price_sgd as number) : "Price unavailable"}</strong>
      </div>
      {product.compatibility_status && (
        <div className="catalogue-card__compatibility">
          <StatusPill
            status={product.compatibility_status}
            label={`Contextual check: ${product.compatibility_status}`}
          />
        </div>
      )}
      <p className="catalogue-card__boundary">
        {USING_DEMO_DATA
          ? "Illustrative demo record; no live retailer connection."
          : "Availability and price are stored observations, not live retailer guarantees."}
      </p>
      <Link
        className="button button--secondary"
        href={productHref}
        aria-label={`Inspect evidence for ${product.canonical_name}`}
        onClick={() => rememberProductImpression(product, impressionContext)}
      >
        Inspect evidence
        <span aria-hidden="true">→</span>
      </Link>
      <Link
        className="catalogue-card__compare"
        href={`/compare?products=${encodeURIComponent(product.product_id)}`}
        aria-label={`Compare ${product.canonical_name}`}
      >
        Compare this product
      </Link>
    </article>
  );
}

function CoverageSummary({ response }: { response: ProductSearchResponse }) {
  const coverage = response.coverage;
  if (!coverage) return null;
  const attributions = coverage.source_attributions ?? [];
  const metrics = [
    ["Canonical products", coverage.canonical_products],
    ["Retailer listings", coverage.retailer_listings],
    ["Permitted sources", coverage.source_count],
    ["Categories", coverage.category_count],
  ] as const;
  return (
    <section className="catalogue-coverage" aria-labelledby="catalogue-coverage-title">
      <div>
        <p className="eyebrow">Provider-reported coverage</p>
        <h2 id="catalogue-coverage-title">{coverage.scope_label}</h2>
        <p>
          {USING_DEMO_DATA
            ? "Counts for the parts included in this demo."
            : "Coverage is reported by the connected catalogue provider and is separate from the current search result count."}
        </p>
        {coverage.as_of && <small>Reported {formatEvidenceTimestamp(coverage.as_of)}</small>}
        {attributions.length > 0 && (
          <ul className="catalogue-attribution" aria-label="Catalogue source attribution">
            {attributions.map((attribution) => (
              <li key={`${attribution.source_name}-${attribution.source_url}`}>
                <a href={attribution.source_url} target="_blank" rel="noreferrer">
                  {attribution.source_name}
                </a>
                <span>{attribution.attribution_notice ?? attribution.licence_or_access_note}</span>
                {attribution.licence_url && (
                  <a
                    className="catalogue-attribution__licence"
                    href={attribution.licence_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    View licence
                  </a>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
      <dl>
        {metrics.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{typeof value === "number" ? value.toLocaleString("en-SG") : "Not reported"}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function CategoryFacets({
  response,
  location,
}: {
  response: ProductSearchResponse;
  location: CatalogueLocation;
}) {
  const facets = response.facets?.categories;
  if (!facets?.length) return null;
  const allCount = facets.reduce((sum, facet) => sum + facet.count, 0);
  return (
    <nav className="catalogue-facets" aria-label="Filter catalogue results by category">
      <div>
        <p className="eyebrow">Category facets</p>
        <small>Counts are supplied by the current catalogue provider.</small>
      </div>
      <div className="catalogue-facets__links">
        <Link
          className={!location.category ? "is-active" : undefined}
          aria-current={!location.category ? "page" : undefined}
          href={catalogueHref({ ...location, category: undefined, page: 1, cursor: undefined })}
        >
          All <span>{allCount}</span>
        </Link>
        {facets.map((facet) => (
          <Link
            key={facet.value}
            className={location.category === facet.value ? "is-active" : undefined}
            aria-current={location.category === facet.value ? "page" : undefined}
            href={catalogueHref({ ...location, category: facet.value, page: 1, cursor: undefined })}
          >
            {categoryLabels[facet.value]} <span>{facet.count}</span>
          </Link>
        ))}
      </div>
    </nav>
  );
}

function CataloguePagination({
  response,
  location,
}: {
  response: ProductSearchResponse;
  location: CatalogueLocation;
}) {
  const pagination = response.pagination;
  if (!pagination) {
    if (response.total <= response.products.length) return null;
    return (
      <p className="catalogue-window-note" role="note">
        Showing {response.products.length.toLocaleString("en-SG")} of {response.total.toLocaleString("en-SG")} matches.
        This provider did not return page metadata, so refine the filters to narrow the result window.
      </p>
    );
  }
  if (pagination.total_pages <= 1) return null;
  const previousPage = Math.max(1, pagination.page - 1);
  const nextPage = Math.min(pagination.total_pages, pagination.page + 1);
  return (
    <nav className="catalogue-pagination" aria-label="Catalogue result pages">
      {pagination.has_previous ? (
        <Link
          className="button button--secondary"
          rel="prev"
          href={catalogueHref({
            ...location,
            page: previousPage,
            cursor: pagination.previous_cursor ?? undefined,
          })}
        >
          ← Previous
        </Link>
      ) : <span aria-hidden="true" />}
      <p>
        Page <strong>{pagination.page.toLocaleString("en-SG")}</strong> of{" "}
        <strong>{pagination.total_pages.toLocaleString("en-SG")}</strong>
      </p>
      {pagination.has_next ? (
        <Link
          className="button button--secondary"
          rel="next"
          href={catalogueHref({
            ...location,
            page: nextPage,
            cursor: pagination.next_cursor ?? undefined,
          })}
        >
          Next →
        </Link>
      ) : <span aria-hidden="true" />}
    </nav>
  );
}

export function CatalogueScreen({
  initialQuery,
  initialCategory,
  initialBrand,
  initialInStockOnly,
  initialPage,
  initialPageSize,
  initialCursor,
}: CatalogueScreenProps) {
  const router = useRouter();
  const effectiveInStockOnly = !USING_DEMO_DATA && initialInStockOnly;
  const [response, setResponse] = useState<ProductSearchResponse | null>(null);
  const [error, setError] = useState("");
  const [retryKey, setRetryKey] = useState(0);
  const location: CatalogueLocation = {
    query: initialQuery,
    category: initialCategory,
    brand: initialBrand,
    inStockOnly: effectiveInStockOnly,
    page: initialPage,
    pageSize: initialPageSize,
    cursor: initialCursor,
  };

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    searchProducts({
      query: initialQuery,
      category: initialCategory,
      brand: initialBrand || undefined,
      in_stock_only: effectiveInStockOnly,
      limit: initialPageSize,
      ...(initialPage > 1 ? { page: initialPage, page_size: initialPageSize } : {}),
      ...(initialCursor ? { cursor: initialCursor } : {}),
    }, { signal: controller.signal })
      .then((result) => {
        if (active) setResponse(result);
      })
      .catch((requestError) => {
        if (active) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "The catalogue could not be searched.",
          );
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [
    initialBrand,
    initialCategory,
    initialCursor,
    effectiveInStockOnly,
    initialPage,
    initialPageSize,
    initialQuery,
    retryKey,
  ]);

  function submitSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const query = String(form.get("query") ?? "").trim();
    const category = String(form.get("category") ?? "") as ComponentCategory | "";
    const brand = String(form.get("brand") ?? "").trim();
    const pageSize = Number(form.get("page_size")) || 24;
    const inStockOnly = form.get("in_stock_only") === "on";
    router.push(catalogueHref({
      query,
      category: category || undefined,
      brand,
      inStockOnly,
      page: 1,
      pageSize,
    }));
  }

  const loadedStart = response?.products.length
    ? ((response.pagination?.page ?? initialPage) - 1) * (response.pagination?.page_size ?? initialPageSize) + 1
    : 0;
  const loadedEnd = response ? loadedStart + response.products.length - (response.products.length ? 1 : 0) : 0;

  return (
    <main className="shell catalogue-page">
      <header className="catalogue-header">
        <div>
          <p className="eyebrow">Canonical product catalogue</p>
          <h1>Inspect the market evidence.</h1>
          <p className="lede">
            Search canonical components across retailer observations, then inspect specifications,
            price history, benchmarks, and review evidence before using them in a build.
          </p>
        </div>
        <div className="catalogue-header__mark" aria-hidden="true">
          <span>{response?.coverage ? "COVERAGE" : "MATCHES"}</span>
          <strong>{response?.coverage?.canonical_products ?? response?.total ?? "—"}</strong>
          <small>{response?.coverage?.scope_label ?? "matching products"}</small>
        </div>
      </header>

      <form className="catalogue-search" role="search" onSubmit={submitSearch}>
        <div className="catalogue-search__query">
          <label htmlFor="catalogue-query">Product, model, or requirement</label>
          <input
            id="catalogue-query"
            name="query"
            type="search"
            defaultValue={initialQuery}
            placeholder="e.g. 16 GB GPU, AM5 motherboard, quiet case"
          />
        </div>
        <div>
          <label htmlFor="catalogue-category">Category</label>
          <select id="catalogue-category" name="category" defaultValue={initialCategory ?? ""}>
            <option value="">All categories</option>
            {categoryOptions.map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="catalogue-brand">Brand</label>
          <input
            id="catalogue-brand"
            name="brand"
            type="search"
            defaultValue={initialBrand}
            list="catalogue-brands"
            placeholder="Any brand"
          />
          <datalist id="catalogue-brands">
            {response?.facets?.brands?.map((facet) => (
              <option key={facet.value} value={facet.value}>{facet.count} products</option>
            ))}
          </datalist>
        </div>
        <div>
          <label htmlFor="catalogue-page-size">Results per page</label>
          <select id="catalogue-page-size" name="page_size" defaultValue={initialPageSize}>
            {[12, 24, 48].map((size) => <option key={size} value={size}>{size}</option>)}
          </select>
        </div>
        <label className="catalogue-stock-filter" htmlFor="catalogue-in-stock">
          <input
            id="catalogue-in-stock"
            name="in_stock_only"
            type="checkbox"
            defaultChecked={effectiveInStockOnly}
            disabled={USING_DEMO_DATA}
          />
          <span>
            <strong>{USING_DEMO_DATA ? "Stock filter unavailable in demo" : "Observed in stock only"}</strong>
            <small>{USING_DEMO_DATA ? "No retailer stock feed is connected" : "Latest stored snapshot; never a live guarantee"}</small>
          </span>
        </label>
        <button className="button button--primary" type="submit">Search catalogue</button>
      </form>

      <div className="public-demo-boundary" role="note">
        <span aria-hidden="true">◎</span>
        <p>
          <strong>{USING_DEMO_DATA ? "Public portfolio demo." : "Evidence boundary."}</strong>{" "}
          {USING_DEMO_DATA
            ? "A fixed set of real parts with prices from August 2026. Not connected to live retailer stock."
            : "Prices and stock are recorded observations, not a live checkout quote."}
        </p>
      </div>

      {error ? (
        <section className="catalogue-state" role="alert">
          <p className="eyebrow">Catalogue connection</p>
          <h2>Search is temporarily unavailable.</h2>
          <p>{error}</p>
          <button
            className="button button--primary"
            type="button"
            onClick={() => {
              setError("");
              setResponse(null);
              setRetryKey((key) => key + 1);
            }}
          >Try again</button>
        </section>
      ) : !response ? (
        <CatalogueSkeleton />
      ) : (
        <>
          <CoverageSummary response={response} />
          <CategoryFacets response={response} location={location} />
          <section className="catalogue-results-heading" aria-live="polite">
            <div>
              <p className="eyebrow">Search result</p>
              <h2>
                {response.total.toLocaleString("en-SG")} product{response.total === 1 ? "" : "s"}
                {initialQuery ? ` for “${initialQuery}”` : " in the current catalogue"}
              </h2>
              <p className="catalogue-result-range">
                {response.products.length
                  ? `Showing ${loadedStart.toLocaleString("en-SG")}–${loadedEnd.toLocaleString("en-SG")}`
                  : "No records loaded for this page"}
              </p>
            </div>
            <dl>
              <div><dt>Data snapshot</dt><dd>{response.data_version}</dd></div>
              <div><dt>Retrieval</dt><dd>{response.retrieval_model}</dd></div>
              <div><dt>Filtered incompatible</dt><dd>{response.filtered_incompatible}</dd></div>
              <div><dt>Missing compatibility data</dt><dd>{response.filtered_unknown}</dd></div>
            </dl>
          </section>

          {response.products.length ? (
            <section className="catalogue-grid" aria-label="Catalogue search results">
              {response.products.map((product, index) => (
                <ProductResultCard
                  key={product.product_id}
                  product={product}
                  queryId={response.query_id}
                  rankPosition={index + 1}
                />
              ))}
            </section>
          ) : (
            <section className="catalogue-state catalogue-state--empty">
              <p className="eyebrow">No matching evidence</p>
              <h2>No products matched this search.</h2>
              <p>Try removing a category filter, searching a model family, or using fewer terms.</p>
              <Link className="button button--secondary" href="/catalogue">Clear filters</Link>
            </section>
          )}
          <CataloguePagination response={response} location={location} />
        </>
      )}
    </main>
  );
}
