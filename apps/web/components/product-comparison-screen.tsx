"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  getProduct,
  getSessionId,
  searchProducts,
  trackInteraction,
  USING_DEMO_DATA,
} from "@/lib/api";
import {
  formatAttributeValue,
  humanizeAttributeKey,
  observedStockLabel,
  stockTone,
} from "@/lib/catalogue";
import {
  comparedAttributeKeys,
  comparedProductsShareCategory,
  comparisonCategory,
  comparisonHref,
  maximumComparedProducts,
  parseComparedProductIds,
} from "@/lib/product-comparison";
import { categoryInlineLabels, categoryLabels, categoryPluralLabels, formatSgd } from "@/lib/format";
import type { ProductDetail, ProductSearchItem } from "@/lib/types";

interface ProductComparisonScreenProps {
  initialProductIds: string[];
}

function comparisonProductIds(products: ProductDetail[]): string[] {
  return products.map((product) => product.product_id);
}

function CandidateCard({
  product,
  selected,
  disabled,
  onSelect,
}: {
  product: ProductSearchItem;
  selected: boolean;
  disabled: boolean;
  onSelect: (productId: string) => void;
}) {
  const priceKnown = typeof product.lowest_price_sgd === "number";
  return (
    <article className="comparison-candidate">
      <div>
        <small>{product.brand ?? "Brand not reported"}</small>
        <h3>{product.canonical_name}</h3>
        {/*
          Every live catalogue record has a null price and a null stock status,
          so this line rendered "Price unavailable · Availability not reported"
          identically on all 24 candidates - a wall of text that said nothing
          and hid the one field that tells the parts apart.
        */}
        {priceKnown ? (
          <p>
            {formatSgd(product.lowest_price_sgd as number)}
            <span aria-hidden="true"> · </span>
            {observedStockLabel(product.stock_status)}
          </p>
        ) : (
          product.headline_spec && <p className="comparison-candidate__spec">{product.headline_spec}</p>
        )}
      </div>
      <button
        className="button button--secondary"
        type="button"
        disabled={disabled}
        onClick={() => onSelect(product.product_id)}
      >
        {selected ? "Added" : "Add"}
      </button>
    </article>
  );
}

export function ProductComparisonScreen({ initialProductIds }: ProductComparisonScreenProps) {
  const router = useRouter();
  const requestedProductIdsKey = parseComparedProductIds(initialProductIds.join(",")).join(",");
  const [products, setProducts] = useState<ProductDetail[]>([]);
  const [candidates, setCandidates] = useState<ProductSearchItem[]>([]);
  const [loading, setLoading] = useState(() => Boolean(requestedProductIdsKey));
  const [updating, setUpdating] = useState<string | null>(null);
  const [error, setError] = useState("");
  const trackedComparison = useRef<string | null>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const requestOptions = { signal: controller.signal };
    const requestedProductIds = parseComparedProductIds(requestedProductIdsKey);

    if (!requestedProductIds.length) {
      return () => controller.abort();
    }

    Promise.all(requestedProductIds.map((productId) => getProduct(productId, requestOptions)))
      .then(async (loadedProducts) => {
        if (!active) return;
        if (!comparedProductsShareCategory(loadedProducts)) {
          setProducts([]);
          setCandidates([]);
          setError("Only products from the same component category can be compared.");
          return;
        }
        setProducts(loadedProducts);
        const category = comparisonCategory(loadedProducts);
        if (!category) return;
        const response = await searchProducts({ query: "", category, limit: 24 }, requestOptions);
        if (active) setCandidates(response.products);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        setProducts([]);
        setCandidates([]);
        setError(
          requestError instanceof Error
            ? requestError.message
            : "The requested product evidence could not be loaded.",
        );
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [requestedProductIdsKey]);

  useEffect(() => {
    if (products.length < 2) return;
    const productIds = comparisonProductIds(products);
    const key = productIds.join(",");
    if (trackedComparison.current === key) return;
    trackedComparison.current = key;
    void trackInteraction({
      event_type: "comparison_opened",
      session_id: getSessionId(),
      metadata: {
        product_ids: productIds,
        category: products[0]?.category,
        surface: "product_comparison",
      },
    });
  }, [products]);

  function setComparedProducts(nextProducts: ProductDetail[]) {
    setProducts(nextProducts);
    router.replace(comparisonHref(comparisonProductIds(nextProducts)), { scroll: false });
  }

  async function addProduct(productId: string) {
    if (products.some((product) => product.product_id === productId) || products.length >= maximumComparedProducts) {
      return;
    }
    setUpdating(productId);
    try {
      const product = await getProduct(productId);
      const category = comparisonCategory(products);
      if (category && product.category !== category) {
        setError("Only products from the same component category can be compared.");
        return;
      }
      setError("");
      setComparedProducts([...products, product]);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "That product could not be added to the comparison.",
      );
    } finally {
      setUpdating(null);
    }
  }

  function removeProduct(productId: string) {
    setError("");
    setComparedProducts(products.filter((product) => product.product_id !== productId));
  }

  const category = comparisonCategory(products);
  const attributeKeys = comparedAttributeKeys(products);
  const selectedIds = new Set(comparisonProductIds(products));
  const canAddMore = products.length < maximumComparedProducts;

  if (loading) {
    return (
      <main className="shell comparison-page" aria-busy="true">
        <div className="skeleton skeleton--eyebrow" />
        <div className="skeleton skeleton--title" />
        <div className="skeleton-card skeleton-card--tall" />
        <p className="sr-only" role="status">Loading product comparison.</p>
      </main>
    );
  }

  if (!products.length) {
    return (
      <main className="shell comparison-page">
        <section className="catalogue-state comparison-empty" role={error ? "alert" : undefined}>
          <p className="eyebrow">Compare parts</p>
          <h1>{error ? "This comparison cannot be shown." : "Choose a component to compare."}</h1>
          <p>
            {error
              ? error
              : "Pick a part from the catalogue, then add up to two alternatives to see them side by side."}
          </p>
          <Link className="button button--primary" href="/catalogue">Browse the catalogue</Link>
        </section>
      </main>
    );
  }

  return (
    <main className="shell comparison-page">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link href="/catalogue">Catalogue</Link>
        <span aria-hidden="true">/</span>
        <span aria-current="page">Compare products</span>
      </nav>

      <header className="comparison-header">
        <div>
          <p className="eyebrow">Compare parts</p>
          <h1>Compare parts side by side.</h1>
          <p className="lede">
            {category ? categoryPluralLabels[category] : "Products"} are compared only on reported catalogue fields.
            Missing values remain visible as not reported.
          </p>
        </div>
        <div className="comparison-header__count">
          <strong>{products.length}/{maximumComparedProducts}</strong>
          <span>products selected</span>
        </div>
      </header>

      {error && (
        <div className="notice-banner" role="alert">
          <span aria-hidden="true">!</span>
          <p>{error}</p>
        </div>
      )}

      <section className="comparison-selected" aria-labelledby="comparison-selected-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Selected records</p>
            <h2 id="comparison-selected-title">{products.length < 2 ? "Add an alternative" : "Comparison set"}</h2>
          </div>
          <p>
            {USING_DEMO_DATA
              ? "Demo catalogue — prices from August 2026."
              : "Prices and availability were recorded earlier."}
          </p>
        </div>
        <div className="comparison-selected__cards">
          {products.map((product) => (
            <article key={product.product_id}>
              <div>
                <span className={`category-chip category-chip--${product.category}`}>
                  {categoryLabels[product.category]}
                </span>
                <p>{product.brand ?? "Brand not reported"}</p>
                <h3>{product.canonical_name}</h3>
                {typeof product.lowest_price_sgd === "number" ? (
                  <small>
                    {formatSgd(product.lowest_price_sgd)}
                    <span aria-hidden="true"> · </span>
                    <span className={`observed-stock observed-stock--${stockTone(product.stock_status)}`}>
                      <span aria-hidden="true" />
                      {observedStockLabel(product.stock_status)}
                    </span>
                  </small>
                ) : (
                  product.headline_spec && <small>{product.headline_spec}</small>
                )}
              </div>
              <div className="comparison-selected__actions">
                <Link className="button button--secondary" href={`/products/${encodeURIComponent(product.product_id)}`}>
                  Inspect evidence
                </Link>
                <button className="comparison-remove" type="button" onClick={() => removeProduct(product.product_id)}>
                  Remove
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>

      {products.length >= 2 && (
        <section className="comparison-table-section" aria-labelledby="comparison-table-title">
          <div className="section-heading">
            <div><p className="eyebrow">Reported specifications</p><h2 id="comparison-table-title">Field-by-field</h2></div>
            <p>Only fields present in at least one selected record are shown.</p>
          </div>
          <div className="comparison-table-scroll" tabIndex={0} aria-label="Scrollable product comparison table">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th scope="col">Field</th>
                  {products.map((product) => <th key={product.product_id} scope="col">{product.canonical_name}</th>)}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Observed price</th>
                  {products.map((product) => <td key={product.product_id}>{typeof product.lowest_price_sgd === "number" ? formatSgd(product.lowest_price_sgd) : "Price unavailable"}</td>)}
                </tr>
                <tr>
                  <th scope="row">Availability</th>
                  {products.map((product) => <td key={product.product_id}>{observedStockLabel(product.stock_status)}</td>)}
                </tr>
                <tr>
                  <th scope="row">Source confidence</th>
                  {products.map((product) => <td key={product.product_id}>{typeof product.source_confidence === "number" ? `${Math.round(product.source_confidence * 100)}%` : "Not reported"}</td>)}
                </tr>
                {attributeKeys.map((attributeKey) => (
                  <tr key={attributeKey}>
                    <th scope="row">{humanizeAttributeKey(attributeKey)}</th>
                    {products.map((product) => <td key={product.product_id}>{formatAttributeValue(product.attributes[attributeKey])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {canAddMore && (
        <section className="comparison-picker" aria-labelledby="comparison-picker-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Catalogue alternatives</p>
              <h2 id="comparison-picker-title">Add {products.length === 1 ? "one or two" : "one"} {category ? categoryInlineLabels[category] : "product"} alternative{products.length === 1 ? "s" : ""}</h2>
            </div>
            <p>Showing up to 24 results from the current category. Add at most three products total.</p>
          </div>
          {candidates.length ? (
            <div className="comparison-candidates">
              {candidates.map((candidate) => (
                <CandidateCard
                  key={candidate.product_id}
                  product={candidate}
                  selected={selectedIds.has(candidate.product_id)}
                  disabled={selectedIds.has(candidate.product_id) || Boolean(updating)}
                  onSelect={addProduct}
                />
              ))}
            </div>
          ) : (
            <div className="empty-evidence"><strong>No alternatives are available in this result window.</strong><p>Browse the category to choose another reported product.</p></div>
          )}
        </section>
      )}
    </main>
  );
}
