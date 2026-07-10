"use client";

import { useEffect, useId, useState } from "react";
import { searchProducts } from "@/lib/api";
import { categoryLabels } from "@/lib/format";
import type { ExistingProductInput, ProductSearchItem } from "@/lib/types";

interface ExistingProductPickerProps {
  selected: ExistingProductInput[];
  onChange(items: ExistingProductInput[]): void;
  disabled?: boolean;
}

export function ExistingProductPicker({
  selected,
  onChange,
  disabled,
}: ExistingProductPickerProps) {
  const listId = useId();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ProductSearchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const open = query.trim().length >= 2;

  useEffect(() => {
    if (!open) return;

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setLoading(true);
      setMessage("");
      searchProducts(
        { query: query.trim(), limit: 8, in_stock_only: false },
        { signal: controller.signal },
      )
        .then((response) => {
          if (controller.signal.aborted) return;
          const filtered = response.products.filter(
            (product) => !selected.some((item) => item.product_id === product.product_id),
          );
          setResults(filtered);
          setActiveIndex(0);
          setMessage(filtered.length ? "" : "No matching catalogue products found.");
        })
        .catch(() => {
          if (!controller.signal.aborted) {
            setResults([]);
            setMessage("Product search is temporarily unavailable.");
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false);
        });
    }, 280);

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [open, query, selected]);

  function addProduct(product: ProductSearchItem) {
    onChange([
      ...selected,
      {
        product_id: product.product_id,
        category: product.category,
        canonical_name: product.canonical_name,
        include_in_budget: false,
      },
    ]);
    setQuery("");
    setResults([]);
  }

  function updateQuery(value: string) {
    setQuery(value);
    if (value.trim().length < 2) {
      setResults([]);
      setMessage("");
      setLoading(false);
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (!results.length) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((current) => (current + 1) % results.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((current) => (current - 1 + results.length) % results.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      addProduct(results[activeIndex]);
    } else if (event.key === "Escape") {
      updateQuery("");
    }
  }

  return (
    <div className="product-picker">
      <label htmlFor={`${listId}-input`}>Existing components</label>
      <p className="field-help" id={`${listId}-help`}>
        Retained parts are locked into every build and excluded from the new-parts budget.
      </p>
      {selected.length > 0 && (
        <ul className="selected-products" aria-label="Selected existing components">
          {selected.map((product) => (
            <li key={product.product_id}>
              <span>
                <small>{categoryLabels[product.category]}</small>
                {product.canonical_name ?? product.product_id}
              </span>
              <button
                type="button"
                onClick={() => onChange(selected.filter((item) => item.product_id !== product.product_id))}
                aria-label={`Remove ${product.canonical_name ?? product.product_id}`}
                disabled={disabled}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="combobox-shell">
        <input
          id={`${listId}-input`}
          type="search"
          value={query}
          onChange={(event) => updateQuery(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search model or part number"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded={open}
          aria-controls={listId}
          aria-activedescendant={results.length ? `${listId}-${activeIndex}` : undefined}
          aria-describedby={`${listId}-help`}
          disabled={disabled}
        />
        {loading && <span className="combobox-shell__loading" aria-hidden="true" />}
        {open && (loading || results.length > 0 || message) && (
          <div className="combobox-popover">
            <ul id={listId} role="listbox">
              {results.map((product, index) => (
                <li
                  id={`${listId}-${index}`}
                  key={product.product_id}
                  role="option"
                  aria-selected={index === activeIndex}
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseMove={() => setActiveIndex(index)}
                  onClick={() => addProduct(product)}
                >
                  <span>
                    <small>{categoryLabels[product.category]}</small>
                    <strong>{product.canonical_name}</strong>
                  </span>
                  {typeof product.lowest_price_sgd === "number" && (
                    <span>S${product.lowest_price_sgd.toLocaleString("en-SG")}</span>
                  )}
                </li>
              ))}
            </ul>
            {(loading || message) && (
              <p className="combobox-message" role="status">
                {loading ? "Searching catalogue…" : message}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
