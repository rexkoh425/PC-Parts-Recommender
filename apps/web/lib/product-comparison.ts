import type { ComponentKind, ProductDetail } from "./types";

export const maximumComparedProducts = 3;

const safeProductId = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$/;

/**
 * Parses the compact, shareable `products` query parameter used by /compare.
 * Invalid, duplicate, and surplus values are ignored before they reach the API.
 */
export function parseComparedProductIds(value?: string | null): string[] {
  if (!value) return [];
  const ids: string[] = [];
  for (const candidate of value.split(",")) {
    const productId = candidate.trim();
    if (!safeProductId.test(productId) || ids.includes(productId)) continue;
    ids.push(productId);
    if (ids.length === maximumComparedProducts) break;
  }
  return ids;
}

export function comparisonHref(productIds: string[]): string {
  const ids = parseComparedProductIds(productIds.join(","));
  if (!ids.length) return "/compare";
  const params = new URLSearchParams({ products: ids.join(",") });
  return `/compare?${params.toString()}`;
}

export function comparedProductsShareCategory(products: ProductDetail[]): boolean {
  return products.length < 2 || products.every((product) => product.category === products[0]?.category);
}

export function comparisonCategory(products: ProductDetail[]): ComponentKind | undefined {
  return comparedProductsShareCategory(products) ? products[0]?.category : undefined;
}

/**
 * Shows only fields that at least one compared product explicitly reports.
 * The sorted list is deterministic so linked comparisons do not shuffle rows.
 */
export function comparedAttributeKeys(products: ProductDetail[]): string[] {
  return [...new Set(products.flatMap((product) => Object.keys(product.attributes)))].sort((left, right) =>
    left.localeCompare(right),
  );
}
