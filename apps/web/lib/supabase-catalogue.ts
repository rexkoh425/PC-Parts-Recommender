/**
 * Live catalogue retrieval from Supabase.
 *
 * Two RPCs, called over plain fetch rather than @supabase/supabase-js: the
 * whole surface is `search_products` and `browse_products`, and this project
 * has already had to fight the Vercel function size limit once. A dependency
 * for two POSTs is not worth the weight.
 *
 * The publishable key is public by design and ships in the browser. It reaches
 * nothing but these two security-definer functions and the read-only policies
 * behind them, which is why the tables have RLS enabled.
 */

import type { ComponentCategory, ProductSearchItem, ProductSearchRequest, ProductSearchResponse } from "./types";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL?.trim() ?? "";
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.trim() ?? "";

/** Live retrieval is only possible when both halves of the credential exist. */
export const SUPABASE_CATALOGUE_ENABLED = Boolean(supabaseUrl && supabaseKey);

/**
 * NEXT_PUBLIC_* values are inlined at build time, so a missing one is invisible
 * at runtime: the page just quietly serves the fixture and looks healthy. Say
 * which half is absent, once, so the cause is legible from the console instead
 * of requiring a bundle grep.
 */
if (typeof window !== "undefined" && !SUPABASE_CATALOGUE_ENABLED) {
  console.info(
    "[catalogue] live retrieval disabled at build time -" +
      ` url:${supabaseUrl ? "set" : "MISSING"} key:${supabaseKey ? "set" : "MISSING"}`,
  );
}

/** Shape returned by both RPCs; search adds the two ranking columns. */
interface RpcRow {
  product_id: string;
  canonical_name: string;
  brand: string | null;
  category: string;
  similarity?: number;
  keyword_rank?: number;
}

const DEFAULT_TIMEOUT_MS = 8000;

async function callRpc(
  name: string,
  body: Record<string, unknown>,
  options: { signal?: AbortSignal } = {},
): Promise<RpcRow[]> {
  if (!SUPABASE_CATALOGUE_ENABLED) {
    throw new Error("Supabase catalogue is not configured.");
  }

  // A slow database must not hang the page. The caller falls back to the
  // bundled fixture when this rejects, so a timeout degrades rather than fails.
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), DEFAULT_TIMEOUT_MS);
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeout.signal])
    : timeout.signal;

  try {
    const response = await fetch(`${supabaseUrl}/rest/v1/rpc/${name}`, {
      method: "POST",
      headers: {
        apikey: supabaseKey,
        Authorization: `Bearer ${supabaseKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
      // Deliberately not "include": the key travels in the header, and
      // credentialed requests would forbid the wildcard CORS Supabase sends.
      credentials: "omit",
    });

    if (!response.ok) {
      throw new Error(`Supabase ${name} responded ${response.status}`);
    }
    return (await response.json()) as RpcRow[];
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The catalogue's categories are a closed set in the UI. A row whose category
 * is not one of them would break rendering, so it is dropped rather than cast.
 */
const KNOWN_CATEGORIES = new Set<string>([
  "cpu",
  "gpu",
  "motherboard",
  "memory",
  "storage",
  "psu",
  "cooler",
  "case",
]);

/**
 * The catalogue names one category differently from the UI's type: it stores
 * `power_supply` where ComponentCategory says `psu`. Without this mapping the
 * filter below silently discarded 3,282 power supplies - 12.8% of the
 * catalogue - and the page simply never showed one.
 */
const CATEGORY_ALIASES: Record<string, ComponentCategory> = {
  power_supply: "psu",
};

function normaliseCategory(raw: string): ComponentCategory | null {
  const aliased = CATEGORY_ALIASES[raw] ?? raw;
  return KNOWN_CATEGORIES.has(aliased) ? (aliased as ComponentCategory) : null;
}

/**
 * The same mismatch in reverse. Filtering by "psu" from the UI has to ask the
 * database for "power_supply", or the query matches nothing at all.
 */
const DB_CATEGORIES: Record<string, string> = Object.fromEntries(
  Object.entries(CATEGORY_ALIASES).map(([dbValue, uiValue]) => [uiValue, dbValue]),
);

function toDatabaseCategory(category: string | null): string | null {
  if (!category) return null;
  return DB_CATEGORIES[category] ?? category;
}

function toSearchItem(row: RpcRow): ProductSearchItem | null {
  const category = normaliseCategory(row.category);
  if (!category) return null;
  return {
    product_id: row.product_id,
    category,
    canonical_name: row.canonical_name,
    brand: row.brand ?? undefined,
    // The live catalogue carries no prices yet: retailer feeds are a separate,
    // consented dataset. Null renders as "Price unavailable" rather than
    // inventing a figure.
    lowest_price_sgd: null,
    stock_status: null,
    compatibility_status: null,
  };
}

export async function searchSupabaseCatalogue(
  request: ProductSearchRequest,
  options: { signal?: AbortSignal } = {},
): Promise<ProductSearchResponse> {
  const pageSize = request.page_size ?? request.limit ?? 24;
  const page = request.page ?? 1;
  const query = request.query?.trim() ?? "";
  const category = request.category ?? null;
  const dbCategory = toDatabaseCategory(category);

  const rows = query
    ? await callRpc(
        "search_products",
        { query_text: query, match_count: pageSize, category_filter: dbCategory },
        options,
      )
    : await callRpc(
        "browse_products",
        { match_count: pageSize, page_offset: (page - 1) * pageSize, category_filter: dbCategory },
        options,
      );

  const products = rows
    .map(toSearchItem)
    .filter((item): item is ProductSearchItem => item !== null);

  return {
    query_id: `supabase:${query || "browse"}:${category ?? "all"}:${page}`,
    products,
    // Postgres would have to count the whole table to report a true total, so
    // report what this page actually holds rather than a guess.
    total: products.length,
    filtered_incompatible: 0,
    filtered_unknown: 0,
    data_version: "buildcores-full-25666-8c738c513661",
    retrieval_model: query ? "postgres-fts-hybrid" : "postgres-browse",
    pagination: {
      page,
      page_size: pageSize,
      total_pages: products.length < pageSize ? page : page + 1,
      has_previous: page > 1,
      has_next: products.length === pageSize,
    },
    coverage: {
      canonical_products: 25666,
      retailer_listings: null,
      source_count: null,
      category_count: KNOWN_CATEGORIES.size,
      as_of: null,
      scope_label: "Live catalogue",
      source_attributions: [],
    },
  };
}
