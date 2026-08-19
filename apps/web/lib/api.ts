import type {
  ApiErrorPayload,
  AdminOperationsResponse,
  BuildRequest,
  BuildShareCreated,
  BuildShareRevoked,
  BuildSummary,
  CompatibilityCheckRequest,
  CompatibilityCheckResponse,
  FreshnessSummary,
  GenerateBuildsResponse,
  InteractionAccepted,
  InteractionEvent,
  ProductBenchmarksResponse,
  ProductDetail,
  ProductPricesResponse,
  ProductReviewsResponse,
  ProductSearchRequest,
  ProductSearchResponse,
  PublicBuildShare,
  ReplacementRequest,
  ReplacementResponse,
} from "./types";
import {
  acceptDemoInteraction,
  checkDemoCompatibility,
  generateDemoBuilds,
  getDemoBenchmarks,
  getDemoFreshness,
  getDemoPrices,
  getDemoProduct,
  getDemoReviews,
  replaceDemoComponent,
  searchDemoProducts,
  demoProductIds,
} from "./demo-api";
import { apiBaseUrl, apiRequestTimeoutMs, usingDemoData } from "./runtime";
import { SUPABASE_CATALOGUE_ENABLED, searchSupabaseCatalogue } from "./supabase-catalogue";
import { readSavedBuilds } from "./saved-builds";

export const API_BASE_URL = apiBaseUrl;
export const USING_DEMO_DATA = usingDemoData;

export interface ApiRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

const requestCachePrefix = "pcbr:request:";
const buildCachePrefix = "pcbr:build:";
const sessionIdKey = "pcbr:session-id";
const productImpressionPrefix = "pcbr:product-impression-context:";
const productImpressionMaximumAgeMs = 10 * 60 * 1000;

export type ProductImpressionSurface = "catalogue_result" | "build_component";

interface StoredProductImpression {
  product_id: string;
  impression_token: string;
  stored_at: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly payload?: ApiErrorPayload,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function errorMessage(payload: ApiErrorPayload | undefined, status: number): string {
  if (payload?.error?.message) return payload.error.message;
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    const message = payload.detail.find((item) => item.msg)?.msg;
    if (message) return message;
  }
  if (payload?.detail && typeof payload.detail === "object" && "message" in payload.detail) {
    if (typeof payload.detail.message === "string" && payload.detail.message) {
      return payload.detail.message;
    }
  }
  if (payload?.message) return payload.message;
  return `The recommendation service returned ${status}.`;
}

export function apiErrorEvidence(error: unknown): string[] {
  if (!(error instanceof ApiError) || !error.payload) return [];
  const payload = error.payload;
  const detail =
    payload.detail && !Array.isArray(payload.detail) && typeof payload.detail === "object"
      ? payload.detail
      : undefined;
  const envelopeDetails =
    payload.error?.details && !Array.isArray(payload.error.details)
      ? payload.error.details
      : undefined;
  const messages = [
    ...(payload.reasons ?? []).map((reason) => reason.message),
    ...(payload.infeasibility?.reasons ?? []).map((reason) => reason.message),
    ...(payload.compatibility_checks ?? []).map((check) => check.message),
    ...(detail?.reasons ?? []).map((reason) => reason.message),
    ...(detail?.compatibility_checks ?? []).map((check) => check.message),
    ...(detail?.checks ?? []).map((check) => check.message),
    ...(detail?.evidence ?? []),
    ...(envelopeDetails?.reasons ?? []).map((reason) => reason.message),
    ...(envelopeDetails?.compatibility_checks ?? []).map((check) => check.message),
    ...(envelopeDetails?.checks ?? []).map((check) => check.message),
    ...(envelopeDetails?.evidence ?? []),
  ].filter((message): message is string => Boolean(message));
  return [...new Set(messages)];
}

export function apiErrorRequestId(error: unknown): string | undefined {
  if (!(error instanceof ApiError)) return undefined;
  return error.payload?.error?.request_id ?? undefined;
}

interface RequestAbortContext {
  signal: AbortSignal;
  didTimeout(): boolean;
  cleanup(): void;
}

function requestAbortContext(options: ApiRequestOptions): RequestAbortContext {
  const controller = new AbortController();
  const requestedTimeout = options.timeoutMs ?? apiRequestTimeoutMs;
  const timeoutMs = Number.isFinite(requestedTimeout)
    ? Math.max(1, Math.min(60_000, Math.round(requestedTimeout)))
    : apiRequestTimeoutMs;
  let timedOut = false;

  const abortFromCaller = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) {
    abortFromCaller();
  } else {
    options.signal?.addEventListener("abort", abortFromCaller, { once: true });
  }

  const timeout = controller.signal.aborted
    ? undefined
    : setTimeout(() => {
        timedOut = true;
        controller.abort(new DOMException("API request timed out", "TimeoutError"));
      }, timeoutMs);

  return {
    signal: controller.signal,
    didTimeout: () => timedOut,
    cleanup: () => {
      if (timeout !== undefined) clearTimeout(timeout);
      options.signal?.removeEventListener("abort", abortFromCaller);
    },
  };
}

async function apiRequest<T>(
  path: string,
  init?: RequestInit,
  options: ApiRequestOptions = {},
): Promise<T> {
  const abortContext = requestAbortContext(options);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      credentials: "include",
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
      signal: abortContext.signal,
    });

    if (!response.ok) {
      let payload: ApiErrorPayload | undefined;
      try {
        payload = (await response.json()) as ApiErrorPayload;
      } catch (error) {
        if (abortContext.signal.aborted) throw error;
        payload = undefined;
      }
      throw new ApiError(errorMessage(payload, response.status), response.status, payload);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (abortContext.didTimeout()) {
      throw new ApiError(
        "The recommendation service did not respond before the request deadline.",
        408,
      );
    }
    throw error;
  } finally {
    abortContext.cleanup();
  }
}

function cacheResponse(response: GenerateBuildsResponse): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem(`${requestCachePrefix}${response.request_id}`, JSON.stringify(response));
  for (const build of response.builds) {
    window.sessionStorage.setItem(
      `${buildCachePrefix}${build.build_id}`,
      JSON.stringify({ ...build, request_id: response.request_id }),
    );
  }
}

function readCachedRequest(requestId: string): GenerateBuildsResponse | undefined {
  if (typeof window === "undefined") return undefined;
  const value = window.sessionStorage.getItem(`${requestCachePrefix}${requestId}`);
  if (!value) return undefined;
  try {
    return JSON.parse(value) as GenerateBuildsResponse;
  } catch {
    return undefined;
  }
}

/** Returns the original structured brief only while its request cache remains in this browser session. */
export function readCachedBuildRequest(requestId: string): BuildRequest | undefined {
  const request = readCachedRequest(requestId)?.request;
  if (
    !request ||
    typeof request.budget_sgd !== "number" ||
    !Number.isFinite(request.budget_sgd) ||
    !Array.isArray(request.workloads) ||
    !Array.isArray(request.existing_products)
  ) {
    return undefined;
  }
  return request;
}

export function readCachedBuild(buildId: string): BuildSummary | undefined {
  if (typeof window === "undefined") return undefined;
  const value = window.sessionStorage.getItem(`${buildCachePrefix}${buildId}`);
  if (!value) return undefined;
  try {
    return JSON.parse(value) as BuildSummary;
  } catch {
    return undefined;
  }
}

function readSavedBuild(buildId: string): BuildSummary | undefined {
  return readSavedBuilds().find((entry) => entry.build.build_id === buildId)?.build;
}

export function productImpressionContext({
  surface,
  sourceId,
  productId,
  rankPosition,
}: {
  surface: ProductImpressionSurface;
  sourceId: string;
  productId: string;
  rankPosition?: number;
}): string {
  const context = `${surface}:${sourceId}:${rankPosition ?? 0}:${productId}`;
  return context.length <= 512 ? context : "";
}

export function rememberProductImpression(
  product: Pick<ProductSearchResponse["products"][number], "product_id" | "impression_token">,
  context: string,
): void {
  if (typeof window === "undefined" || !product.impression_token || !context) return;
  const value: StoredProductImpression = {
    product_id: product.product_id,
    impression_token: product.impression_token,
    stored_at: Date.now(),
  };
  window.sessionStorage.setItem(
    `${productImpressionPrefix}${context}`,
    JSON.stringify(value),
  );
}

export function readProductImpression(productId: string, context: string | null): string | undefined {
  if (typeof window === "undefined" || !context || context.length > 512) return undefined;
  const key = `${productImpressionPrefix}${context}`;
  const raw = window.sessionStorage.getItem(key);
  if (!raw) return undefined;
  try {
    const value = JSON.parse(raw) as Partial<StoredProductImpression>;
    if (
      value.product_id !== productId ||
      typeof value.impression_token !== "string" ||
      typeof value.stored_at !== "number" ||
      Date.now() - value.stored_at > productImpressionMaximumAgeMs
    ) {
      window.sessionStorage.removeItem(key);
      return undefined;
    }
    return value.impression_token;
  } catch {
    window.sessionStorage.removeItem(key);
    return undefined;
  }
}

export async function generateBuilds(
  request: BuildRequest,
  options: ApiRequestOptions = {},
): Promise<GenerateBuildsResponse> {
  const response = USING_DEMO_DATA
    ? generateDemoBuilds(request)
    : await apiRequest<GenerateBuildsResponse>(
        "/v1/builds/generate",
        {
          method: "POST",
          body: JSON.stringify(request),
        },
        options,
      );
  const enriched = { ...response, request: response.request ?? request };
  cacheResponse(enriched);
  return enriched;
}

export async function getRequestBuilds(
  requestId: string,
  options: ApiRequestOptions = {},
): Promise<GenerateBuildsResponse> {
  if (USING_DEMO_DATA) {
    const cached = readCachedRequest(requestId);
    if (cached) return cached;
    throw new ApiError("This demo recommendation is not available in this browser session.", 404);
  }
  const cached = readCachedRequest(requestId);
  try {
    const response = await apiRequest<GenerateBuildsResponse>(
      `/v1/requests/${encodeURIComponent(requestId)}/builds`,
      undefined,
      options,
    );
    const enriched = {
      ...response,
      request: response.request ?? cached?.request,
    };
    cacheResponse(enriched);
    return enriched;
  } catch (error) {
    if (options.signal?.aborted) throw error;
    if (cached) return cached;
    throw error;
  }
}

export async function getBuild(
  buildId: string,
  options: ApiRequestOptions = {},
): Promise<BuildSummary> {
  if (USING_DEMO_DATA) {
    const cached = readCachedBuild(buildId) ?? readSavedBuild(buildId);
    if (cached) return cached;
    throw new ApiError("This demo build is not available in this browser session.", 404);
  }
  try {
    const response = await apiRequest<BuildSummary | { build: BuildSummary }>(
      `/v1/builds/${encodeURIComponent(buildId)}`,
      undefined,
      options,
    );
    const build = "build" in response ? response.build : response;
    const cached = readCachedBuild(buildId);
    const enriched = {
      ...build,
      request_id: build.request_id ?? cached?.request_id,
      ...(!build.impression_token && cached?.impression_token
        ? {
            impression_token: cached.impression_token,
            components: build.components.map((component) => ({
              ...component,
              impression_token: cached.components.find(
                (cachedComponent) => cachedComponent.product_id === component.product_id,
              )?.impression_token,
            })),
          }
        : {}),
    };
    if (typeof window !== "undefined") {
      window.sessionStorage.setItem(
        `${buildCachePrefix}${enriched.build_id}`,
        JSON.stringify(enriched),
      );
    }
    return enriched;
  } catch (error) {
    if (options.signal?.aborted) throw error;
    const cached = readCachedBuild(buildId) ?? readSavedBuild(buildId);
    if (cached) return cached;
    throw error;
  }
}

export function createBuildShare(
  buildId: string,
  options: ApiRequestOptions = {},
): Promise<BuildShareCreated> {
  if (USING_DEMO_DATA) {
    return Promise.reject(
      new ApiError("Durable build sharing is not available in the public demo.", 503),
    );
  }
  return apiRequest<BuildShareCreated>(
    `/v1/builds/${encodeURIComponent(buildId)}/shares`,
    { method: "POST" },
    options,
  );
}

export function getBuildShare(
  shareId: string,
  options: ApiRequestOptions = {},
): Promise<PublicBuildShare> {
  if (USING_DEMO_DATA) {
    return Promise.reject(
      new ApiError("Durable build sharing is not available in the public demo.", 503),
    );
  }
  return apiRequest<PublicBuildShare>(
    `/v1/build-shares/${encodeURIComponent(shareId)}`,
    undefined,
    options,
  );
}

export function revokeBuildShare(
  shareId: string,
  revocationToken: string,
  options: ApiRequestOptions = {},
): Promise<BuildShareRevoked> {
  if (USING_DEMO_DATA) {
    return Promise.reject(
      new ApiError("Durable build sharing is not available in the public demo.", 503),
    );
  }
  return apiRequest<BuildShareRevoked>(
    `/v1/build-shares/${encodeURIComponent(shareId)}/revoke`,
    {
      method: "POST",
      body: JSON.stringify({ revocation_token: revocationToken }),
    },
    options,
  );
}

export async function searchProducts(
  request: ProductSearchRequest,
  options: ApiRequestOptions = {},
): Promise<ProductSearchResponse> {
  // Prefer the live catalogue when it is configured: 25,666 real parts rather
  // than the 21-part fixture. If it is slow or unreachable the fixture still
  // answers, so an outage degrades the result instead of breaking the page.
  if (SUPABASE_CATALOGUE_ENABLED) {
    try {
      return await searchSupabaseCatalogue(request, { signal: options.signal });
    } catch (error) {
      if (options.signal?.aborted) throw error;
      console.warn("[catalogue] live retrieval failed, serving the fixture instead.", error);
    }
  }
  if (USING_DEMO_DATA) return searchDemoProducts(request);
  return apiRequest<ProductSearchResponse>(
    "/v1/products/search",
    {
      method: "POST",
      body: JSON.stringify(request),
    },
    options,
  );
}

/**
 * Product IDs that can be prerendered at build time.
 *
 * Empty against the live API: the catalogue is not known at build time there,
 * so those routes stay server-rendered on demand.
 */
export function prerenderableProductIds(): string[] {
  return USING_DEMO_DATA ? demoProductIds : [];
}

export function getProduct(
  productId: string,
  options: ApiRequestOptions = {},
): Promise<ProductDetail> {
  // Defer controlled-demo lookup so an invalid shared product ID rejects the
  // promise just like an API 404 instead of escaping synchronously in React.
  if (USING_DEMO_DATA) return Promise.resolve().then(() => getDemoProduct(productId));
  return apiRequest<ProductDetail>(
    `/v1/products/${encodeURIComponent(productId)}`,
    undefined,
    options,
  );
}

export function getProductPrices(
  productId: string,
  options: ApiRequestOptions = {},
): Promise<ProductPricesResponse> {
  if (USING_DEMO_DATA) return Promise.resolve().then(() => getDemoPrices(productId));
  return apiRequest<ProductPricesResponse>(
    `/v1/products/${encodeURIComponent(productId)}/prices`,
    undefined,
    options,
  );
}

export function getProductBenchmarks(
  productId: string,
  options: ApiRequestOptions = {},
): Promise<ProductBenchmarksResponse> {
  if (USING_DEMO_DATA) return Promise.resolve().then(() => getDemoBenchmarks(productId));
  return apiRequest<ProductBenchmarksResponse>(
    `/v1/products/${encodeURIComponent(productId)}/benchmarks`,
    undefined,
    options,
  );
}

export function getProductReviews(
  productId: string,
  options: ApiRequestOptions = {},
): Promise<ProductReviewsResponse> {
  if (USING_DEMO_DATA) return Promise.resolve().then(() => getDemoReviews(productId));
  return apiRequest<ProductReviewsResponse>(
    `/v1/products/${encodeURIComponent(productId)}/reviews`,
    undefined,
    options,
  );
}

export function checkCompatibility(
  request: CompatibilityCheckRequest,
  options: ApiRequestOptions = {},
): Promise<CompatibilityCheckResponse> {
  if (USING_DEMO_DATA) return Promise.resolve(checkDemoCompatibility(request));
  return apiRequest<CompatibilityCheckResponse>(
    "/v1/compatibility/check",
    {
      method: "POST",
      body: JSON.stringify(request),
    },
    options,
  );
}

export async function replaceComponent(
  buildId: string,
  request: ReplacementRequest,
  options: ApiRequestOptions = {},
): Promise<ReplacementResponse> {
  const cachedBuild = readCachedBuild(buildId) ?? readSavedBuild(buildId);
  if (USING_DEMO_DATA && !cachedBuild) {
    throw new ApiError("This demo build is not available in this browser session.", 404);
  }
  const response = USING_DEMO_DATA
    ? replaceDemoComponent(cachedBuild as BuildSummary, request)
    : await apiRequest<ReplacementResponse>(
        `/v1/builds/${encodeURIComponent(buildId)}/replace`,
        {
          method: "POST",
          body: JSON.stringify(request),
        },
        options,
      );
  const enriched = {
    ...response,
    build: {
      ...response.build,
      request_id: response.build.request_id ?? cachedBuild?.request_id,
    },
  };
  if (typeof window !== "undefined") {
    window.sessionStorage.setItem(
      `${buildCachePrefix}${enriched.build.build_id}`,
      JSON.stringify(enriched.build),
    );
  }
  return enriched;
}

export function getFreshness(options: ApiRequestOptions = {}): Promise<FreshnessSummary> {
  if (USING_DEMO_DATA) return Promise.resolve(getDemoFreshness());
  return apiRequest<FreshnessSummary>("/v1/system/freshness", undefined, options);
}

export function getAdminOperations(
  adminToken: string,
  options: ApiRequestOptions = {},
): Promise<AdminOperationsResponse> {
  if (USING_DEMO_DATA) {
    return Promise.reject(
      new ApiError("The protected operations surface is not available in the public demo.", 503),
    );
  }
  return apiRequest<AdminOperationsResponse>(
    "/v1/admin/operations",
    { headers: { "X-PCBR-Admin-Token": adminToken } },
    options,
  );
}

export function getSessionId(): string {
  if (typeof window === "undefined") return "server";
  const current = window.sessionStorage.getItem(sessionIdKey);
  if (current) return current;
  const created = globalThis.crypto?.randomUUID?.() ?? `session-${Date.now()}`;
  window.sessionStorage.setItem(sessionIdKey, created);
  return created;
}

export async function trackInteraction(
  event: InteractionEvent,
  options: ApiRequestOptions = {},
): Promise<void> {
  if (USING_DEMO_DATA) {
    acceptDemoInteraction(event);
    return;
  }
  try {
    const idempotencyKey = event.impression_token
      ? await interactionIdempotencyKey(event)
      : undefined;
    await apiRequest<InteractionAccepted>(
      "/v1/interactions",
      {
        method: "POST",
        body: JSON.stringify(event),
        keepalive: true,
        headers: idempotencyKey ? { "Idempotency-Key": idempotencyKey } : undefined,
      },
      options,
    );
  } catch {
    // Analytics is deliberately non-blocking for the recommendation flow.
  }
}

async function interactionIdempotencyKey(event: InteractionEvent): Promise<string> {
  const canonical = `${event.session_id}\u0000${event.event_type}\u0000${event.impression_token ?? ""}`;
  const bytes = new TextEncoder().encode(canonical);
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
    const hex = Array.from(new Uint8Array(digest), (value) =>
      value.toString(16).padStart(2, "0"),
    ).join("");
    return `interaction-${hex}`;
  }
  let hash = 2166136261;
  for (const value of bytes) {
    hash ^= value;
    hash = Math.imul(hash, 16777619);
  }
  return `interaction-fallback-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}
