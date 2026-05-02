import type {
  ApiErrorPayload,
  AdminOperationsResponse,
  BuildRequest,
  BuildShareCreated,
  BuildResult,
  CompatibilityCheckRequest,
  CompatibilityCheckResponse,
  FreshnessSummary,
  GenerateBuildsResponse,
  InteractionAccepted,
  InteractionRecord,
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
} from "./demo-api";
import { apiBaseUrl, apiRequestTimeoutMs, usingDemoData } from "./runtime";

export const API_BASE_URL = apiBaseUrl;
export const USING_DEMO_DATA = usingDemoData;

export interface ApiRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

const requestCachePrefix = "pcbr:request:";
const buildCachePrefix = "pcbr:build:";
const sessionIdKey = "pcbr:session-id";

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

export function readCachedBuild(buildId: string): BuildResult | undefined {
  if (typeof window === "undefined") return undefined;
  const value = window.sessionStorage.getItem(`${buildCachePrefix}${buildId}`);
  if (!value) return undefined;
  try {
    return JSON.parse(value) as BuildResult;
  } catch {
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
  try {
    const response = await apiRequest<GenerateBuildsResponse>(
      `/v1/requests/${encodeURIComponent(requestId)}/builds`,
      undefined,
      options,
    );
    cacheResponse(response);
    return response;
  } catch (error) {
    if (options.signal?.aborted) throw error;
    const cached = readCachedRequest(requestId);
    if (cached) return cached;
    throw error;
  }
}

export async function getBuild(
  buildId: string,
  options: ApiRequestOptions = {},
): Promise<BuildResult> {
  if (USING_DEMO_DATA) {
    const cached = readCachedBuild(buildId);
    if (cached) return cached;
    throw new ApiError("This demo build is not available in this browser session.", 404);
  }
  try {
    const response = await apiRequest<BuildResult | { build: BuildResult }>(
      `/v1/builds/${encodeURIComponent(buildId)}`,
      undefined,
      options,
    );
    const build = "build" in response ? response.build : response;
    if (typeof window !== "undefined") {
      window.sessionStorage.setItem(`${buildCachePrefix}${build.build_id}`, JSON.stringify(build));
    }
    return build;
  } catch (error) {
    if (options.signal?.aborted) throw error;
    const cached = readCachedBuild(buildId);
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

export async function searchProducts(
  request: ProductSearchRequest,
  options: ApiRequestOptions = {},
): Promise<ProductSearchResponse> {
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

// TODO: rest of this module still to come.
