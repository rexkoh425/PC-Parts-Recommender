const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim();

export const DEFAULT_API_REQUEST_TIMEOUT_MS = 15_000;
const MINIMUM_API_REQUEST_TIMEOUT_MS = 1_000;
const MAXIMUM_API_REQUEST_TIMEOUT_MS = 60_000;

export function parseApiRequestTimeout(value: string | undefined): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_API_REQUEST_TIMEOUT_MS;
  return Math.max(
    MINIMUM_API_REQUEST_TIMEOUT_MS,
    Math.min(MAXIMUM_API_REQUEST_TIMEOUT_MS, Math.round(parsed)),
  );
}

/**
 * An empty base URL deliberately targets the current origin.  This is used by
 * the single-domain Vercel Services deployment, where the FastAPI service is
 * routed alongside the Next.js application at `/v1/*`.
 */
export function resolveApiBaseUrl(value: string | undefined): string {
  const configured = value?.trim();
  if (configured === "/") return "";
  return (configured || "http://localhost:8000").replace(/\/$/, "");
}

export const dataMode =
  process.env.NEXT_PUBLIC_DATA_MODE ??
  (configuredApiUrl ? "api" : process.env.NODE_ENV === "production" ? "demo" : "api");

export const usingDemoData = dataMode === "demo";

export interface RuntimeCapabilities {
  reoptimizeUnlockedReplacement: boolean;
}

export function runtimeCapabilitiesForDataMode(mode: string): RuntimeCapabilities {
  return {
    reoptimizeUnlockedReplacement: mode === "api",
  };
}

export const runtimeCapabilities = runtimeCapabilitiesForDataMode(dataMode);

export const apiBaseUrl = resolveApiBaseUrl(configuredApiUrl);

export const apiRequestTimeoutMs = parseApiRequestTimeout(
  process.env.NEXT_PUBLIC_API_TIMEOUT_MS,
);
