function configuredApiOrigin(): string | undefined {
  const configured = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (!configured) {
    return process.env.NODE_ENV === "production" ? undefined : "http://localhost:8000";
  }
  try {
    const url = new URL(configured);
    return url.protocol === "http:" || url.protocol === "https:" ? url.origin : undefined;
  } catch {
    return undefined;
  }
}

/**
 * The catalogue reads directly from Supabase over PostgREST, so its origin has
 * to be allowed explicitly - `connect-src 'self'` blocks it otherwise, and the
 * page silently falls back to the bundled fixture.
 *
 * Derived from the configured URL rather than hard-coded, so a different
 * project or a local stack does not require editing the policy. Only the
 * origin is taken, never a path, so a malformed value cannot widen the policy
 * beyond one host.
 */
function configuredSupabaseOrigin(): string | undefined {
  const configured = process.env.NEXT_PUBLIC_SUPABASE_URL?.trim();
  if (!configured) return undefined;
  try {
    const url = new URL(configured);
    return url.protocol === "https:" ? url.origin : undefined;
  } catch {
    return undefined;
  }
}

const apiOrigin = configuredApiOrigin();
const supabaseOrigin = configuredSupabaseOrigin();
const scriptSources = ["'self'", "'unsafe-inline'"];
if (process.env.NODE_ENV === "development") scriptSources.push("'unsafe-eval'");

export const contentSecurityPolicy = [
  "default-src 'self'",
  "base-uri 'self'",
  `connect-src ${["'self'", apiOrigin, supabaseOrigin].filter(Boolean).join(" ")}`,
  "font-src 'self' data:",
  "form-action 'self'",
  "frame-ancestors 'none'",
  "img-src 'self' blob: data:",
  "object-src 'none'",
  `script-src ${scriptSources.join(" ")}`,
  "style-src 'self' 'unsafe-inline'",
  "worker-src 'self' blob:",
  ...(process.env.NODE_ENV === "production" ? ["upgrade-insecure-requests"] : []),
].join("; ");

export const securityHeaders = [
  { key: "Content-Security-Policy", value: contentSecurityPolicy },
  {
    key: "Strict-Transport-Security",
    value: "max-age=63072000; includeSubDomains; preload",
  },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  {
    key: "Permissions-Policy",
    value: "camera=(), geolocation=(), microphone=(), payment=(), usb=()",
  },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
] as const;

export function withSecurityHeaders(response: Response): Response {
  const headers = new Headers(response.headers);
  for (const { key, value } of securityHeaders) headers.set(key, value);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
