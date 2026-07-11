// No deployment host is pinned in source. Vercel exposes the project's own
// production domain, and NEXT_PUBLIC_SITE_URL overrides it for a custom domain.
// The production fallback stays https so metadata origins can never downgrade.
const vercelProductionHost = process.env.NEXT_PUBLIC_VERCEL_PROJECT_PRODUCTION_URL;
const productionSiteUrl = vercelProductionHost
  ? `https://${vercelProductionHost}`
  : "https://localhost";
const localSiteUrl = "http://localhost:3000";

export function resolveSiteUrl(
  configuredUrl: string | undefined,
  environment = process.env.NODE_ENV,
): URL {
  const fallback = new URL(environment === "production" ? productionSiteUrl : localSiteUrl);
  if (!configuredUrl?.trim()) return fallback;

  try {
    const url = new URL(configuredUrl.trim());
    if (url.protocol !== "http:" && url.protocol !== "https:") return fallback;
    if (environment === "production" && url.protocol !== "https:") return fallback;
    url.pathname = "/";
    url.search = "";
    url.hash = "";
    return url;
  } catch {
    return fallback;
  }
}

export const siteUrl = resolveSiteUrl(process.env.NEXT_PUBLIC_SITE_URL);
