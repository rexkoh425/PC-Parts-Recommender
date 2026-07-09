// No deployment is pinned here. Set NEXT_PUBLIC_SITE_URL for a real origin;
// without it both environments fall back to the local development server.
const productionSiteUrl = "http://localhost:3000";
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
