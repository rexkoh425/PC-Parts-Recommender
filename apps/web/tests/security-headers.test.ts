import { describe, expect, it } from "vitest";
import nextConfig, {
  contentSecurityPolicy,
  securityHeaders,
} from "../next.config";
import { withSecurityHeaders } from "../lib/security-headers";

describe("web security headers", () => {
  it("applies the security policy to every application route", async () => {
    const rules = await nextConfig.headers?.();

    expect(rules).toEqual([
      {
        source: "/:path*",
        headers: [...securityHeaders],
      },
    ]);
  });

  it("sets the required browser security controls", () => {
    const headers = new Map(securityHeaders.map(({ key, value }) => [key, value]));

    expect(headers.get("Strict-Transport-Security")).toContain("max-age=63072000");
    expect(headers.get("Referrer-Policy")).toBe("strict-origin-when-cross-origin");
    expect(headers.get("Permissions-Policy")).toContain("camera=()");
    expect(headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(headers.get("X-Frame-Options")).toBe("DENY");
  });

  it("locks down active content and framing in CSP", () => {
    expect(contentSecurityPolicy).toContain("default-src 'self'");
    expect(contentSecurityPolicy).toContain("base-uri 'self'");
    expect(contentSecurityPolicy).toContain("frame-ancestors 'none'");
    expect(contentSecurityPolicy).toContain("object-src 'none'");
    expect(contentSecurityPolicy).toMatch(/connect-src 'self'(?: |;)/);
    expect(contentSecurityPolicy).not.toContain("default-src *");
    expect(contentSecurityPolicy).not.toContain("object-src *");
  });

  it("applies the same policy at the hosting worker boundary", async () => {
    const response = withSecurityHeaders(
      new Response("ok", { headers: { "Cache-Control": "public, max-age=60" } }),
    );

    expect(response.headers.get("Content-Security-Policy")).toBe(contentSecurityPolicy);
    expect(response.headers.get("Strict-Transport-Security")).toContain("max-age=63072000");
    expect(response.headers.get("Cache-Control")).toBe("public, max-age=60");
    await expect(response.text()).resolves.toBe("ok");
  });
});
