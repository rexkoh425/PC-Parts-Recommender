import type { Metadata, Viewport } from "next";
import { Fraunces, Inter } from "next/font/google";
import type { ReactNode } from "react";
import { SiteHeader } from "@/components/site-header";
import { siteUrl } from "@/lib/site-url";
import "./globals.css";

/*
 * The stylesheet named Inter but nothing ever loaded it, so the whole site had
 * been rendering in the system fallback — Segoe UI on Windows. Both faces are
 * self-hosted by next/font at build time, so no request leaves the page and
 * the CSP stays untouched.
 */
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
});

/*
 * A display face so headings have a voice of their own. Fraunces is a variable
 * serif with optical sizing; at low softness and wonk it reads as considered
 * rather than decorative, and gives the page the editorial weight the
 * all-Inter setting never had.
 */
const fraunces = Fraunces({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-display",
  axes: ["SOFT", "WONK", "opsz"],
});

export function generateMetadata(): Metadata {
  const metadataBase = siteUrl;
  const socialImage = new URL("/og.png", metadataBase).toString();
  return {
    metadataBase,
    title: {
      default: "BuildSignal — Evidence-backed PC builds",
      template: "%s · BuildSignal",
    },
    description:
      "An interactive portfolio demo for compatible PC builds ranked by workload, budget, existing hardware, and explicit evidence boundaries.",
    applicationName: "BuildSignal",
    keywords: ["PC build recommender", "compatibility", "recommendation system", "Singapore"],
    openGraph: {
      title: "BuildSignal — Build for the work, not the hype",
      description:
        "Explore an evidence-backed PC build recommendation system with compatibility-first constraints.",
      type: "website",
      images: [{ url: socialImage, width: 1731, height: 909, alt: "BuildSignal recommendation system" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "BuildSignal — Evidence-backed PC builds",
      description: "Compatibility-first PC build recommendations with visible evidence boundaries.",
      images: [socialImage],
    },
  };
}

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#f4f1e8",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en-SG" data-scroll-behavior="smooth">
      <body className={`${inter.variable} ${fraunces.variable}`}>
        <a className="skip-link" href="#main-content">
          Skip to main content
        </a>
        <SiteHeader />
        <div id="main-content">{children}</div>
        <footer className="site-footer">
          <div className="shell site-footer__inner">
            <p>
              <strong>BuildSignal</strong> — complete PC builds, ranked for the work you do.
            </p>
            <p>A portfolio demo. Prices recorded August 2026.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
