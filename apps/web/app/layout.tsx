import type { Metadata, Viewport } from "next";
import { Fraunces, Inter } from "next/font/google";
import type { ReactNode } from "react";
import { SiteHeader } from "@/components/site-header";
import { siteUrl } from "@/lib/site-url";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
});

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
      default: "BuildSignal · PC builds that fit together",
      template: "%s · BuildSignal",
    },
    description:
      "Tell it your budget and what you do. It picks parts that fit together, shows what each one costs, and says why it chose them.",
    applicationName: "BuildSignal",
    keywords: ["PC build recommender", "compatibility", "recommendation system", "Singapore"],
    openGraph: {
      title: "BuildSignal · Build for the work",
      description:
        "Tell it your budget and what you do. It picks parts that fit together and shows why.",
      type: "website",
      images: [{ url: socialImage, width: 1731, height: 909, alt: "BuildSignal - PC builds that fit together" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "BuildSignal · PC builds that fit together",
      description: "PC builds where every part fits, with the reasoning shown.",
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
              <strong>BuildSignal</strong> · complete PC builds, ranked for the work you do.
            </p>
            <p>A portfolio demo. Prices recorded August 2026.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
