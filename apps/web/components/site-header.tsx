"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { USING_DEMO_DATA } from "@/lib/api";

interface NavItem {
  href: string;
  label: string;
  className?: string;
}

const navItems: readonly NavItem[] = [
  { href: "/how-it-works", label: "How it works", className: "header-nav__secondary header-nav__evidence" },
  { href: "/catalogue", label: "Catalogue" },
  { href: "/compare", label: "Compare", className: "header-nav__comparison" },
  { href: "/saved", label: "Saved" },
];

/**
 * A section is current when the path is the link or sits beneath it, so a
 * product record still marks Catalogue and a saved build still marks Saved.
 * Exact-match alone would leave the nav blank on most pages.
 */
function isCurrent(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function SiteHeader() {
  const pathname = usePathname() ?? "/";
  return (
    <header className="site-header">
      {USING_DEMO_DATA && (
        <div className="demo-banner" role="status">
          <div className="shell demo-banner__inner">
            <strong>Portfolio demo</strong>
            <span>Real parts, real prices from August 2026 — not live retailer stock.</span>
          </div>
        </div>
      )}
      <div className="site-header__inner shell">
        <Link className="brand" href="/" aria-label="PC Build Recommender home">
          <span className="brand__mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span>
            <strong>BuildSignal</strong>
            <small>PC Build Recommender</small>
          </span>
        </Link>

        <nav className="header-nav" aria-label="Primary navigation">
          {navItems.map((item) => {
            const current = isCurrent(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={[item.className, current ? "is-current" : ""].filter(Boolean).join(" ")}
                // aria-current carries this to a screen reader, which cannot
                // see the underline.
                aria-current={current ? "page" : undefined}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
