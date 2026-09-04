import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Page not found",
  robots: { index: false, follow: true },
};

export default function NotFound() {
  return (
    <main className="shell state-page">
      <div className="state-card">
        <span className="state-card__icon" aria-hidden="true">404</span>
        <p className="eyebrow">Page not found</p>
        {}
        <h1>We could not find that page.</h1>
        <p>It may have moved, or the link may be wrong. Here is the way back.</p>
        <div className="button-row">
          <Link className="button button--primary" href="/">Build a PC</Link>
          <Link className="button button--secondary" href="/catalogue">Browse parts</Link>
          <Link className="button button--secondary" href="/saved">Saved builds</Link>
        </div>
      </div>
    </main>
  );
}
