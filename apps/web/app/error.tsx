"use client";

import { TriangleAlert } from "lucide-react";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset(): void }) {
  return (
    <main className="shell state-page">
      <div className="state-card" role="alert">
        <span className="state-card__icon" aria-hidden="true">
          <TriangleAlert size={24} strokeWidth={1.8} />
        </span>
        <p className="eyebrow">Something went wrong</p>
        <h1>This page needs another attempt.</h1>
        <p>Your saved builds and build brief are still here.</p>
        <button className="button button--primary" type="button" onClick={reset}>Try again</button>
      </div>
    </main>
  );
}
