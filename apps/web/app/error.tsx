"use client";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset(): void }) {
  return (
    <main className="shell state-page">
      <div className="state-card" role="alert">
        <span className="state-card__icon" aria-hidden="true">!</span>
        <p className="eyebrow">Unexpected interface error</p>
        <h1>This page needs another attempt.</h1>
        <p>Your saved builds and build brief have not been cleared.</p>
        <button className="button button--primary" type="button" onClick={reset}>Try again</button>
      </div>
    </main>
  );
}
