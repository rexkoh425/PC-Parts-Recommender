import Link from "next/link";

export default function NotFound() {
  return (
    <main className="shell state-page">
      <div className="state-card">
        <span className="state-card__icon" aria-hidden="true">404</span>
        <p className="eyebrow">Page not found</p>
        <h1>This route is not in the current build.</h1>
        <p>Start a new recommendation or return to a build you saved in this browser.</p>
        <div className="button-row">
          <Link className="button button--primary" href="/">Start a build</Link>
          <Link className="button button--secondary" href="/catalogue">Browse catalogue</Link>
          <Link className="button button--secondary" href="/saved">Saved builds</Link>
        </div>
      </div>
    </main>
  );
}
