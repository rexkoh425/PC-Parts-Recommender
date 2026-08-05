import type { Metadata } from "next";
import { BuildForm } from "@/components/build-form";

export const metadata: Metadata = {
  alternates: { canonical: "/" },
};

export default function HomePage() {
  return (
    <main>
      <section className="hero shell" aria-labelledby="hero-title">
        <div className="hero__copy">
          <p className="eyebrow">
            <span aria-hidden="true">●</span>
            Built for Singapore prices
          </p>
          <h1 id="hero-title">
            Build for the work.
            <span>Not the hype.</span>
          </h1>
          <p className="hero__lede">
            Turn your budget, workload, and existing hardware into a ranked shortlist of complete,
            compatible PCs—with the evidence behind every choice.
          </p>
          <div className="hero__actions">
            <a className="button button--primary button--large" href="#builder">
              Build my PC
              <span aria-hidden="true">↓</span>
            </a>
            <a className="button button--secondary button--large" href="#evidence">
              See how it works
            </a>
          </div>
        </div>
        {/* Show the output, not a description of it. These are the real figures
            the demo returns for a S$3,500 gaming + local-AI brief. */}
        <aside className="hero__preview" aria-label="Example result">
          <div className="hero__preview-head">
            <span className="profile-kicker">Best overall</span>
            <span className="recommended-chip">Recommended</span>
          </div>
          <p className="hero__preview-price">$2,672</p>
          <p className="hero__preview-sub">88 / 100 fit · $828 under budget</p>
          <ul className="hero__preview-parts">
            <li><span>Processor</span><strong>AMD Ryzen 7 7700</strong></li>
            <li><span>Graphics</span><strong>GeForce RTX 5060 Ti 16 GB</strong></li>
            <li><span>Memory</span><strong>32 GB DDR5-6000 CL30</strong></li>
          </ul>
          <p className="hero__preview-foot">+ 5 more parts, all checked against each other</p>
        </aside>
      </section>

      <section className="method-strip" id="method" aria-label="How recommendations are generated">
        <div className="shell method-strip__inner">
          <div><span>01</span><strong>Search</strong><small>25,666 parts, by name and by meaning</small></div>
          <div><span>02</span><strong>Check</strong><small>Every part against every other</small></div>
          <div><span>03</span><strong>Rank</strong><small>Scored against your workload</small></div>
          <div><span>04</span><strong>Optimise</strong><small>The best whole machine for the money</small></div>
        </div>
      </section>

      <section className="builder-section shell" id="builder" aria-labelledby="builder-title">
        <div className="builder-section__heading">
          <p className="eyebrow">Start here</p>
          <h2 id="builder-title">Describe the system you need</h2>
          <p>
            Your must-haves come first. Preferences only break ties between builds that already fit.
          </p>
        </div>
        <BuildForm />
      </section>

      <section className="evidence-section shell" id="evidence" aria-labelledby="evidence-title">
        <div className="evidence-section__heading">
          <div>
            <p className="eyebrow">How it was built</p>
            <h2 id="evidence-title">What it runs on.</h2>
          </div>
          <p>
            Each figure comes from a file in the repository.
          </p>
        </div>
        <div className="evidence-stat-grid">
          <article>
            <strong>25,666</strong>
            <span>PC parts in the catalogue</span>
          </article>
          <article>
            <strong>9.8M</strong>
            <span>search vectors, generated on GPU</span>
          </article>
          <article>
            <strong>10,000</strong>
            <span>builds solved and independently re-checked</span>
          </article>
          <article>
            <strong>32</strong>
            <span>search-quality test queries</span>
          </article>
        </div>
        <details className="evidence-ledger">
          <summary>Where these numbers come from</summary>
          <ol>
            <li>
              <strong>Catalogue</strong>
              <code>data/processed/buildcores_open_db/&hellip;/manifest.json</code>
              <small>SHA-256 72fe9ef3&hellip;e5dd39</small>
            </li>
            <li>
              <strong>Search vectors</strong>
              <code>artifacts/retrieval/buildcores-full-embeddings-pinned/manifest.json</code>
              <small>SHA-256 4e9ccfc6&hellip;2225c0</small>
            </li>
            <li>
              <strong>Solved builds</strong>
              <code>artifacts/evaluation/optimizer-generated-builds-v1/&hellip;.json</code>
              <small>SHA-256 169c0387&hellip;e7e8e</small>
            </li>
            <li>
              <strong>Search quality</strong>
              <code>artifacts/evaluation/retrieval-silver-full-v2/metrics.json</code>
              <small>SHA-256 13ba6c7e&hellip;83214cf</small>
            </li>
          </ol>
        </details>
      </section>

      <section className="trust-section shell" id="guardrails" aria-labelledby="trust-title">
        <div>
          <p className="eyebrow">Why you can trust it</p>
          <h2 id="trust-title">Every recommendation shows its working.</h2>
        </div>
        <div className="trust-grid">
          <article>
            <span aria-hidden="true">◇</span>
            <h3>Nothing is ranked until it fits</h3>
            <p>Sockets, clearances, connectors and power draw are checked before a part can appear in a build at all.</p>
          </article>
          <article>
            <span aria-hidden="true">◌</span>
            <h3>We say where numbers come from</h3>
            <p>Measured results, estimates and gaps stay separate. Nothing is averaged into a single figure that hides what it is.</p>
          </article>
          <article>
            <span aria-hidden="true">↗</span>
            <h3>Prices carry a date</h3>
            <p>Every price says when it was recorded and where it came from, so you can tell a current figure from a stale one.</p>
          </article>
        </div>
      </section>
    </main>
  );
}
