import type { Metadata } from "next";
import { BuildForm } from "@/components/build-form";
import { CatalogueReadinessPanel } from "@/components/catalogue-readiness-panel";

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
            Singapore-focused recommendation system
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
              Try the interactive demo
              <span aria-hidden="true">↓</span>
            </a>
            <a className="button button--secondary button--large" href="#evidence">
              Inspect the evidence
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
          <div><span>01</span><strong>Retrieve</strong><small>Keyword + semantic search</small></div>
          <div><span>02</span><strong>Verify</strong><small>Versioned compatibility rules</small></div>
          <div><span>03</span><strong>Rank</strong><small>Workload and value model</small></div>
          <div><span>04</span><strong>Optimise</strong><small>Complete-build constraint solver</small></div>
        </div>
      </section>

      <section className="builder-section shell" id="builder" aria-labelledby="builder-title">
        <div className="builder-section__heading">
          <p className="eyebrow">Your requirements are authoritative</p>
          <h2 id="builder-title">Describe the system you need</h2>
          <p>
            Set hard limits first. Preferences shape the ranking only after every compatible candidate
            has passed.
          </p>
        </div>
        <BuildForm />
      </section>

      <section className="evidence-section shell" id="evidence" aria-labelledby="evidence-title">
        <div className="evidence-section__heading">
          <div>
            <p className="eyebrow">How it was built</p>
            <h2 id="evidence-title">Real scale, not a toy project.</h2>
          </div>
          <p>
            Every number below comes from a build artifact you can open in the repository.
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

      <CatalogueReadinessPanel />

      <section className="trust-section shell" id="guardrails" aria-labelledby="trust-title">
        <div>
          <p className="eyebrow">Built for scrutiny</p>
          <h2 id="trust-title">You can inspect every recommendation.</h2>
        </div>
        <div className="trust-grid">
          <article>
            <span aria-hidden="true">◇</span>
            <h3>Compatibility before scores</h3>
            <p>A ranked part cannot enter a build until sockets, dimensions, connectors, and power pass.</p>
          </article>
          <article>
            <span aria-hidden="true">◌</span>
            <h3>Evidence stays labelled</h3>
            <p>Direct measurements, model estimates, and missing evidence never blur into one number.</p>
          </article>
          <article>
            <span aria-hidden="true">↗</span>
            <h3>Market context contract</h3>
            <p>Rights-cleared releases carry seller, stock, price freshness, and value context; demo values stay labelled.</p>
          </article>
        </div>
      </section>
    </main>
  );
}
