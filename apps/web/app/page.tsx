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
        <div className="hero__method" aria-label="Recommendation method">
          <span className="hero__method-number">03–05</span>
          <p>
            <strong>Distinct complete builds</strong>
            The public experience uses prevalidated sample builds; the full local system runs retrieval,
            versioned rules, ranking, and constraint optimisation.
          </p>
        </div>
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
            <p className="eyebrow">Measured project evidence</p>
            <h2 id="evidence-title">Engineering proof, with the boundaries left visible.</h2>
          </div>
          <p>
            These are reproducible development artifacts. Synthetic and silver-label results remain
            blocked from production claims until licensed retailer data and human judgements exist.
          </p>
        </div>
        <div className="evidence-stat-grid">
          <article><strong>25,666</strong><span>licensed catalogue records materialised</span></article>
          <article><strong>25,666 × 384</strong><span>revision-pinned CUDA catalogue embeddings</span></article>
          <article><strong>10,000</strong><span>retained CP-SAT outputs passed an independent oracle and compat_v2</span></article>
          <article><strong>32 queries</strong><span>silver retrieval diagnostic, not a human-labelled claim</span></article>
        </div>
        <div className="evidence-boundary">
          <span aria-hidden="true">i</span>
          <p>
            <strong>Current public limitation:</strong> verified live stock is not connected. Demo
            prices are illustrative, and arbitrary compatibility checks remain fail-closed.
          </p>
        </div>
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
