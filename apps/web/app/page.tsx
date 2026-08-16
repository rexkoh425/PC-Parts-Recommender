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
          <article id="evidence-catalogue">
            <strong>25,666</strong>
            <span>licensed catalogue records materialised</span>
            <a href="#evidence-catalogue-source">Inspect manifest</a>
          </article>
          <article id="evidence-embeddings">
            <strong>25,666 × 384</strong>
            <span>revision-pinned CUDA catalogue embeddings</span>
            <a href="#evidence-embeddings-source">Inspect manifest</a>
          </article>
          <article id="evidence-optimizer">
            <strong>10,000</strong>
            <span>retained CP-SAT outputs passed an independent oracle and compat_v2</span>
            <a href="#evidence-optimizer-source">Inspect evaluation</a>
          </article>
          <article id="evidence-retrieval">
            <strong>32 queries</strong>
            <span>silver retrieval diagnostic, not a human-labelled claim</span>
            <a href="#evidence-retrieval-source">Inspect evaluation</a>
          </article>
        </div>
        <section className="evidence-ledger" aria-labelledby="evidence-ledger-title">
          <div>
            <p className="eyebrow">Immutable local evidence ledger</p>
            <h3 id="evidence-ledger-title">Every figure resolves to a versioned repository artifact.</h3>
            <p>
              Paths identify the checked-in project evidence. Full SHA-256 digests prevent a later
              file from silently standing in for the result described here.
            </p>
          </div>
          <ol>
            <li id="evidence-catalogue-source">
              <strong>Catalogue batch manifest</strong>
              <code>data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/full/manifest.json</code>
              <small>SHA-256 72fe9ef33e06452d795b14f13aa8742fdc0767b32ec25c008a4c683777e5dd39</small>
              <a href="#evidence-catalogue">Back to figure</a>
            </li>
            <li id="evidence-embeddings-source">
              <strong>Revision-pinned embedding manifest</strong>
              <code>artifacts/retrieval/buildcores-full-embeddings-pinned/manifest.json</code>
              <small>SHA-256 4e9ccfc65af09962fb3caf8a259681464255edaa8f6a5984f9ac9520f62225c0</small>
              <a href="#evidence-embeddings">Back to figure</a>
            </li>
            <li id="evidence-optimizer-source">
              <strong>Retained optimizer-output evaluation</strong>
              <code>artifacts/evaluation/optimizer-generated-builds-v1/optimizer-generated-seed-20260723-n-10000-12c1305bec5666d4.json</code>
              <small>SHA-256 169c03872943920c741589f11dd450d8b00732e6ff4620022a968dbbe26e7e8e</small>
              <a href="#evidence-optimizer">Back to figure</a>
            </li>
            <li id="evidence-retrieval-source">
              <strong>Silver retrieval diagnostic</strong>
              <code>artifacts/evaluation/retrieval-silver-full-v2/metrics.json</code>
              <small>SHA-256 13ba6c7e074157d6bb654c1b4be5d244dfb9b02f3004b69464183e214efb2caf · not promotion eligible</small>
              <a href="#evidence-retrieval">Back to figure</a>
            </li>
          </ol>
        </section>
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
