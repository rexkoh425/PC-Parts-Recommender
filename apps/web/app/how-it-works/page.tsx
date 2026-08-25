import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How it works",
  description:
    "How BuildSignal picks parts, checks they fit, and where its prices and figures come from.",
  alternates: { canonical: "/how-it-works" },
};

const faqs = [
  {
    q: "Are these prices live?",
    a: "No. Every price was recorded in August 2026 and converted from USD at 1.2775 SGD. They are real published figures, not a live retailer feed, so treat them as a guide rather than a quote.",
  },
  {
    q: "Why is the memory so expensive?",
    a: "Memory makers moved production capacity to HBM for AI datacentres through 2025 and 2026. DDR5 is up roughly four times on late-2025 prices and NAND storage roughly double. A 32 GB kit that cost under $90 now runs closer to $390, which is why RAM and storage can exceed the cost of the processor.",
  },
  {
    q: "Can I buy a build from here?",
    a: "No. Each part links to its manufacturer's own page so you can check the specification, then buy wherever you prefer. Nothing here is affiliate-linked or sponsored.",
  },
  {
    q: "How do you know the parts fit together?",
    a: "Socket, memory type, graphics card length, cooler height, case clearance and power draw are all checked between every pair of parts before a build can be ranked. A part that fails any check never appears in a result.",
  },
  {
    q: "What does the score out of 100 mean?",
    a: "It compares the builds in your shortlist against each other for the workload you gave. It is not a rating of the product in general: the same part scores differently under a different brief.",
  },
  {
    q: "Is this a real shop?",
    a: "It is a portfolio demo. The catalogue, compatibility rules and optimiser are real; the storefront is not, and nothing here is for sale.",
  },
];

export default function HowItWorksPage() {
  return (
    <main>
      <section className="shell page-intro" aria-labelledby="hiw-title">
        <p className="eyebrow">How it works</p>
        <h1 id="hiw-title">From a brief to a machine that fits together.</h1>
        <p className="lede">
          You give a budget and the work you do. Every part is searched, checked against every other
          part, scored for that workload, then assembled into complete builds.
        </p>
      </section>

      <section className="method-strip" aria-label="How recommendations are generated">
        <div className="shell method-strip__inner">
          <div><span>01</span><strong>Search</strong><small>25,666 parts, by name and by meaning</small></div>
          <div><span>02</span><strong>Check</strong><small>Every part against every other</small></div>
          <div><span>03</span><strong>Rank</strong><small>Scored against your workload</small></div>
          <div><span>04</span><strong>Optimise</strong><small>The best whole machine for the money</small></div>
        </div>
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
      <section className="shell faq-section" aria-labelledby="faq-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Questions</p>
            <h2 id="faq-title">Common questions</h2>
          </div>
        </div>
        <div className="faq-list">
          {faqs.map((item) => (
            <details key={item.q} className="faq-item">
              <summary>{item.q}</summary>
              <p>{item.a}</p>
            </details>
          ))}
        </div>
        <p className="faq-cta">
          <Link className="button button--primary" href="/">Build my PC</Link>
        </p>
      </section>
    </main>
  );
}
