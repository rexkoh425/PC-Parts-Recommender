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
            Tell us your budget and what you actually do. You get complete machines whose parts
            are checked against each other, and you can see why each one was picked.
          </p>
          <div className="hero__actions">
            <a className="button button--primary button--large" href="#builder">
              Build my PC
              <span aria-hidden="true">↓</span>
            </a>
            <a className="button button--secondary button--large" href="/how-it-works">
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

    </main>
  );
}
