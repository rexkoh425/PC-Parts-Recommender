# PC Build Recommender

A search and recommendation system for complete desktop PC builds in Singapore.

Status: early. Nothing here is production ready and no model has been trained.

## Goal

Given a budget and a workload (gaming, content creation, general use), return
complete builds that are actually compatible, with an explanation for every
component choice and a visible line back to the evidence behind it.

The hard part is not the ranking. It is that component data is messy, retailer
prices are rights-encumbered, and "compatible" is a claim that has to be
defensible rather than a vibe.

## Planned shape

```text
packages/core/src/   Domain models, rules, retrieval, ranking
pipelines/           Source adapters and parsing
tests/               Unit and property tests
docs/                Requirements and design
```

## Ground rules

- Every ingested record keeps its source provenance.
- No retailer price is displayed, cached, embedded, or trained on without
  explicit written rights.
- Observed evidence, model predictions, and targets never get mixed together in
  the same field.
- Synthetic fixtures are for smoke tests only and can never be promoted.

## Local setup

```powershell
uv sync
npm install
```

## Status of the evidence

Nothing measured yet.
