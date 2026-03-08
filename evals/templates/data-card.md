# Dataset card: <dataset name>

## Summary

- Dataset version:
- Intended task:
- Owner:
- Created at:
- Data snapshot range:
- Manifest SHA-256:
- Licence or access notes:

## Composition

| Item | Count |
| --- | ---: |
| Rows | |
| Leakage groups | |
| Train groups / rows | |
| Validation groups / rows | |
| Test groups / rows | |
| Synthetic rows | |
| Synthetic rows excluded from metrics | |

Describe category, source, label, product-family, workload, budget, and time distributions.

## Collection and provenance

List source URLs or source IDs, retrieval times, raw content hashes, parser versions, licences,
and manual labelling procedure. Explain how duplicate or revised observations were handled.

## Synthetic-data declaration

- Are row-level synthetic flags complete?
- Why were synthetic rows created?
- Which commands consume them?
- Evidence that reported metrics exclude them:
- Claim eligibility: `eligible | blocked`

Synthetic data may validate code paths; it must not support model-quality or résumé claims.

## Label quality

- Label rubric version:
- Reviewer count:
- Double-labelled test coverage:
- Agreement statistic:
- Adjudication policy:
- Unresolved examples:

## Leakage controls

State the leakage unit and prove that no unit crosses splits. Document point-in-time feature
cutoffs, train-only normalisation, out-of-fold model features, and any leave-source or
leave-generation robustness split.

## Evaluation scope

List supported metrics, confidence-interval unit, baselines, negative controls, and slices.
For retrieval, state whether recall is corpus-complete or pooled. For entity resolution, state
whether precision was measured on a representative candidate stream or an enriched pair set.

## Known limitations and prohibited claims

Include missing categories, source bias, incomplete judgements, small test slices, uncertain
compatibility fields, and unsupported workloads. Name metrics that remain provisional.

## Change log

| Version | Change | Split impact | Owner |
| --- | --- | --- | --- |
| | | | |
