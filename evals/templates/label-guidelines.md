# Label guidelines: <dataset name>

## Decision supported

Describe the exact model or gate this dataset evaluates. Do not combine unrelated decisions.

## Version and scope

- Dataset version:
- Data snapshot time:
- Categories:
- Retailers or benchmark sources:
- Allowed source licences:
- Excluded sources:
- Split unit:
- Frozen test-set owner:

## Synthetic-data policy

Every row must contain `is_synthetic: true | false`. Synthetic examples are allowed only for
pipeline, schema, compatibility, and failure-path tests. They must be removed before reported
ML metrics are calculated. An artifact containing synthetic rows is not claim-eligible.

## Entity-resolution labels

- `1 — same SKU`: both records identify the same manufacturer model and variant.
- `0 — different SKU`: any material variant differs, including capacity, form factor, memory
  generation, suffix, connector layout, clocked edition, or bundle.
- `needs_adjudication`: evidence is insufficient or sources conflict. Do not coerce this state
  into a training label.

Record the evidence fields used. Exact numeric conflicts are hard negatives even when titles
are nearly identical. Test examples require two reviewers and adjudication.

## Retrieval and ranking labels

Judge a product within one `query_id + component_category` group using facts available at the
snapshot time. Reviewers must not see model scores.

- `0 — irrelevant or unsuitable`: wrong category, violates a hard requirement, or does not
  address the workload.
- `1 — weak fit`: technically possible but materially poor for budget or workload.
- `2 — acceptable`: satisfies requirements with ordinary value and performance.
- `3 — strong fit`: clearly good workload, budget, and preference fit.
- `4 — best fit`: among the strongest defensible choices in the judged candidate pool.

Record hard-constraint failures separately from subjective relevance. Grade every frozen-test
item twice; adjudicate differences of two or more grades. Track weighted reviewer agreement.

## Performance observations

Performance rows are measurements rather than subjective grades. Preserve benchmark name,
version, resolution, preset, operating system, driver, source, unit, direction, and observation
time. Do not normalise or merge incomparable configurations. Mark imputed values separately;
an imputed target is not an observed benchmark.

## Review procedure

1. Read the structured request and evidence available at the snapshot time.
2. Apply hard conflicts before subjective judgement.
3. Assign a label and one-sentence rationale.
4. Mark uncertainty rather than guessing.
5. Escalate source conflicts or possible catalogue corruption.

## Required row fields

`row_id`, `dataset_version`, `split_group_id`, `category`, `label`, `label_status`,
`reviewer_id`, `reviewed_at`, `rationale`, `source_ids`, `is_synthetic`.

Add task-specific IDs such as `listing_id`, `product_id`, `query_id`, `benchmark_id`, and
`query_intent_family_id`.

## Disagreement log

Document disagreement rate, adjudicator, recurring ambiguity, and every rubric revision.
