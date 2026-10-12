# Independent 200-query evaluation status

Status: not complete. The candidate cohort was frozen provisionally, but the
independence audit found semantic overlaps and the evaluation was not scored.

- Query count: 200
- Candidate tasks: 5,600 (28 candidates per query)
- Frozen seed: 20261110
- Automated similarity threshold: cosine similarity 0.80
- Cross-split matches: 14
- Matches against the previous fresh-v1 query set: 7
- Maximum observed previous-query similarity: 0.84655

Because the frozen cohort failed its independence gate, no new relevance labels,
ranking metrics, bootstrap intervals, or promotion decision are reported. The
existing 53-query sealed result remains unchanged. A human annotation package is
not yet valid until the overlapping queries are replaced and the audit passes.

The proposal source is synthetic model-authored text. No independent human labels
were obtained. Any future LLM judging must be blind to method identity and must be
reported as diagnostic only.
