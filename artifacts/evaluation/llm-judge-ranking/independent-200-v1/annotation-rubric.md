# Independent held-out relevance annotation rubric

Annotate each candidate independently of its retrieval or reranking source. Do not
infer the hidden system from ordering. Use the displayed query and catalogue evidence.

- 4: Excellent match. Satisfies the explicit requirements and is a strong choice.
- 3: Good match. Satisfies the important requirements with only minor uncertainty.
- 2: Partially relevant. Useful but misses, weakens, or leaves an important requirement uncertain.
- 1: Marginal. Related to the category but not a meaningful answer to the request.
- 0: Irrelevant or contradicted by an explicit requirement.

Record evidence for every non-zero label. Mark unsupported or missing specifications as
unknown rather than assuming compatibility. Human labels, if collected, must retain
reviewer ID, timestamp, rubric version, and disagreement resolution notes.

This cohort is synthetic and the automated independence screen is not proof of human
semantic independence. LLM-only labels remain diagnostic and are not human evidence.
