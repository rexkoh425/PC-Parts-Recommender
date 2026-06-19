# Canonical identity resolution contract

Production catalogue import is fail-closed when the canonical-identity preflight finds a
missing manufacturer part number or more than one product with the same normalized
brand/manufacturer-part-number key. Development inspection retains every source row and
reports those conflicts; it never applies a resolution artifact.

The v1 resolution artifact is immutable and release-bound. Its exact file SHA-256 must be
provided separately by the caller, while its payload binds all of the following:

- the exact source catalogue file SHA-256 and byte size;
- the canonical-identity preflight schema and content SHA-256;
- the complete deterministic conflict-set SHA-256, including every finding member;
- exactly one resolution for every missing-MPN and duplicate brand/MPN finding;
- two distinct reviewer identities and review IDs per finding, each with rationale and
  hashed HTTPS evidence;
- a distinct adjudicator, later adjudication timestamp, rationale, hashed evidence, and
  final explicit assignment for every finding member.

The loader rejects duplicate JSON keys, unknown fields, invalid self-hashes, stale source
bindings, missing or extra findings, incomplete member sets, non-independent reviewers,
cross-category aliases, and decisions that do not make the whole effective catalogue pass
the identity preflight. Missing brands and duplicate product IDs remain source-data errors
and cannot be waived by this artifact version.

Adjudicated assignments may either give distinct products verified distinct MPNs or map
true duplicate rows to an explicitly retained member. Application creates new Pydantic
product instances for MPN corrections and omits only explicitly adjudicated aliases; the
source JSONL and source product objects are never modified. The applied artifact hash,
source/effective preflight reports, MPN overrides, and aliases are returned as release
lineage and the artifact hash contributes to the imported data version.

No resolution decisions have been invented for the current 3,000-product catalogue. It
remains blocked until operators collect the required independent evidence and publish an
artifact whose source and conflict-set bindings match that exact catalogue.
