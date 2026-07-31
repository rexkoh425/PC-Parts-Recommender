# Seeds

Seed data is generated through the versioned ingestion pipeline rather than embedded in the
schema migration. This keeps provenance, parser versions, and raw-content hashes mandatory.
Synthetic fixtures are permitted for tests and demonstrations, but evaluation manifests must
mark them synthetic and exclude them from externally reported model metrics.
