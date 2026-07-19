# Signed local Awin feed ingestion

`awin_feed` is a bounded local importer, not a downloader. An authorised operator must obtain
the Awin feed outside this application and provide the resulting regular local CSV or gzip file.
The CLI deliberately accepts no network URL or API credential. Awin download URLs can contain
API keys, so they must never be used as provenance, command arguments, policy text, CSV values,
logs, or checked-in configuration. Normalised provenance uses only
`awin://advertisers/<advertiser-id>/feeds/<feed-id>`; feed bytes and the
validated URL-bearing policy, receipt, manifest, and record fields are
rejected if credential-bearing URL material is detected. Operators must keep
credentials out of all remaining free-text policy fields as an operational
control.

## External prerequisites

Before a run, an operator needs all of the following from parties authorised to provide them:

- active Awin access and an already-downloaded feed whose acquisition and downstream uses are
  covered by the applicable publisher/advertiser agreement;
- an exact, current policy envelope signed by an approved Ed25519 key;
- the detached signature document and a trust-root document, with the trust-root SHA-256 pinned
  through an independent trusted channel rather than copied from the policy bundle;
- reviewed advertiser/feed IDs, merchant name, exact listing-link hosts, category mappings,
  territory, retention/deletion terms, and data-use grants; and
- operational retention and deletion controls for raw and processed artifacts.

No real feed, agreement, signature, trust root, public key, private key, or grant is bundled.

## Signature and policy contract

The signed envelope has exactly `schema_version`, `policy_id`, `issued_at`, `expires_at`, and
`payload`. The envelope schema is `pc-build-recommender.signed-policy.v1`. The detached signature
document has exactly `schema_version`, `key_id`, `policy_sha256`, and `signature`, uses schema
`pc-build-recommender.detached-policy-signature.v1`, and contains a canonical-base64 Ed25519
signature over the exact policy bytes. The trust root has schema
`pc-build-recommender.policy-trust-root.v1` and an exact `keys` array; each key declares
`key_id`, `algorithm` (`Ed25519`), `public_key`, `status`, `valid_from`, and `valid_until`.
Private signing keys must not be stored in this repository.

The Awin payload schema is `pc-build-recommender.awin-feed-policy.v1`. Required fields are
`schema_version`, `advertiser_id`, `feed_id`, `retailer`, `licence_or_access_note`, `rights`,
`allowed_currencies`, `allowed_link_hosts`, `category_mappings`, `feed`, and
`default_condition`. Optional fail-closed fields are `production_catalog_eligible`,
`training_eligible`, `published_claims_eligible`, `published_claims_grant_reference`,
`allow_non_new`, `expected_input_sha256`, and `limits`. The intentionally expired, zero-grant
[synthetic policy template](awin-policy.DO-NOT-USE.synthetic.json) is a schema example only and
cannot authorise ingestion. An operator must not make it operational by self-asserting rights;
an authorised policy issuer must generate and sign a new policy from the governing agreement.

`rights` contains exactly `contract_reference`, `contract_version_url`,
`consent_effective_on`, `consent_expires_on`, `retention_days`,
`deletion_required_on_termination`, `deletion_sla_days`, `territories`, and `grants`.
Recognised grants are `display`, `cache`, `store_history`, `redistribute`, `embed`, `train`, and
`derive`; an omitted grant is denied. Ingestion requires active SG rights for `cache` and
`derive`. Production-catalog eligibility additionally requires `display` and `store_history`.
Training requires a separate `train` grant. `published_claims_eligible=true` requires display
and derivation rights plus a non-empty, signed `published_claims_grant_reference` citing the
specific contractual authority; a generic access note is insufficient.

## Feed constraints

Policy and row enforcement is exact:

- advertiser and feed IDs are numeric; every row's `merchant_id` must equal the signed
  advertiser ID and `merchant_name` must equal the signed retailer name case-insensitively;
- `allowed_currencies` must be exactly `["SGD"]`; positive product price and explicit shipping
  cost, including an explicit free-shipping marker, are required;
- listing URLs must use HTTPS on port 443 or the default port and their host must exactly match
  an `allowed_link_hosts` entry; user info and credential-like paths/query keys are forbidden;
- category keys are explicit `id:`, `path:`, `merchant:`, or `name:` mappings to one of `cpu`,
  `gpu`, `motherboard`, `memory`, `storage`, `power_supply`, `cooler`, or `case`; conflicting
  matches are rejected rather than guessed;
- CSV format, compression (`none` or `gzip`), and delimiter (comma, semicolon, pipe, or tab) are
  signed; XML is deferred and is not accepted by this adapter; and
- signed limits bound `maximum_input_bytes`, `maximum_decompressed_bytes`, `maximum_records`,
  `maximum_rejections`, `maximum_field_characters`, `maximum_columns`, `maximum_output_bytes`,
  `maximum_record_bytes`, `maximum_price_sgd`, and `maximum_rejection_rate`. Built-in hard
  ceilings remain in force even if the signed policy requests larger values.

The adapter streams input and output, uses a disk-backed duplicate-ID index, and stages
content-hashed artifacts before atomic publication. Accepted records remain unmatched until
entity resolution.

## Run the local importer

Acquire the authorised feed separately, keep all source download URLs and credentials outside
the repository, then run:

```powershell
uv run --no-sync python scripts/fetch_open_data.py --source awin_feed `
  --awin-feed D:/secure-input/authorised-awin-feed.csv.gz `
  --awin-policy-json D:/secure-policy/awin-policy.json `
  --awin-policy-signature D:/secure-policy/awin-policy.signature.json `
  --awin-trust-root D:/secure-policy/awin-trust-root.json `
  --awin-trust-root-sha256 REPLACE_WITH_64_HEX_SHA256
```

The processed run is written beneath
`data/processed/awin_<advertiser-id>_<feed-id>/<processed-run-hash>/`. It includes
`records.jsonl`, `rejections.jsonl`, `manifest.json`, and `data-quality.json`; raw and processed
artifacts remain gitignored. CLI output identifies safe source provenance and content hashes,
not the original feed download URL.

## Production boundary

Cryptographic verification and bounded publication do not create licence authority. No Awin
catalogue size, coverage, model-quality, or published-claims metric is established until an
authorised feed has been processed and evaluated. The current downstream catalogue release path
also does not independently re-verify the detached-signature chain. Keep Awin artifacts out of
production release, training, and public claims until downstream signature/receipt verification,
retention enforcement, and an end-to-end authorised-feed evaluation are implemented.
