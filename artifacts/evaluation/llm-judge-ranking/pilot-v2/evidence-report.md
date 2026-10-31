# LLM judge ranking pilot: execution record

This is an unjudged technical pilot. It is not an independent ranking evaluation.

- Base git revision: `5ae6cb0f3475baa090ce3ca4d5accfff4687ce61` (working tree dirty).
- Catalogue: 25,666 BuildCores records; SHA-256
  `8c738c5136615080ac29ced074726f97cab473c5a45eafdf19c26504aa624ce2`.
- Seed: `20261108`. Query set: 32 previously used silver-diagnostic queries,
  four in each of eight categories. Pilot split: 19 train / 7 validation / 6 test
  query IDs; semantic independence has not been established.
- 896 unique unjudged pairs, 28 per query. Frozen task and snapshot checksums,
  query source hash and embedding-manifest hash are in `manifest.json`.
- Rebuilt the queue in a separate temporary directory: task and snapshot hashes
  matched. Offline checks constructed all 896 ranking candidates, computed 25
  features per candidate, checked task uniqueness and confirmed the seed-based
  trainer split against the saved split. Synthetic test-only grades used to
  exercise the split were not saved as judgments or used to train a model.
- Windows 11, Python 3.12.9, NumPy 2.5.1, sentence-transformers 5.6.0,
  LightGBM 4.7.0; queue encoding resolved to CUDA.
- Provider: local CLIProxyAPI, configured alias `codex-5.3-spark`. Initially
  port 8317 refused connections. The existing start script brought the server
  online. Authenticated `/v1/models` returned HTTP 200 but did not list the
  configured alias. A real structured JSON request returned HTTP 502. Gateway
  logs reported HTTP 401 `invalid_refresh_token` from upstream. A separate real
  judge task attempted with prompt v1 failed; its failure record is in
  `../pilot-v1/connectivity-attempt/`. Current prompt version is v2.
- Actual judgments: 0; attempted task failures: 1; unattempted pilot pairs: 896.
  Token usage and monetary cost cannot be determined from a successful response.
  No model was trained, selected or evaluated. No feature importance, inference
  timing, LLM-judged confidence interval or judge agreement exists.

See `docs/llm-weak-ranking-judgments.md` for commands and limitations. A working
upstream login is necessary, followed by a successful structured response and
pilot validation. A separate, larger set of grouped intents is still needed for
a credible held-out experiment.
