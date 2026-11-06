# Gateway check after reported login

- The existing CLIProxyAPI listener on port 8317 responded to `/v1/models` with
  HTTP 200. It listed `gpt-6-luna` and `gpt-6.1-sol`, among other models; the
  earlier `codex-5.3-spark` alias was absent.
- A real pilot-task structured request using `gpt-6-luna` returned HTTP 401.
  Further minimal requests returned HTTP 503 for `gpt-6-luna` and HTTP 401
  `auth_unavailable` for `gpt-6.1-sol`. An ordinary, non-JSON request also
  returned HTTP 503. No inference response or usage record was received.
- The running server is the existing `CodingAgents` CLIProxyAPI executable.
  Its `config.codex-vercel-ai.yaml` specifies `./auths`; the start and login
  scripts both run the executable from the proxy directory. That directory had
  one authentication JSON file, last modified on June 8, 2026; no credential
  contents were read. This observation does not establish where the recently
  completed login saved its credentials.
- The 896-task pilot was not started: no selected route passed a real
  inference check. The existing queue and any prior checkpoints are preserved.
  No real judgments, model training, or holdout evaluation resulted from this
  check. No spending can be calculated without provider usage information.
