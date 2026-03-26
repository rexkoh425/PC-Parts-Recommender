# Revoked diagnostic evidence

This diagnostic was revoked on 2026-07-23.

It inherited the same invalid reciprocal target as v2: native higher-is-better
`samples_per_minute` observations were labelled as seconds and inverted. Its development and
reused-holdout values are invalid as model-quality evidence. In addition, the holdout was observed
during adaptive development and cannot be reused for promotion after correcting the target.

These values must not be used as promotion, production, or resume evidence. The manifest remains
`production_loadable=false` and is explicitly marked `revoked=true`.
