# Revoked performance evidence

This artifact was revoked on 2026-07-23.

The underlying Blender 4.0 observations are native higher-is-better
`samples_per_minute` values with unit `samples/minute`. The preparation path incorrectly labelled
them as seconds and trained on the reciprocal target `1000 / median(score)`. All evaluation values
in `metadata.json` and `training_report.json` were therefore computed against an invalid target.

Those values must not be used as model-quality, promotion, production, or resume evidence. The
runtime loader rejects this artifact through the `revoked` flag in `artifact_manifest.json`.
