# Backup and restore policy

Status: scripts and operator contract implemented; recovery time is not yet measured
Last updated: 2026-07-22

The production Compose file contains opt-in `operations` backup services and an explicit `restore`
service. They never run on application startup. A backup is useful only after an independent
restore drill, checksum verification, and an encrypted off-host copy.

## Protected state

| State | Mechanism | Consistency boundary |
| --- | --- | --- |
| Application, Dagster, and MLflow databases | PostgreSQL custom-format logical dumps plus globals | Each database dump is transactionally consistent at its own start time; the three dumps are not one cross-database transaction. |
| Dagster compute logs/local artifacts | Compressed archive | Quiesce Dagster for a consistent archive. Run/event metadata is in PostgreSQL. |
| MLflow artifacts | Compressed archive | Quiesce training/model writes or use versioned object storage snapshots. Metadata is in PostgreSQL. |
| Pipeline artifacts | Compressed archive | Quiesce asset materialization or copy immutable content-addressed objects. |
| Raw/processed source data | Source manifest hashes and separately replicated data directory | Re-fetch only where licence/access and immutable source guarantees permit it. |
| Images and configuration | Registry digests plus release record | Never rely on a mutable tag. Secret values are reconstructed from the secret manager, not backed up into Git. |

The checked-in configuration provides logical backups, not WAL archiving or point-in-time
recovery. Use a managed PostgreSQL service with encrypted PITR for a higher availability/RPO
requirement. RPO and RTO remain unmeasured until a timed restore drill is recorded.

## Create a backup

Validate the environment first. For artifact consistency, pause Dagster schedules/daemon and stop
new MLflow-writing jobs. The API can remain available during a logical database dump if the
accepted cross-database consistency boundary is documented.

```powershell
python scripts/validate_production_env.py --env-file .env.production

docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile operations run --rm backup-postgres

docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile operations run --rm backup-artifacts
```

Each service writes into a new UTC-named directory below `PCBR_BACKUP_DIR`, stages it as a partial
directory, calculates `SHA256SUMS`, and atomically renames it after success. PostgreSQL backups
contain separate custom-format dumps for the application, Dagster, and MLflow databases. The
globals file is retained for audit/emergency reconstruction; the normal restore uses roles created
from current secret-manager values and does not replay old passwords.

After each backup:

1. inspect the manifest and run `sha256sum --check SHA256SUMS` in both backup-set directories;
2. copy to encrypted, access-controlled off-host storage with versioning/immutability;
3. record the deployment ID, image digests, database revision, data/model/rule versions, byte
   counts, backup-set names, and copy destination;
4. enforce retention outside the container after confirming an off-host copy and restore drill;
5. monitor free space before and after the run.

The repository deliberately does not auto-delete old backups.

## PostgreSQL recovery

Prefer a separate recovery host and fresh volumes. Never use the first restore attempt against the
only production copy. The restore script is destructive: it disconnects clients, drops the three
target databases, recreates them with their least-privilege owners, and restores with
`--no-owner`.

1. Prove the backup checksums and inventory.
2. Stop API, migration, Dagster, MLflow, and ingestion writers.
3. Confirm the target host, Compose project, secret paths, target database names, and backup set.
4. Keep the PostgreSQL service running and healthy.
5. Set the exact backup name and matching confirmation in the current shell:

```powershell
$env:PCBR_RESTORE_SET = "20260722T130000Z"
$env:PCBR_RESTORE_CONFIRM = "RESTORE:$env:PCBR_RESTORE_SET"

docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile restore run --rm restore-postgres
```

6. Clear the two confirmation variables immediately.
7. Run the application migration and MLflow schema-upgrade jobs against the restored databases.
8. Verify Alembic revision, extension versions, row counts, provenance hashes, Dagster code
   location/daemon state, MLflow experiments/artifact references, API readiness, and a safe
   end-to-end request.
9. Compare the measured recovery time and recovered timestamp with the declared RTO/RPO.

If any checksum, role, extension, schema, or provenance check fails, keep the deployment isolated.
Do not weaken the check to make the restore pass.

## Artifact recovery

Restore archives into new empty directories first. List every archive before extraction and reject
absolute paths or `..` traversal. Verify `SHA256SUMS`, extract, compare manifests and expected
artifact hashes, then point a staging deployment at the recovered directories. Switch production
paths only after Dagster and MLflow consistency checks pass.

Artifact state and its database metadata must come from compatible backup intervals. Restoring an
MLflow database without the matching artifact objects can produce healthy HTTP probes but broken
models. Restoring Dagster metadata without compute logs affects auditability even when schedules
resume.

## Drill cadence and evidence

Run a restore drill after any storage/topology change and at a regular cadence chosen by the
deployment owner. Preserve:

- source backup names and hashes;
- target isolation evidence;
- start/end UTC timestamps and measured recovery time;
- database, table, row-count, and extension checks;
- sampled raw/data/model artifact hash checks;
- application, Dagster, MLflow, and Prometheus health results;
- issues, operator decisions, and follow-up owners.

Until such an artifact exists, the project may claim backup/restore tooling, not a proven RPO,
RTO, disaster-recovery capability, or zero-data-loss guarantee.
