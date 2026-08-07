#!/bin/sh
set -eu
umask 077

: "${POSTGRES_HOST:?POSTGRES_HOST is required}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${PCBR_DATABASE_NAME:?PCBR_DATABASE_NAME is required}"
: "${PCBR_DAGSTER_DATABASE:?PCBR_DAGSTER_DATABASE is required}"
: "${PCBR_MLFLOW_DATABASE:?PCBR_MLFLOW_DATABASE is required}"

backup_root=${PCBR_BACKUP_ROOT:-/backups/postgres}
case "$backup_root" in
    /backups | /backups/*) ;;
    *)
        echo "Backup root must remain below /backups: $backup_root" >&2
        exit 64
        ;;
esac

password_file=${POSTGRES_PASSWORD_FILE:-/run/secrets/postgres_admin_password}
if [ ! -r "$password_file" ]; then
    echo "PostgreSQL backup password file is unreadable: $password_file" >&2
    exit 64
fi
PGPASSWORD=$(tr -d '\r\n' < "$password_file")
export PGPASSWORD

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
partial_dir="$backup_root/.${timestamp}.partial"
final_dir="$backup_root/$timestamp"
mkdir -p "$partial_dir"

pg_dumpall \
    --host "$POSTGRES_HOST" \
    --port "${POSTGRES_PORT:-5432}" \
    --username "$POSTGRES_USER" \
    --globals-only \
    --file "$partial_dir/globals.sql"

for database in "$PCBR_DATABASE_NAME" "$PCBR_DAGSTER_DATABASE" "$PCBR_MLFLOW_DATABASE"; do
    pg_dump \
        --host "$POSTGRES_HOST" \
        --port "${POSTGRES_PORT:-5432}" \
        --username "$POSTGRES_USER" \
        --dbname "$database" \
        --format custom \
        --compress 6 \
        --file "$partial_dir/${database}.dump"
done

cat > "$partial_dir/manifest.json" <<EOF
{
  "created_at_utc": "$timestamp",
  "format": "pg_dump-custom-v1",
  "databases": [
    "$PCBR_DATABASE_NAME",
    "$PCBR_DAGSTER_DATABASE",
    "$PCBR_MLFLOW_DATABASE"
  ],
  "restore_contract": "infra/ops/restore-postgres.sh"
}
EOF

(
    cd "$partial_dir"
    sha256sum globals.sql manifest.json ./*.dump > SHA256SUMS
)
mv "$partial_dir" "$final_dir"
unset PGPASSWORD
echo "$final_dir"
