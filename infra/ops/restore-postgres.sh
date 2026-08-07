#!/bin/sh
set -eu

: "${PCBR_RESTORE_SET:?PCBR_RESTORE_SET is required}"
: "${PCBR_RESTORE_CONFIRM:?PCBR_RESTORE_CONFIRM is required}"
: "${POSTGRES_HOST:?POSTGRES_HOST is required}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${PCBR_DATABASE_NAME:?PCBR_DATABASE_NAME is required}"
: "${PCBR_MIGRATOR_USER:?PCBR_MIGRATOR_USER is required}"
: "${PCBR_DAGSTER_DATABASE:?PCBR_DAGSTER_DATABASE is required}"
: "${PCBR_DAGSTER_USER:?PCBR_DAGSTER_USER is required}"
: "${PCBR_MLFLOW_DATABASE:?PCBR_MLFLOW_DATABASE is required}"
: "${PCBR_MLFLOW_USER:?PCBR_MLFLOW_USER is required}"

case "$PCBR_RESTORE_SET" in
    [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z) ;;
    *)
        echo "PCBR_RESTORE_SET must be a UTC backup-set name such as 20260722T130000Z" >&2
        exit 64
        ;;
esac

expected_confirmation="RESTORE:${PCBR_RESTORE_SET}"
if [ "$PCBR_RESTORE_CONFIRM" != "$expected_confirmation" ]; then
    echo "Restore refused. Set PCBR_RESTORE_CONFIRM=$expected_confirmation" >&2
    exit 64
fi

restore_dir="/backups/postgres/$PCBR_RESTORE_SET"
if [ ! -d "$restore_dir" ]; then
    echo "Backup set not found: $restore_dir" >&2
    exit 66
fi
(
    cd "$restore_dir"
    sha256sum --check SHA256SUMS
)

password_file=${POSTGRES_PASSWORD_FILE:-/run/secrets/postgres_admin_password}
if [ ! -r "$password_file" ]; then
    echo "PostgreSQL restore password file is unreadable: $password_file" >&2
    exit 64
fi
PGPASSWORD=$(tr -d '\r\n' < "$password_file")
export PGPASSWORD

restore_database() {
    database=$1
    owner=$2
    archive="$restore_dir/${database}.dump"
    if [ ! -f "$archive" ]; then
        echo "Database archive not found: $archive" >&2
        exit 66
    fi

    psql --set ON_ERROR_STOP=1 \
        --host "$POSTGRES_HOST" \
        --port "${POSTGRES_PORT:-5432}" \
        --username "$POSTGRES_USER" \
        --dbname postgres \
        --set target_db="$database" \
        --set target_owner="$owner" <<'SQL'
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = :'target_db' AND pid <> pg_backend_pid();
DROP DATABASE IF EXISTS :"target_db";
CREATE DATABASE :"target_db" OWNER :"target_owner";
SQL

    pg_restore \
        --host "$POSTGRES_HOST" \
        --port "${POSTGRES_PORT:-5432}" \
        --username "$POSTGRES_USER" \
        --dbname "$database" \
        --no-owner \
        --role "$owner" \
        --exit-on-error \
        "$archive"
}

restore_database "$PCBR_DATABASE_NAME" "$PCBR_MIGRATOR_USER"
restore_database "$PCBR_DAGSTER_DATABASE" "$PCBR_DAGSTER_USER"
restore_database "$PCBR_MLFLOW_DATABASE" "$PCBR_MLFLOW_USER"

unset PGPASSWORD
echo "Restore completed for backup set $PCBR_RESTORE_SET"
