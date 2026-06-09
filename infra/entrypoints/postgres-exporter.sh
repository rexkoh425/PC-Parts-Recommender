#!/bin/sh
set -eu

: "${POSTGRES_EXPORTER_HOST:?POSTGRES_EXPORTER_HOST is required}"
: "${POSTGRES_EXPORTER_DATABASE:?POSTGRES_EXPORTER_DATABASE is required}"
: "${POSTGRES_EXPORTER_USER:?POSTGRES_EXPORTER_USER is required}"
password_file="${POSTGRES_EXPORTER_PASSWORD_FILE:-/run/secrets/postgres_monitor_password}"
if [ ! -r "$password_file" ]; then
    echo "PostgreSQL exporter password file is not readable: $password_file" >&2
    exit 64
fi

export DATA_SOURCE_URI="${POSTGRES_EXPORTER_HOST}:${POSTGRES_EXPORTER_PORT:-5432}/${POSTGRES_EXPORTER_DATABASE}?sslmode=${POSTGRES_EXPORTER_SSLMODE:-disable}"
export DATA_SOURCE_USER="$POSTGRES_EXPORTER_USER"
DATA_SOURCE_PASS=$(tr -d '\r\n' < "$password_file")
export DATA_SOURCE_PASS

exec /bin/postgres_exporter "$@"
