#!/bin/sh
set -eu

if [ -n "${DAGSTER_POSTGRES_PASSWORD_FILE:-}" ]; then
    if [ ! -r "$DAGSTER_POSTGRES_PASSWORD_FILE" ]; then
        echo "Dagster password file is not readable: $DAGSTER_POSTGRES_PASSWORD_FILE" >&2
        exit 64
    fi
    DAGSTER_POSTGRES_PASSWORD=$(tr -d '\r\n' < "$DAGSTER_POSTGRES_PASSWORD_FILE")
    export DAGSTER_POSTGRES_PASSWORD
fi

exec "$@"
