#!/bin/sh
set -eu

if [ -z "${DATABASE_URL:-}" ]; then
    : "${PCBR_DATABASE_HOST:?PCBR_DATABASE_HOST is required when DATABASE_URL is absent}"
    : "${PCBR_DATABASE_NAME:?PCBR_DATABASE_NAME is required when DATABASE_URL is absent}"
    : "${PCBR_DATABASE_USER:?PCBR_DATABASE_USER is required when DATABASE_URL is absent}"
    password_file="${PCBR_DATABASE_PASSWORD_FILE:-/run/secrets/database_password}"
    if [ ! -r "$password_file" ]; then
        echo "Database password file is not readable: $password_file" >&2
        exit 64
    fi
    PCBR_DATABASE_PASSWORD=$(tr -d '\r\n' < "$password_file")
    export PCBR_DATABASE_PASSWORD
    PCBR_PGPASS_FILE=${PCBR_PGPASS_FILE:-/tmp/pcbr.pgpass}
    export PCBR_PGPASS_FILE
    DATABASE_URL=$(
        python -c 'import os, pathlib, urllib.parse; esc=lambda v:v.replace("\\", "\\\\").replace(":", "\\:"); host=os.environ["PCBR_DATABASE_HOST"]; port=os.environ.get("PCBR_DATABASE_PORT", "5432"); db=os.environ["PCBR_DATABASE_NAME"]; user=os.environ["PCBR_DATABASE_USER"]; p=pathlib.Path(os.environ["PCBR_PGPASS_FILE"]); p.write_text(f"{esc(host)}:{esc(port)}:{esc(db)}:{esc(user)}:{esc(os.environ["PCBR_DATABASE_PASSWORD"])}\n", encoding="utf-8"); p.chmod(0o600); print("postgresql+psycopg://{}@{}:{}/{}".format(urllib.parse.quote(user, safe=""), host, port, urllib.parse.quote(db, safe="")))'
    )
    unset PCBR_DATABASE_PASSWORD
    PGPASSFILE=$PCBR_PGPASS_FILE
    export PGPASSFILE
    export DATABASE_URL
fi

case "${PCBR_RUN_MIGRATIONS:-true}" in
    true)
        alembic -c db/alembic.ini upgrade head
        ;;
    false)
        ;;
    *)
        echo "PCBR_RUN_MIGRATIONS must be true or false" >&2
        exit 64
        ;;
esac

exec "$@"
