#!/bin/sh
set -eu

if [ -z "${MLFLOW_BACKEND_STORE_URI:-}" ]; then
    : "${MLFLOW_POSTGRES_HOST:?MLFLOW_POSTGRES_HOST is required}"
    : "${MLFLOW_POSTGRES_DATABASE:?MLFLOW_POSTGRES_DATABASE is required}"
    : "${MLFLOW_POSTGRES_USER:?MLFLOW_POSTGRES_USER is required}"
    password_file="${MLFLOW_POSTGRES_PASSWORD_FILE:-/run/secrets/mlflow_password}"
    if [ ! -r "$password_file" ]; then
        echo "MLflow password file is not readable: $password_file" >&2
        exit 64
    fi
    MLFLOW_POSTGRES_PASSWORD=$(tr -d '\r\n' < "$password_file")
    export MLFLOW_POSTGRES_PASSWORD
    MLFLOW_PGPASS_FILE=${MLFLOW_PGPASS_FILE:-/tmp/mlflow.pgpass}
    export MLFLOW_PGPASS_FILE
    MLFLOW_BACKEND_STORE_URI=$(
        python -c 'import os, pathlib, urllib.parse; esc=lambda v:v.replace("\\", "\\\\").replace(":", "\\:"); host=os.environ["MLFLOW_POSTGRES_HOST"]; port=os.environ.get("MLFLOW_POSTGRES_PORT", "5432"); db=os.environ["MLFLOW_POSTGRES_DATABASE"]; user=os.environ["MLFLOW_POSTGRES_USER"]; p=pathlib.Path(os.environ["MLFLOW_PGPASS_FILE"]); p.write_text(f"{esc(host)}:{esc(port)}:{esc(db)}:{esc(user)}:{esc(os.environ["MLFLOW_POSTGRES_PASSWORD"])}\n", encoding="utf-8"); p.chmod(0o600); print("postgresql+psycopg://{}@{}:{}/{}".format(urllib.parse.quote(user, safe=""), host, port, urllib.parse.quote(db, safe="")))'
    )
    unset MLFLOW_POSTGRES_PASSWORD
    PGPASSFILE=$MLFLOW_PGPASS_FILE
    export PGPASSFILE
    export MLFLOW_BACKEND_STORE_URI
fi

case "${1:-server}" in
    migrate)
        exec mlflow db upgrade "$MLFLOW_BACKEND_STORE_URI"
        ;;
    server)
        : "${MLFLOW_ARTIFACT_ROOT:?MLFLOW_ARTIFACT_ROOT is required}"
        exec mlflow server \
            --host 0.0.0.0 \
            --port 5000 \
            --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
            --artifacts-destination "$MLFLOW_ARTIFACT_ROOT" \
            --serve-artifacts \
            --workers "${MLFLOW_WORKERS:-2}" \
            --gunicorn-opts "--access-logfile - --error-logfile - --timeout ${MLFLOW_REQUEST_TIMEOUT_SECONDS:-120}"
        ;;
    *)
        exec "$@"
        ;;
esac
