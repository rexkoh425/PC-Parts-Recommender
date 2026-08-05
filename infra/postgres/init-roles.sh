#!/bin/sh
set -eu

read_secret() {
    secret_file=$1
    if [ ! -r "$secret_file" ]; then
        echo "Required PostgreSQL bootstrap secret is unreadable: $secret_file" >&2
        exit 64
    fi
    tr -d '\r\n' < "$secret_file"
}

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${PCBR_DATABASE_NAME:?PCBR_DATABASE_NAME is required}"
: "${PCBR_MIGRATOR_USER:?PCBR_MIGRATOR_USER is required}"
: "${PCBR_APP_USER:?PCBR_APP_USER is required}"
: "${PCBR_DAGSTER_DATABASE:?PCBR_DAGSTER_DATABASE is required}"
: "${PCBR_DAGSTER_USER:?PCBR_DAGSTER_USER is required}"
: "${PCBR_MLFLOW_DATABASE:?PCBR_MLFLOW_DATABASE is required}"
: "${PCBR_MLFLOW_USER:?PCBR_MLFLOW_USER is required}"
: "${PCBR_MONITOR_USER:?PCBR_MONITOR_USER is required}"

migrator_password=$(read_secret /run/secrets/postgres_migrator_password)
app_password=$(read_secret /run/secrets/postgres_app_password)
dagster_password=$(read_secret /run/secrets/postgres_dagster_password)
mlflow_password=$(read_secret /run/secrets/postgres_mlflow_password)
monitor_password=$(read_secret /run/secrets/postgres_monitor_password)

psql --set ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
    --set migrator_user="$PCBR_MIGRATOR_USER" \
    --set migrator_password="$migrator_password" \
    --set app_user="$PCBR_APP_USER" \
    --set app_password="$app_password" \
    --set dagster_user="$PCBR_DAGSTER_USER" \
    --set dagster_password="$dagster_password" \
    --set mlflow_user="$PCBR_MLFLOW_USER" \
    --set mlflow_password="$mlflow_password" \
    --set monitor_user="$PCBR_MONITOR_USER" \
    --set monitor_password="$monitor_password" \
    --set app_db="$PCBR_DATABASE_NAME" \
    --set dagster_db="$PCBR_DAGSTER_DATABASE" \
    --set mlflow_db="$PCBR_MLFLOW_DATABASE" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'migrator_user', :'migrator_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'migrator_user')
\gexec
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'app_user', :'app_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user')
\gexec
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'dagster_user', :'dagster_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'dagster_user')
\gexec
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'mlflow_user', :'mlflow_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'mlflow_user')
\gexec
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'monitor_user', :'monitor_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'monitor_user')
\gexec

SELECT format('CREATE DATABASE %I OWNER %I', :'app_db', :'migrator_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'app_db')
\gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'dagster_db', :'dagster_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'dagster_db')
\gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'mlflow_db', :'mlflow_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'mlflow_db')
\gexec

GRANT pg_monitor TO :"monitor_user";
GRANT CONNECT ON DATABASE :"app_db" TO :"app_user";
SQL

psql --set ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$PCBR_DATABASE_NAME" \
    --set migrator_user="$PCBR_MIGRATOR_USER" \
    --set app_user="$PCBR_APP_USER" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
GRANT USAGE ON SCHEMA public TO :"app_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_user" IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"app_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_user" IN SCHEMA public
    GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO :"app_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_user" IN SCHEMA public
    GRANT EXECUTE ON FUNCTIONS TO :"app_user";
SQL

unset migrator_password app_password dagster_password mlflow_password monitor_password
