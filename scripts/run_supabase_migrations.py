"""Apply the Supabase migrations in order.

Reads SUPABASE_DB_URL from the environment or a gitignored .env so the
connection string never has to be pasted anywhere it would be recorded. The
value is never printed; only the host is shown, so a run can be identified
without disclosing the password.

Usage:
    python scripts/run_supabase_migrations.py --check      # connectivity only
    python scripts/run_supabase_migrations.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg

REPO = Path(__file__).resolve().parents[1]
MIGRATIONS = REPO / "supabase" / "migrations"


def load_dsn() -> str:
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        env_file = REPO / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("SUPABASE_DB_URL="):
                    dsn = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not dsn:
        raise SystemExit(
            "SUPABASE_DB_URL is not set.\n"
            "Add it to a .env file at the repo root (already gitignored):\n"
            "    SUPABASE_DB_URL=postgresql://...\n"
        )
    return dsn


def describe(dsn: str) -> str:
    """Host and database only - never the password."""
    parsed = urlparse(dsn)
    return f"{parsed.hostname}:{parsed.port or 5432}{parsed.path}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="test the connection and exit")
    args = parser.parse_args()

    dsn = load_dsn()
    print(f"Connecting to {describe(dsn)}")

    with psycopg.connect(dsn, connect_timeout=20) as conn:
        with conn.cursor() as cur:
            cur.execute("select current_database(), version()")
            database, version = cur.fetchone()
            print(f"  connected to {database}")
            print(f"  {version.split(',')[0]}")

            cur.execute("select extname from pg_extension where extname = 'vector'")
            has_vector = cur.fetchone() is not None
            print(f"  pgvector extension: {'enabled' if has_vector else 'NOT enabled'}")

            if args.check:
                return 0

            if not has_vector:
                # Creating it needs elevated rights the pooled role may lack, so
                # try, and say plainly what to click if it is refused.
                print("\nEnabling pgvector")
                try:
                    cur.execute("create extension if not exists vector")
                    conn.commit()
                    print("  enabled")
                except psycopg.Error as error:
                    conn.rollback()
                    raise SystemExit(
                        f"  could not enable it: {error}\n"
                        "  Enable it by hand: Dashboard > Database > Extensions > vector"
                    ) from error

            files = sorted(MIGRATIONS.glob("*.sql"))
            if not files:
                raise SystemExit(f"no .sql files in {MIGRATIONS}")

            print(f"\nApplying {len(files)} migration(s)")
            for path in files:
                sql = path.read_text(encoding="utf-8")
                print(f"  {path.name} ... ", end="", flush=True)
                try:
                    cur.execute(sql)
                    conn.commit()
                    print("ok")
                except psycopg.Error as error:
                    conn.rollback()
                    print("FAILED")
                    raise SystemExit(f"    {error}") from error

            # Report what now exists rather than assuming the DDL did what it said.
            print("\nVerifying")
            cur.execute(
                """
                select table_name from information_schema.tables
                where table_schema = 'public'
                  and table_name in ('canonical_products', 'product_embeddings')
                order by table_name
                """
            )
            tables = [row[0] for row in cur.fetchall()]
            for name in ("canonical_products", "product_embeddings"):
                print(f"  table {name}: {'present' if name in tables else 'MISSING'}")

            cur.execute("select proname from pg_proc where proname = 'search_products'")
            print(f"  function search_products: {'present' if cur.fetchone() else 'MISSING'}")

            cur.execute(
                "select indexname from pg_indexes "
                "where schemaname = 'public' and indexname = 'ix_product_embeddings_hnsw'"
            )
            print(f"  hnsw index: {'present' if cur.fetchone() else 'MISSING'}")

            cur.execute(
                """
                select tablename, rowsecurity from pg_tables
                where schemaname = 'public'
                  and tablename in ('canonical_products', 'product_embeddings')
                order by tablename
                """
            )
            for table, rls in cur.fetchall():
                print(f"  RLS on {table}: {'enabled' if rls else 'DISABLED'}")

    print("\nMigrations applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
