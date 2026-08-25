"""Backfill each product's manufacturer page into Supabase.

Only real manufacturer hosts are written. Most provenance entries point at the
raw dataset on GitHub, which is where the data came from rather than a page a
visitor should be sent to, so those are skipped and the column stays null.

Usage:
    python scripts/backfill_source_urls.py --dry-run
    python scripts/backfill_source_urls.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg

REPO = Path(__file__).resolve().parents[1]
RECORDS = (
    REPO
    / "data/processed/buildcores_open_db"
    / "f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f"
    / "full/records.jsonl"
)

# Hosts that serve the dataset itself rather than a product page.
DATASET_HOSTS = {"raw.githubusercontent.com", "github.com", "objects.githubusercontent.com"}


def load_dsn() -> str:
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        env_file = REPO / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("SUPABASE_DB_URL="):
                    dsn = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not dsn:
        raise SystemExit("SUPABASE_DB_URL is not set.")
    return dsn


def manufacturer_url(record: dict) -> str | None:
    for entry in record.get("provenance") or []:
        url = (entry.get("source_url") or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme != "https":
            continue
        host = parsed.hostname or ""
        if not host or host in DATASET_HOSTS:
            continue
        return url
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch", type=int, default=1000)
    args = parser.parse_args()

    pairs: list[tuple[str, str]] = []
    total = 0
    with RECORDS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") != "canonical_product":
                continue
            total += 1
            url = manufacturer_url(record["data"])
            if url:
                pairs.append((url, record["data"]["product_id"]))

    print(f"  {len(pairs):,} of {total:,} products have a manufacturer page")
    if args.dry_run:
        for url, _ in pairs[:3]:
            print(f"    {url[:72]}")
        print("\nDry run - nothing written.")
        return 0

    with psycopg.connect(load_dsn(), connect_timeout=20) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(pairs), args.batch):
                chunk = pairs[start : start + args.batch]
                cur.executemany(
                    "update public.canonical_products set source_url = %s where product_id = %s",
                    chunk,
                )
                conn.commit()
                print(f"  {min(start + args.batch, len(pairs)):,}/{len(pairs):,}")

            cur.execute("select count(*) from public.canonical_products where source_url is not null")
            written = cur.fetchone()[0]

    print(f"\nDone. {written:,} products now carry a manufacturer link.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
