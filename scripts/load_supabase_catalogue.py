"""Stream the pinned catalogue and its embeddings into Supabase.

Reads the same immutable artifacts the site cites, verifies their SHA-256
against the manifest before writing anything, and upserts in batches so an
interrupted run can simply be repeated.

Usage:
    export SUPABASE_DB_URL='postgresql://...'      # never commit this
    python scripts/load_supabase_catalogue.py --dry-run
    python scripts/load_supabase_catalogue.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

REPO = Path(__file__).resolve().parents[1]
EMB_DIR = REPO / "artifacts/retrieval/buildcores-full-embeddings-pinned"
RECORDS = (
    REPO
    / "data/processed/buildcores_open_db"
    / "f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f"
    / "full/records.jsonl"
)
EMBEDDING_DIMENSION = 384

PRODUCT_UPSERT = """
    insert into public.canonical_products
        (product_id, category, brand, model, canonical_name,
         manufacturer_part_number, gtin, status,
         common_attributes, category_attributes,
         source_confidence, search_document, data_version)
    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    on conflict (product_id) do update set
        category = excluded.category,
        brand = excluded.brand,
        model = excluded.model,
        canonical_name = excluded.canonical_name,
        manufacturer_part_number = excluded.manufacturer_part_number,
        gtin = excluded.gtin,
        status = excluded.status,
        common_attributes = excluded.common_attributes,
        category_attributes = excluded.category_attributes,
        source_confidence = excluded.source_confidence,
        search_document = excluded.search_document,
        data_version = excluded.data_version,
        updated_at = now()
"""

EMBEDDING_UPSERT = """
    insert into public.product_embeddings
        (product_id, embedding, embedding_model, data_version,
         embeddings_artifact_sha256, id_map_artifact_sha256)
    values (%s,%s,%s,%s,%s,%s)
    on conflict (product_id) do update set
        embedding = excluded.embedding,
        embedding_model = excluded.embedding_model,
        data_version = excluded.data_version,
        embeddings_artifact_sha256 = excluded.embeddings_artifact_sha256,
        id_map_artifact_sha256 = excluded.id_map_artifact_sha256,
        updated_at = now()
"""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def search_document(data: dict) -> str:
    """Mirror catalog/repository.py so keyword hits match the local index."""
    return " ".join(
        part
        for part in (
            data.get("category"),
            data.get("brand"),
            data.get("model"),
            data.get("manufacturer_part_number"),
            data.get("canonical_name"),
        )
        if part
    )


def load_manifest() -> dict:
    manifest = json.loads((EMB_DIR / "manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    # The site publishes these digests; refuse to upload anything whose bytes
    # do not match the files they name.
    for key, path in (
        ("embeddings", EMB_DIR / "embeddings.npy"),
        ("id_map", EMB_DIR / "ids.jsonl"),
    ):
        expected = artifacts[key]["sha256"]
        actual = sha256_of(path)
        if actual != expected:
            raise SystemExit(
                f"{path.name} digest mismatch\n  manifest {expected}\n  actual   {actual}"
            )
        print(f"  verified {path.name}  {actual[:12]}...")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="verify and report, write nothing")
    parser.add_argument("--batch", type=int, default=500)
    args = parser.parse_args()

    print("Verifying artifacts against the manifest")
    manifest = load_manifest()
    data_version = manifest["data_version"]
    encoder = manifest.get("encoder", {})
    model_name = (
        encoder.get("model")
        or encoder.get("name")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = np.load(EMB_DIR / "embeddings.npy")
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise SystemExit(f"expected {EMBEDDING_DIMENSION} dimensions, found {embeddings.shape[1]}")

    id_lines = (EMB_DIR / "ids.jsonl").read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line) for line in id_lines if line.strip()]
    if len(ids) != embeddings.shape[0]:
        raise SystemExit(f"id map has {len(ids)} rows, embeddings have {embeddings.shape[0]}")

    # row_index is the embedding's position. Trusting file order instead would
    # silently pair a product with someone else's vector.
    row_for = {entry["product_id"]: entry["row_index"] for entry in ids}
    print(f"  {embeddings.shape[0]:,} embeddings x {embeddings.shape[1]} dims")
    print(f"  data_version {data_version}")

    products: list[dict] = []
    with RECORDS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") != "canonical_product":
                continue
            data = record["data"]
            if data["product_id"] in row_for:
                products.append(data)
    print(f"  {len(products):,} products matched to an embedding")

    orphaned = len(row_for) - len(products)
    if orphaned:
        print(f"  note: {orphaned:,} embedded ids had no product record and are skipped")

    if args.dry_run:
        print("\nDry run - nothing written.")
        return 0

    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        raise SystemExit("SUPABASE_DB_URL is not set. Export it; never hard-code it.")

    emb_sha = manifest["artifacts"]["embeddings"]["sha256"]
    idmap_sha = manifest["artifacts"]["id_map"]["sha256"]

    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            print("\nWriting products")
            for start in range(0, len(products), args.batch):
                chunk = products[start : start + args.batch]
                cur.executemany(
                    PRODUCT_UPSERT,
                    [
                        (
                            item["product_id"],
                            item["category"],
                            item["brand"],
                            item["model"],
                            item["canonical_name"],
                            item.get("manufacturer_part_number"),
                            item.get("gtin"),
                            item.get("status", "active"),
                            Jsonb(item.get("common_attributes") or {}),
                            Jsonb(item.get("category_attributes") or {}),
                            float(item.get("source_confidence") or 1.0),
                            search_document(item),
                            data_version,
                        )
                        for item in chunk
                    ],
                )
                conn.commit()
                print(f"  products {min(start + args.batch, len(products)):,}/{len(products):,}")

            print("Writing embeddings")
            for start in range(0, len(products), args.batch):
                chunk = products[start : start + args.batch]
                cur.executemany(
                    EMBEDDING_UPSERT,
                    [
                        (
                            item["product_id"],
                            embeddings[row_for[item["product_id"]]],
                            model_name,
                            data_version,
                            emb_sha,
                            idmap_sha,
                        )
                        for item in chunk
                    ],
                )
                conn.commit()
                print(f"  embeddings {min(start + args.batch, len(products)):,}/{len(products):,}")

            cur.execute("select count(*) from public.canonical_products")
            product_count = cur.fetchone()[0]
            cur.execute("select count(*) from public.product_embeddings")
            embedding_count = cur.fetchone()[0]

    print(f"\nDone. {product_count:,} products, {embedding_count:,} embeddings in Supabase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
