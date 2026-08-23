-- Catalogue + pgvector retrieval for the public demo.
--
-- Mirrors the subset of packages/core/.../catalog/orm.py that the deployed
-- site actually reads: identity, search text, and the 384-dimensional MiniLM
-- embedding. The write-side tables (listings, price observations, provenance)
-- stay in the local Postgres; nothing public needs them yet.
--
-- Run in the Supabase SQL editor, or: psql "$SUPABASE_DB_URL" -f this-file

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- Products
-- ---------------------------------------------------------------------------

create table if not exists public.canonical_products (
    product_id          text primary key,
    category            text        not null,
    brand               text        not null,
    model               text        not null,
    canonical_name      text        not null,
    manufacturer_part_number text,
    gtin                text unique,
    status              text        not null default 'active',
    common_attributes   jsonb       not null default '{}'::jsonb,
    category_attributes jsonb       not null default '{}'::jsonb,
    source_confidence   real        not null default 1.0,
    search_document     text        not null default '',
    data_version        text        not null,
    updated_at          timestamptz not null default now()
);

create index if not exists ix_products_category on public.canonical_products (category);
create index if not exists ix_products_brand    on public.canonical_products (brand);

-- Keyword half of the hybrid retrieval. Generated, so it can never drift from
-- the text it indexes.
alter table public.canonical_products
    add column if not exists search_tsv tsvector
    generated always as (to_tsvector('english', coalesce(search_document, ''))) stored;

create index if not exists ix_products_search_tsv
    on public.canonical_products using gin (search_tsv);

-- ---------------------------------------------------------------------------
-- Embeddings
-- ---------------------------------------------------------------------------

create table if not exists public.product_embeddings (
    product_id  text primary key
                references public.canonical_products (product_id) on delete cascade,
    embedding   vector(384) not null,
    -- Which immutable artifacts produced this row, so a served result can be
    -- traced back to the files that generated it rather than a model name.
    embedding_model            text not null,
    data_version               text not null,
    embeddings_artifact_sha256 text not null,
    id_map_artifact_sha256     text not null,
    updated_at  timestamptz not null default now()
);

-- HNSW over cosine distance, matching the local index. Built after load in
-- 0002; creating it on an empty table is fine but slower to populate.
create index if not exists ix_product_embeddings_hnsw
    on public.product_embeddings
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- Row level security
--
-- The publishable key is public and ships in the browser, so every table it
-- can reach must state its own policy. Read-only for anon; writes only via a
-- secret key, which never leaves the loader.
-- ---------------------------------------------------------------------------

alter table public.canonical_products enable row level security;
alter table public.product_embeddings enable row level security;

drop policy if exists "catalogue is publicly readable" on public.canonical_products;
create policy "catalogue is publicly readable"
    on public.canonical_products for select
    to anon, authenticated
    using (true);

drop policy if exists "embeddings are publicly readable" on public.product_embeddings;
create policy "embeddings are publicly readable"
    on public.product_embeddings for select
    to anon, authenticated
    using (true);
