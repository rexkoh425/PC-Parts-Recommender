-- Return the attributes a catalogue card needs.
--
-- Cards were showing a name, a brand, and nothing else, so a page of 24 was
-- unscannable: no way to tell a 550 W supply from a 1200 W one without opening
-- each record. Both functions now return category_attributes, and the client
-- picks the one spec that distinguishes parts within a category.
--
-- Returned whole rather than as a pre-picked string so the choice of which
-- spec matters stays in the UI, where it can change without a migration.

-- Adding a returned column changes the row type, which CREATE OR REPLACE
-- cannot do. Dropping first is required; both functions are recreated below
-- in the same transaction, so nothing is left without a definition.
drop function if exists public.search_products(vector, text, integer, text);
drop function if exists public.browse_products(integer, integer, text);

create or replace function public.search_products(
    query_embedding vector(384) default null,
    query_text      text default '',
    match_count     integer default 24,
    category_filter text default null
)
returns table (
    product_id     text,
    canonical_name text,
    brand          text,
    category       text,
    similarity     real,
    keyword_rank   real,
    attributes     jsonb
)
language sql
stable
security definer
set search_path = public, extensions
as $$
    with semantic as (
        select * from (
            select
                e.product_id,
                (1 - (e.embedding <=> query_embedding))::real as similarity
            from public.product_embeddings e
            where query_embedding is not null
            order by e.embedding <=> query_embedding
            limit greatest(match_count * 4, 96)
        ) ranked
        -- Cosine distance against a degenerate vector is NaN, and Postgres
        -- sorts NaN above every real number.
        where ranked.similarity <> 'NaN'::real
    ),
    keyword as (
        select
            p.product_id,
            ts_rank(p.search_tsv, websearch_to_tsquery('english', query_text))::real as keyword_rank
        from public.canonical_products p
        where query_text <> ''
          and p.search_tsv @@ websearch_to_tsquery('english', query_text)
        order by ts_rank(p.search_tsv, websearch_to_tsquery('english', query_text)) desc
        limit greatest(match_count * 4, 96)
    )
    select
        p.product_id,
        p.canonical_name,
        p.brand,
        p.category,
        coalesce(s.similarity, 0)::real,
        coalesce(k.keyword_rank, 0)::real,
        p.category_attributes
    from public.canonical_products p
    left join semantic s on s.product_id = p.product_id
    left join keyword  k on k.product_id = p.product_id
    where (s.product_id is not null or k.product_id is not null)
      and (category_filter is null or p.category = category_filter)
    order by (coalesce(s.similarity, 0) * 0.7 + coalesce(k.keyword_rank, 0) * 0.3) desc
    limit match_count;
$$;

revoke all on function public.search_products(vector, text, integer, text) from public;
grant execute on function public.search_products(vector, text, integer, text) to anon, authenticated;

create or replace function public.browse_products(
    match_count     integer default 24,
    page_offset     integer default 0,
    category_filter text default null
)
returns table (
    product_id     text,
    canonical_name text,
    brand          text,
    category       text,
    attributes     jsonb
)
language sql
stable
security definer
set search_path = public, extensions
as $$
    select p.product_id, p.canonical_name, p.brand, p.category, p.category_attributes
    from public.canonical_products p
    where category_filter is null or p.category = category_filter
    order by p.brand, p.canonical_name
    limit match_count offset page_offset;
$$;

revoke all on function public.browse_products(integer, integer, text) from public;
grant execute on function public.browse_products(integer, integer, text) to anon, authenticated;
