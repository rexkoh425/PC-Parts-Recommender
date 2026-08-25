-- Carry each product's manufacturer page.
--
-- 3,553 of the 25,666 records cite a real manufacturer URL in their
-- provenance. Surfacing it gives a live record the same "Manufacturer spec"
-- link the 21 curated parts already had, so a card is a way through to the
-- authoritative page rather than a dead end.
--
-- Only genuine manufacturer hosts are stored. The rest of the provenance
-- points at the raw dataset on GitHub, which is where the data came from, not
-- a page any visitor should be sent to.

alter table public.canonical_products
    add column if not exists source_url text;

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
    attributes     jsonb,
    source_url     text
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
        p.category_attributes,
        p.source_url
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
    attributes     jsonb,
    source_url     text
)
language sql
stable
security definer
set search_path = public, extensions
as $$
    select p.product_id, p.canonical_name, p.brand, p.category,
           p.category_attributes, p.source_url
    from public.canonical_products p
    where category_filter is null or p.category = category_filter
    order by p.brand, p.canonical_name
    limit match_count offset page_offset;
$$;

revoke all on function public.browse_products(integer, integer, text) from public;
grant execute on function public.browse_products(integer, integer, text) to anon, authenticated;
