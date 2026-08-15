-- Hybrid retrieval as a single round trip.
--
-- The browser holds a publishable key, so it must not be able to run arbitrary
-- SQL. Exposing retrieval as one security-definer function means the client
-- calls a named operation with typed arguments and gets rows back — it cannot
-- reach past what this function returns.

create or replace function public.search_products(
    query_embedding vector(384),
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
    keyword_rank   real
)
language sql
stable
security definer
set search_path = public
as $$
    with semantic as (
        select
            e.product_id,
            -- pgvector's <=> is cosine DISTANCE; similarity is its complement,
            -- so a higher number means a closer match.
            (1 - (e.embedding <=> query_embedding))::real as similarity
        from public.product_embeddings e
        order by e.embedding <=> query_embedding
        limit greatest(match_count * 4, 96)
    ),
    keyword as (
        select
            p.product_id,
            ts_rank(p.search_tsv, websearch_to_tsquery('english', query_text))::real as keyword_rank
        from public.canonical_products p
        where query_text <> ''
          and p.search_tsv @@ websearch_to_tsquery('english', query_text)
        limit greatest(match_count * 4, 96)
    )
    select
        p.product_id,
        p.canonical_name,
        p.brand,
        p.category,
        coalesce(s.similarity, 0)::real,
        coalesce(k.keyword_rank, 0)::real
    from public.canonical_products p
    left join semantic s on s.product_id = p.product_id
    left join keyword  k on k.product_id = p.product_id
    where (s.product_id is not null or k.product_id is not null)
      and (category_filter is null or p.category = category_filter)
    -- Reciprocal-rank style blend: the semantic score dominates, keyword
    -- breaks ties. Matches the weighting the local retriever uses.
    order by (coalesce(s.similarity, 0) * 0.7 + coalesce(k.keyword_rank, 0) * 0.3) desc
    limit match_count;
$$;

revoke all on function public.search_products(vector, text, integer, text) from public;
grant execute on function public.search_products(vector, text, integer, text) to anon, authenticated;
