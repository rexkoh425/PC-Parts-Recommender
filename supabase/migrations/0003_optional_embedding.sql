-- Make the query embedding optional.
--
-- The browser has no way to produce a 384-dimensional MiniLM vector: that
-- needs the encoder, which is not deployed. Until an embedding endpoint
-- exists, the client can only search by text.
--
-- The NaN guard in 0002 already made a zero vector behave (keyword answers
-- alone), but that meant sending 384 zeros on every request and relying on a
-- degenerate float to mean "no vector". A null says it plainly, skips the
-- vector scan entirely, and costs nothing on the wire.

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
    keyword_rank   real
)
language sql
stable
security definer
-- Supabase installs pgvector into the extensions schema, so a search_path of
-- public alone hides the <=> operator. Still pinned, just naming both.
set search_path = public, extensions
as $$
    with semantic as (
        select * from (
            select
                e.product_id,
                -- <=> is cosine DISTANCE; similarity is its complement, so a
                -- higher number means a closer match.
                (1 - (e.embedding <=> query_embedding))::real as similarity
            from public.product_embeddings e
            where query_embedding is not null
            order by e.embedding <=> query_embedding
            limit greatest(match_count * 4, 96)
        ) ranked
        -- Cosine distance against a degenerate vector is NaN, and Postgres
        -- sorts NaN ABOVE every real number, so without this the worst
        -- possible matches would rank first.
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
        coalesce(k.keyword_rank, 0)::real
    from public.canonical_products p
    left join semantic s on s.product_id = p.product_id
    left join keyword  k on k.product_id = p.product_id
    where (s.product_id is not null or k.product_id is not null)
      and (category_filter is null or p.category = category_filter)
    -- Semantic dominates when present; keyword breaks ties. With no vector the
    -- first term is zero throughout and this degrades to pure keyword rank.
    order by (coalesce(s.similarity, 0) * 0.7 + coalesce(k.keyword_rank, 0) * 0.3) desc
    limit match_count;
$$;

revoke all on function public.search_products(vector, text, integer, text) from public;
grant execute on function public.search_products(vector, text, integer, text) to anon, authenticated;

-- Browsing with no query at all is the catalogue's default view, so give it a
-- dedicated path rather than making the client fake a search.
create or replace function public.browse_products(
    match_count     integer default 24,
    page_offset     integer default 0,
    category_filter text default null
)
returns table (
    product_id     text,
    canonical_name text,
    brand          text,
    category       text
)
language sql
stable
security definer
set search_path = public, extensions
as $$
    select p.product_id, p.canonical_name, p.brand, p.category
    from public.canonical_products p
    where category_filter is null or p.category = category_filter
    order by p.brand, p.canonical_name
    limit match_count offset page_offset;
$$;

revoke all on function public.browse_products(integer, integer, text) from public;
grant execute on function public.browse_products(integer, integer, text) to anon, authenticated;
