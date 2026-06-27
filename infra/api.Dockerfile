FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN groupadd --gid 10001 pcbr \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin pcbr

COPY --chown=pcbr:pcbr pyproject.toml uv.lock README.md ./
COPY --chown=pcbr:pcbr packages/core ./packages/core
RUN uv sync --locked --no-dev --no-install-project --extra serving

COPY --chown=pcbr:pcbr services ./services
COPY --chown=pcbr:pcbr db ./db
COPY --chown=pcbr:pcbr scripts/import_processed_catalog.py ./scripts/import_processed_catalog.py
COPY --chown=pcbr:pcbr infra/entrypoints/api.sh /usr/local/bin/pcbr-api-entrypoint
RUN uv sync --locked --no-dev --extra serving

RUN chmod 0555 /usr/local/bin/pcbr-api-entrypoint

USER 10001:10001

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/pcbr-api-entrypoint"]
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
