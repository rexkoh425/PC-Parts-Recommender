FROM python:3.12-slim AS runtime

# Patch base-image OS packages. The published python:3.12-slim tag lags Debian
# security updates, which trivy reports as fixed HIGH advisories.
RUN apt-get update \n    && apt-get upgrade -y --no-install-recommends \n    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    DAGSTER_HOME=/app/.dagster

WORKDIR /app

RUN groupadd --gid 10002 pcbr \
    && useradd --uid 10002 --gid 10002 --no-create-home --shell /usr/sbin/nologin pcbr \
    && mkdir -p /app/.dagster /app/data /app/artifacts /app/config \
    && chown -R pcbr:pcbr /app

COPY --chown=pcbr:pcbr pyproject.toml uv.lock README.md ./
COPY --chown=pcbr:pcbr packages/core ./packages/core
RUN uv sync --locked --no-dev --extra pipeline --no-install-project

COPY --chown=pcbr:pcbr pipelines ./pipelines
COPY --chown=pcbr:pcbr data/source_registry.yaml ./config/source_registry.yaml
COPY --chown=pcbr:pcbr infra/dagster.yaml /app/.dagster/dagster.yaml
COPY --chown=pcbr:pcbr infra/dagster.workspace.yaml /app/.dagster/workspace.yaml
COPY --chown=pcbr:pcbr infra/entrypoints/dagster.sh /usr/local/bin/pcbr-dagster-entrypoint
RUN uv sync --locked --no-dev --extra pipeline

RUN chmod 0555 /usr/local/bin/pcbr-dagster-entrypoint

USER 10002:10002

EXPOSE 3001

HEALTHCHECK --interval=60s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import sys; sys.exit(0)"]

ENTRYPOINT ["/usr/local/bin/pcbr-dagster-entrypoint"]
CMD ["dagster", "dev", "-h", "0.0.0.0", "-p", "3001", "-m", "pipelines.definitions"]
