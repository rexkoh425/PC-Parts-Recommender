FROM python:3.12-slim AS runtime

# Patch base-image OS packages. The published python:3.12-slim tag lags Debian
# security updates, which the container scan reports as fixed HIGH advisories.
# DL3005 discourages blanket upgrades; here it is deliberate and the only way to
# pick up published security fixes without pinning every transitive OS package.
# hadolint ignore=DL3005
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

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
RUN uv sync --locked --no-dev --no-install-project --extra serving --extra modeling

COPY --chown=pcbr:pcbr services ./services
COPY --chown=pcbr:pcbr db ./db
COPY --chown=pcbr:pcbr pipelines ./pipelines
COPY --chown=pcbr:pcbr scripts/__init__.py scripts/import_catalog_release.py scripts/import_processed_catalog.py ./scripts/
COPY --chown=pcbr:pcbr infra/entrypoints/api.sh /usr/local/bin/pcbr-api-entrypoint
RUN uv sync --locked --no-dev --extra serving --extra modeling
RUN uv run --no-sync python -c "import yaml; from pipelines.source_release import verify_awin_production_batch_release"

RUN chmod 0555 /usr/local/bin/pcbr-api-entrypoint

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live',timeout=4).status==200 else 1)"]

ENTRYPOINT ["/usr/local/bin/pcbr-api-entrypoint"]
# Managed hosts inject the listening port; default to 8000 for Compose and local runs.
CMD ["sh", "-c", "exec uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-server-header"]
