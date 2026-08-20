FROM python:3.14-slim AS runtime

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
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN groupadd --gid 10003 pcbr \
    && useradd --uid 10003 --gid 10003 --no-create-home --shell /usr/sbin/nologin pcbr \
    && mkdir -p /mlartifacts \
    && chown -R pcbr:pcbr /app /mlartifacts

COPY --chown=pcbr:pcbr pyproject.toml uv.lock README.md ./
COPY --chown=pcbr:pcbr packages/core ./packages/core
COPY --chown=pcbr:pcbr infra/entrypoints/mlflow.sh /usr/local/bin/pcbr-mlflow-entrypoint
# MLflow is no longer a locked extra: every release caps cryptography below 50,
# which would hold the project lock on a vulnerable version. The tracking server
# installs it here instead, isolated from the API and pipeline images.
RUN uv sync --locked --no-dev \
    && uv pip install "mlflow>=2.19,<4"

RUN chmod 0555 /usr/local/bin/pcbr-mlflow-entrypoint

USER 10003:10003

VOLUME ["/mlartifacts"]
EXPOSE 5000

HEALTHCHECK --interval=60s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import sys; sys.exit(0)"]

ENTRYPOINT ["/usr/local/bin/pcbr-mlflow-entrypoint"]
CMD ["server"]
