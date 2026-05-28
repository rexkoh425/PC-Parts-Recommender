FROM python:3.12-slim AS runtime

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
RUN uv sync --locked --no-dev --extra mlops

RUN chmod 0555 /usr/local/bin/pcbr-mlflow-entrypoint

USER 10003:10003

VOLUME ["/mlartifacts"]
EXPOSE 5000

ENTRYPOINT ["/usr/local/bin/pcbr-mlflow-entrypoint"]
CMD ["server"]
