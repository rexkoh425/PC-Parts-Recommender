FROM quay.io/prometheuscommunity/postgres-exporter:v0.17.1 AS exporter

FROM alpine:3.22

RUN addgroup -g 10004 -S pcbr \
    && adduser -u 10004 -S -D -H -G pcbr pcbr

COPY --from=exporter /bin/postgres_exporter /bin/postgres_exporter
COPY infra/entrypoints/postgres-exporter.sh /usr/local/bin/pcbr-postgres-exporter-entrypoint
RUN chmod 0555 /bin/postgres_exporter /usr/local/bin/pcbr-postgres-exporter-entrypoint

USER 10004:10004
EXPOSE 9187
ENTRYPOINT ["/usr/local/bin/pcbr-postgres-exporter-entrypoint"]
