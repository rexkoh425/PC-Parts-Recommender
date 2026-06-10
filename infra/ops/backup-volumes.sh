#!/bin/sh
set -eu
umask 077

backup_root=${PCBR_BACKUP_ROOT:-/backups/artifacts}
case "$backup_root" in
    /backups | /backups/*) ;;
    *)
        echo "Backup root must remain below /backups: $backup_root" >&2
        exit 64
        ;;
esac

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
partial_dir="$backup_root/.${timestamp}.partial"
final_dir="$backup_root/$timestamp"
mkdir -p "$partial_dir"

tar -czf "$partial_dir/dagster-state.tar.gz" -C /dagster-state .
tar -czf "$partial_dir/mlflow-artifacts.tar.gz" -C /mlflow-artifacts .
tar -czf "$partial_dir/pipeline-artifacts.tar.gz" -C /pipeline-artifacts .

cat > "$partial_dir/manifest.json" <<EOF
{
  "created_at_utc": "$timestamp",
  "format": "tar-gzip-v1",
  "contents": ["dagster-state", "mlflow-artifacts", "pipeline-artifacts"]
}
EOF

(
    cd "$partial_dir"
    sha256sum manifest.json ./*.tar.gz > SHA256SUMS
)
mv "$partial_dir" "$final_dir"
echo "$final_dir"
