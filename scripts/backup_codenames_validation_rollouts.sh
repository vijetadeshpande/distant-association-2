#!/usr/bin/env bash

set -uo pipefail

readonly SOURCE_DIR="/home/dpsk_a2a/distant-association-2/outputs/codenames_validation_rollouts"
readonly DESTINATION="gdrive-distant-association:distant-association-drive/outputs/codenames_validation_rollouts"
readonly RCLONE_CONFIG="/root/.config/rclone/rclone.conf"
readonly LOCK_FILE="/tmp/codenames_validation_rollouts_backup.lock"
readonly PID_FILE="/tmp/codenames_validation_rollouts_backup.pid"
readonly INTERVAL_SECONDS="${BACKUP_INTERVAL_SECONDS:-1800}"

run_backup() {
    printf '[%s] Starting backup\n' "$(date --iso-8601=seconds)"

    if /usr/bin/rclone copy \
        "$SOURCE_DIR" \
        "$DESTINATION" \
        --config "$RCLONE_CONFIG" \
        --stats 30s \
        --stats-one-line; then
        printf '[%s] Backup completed successfully\n' "$(date --iso-8601=seconds)"
    else
        status=$?
        printf '[%s] Backup failed with exit code %s\n' "$(date --iso-8601=seconds)" "$status"
    fi
}

if [[ "${1:-}" == "--once" ]]; then
    run_backup
    exit
fi

exec 9>"$LOCK_FILE"
if ! /usr/bin/flock -n 9; then
    printf '[%s] Another backup watcher is already running\n' "$(date --iso-8601=seconds)"
    exit 0
fi

printf '%s\n' "$$" >"$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

while true; do
    run_backup
    sleep "$INTERVAL_SECONDS"
done
