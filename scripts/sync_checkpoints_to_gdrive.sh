#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Mirror checkpoints/ to Google Drive via rclone — runs OUTSIDE training.
#
# A lightweight polling daemon: every SYNC_INTERVAL seconds (default 2h) it
# checks whether anything under checkpoints/ has changed and, if so, runs
#
#     rclone copy <CKPT_DIR> <RCLONE_REMOTE>
#
# `rclone copy` (not `sync`) never deletes on the destination, so a checkpoint
# caught half-written is simply completed on a later pass. To avoid uploading
# files that are still being written, BOTH the change detection and rclone
# itself ignore files younger than MIN_AGE_MIN minutes.
#
# This is intentionally decoupled from train_codenames_dapo.sh: start it once,
# leave it running, and it will pick up every checkpoint the trainer drops.
#
# Usage:
#   scripts/sync_checkpoints_to_gdrive.sh start    # launch daemon (background)
#   scripts/sync_checkpoints_to_gdrive.sh stop     # final sync, then stop daemon
#   scripts/sync_checkpoints_to_gdrive.sh status   # running? last sync time?
#   scripts/sync_checkpoints_to_gdrive.sh once     # one sync now, then exit
#   scripts/sync_checkpoints_to_gdrive.sh run      # run the loop in foreground
#
# `stop` signals the daemon to run ONE last sync (covering the final
# checkpoint) before exiting, and blocks until that upload finishes.
#
# Tunables (env vars):
#   SYNC_INTERVAL   seconds between checks             (default 7200 = 2h)
#   CKPT_DIR        local checkpoints dir              (default <repo>/checkpoints)
#   RCLONE_REMOTE   rclone destination                 (default gdrive-... below)
#   MIN_AGE_MIN     ignore files younger than N min    (default 2)
#   STOP_TIMEOUT    max seconds to wait for final sync (default 1800 = 30m)
#   RCLONE_FLAGS    extra flags passed to `rclone copy`
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

SYNC_INTERVAL="${SYNC_INTERVAL:-7200}"                       # 2 hours
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive-distant-association:distant-association-drive/checkpoints}"
MIN_AGE_MIN="${MIN_AGE_MIN:-2}"
STOP_TIMEOUT="${STOP_TIMEOUT:-1800}"                         # cap on final-sync wait
# Log-friendly flags (the interactive `-P` spams the logfile with \r updates).
RCLONE_FLAGS="${RCLONE_FLAGS:---transfers 4 --checkers 8 --stats 5m --stats-one-line -v}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
SYNC_LOG="${SYNC_LOG:-${LOG_DIR}/checkpoint_sync.log}"
SYNC_PIDFILE="${SYNC_PIDFILE:-${LOG_DIR}/checkpoint_sync.pid}"
SYNC_STATE="${SYNC_STATE:-${LOG_DIR}/.checkpoint_sync_state}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Newest mtime (epoch seconds) among checkpoint files older than MIN_AGE_MIN.
# Prints nothing when there is no settled file to sync.
newest_settled_mtime() {
  [ -d "${CKPT_DIR}" ] || return 0
  find "${CKPT_DIR}" -type f -mmin "+${MIN_AGE_MIN}" -printf '%T@\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d. -f1
}

# Run rclone copy. Optional $1 overrides MIN_AGE_MIN — used for the final
# shutdown flush, when training has stopped and even fresh files are safe.
do_sync() {
  local min_age="${1:-${MIN_AGE_MIN}}"
  log "syncing ${CKPT_DIR} -> ${RCLONE_REMOTE} (min-age=${min_age}m)"
  if rclone copy "${CKPT_DIR}" "${RCLONE_REMOTE}" \
        --min-age "${min_age}m" ${RCLONE_FLAGS}; then
    log "sync OK"
    return 0
  fi
  local rc=$?
  log "sync FAILED (rclone exit ${rc}) — will retry next cycle"
  return "${rc}"
}

# SIGTERM/SIGINT handler for the daemon: `stop` sends SIGTERM, which lands
# here. Training has already ended, so sync with min-age 0 to capture the
# just-saved final checkpoint, then exit cleanly.
on_term() {
  log "received stop signal — running final sync"
  kill "${SLEEP_PID:-}" 2>/dev/null || true
  do_sync 0 || true
  log "daemon exiting"
  exit 0
}

cmd_run() {
  log "checkpoint-sync daemon up: interval=${SYNC_INTERVAL}s  remote=${RCLONE_REMOTE}"
  trap on_term TERM INT
  local last="" cur=""
  [ -f "${SYNC_STATE}" ] && last="$(cat "${SYNC_STATE}")"
  while true; do
    cur="$(newest_settled_mtime || true)"
    if [ -z "${cur}" ]; then
      log "no settled checkpoint files under ${CKPT_DIR} — nothing to do"
    elif [ "${cur}" = "${last}" ]; then
      log "no change since last sync (mtime=${cur}) — skipping"
    else
      log "change detected (mtime ${last:-none} -> ${cur})"
      if do_sync; then
        last="${cur}"
        echo "${cur}" > "${SYNC_STATE}"
      fi
    fi
    # Background the sleep and wait on it, so a stop signal interrupts the
    # wait immediately instead of stalling for the whole interval.
    sleep "${SYNC_INTERVAL}" &
    SLEEP_PID=$!
    wait "${SLEEP_PID}" || true
  done
}

cmd_once() {
  if do_sync; then
    local cur; cur="$(newest_settled_mtime || true)"
    [ -n "${cur}" ] && echo "${cur}" > "${SYNC_STATE}"
    log "one-shot sync complete"
  else
    exit 1
  fi
}

cmd_start() {
  if [ -f "${SYNC_PIDFILE}" ] && kill -0 "$(cat "${SYNC_PIDFILE}")" 2>/dev/null; then
    echo "[sync] already running, pid=$(cat "${SYNC_PIDFILE}")"
    exit 0
  fi
  local remote_name="${RCLONE_REMOTE%%:*}"
  if ! rclone listremotes 2>/dev/null | grep -q "^${remote_name}:"; then
    echo "[sync] ERROR: rclone remote '${remote_name}:' not found." >&2
    echo "[sync]   configure it with: rclone config" >&2
    exit 1
  fi
  nohup bash "${BASH_SOURCE[0]}" run >> "${SYNC_LOG}" 2>&1 &
  echo $! > "${SYNC_PIDFILE}"
  echo "[sync] started, pid=$(cat "${SYNC_PIDFILE}")  log=${SYNC_LOG}"
}

cmd_stop() {
  if [ ! -f "${SYNC_PIDFILE}" ]; then
    echo "[sync] not running (no pidfile)"
    return 0
  fi
  local pid; pid="$(cat "${SYNC_PIDFILE}")"
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[sync] not running (stale pidfile)"
    rm -f "${SYNC_PIDFILE}"
    return 0
  fi
  echo "[sync] stopping pid=${pid} — it runs one final sync first..."
  kill "${pid}" 2>/dev/null || true
  # Block until the daemon's final sync completes (capped at STOP_TIMEOUT).
  local waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    if [ "${waited}" -ge "${STOP_TIMEOUT}" ]; then
      echo "[sync] final sync exceeded ${STOP_TIMEOUT}s — sending SIGKILL" >&2
      kill -9 "${pid}" 2>/dev/null || true
      break
    fi
    if [ "${waited}" -gt 0 ] && [ $(( waited % 60 )) -eq 0 ]; then
      echo "[sync]   ...still finishing final sync (${waited}s elapsed)"
    fi
    sleep 2
    waited=$(( waited + 2 ))
  done
  echo "[sync] daemon stopped (pid=${pid})"
  rm -f "${SYNC_PIDFILE}"
}

cmd_status() {
  if [ -f "${SYNC_PIDFILE}" ] && kill -0 "$(cat "${SYNC_PIDFILE}")" 2>/dev/null; then
    echo "[sync] RUNNING  pid=$(cat "${SYNC_PIDFILE}")"
  else
    echo "[sync] not running"
  fi
  if [ -f "${SYNC_STATE}" ]; then
    local m; m="$(cat "${SYNC_STATE}")"
    echo "[sync] last synced checkpoint mtime: $(date -d "@${m}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "${m}")"
  fi
  echo "[sync] log: ${SYNC_LOG}"
}

case "${1:-}" in
  start)  cmd_start  ;;
  stop)   cmd_stop   ;;
  status) cmd_status ;;
  once)   cmd_once   ;;
  run)    cmd_run    ;;
  *) echo "usage: $0 {start|stop|status|once|run}" >&2; exit 1 ;;
esac
