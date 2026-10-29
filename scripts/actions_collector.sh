#!/bin/bash
# Hosted collector, run by .github/workflows/collector.yml (one run at a time). Restores the
# encrypted state, then every 5 minutes:
#   1. once a day after 10:00 UTC (all NBA games are final by then): takes scores, tip-off times and
#      player box scores from ESPN (results-update), then reconciles (update-results --retrain-if-due),
#   2. ingests new official injury reports,
#   3. collects odds if a snapshot target is due, then logs predictions and paper decisions,
#   4. pushes the state branch when anything changed.
# It stops at its time budget (the next queued run takes over) or as soon as no snapshot target
# opens within 6 hours. Logs: logs/actions/ (inside the state, so `make pull` brings them to the Mac).
set -uo pipefail
cd "$(dirname "$0")/.."
NBA=(.venv/bin/python -W ignore -m nbamodel)
SYNC=scripts/state_sync.sh
LOG=logs/actions
stop=$(( $(date +%s) + ${COLLECTOR_SECONDS:-20400} ))   # 5h40m, inside the job's 6h limit
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

$SYNC pull-state || exit 1
# The games file lives in the state. Before it did, the Mac pushed it to the results branch.
if [ ! -f data/raw/nba_games.csv ] && ! $SYNC pull-results >/dev/null; then
  echo "$(stamp) no games file in the state or on the results branch; cannot collect" >&2
  exit 1
fi
mkdir -p "$LOG"
last_failed=0
while :; do
  changed=0
  # Daily reconcile, retried at most once an hour.
  done_today="$LOG/.reconciled-$(date -u +%Y-%m-%d)"
  if [ "$(date -u +%H)" -ge 10 ] && [ ! -e "$done_today" ] && [ $(( $(date +%s) - last_failed )) -ge 3600 ]; then
    changed=1
    echo "=== $(stamp) daily reconcile" >> "$LOG/reconcile.log"
    if "${NBA[@]}" results-update >> "$LOG/reconcile.log" 2>&1 \
       && "${NBA[@]}" update-results --retrain-if-due >> "$LOG/reconcile.log" 2>&1; then
      find "$LOG" -name '.reconciled-*' -mtime +7 -delete 2>/dev/null
      touch "$done_today"
      [ -n "${RECONCILED_FLAG:-}" ] && touch "$RECONCILED_FLAG"
    else
      last_failed=$(date +%s)
    fi
  fi

  # Official injury reports (free; new files only). Each new report is a change to keep.
  inj=$("${NBA[@]}" injuries-refresh 2>&1)
  case "$inj" in
    "injury reports: no new reports") ;;
    "injury reports: refresh failed"*) printf '=== %s\n%s\n' "$(stamp)" "$inj" >> "$LOG/injuries.log" ;;
    *) changed=1; printf '=== %s\n%s\n' "$(stamp)" "$inj" >> "$LOG/injuries.log" ;;
  esac

  out=$("${NBA[@]}" collect-odds --then-predict 2>&1)
  case "$out" in
    "no snapshot due"*) ;;
    *) changed=1; printf '=== %s\n%s\n' "$(stamp)" "$out" >> "$LOG/odds_collect.log" ;;
  esac

  if [ "$changed" = 1 ] && ! $SYNC push-state >/dev/null; then
    echo "$(stamp) state push failed" >&2
    exit 1
  fi

  [ "$(date +%s)" -ge "$stop" ] && { echo "$(stamp) time budget used; the next run takes over"; break; }
  if ! next=$("${NBA[@]}" odds-next --within-hours 6); then
    echo "$(stamp) idle: $next"
    break
  fi
  sleep $(( 300 - $(date +%s) % 300 ))                 # next 5-minute boundary
done
