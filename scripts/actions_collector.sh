#!/bin/bash
# Hosted collector, run by .github/workflows/collector.yml (one run at a time). Restores the
# encrypted state, then every 5 minutes:
#   1. scores results the Mac pushed to the results branch (update-results --retrain-if-due),
#   2. collects odds if a snapshot target is due, then logs predictions and paper decisions,
#   3. pushes the state branch when anything changed.
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
mkdir -p "$LOG"
last_failed=0
while :; do
  changed=0
  if sha=$($SYNC pull-results); then
    # Retry a failed reconcile at most once an hour.
    if [ "$sha" != "$(cat "$LOG/results_sha" 2>/dev/null)" ] && [ $(( $(date +%s) - last_failed )) -ge 3600 ]; then
      changed=1
      echo "=== $(stamp) reconcile (results ${sha:0:12})" >> "$LOG/reconcile.log"
      if "${NBA[@]}" update-results --retrain-if-due >> "$LOG/reconcile.log" 2>&1; then
        echo "$sha" > "$LOG/results_sha"
        [ -n "${RECONCILED_FLAG:-}" ] && touch "$RECONCILED_FLAG"
      else
        last_failed=$(date +%s)
      fi
    fi
  elif [ ! -f data/raw/nba_games.csv ]; then
    echo "$(stamp) no results file; cannot collect" >&2
    exit 1
  fi

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
