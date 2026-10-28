#!/bin/bash
# Mac side of hosted collection (every 5 minutes via scripts/install_results_schedule.sh).
# GitHub's runners can't reach stats.nba.com, so once a day after 10:00 UTC (all NBA games are
# final by then) this Mac refreshes the season's results and schedule and pushes them, encrypted,
# to the results branch. The hosted collector scores them on its next pass. If the Mac is asleep,
# launchd runs this on wake. Output: logs/results_push.log
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

# One attempt per hour at most, so a failing nba_api fetch is not retried every 5 minutes.
stamp="logs/.results-pushed-$(date -u +%Y-%m-%d)"
attempt="logs/.results-attempt-$(date -u +%Y-%m-%dT%H)"
if [ "$(date -u +%H)" -ge 10 ] && [ ! -e "$stamp" ] && [ ! -e "$attempt" ]; then
  touch "$attempt"
  find logs -name '.results-attempt-*' -mtime +2 -delete 2>/dev/null
  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) results push"
    make -s data-update && scripts/state_sync.sh push-results && touch "$stamp"
  } >> logs/results_push.log 2>&1
fi
