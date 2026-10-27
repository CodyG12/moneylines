#!/bin/bash
# Scheduled job (every 5 minutes via scripts/install_odds_schedule.sh, or cron:
#   */5 * * * * /path/to/moneylines/scripts/collect_odds_cron.sh)
# 1. Odds: spends an API credit only when a scheduled snapshot target is due, then logs
#    predictions (and paper decisions) against the fresh prices.
# 2. Once a day after 10:00 UTC (all NBA games are final by then): refresh results and the
#    schedule, reconcile predictions, settle paper decisions, compute CLV, rebuild the dashboard.
# Output: logs/odds_collect.log and logs/reconcile.log
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
.venv/bin/python -W ignore -m nbamodel collect-odds --then-predict >> logs/odds_collect.log 2>&1

# One attempt per hour at most, so a failing nba_api fetch is not retried every 5 minutes.
stamp="logs/.reconciled-$(date -u +%Y-%m-%d)"
attempt="logs/.reconcile-attempt-$(date -u +%Y-%m-%dT%H)"
if [ "$(date -u +%H)" -ge 10 ] && [ ! -e "$stamp" ] && [ ! -e "$attempt" ]; then
  touch "$attempt"
  find logs -name '.reconcile-attempt-*' -mtime +2 -delete 2>/dev/null
  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) daily reconcile"
    make -s data-update && make -s retrain && make -s dashboard && touch "$stamp"
  } >> logs/reconcile.log 2>&1
fi
