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

stamp="logs/.reconciled-$(date -u +%Y-%m-%d)"
if [ "$(date -u +%H)" -ge 10 ] && [ ! -e "$stamp" ]; then
  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) daily reconcile"
    make -s data-update && make -s retrain && make -s dashboard && touch "$stamp"
  } >> logs/reconcile.log 2>&1
fi
