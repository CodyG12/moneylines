#!/bin/bash
# Scheduled odds collection (install with `crontab -e`, e.g. every 5 minutes):
#   */5 * * * * /path/to/moneylines/scripts/collect_odds_cron.sh
# Each run spends an API credit only when a scheduled snapshot target is due, then logs
# predictions (and paper decisions) against the fresh prices. Output: logs/odds_collect.log
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
.venv/bin/python -W ignore -m nbamodel collect-odds --then-predict >> logs/odds_collect.log 2>&1
