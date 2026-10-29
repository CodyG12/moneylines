#!/bin/bash
# Removes the retired Mac results-push LaunchAgent (com.moneylines.results). Results now come from
# ESPN inside the hosted collector, so the Mac has no scheduled job. Safe to run more than once.
#   scripts/install_results_schedule.sh --uninstall
# Back to local collection instead: scripts/install_odds_schedule.sh
set -euo pipefail
LABEL=com.moneylines.results
if [ "${1:-}" != "--uninstall" ]; then
  echo "the results-push job is retired (results come from ESPN in the hosted collector)." >&2
  echo "run: $0 --uninstall" >&2
  exit 1
fi
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "removed $LABEL"
