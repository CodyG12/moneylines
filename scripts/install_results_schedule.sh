#!/bin/bash
# Switch this Mac to hosted collection: remove the local odds-collection LaunchAgent and run
# scripts/results_cron.sh every 5 minutes instead (daily results push for the GitHub collector).
#   install:   scripts/install_results_schedule.sh
#   uninstall: scripts/install_results_schedule.sh --uninstall
# Back to local collection: uninstall this, then run scripts/install_odds_schedule.sh.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=com.moneylines.results
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
"$REPO/scripts/install_odds_schedule.sh" --uninstall
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [ "${1:-}" = "--uninstall" ]; then
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi
mkdir -p "$HOME/Library/LaunchAgents" "$REPO/logs"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$REPO/scripts/results_cron.sh</string></array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$REPO/logs/launchd.err</string>
</dict>
</plist>
PLIST
plutil -lint "$PLIST" >/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL: daily results push after 10:00 UTC -> $REPO/logs/results_push.log"
