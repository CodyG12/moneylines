#!/bin/bash
# Run scripts/collect_odds_cron.sh every 5 minutes with a macOS LaunchAgent (no Full Disk Access
# needed, unlike cron). Each run spends an API credit only when a snapshot target is due.
#   install:   scripts/install_odds_schedule.sh
#   uninstall: scripts/install_odds_schedule.sh --uninstall
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=com.moneylines.collect-odds
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
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
  <key>ProgramArguments</key><array><string>$REPO/scripts/collect_odds_cron.sh</string></array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$REPO/logs/launchd.err</string>
</dict>
</plist>
PLIST
plutil -lint "$PLIST" >/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL: every 5 minutes -> $REPO/logs/odds_collect.log"
