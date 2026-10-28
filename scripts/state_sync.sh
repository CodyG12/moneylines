#!/bin/bash
# Encrypted state for hosted collection, kept on two single-writer orphan branches. The repo is
# public, so only ciphertext is ever pushed.
#   state:   written only by the GitHub Actions collector: data/predictions.db, model artifacts, the
#            walk-forward reference report, the latest upcoming-predictions CSV and logs/actions/.
#   results: written only by the Mac (stats.nba.com blocks GitHub's runners): data/raw/nba_games.csv.
# Each push replaces the branch with one new commit, so the branches never accumulate history.
#
#   scripts/state_sync.sh push-state     pack + encrypt + force-push the state branch
#   scripts/state_sync.sh pull-state     fetch + decrypt + unpack it over the working copy
#   scripts/state_sync.sh push-results   encrypt + force-push data/raw/nba_games.csv
#   scripts/state_sync.sh pull-results   fetch + decrypt it; prints the results commit id
#
# Key: STATE_KEY from the environment, else from .env (the same value as the STATE_KEY repo secret).
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${STATE_REMOTE:-origin}
PY=${PY:-.venv/bin/python}
STATE_PATHS=(artifacts out/walkforward/report.json out/upcoming_predictions.csv logs/actions)

if [ -z "${STATE_KEY:-}" ] && [ -f .env ]; then
  STATE_KEY=$(sed -n 's/^STATE_KEY=//p' .env | tail -1 | tr -d "'\"")
fi
[ -n "${STATE_KEY:-}" ] || { echo "STATE_KEY is not set (environment or .env)" >&2; exit 2; }
export STATE_KEY

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

crypt() {  # crypt -e|-d IN OUT
  openssl enc "$1" -aes-256-cbc -md sha256 -pbkdf2 -iter 200000 -salt -pass env:STATE_KEY -in "$2" -out "$3"
}

publish() {  # publish BRANCH FILE NAME
  local blob tree commit
  blob=$(git hash-object -w "$2")
  tree=$(printf '100644 blob %s\t%s\n' "$blob" "$3" | git mktree)
  commit=$(git commit-tree "$tree" -m "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)")
  git push -q -f "$REMOTE" "$commit:refs/heads/$1"
  echo "$commit"
}

fetch() {  # fetch BRANCH NAME OUT
  if ! git fetch -q "$REMOTE" "+refs/heads/$1:refs/remotes/$REMOTE/$1" 2>/dev/null; then
    echo "no $1 branch on $REMOTE yet (seed it from the Mac with: make seed-state)" >&2
    exit 3
  fi
  git cat-file blob "$REMOTE/$1:$2" > "$3"
}

case "${1:-}" in
  push-state)
    mkdir -p "$tmp/db/data"
    # Consistent single-file copy (the live file may be in WAL mode with a -wal side file).
    "$PY" - "$tmp/db/data/predictions.db" <<'PY'
import sqlite3, sys
src, dst = sqlite3.connect("data/predictions.db"), sqlite3.connect(sys.argv[1])
src.backup(dst)
dst.execute("PRAGMA journal_mode=DELETE")
dst.close(); src.close()
PY
    present=()
    for p in "${STATE_PATHS[@]}"; do [ -e "$p" ] && present+=("$p"); done
    tar -czf "$tmp/state.tgz" -C "$tmp/db" data/predictions.db -C "$PWD" ${present[@]+"${present[@]}"}
    crypt -e "$tmp/state.tgz" "$tmp/state.enc"
    publish state "$tmp/state.enc" state.tar.gz.enc
    ;;
  pull-state)
    if command -v lsof >/dev/null && lsof data/predictions.db >/dev/null 2>&1; then
      echo "data/predictions.db is open in another program (DBCode?); disconnect it and rerun" >&2
      exit 4
    fi
    fetch state state.tar.gz.enc "$tmp/state.enc"
    crypt -d "$tmp/state.enc" "$tmp/state.tgz"
    tar -tzf "$tmp/state.tgz" >/dev/null           # verify before touching the working copy
    rm -f data/predictions.db-wal data/predictions.db-shm
    tar -xzf "$tmp/state.tgz"
    echo "restored state from $(git rev-parse --short "$REMOTE/state") ($(git log -1 --format=%s "$REMOTE/state"))"
    ;;
  push-results)
    crypt -e data/raw/nba_games.csv "$tmp/games.enc"
    publish results "$tmp/games.enc" nba_games.csv.enc
    ;;
  pull-results)
    fetch results nba_games.csv.enc "$tmp/games.enc"
    crypt -d "$tmp/games.enc" "$tmp/games.csv"
    mkdir -p data/raw
    mv "$tmp/games.csv" data/raw/nba_games.csv
    git rev-parse "$REMOTE/results"
    ;;
  *)
    echo "usage: $0 push-state|pull-state|push-results|pull-results" >&2
    exit 1
    ;;
esac
