#!/usr/bin/env python3
"""Refresh one season in a games CSV from the NBA schedule (finals + upcoming games).

    python scripts/update_schedule.py --games data/raw/nba_games.csv --season 2026-27

Rows for that season are replaced by the schedule's view: completed games with
scores, and scheduled games with blank scores plus UTC tip-off times. Other
seasons are left untouched. Same nba_api / stats.nba.com terms as fetch_nba_api.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nbamodel.sources import schedule_to_games  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="existing games CSV to update in place")
    p.add_argument("--season", required=True, help="e.g. 2026-27")
    p.add_argument("--out", help="write here instead of overwriting --games")
    args = p.parse_args()
    try:
        from nba_api.stats.endpoints import scheduleleaguev2
    except ImportError:
        sys.exit("nba_api is not installed: pip install -r requirements-fetch.txt")

    sched = scheduleleaguev2.ScheduleLeagueV2(season=args.season, league_id="00", timeout=60).get_data_frames()[0]
    fresh = schedule_to_games(sched)
    season_year = int(args.season[:4])
    existing = pd.read_csv(args.games)
    kept = existing[existing["season"] != season_year]
    merged = pd.concat([kept, fresh], ignore_index=True).sort_values(["date", "home_team"])
    out = Path(args.out or args.games)
    merged.to_csv(out, index=False)
    n_final = int(fresh["home_score"].notna().sum())
    print(f"{args.season}: {n_final} final + {len(fresh) - n_final} scheduled games -> {out} "
          f"({len(merged)} rows total)")


if __name__ == "__main__":
    main()
