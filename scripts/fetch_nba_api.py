#!/usr/bin/env python3
"""Download NBA regular-season results via the unofficial `nba_api` package.

    pip install -r requirements-fetch.txt
    python scripts/fetch_nba_api.py --seasons 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 \
        --out data/raw/nba_games.csv

Each game keeps team box-score totals (FGA, FTA, OREB, TOV, ...) so possessions and
offensive/defensive ratings can be estimated. Where the schedule endpoint covers a
season (2019-20 onward in practice), tip-off times and neutral-site flags are added
from it; older seasons keep blank tip-off times. `fetched_at` records when the rows
were downloaded.

stats.nba.com is not an official public API. Its terms restrict use to personal,
non-commercial purposes; keep requests slow and do not redistribute the data.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nbamodel.sources import merge_schedule_info, schedule_to_games, team_logs_to_games  # noqa: E402


def fetch_schedule(season: str):
    from nba_api.stats.endpoints import scheduleleaguev2
    try:
        sched = scheduleleaguev2.ScheduleLeagueV2(season=season, league_id="00", timeout=60).get_data_frames()[0]
    except Exception as exc:  # the endpoint has no data for older seasons
        print(f"  no schedule for {season} ({type(exc).__name__}); tip-off times left blank")
        return None
    return schedule_to_games(sched)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", required=True, help="e.g. 2022-23 2023-24")
    p.add_argument("--season-type", default="Regular Season")
    p.add_argument("--out", required=True)
    p.add_argument("--sleep", type=float, default=2.0, help="seconds between requests")
    p.add_argument("--no-schedule", action="store_true", help="skip the tip-off time / neutral-site lookup")
    args = p.parse_args()

    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        sys.exit("nba_api is not installed: pip install -r requirements-fetch.txt")

    frames = []
    for season in args.seasons:
        print(f"fetching {season} ...")
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season, season_type_nullable=args.season_type,
            league_id_nullable="00", timeout=60)
        games = team_logs_to_games(finder.get_data_frames()[0])
        time.sleep(args.sleep)
        if not args.no_schedule:
            sched = fetch_schedule(season)
            if sched is not None:
                games = merge_schedule_info(games, sched)
            time.sleep(args.sleep)
        frames.append(games)
    games = pd.concat(frames, ignore_index=True).sort_values(["date", "home_team"])
    games["fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(out, index=False)
    print(f"wrote {len(games)} games to {out}")


if __name__ == "__main__":
    main()
