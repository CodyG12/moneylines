#!/usr/bin/env python3
"""Download NBA regular-season results via the unofficial `nba_api` package.

    pip install -r requirements-fetch.txt
    python scripts/fetch_nba_api.py --seasons 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 \
        --out data/raw/nba_games.csv

stats.nba.com is not an official public API. Its terms restrict use to personal,
non-commercial purposes; keep requests slow and do not redistribute the data.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nbamodel.sources import team_logs_to_games  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", required=True, help="e.g. 2022-23 2023-24")
    p.add_argument("--season-type", default="Regular Season")
    p.add_argument("--out", required=True)
    p.add_argument("--sleep", type=float, default=2.0, help="seconds between requests")
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
        frames.append(finder.get_data_frames()[0])
        time.sleep(args.sleep)
    games = team_logs_to_games(pd.concat(frames, ignore_index=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(out, index=False)
    print(f"wrote {len(games)} games to {out}")


if __name__ == "__main__":
    main()
