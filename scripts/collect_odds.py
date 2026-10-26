#!/usr/bin/env python3
"""Append timestamped NBA moneyline snapshots from The Odds API to a CSV archive.

Current odds (free plan, 1 credit per call). Run this on a schedule, e.g. a
few times on each game day, to build your own timestamped archive:
    python scripts/collect_odds.py --out data/raw/odds_snapshots.csv

Historical backfill (paid plan only, 10 credits per call, from June 2020).
One snapshot per date at each --time (UTC):
    python scripts/collect_odds.py --out data/raw/odds_snapshots.csv \
        --historical --start 2024-10-22 --end 2025-04-13 --time 22:00

The key is read from ODDS_API_KEY or a local .env file (see .env.example).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nbamodel.odds_api import OddsApiError, fetch_current, fetch_historical, load_api_key  # noqa: E402


def append(out: Path, rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out, dtype=str)
        rows = rows.astype(str)
        keys = ["observed_at", "event_id", "book"]
        rows = rows[~rows.set_index(keys).index.isin(old.set_index(keys).index)]
    rows.to_csv(out, mode="a", header=not out.exists(), index=False)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--regions", default="us")
    p.add_argument("--bookmakers", help="comma-separated book keys (overrides --regions)")
    p.add_argument("--historical", action="store_true")
    p.add_argument("--start", help="first date (YYYY-MM-DD) for --historical")
    p.add_argument("--end", help="last date (YYYY-MM-DD) for --historical")
    p.add_argument("--time", nargs="+", default=["22:00"], help="UTC snapshot time(s) for --historical")
    p.add_argument("--sleep", type=float, default=1.0)
    args = p.parse_args()

    out = Path(args.out)
    try:
        key = load_api_key()
        if not args.historical:
            rows, usage = fetch_current(key, args.regions, args.bookmakers)
            n = append(out, rows)
            print(f"appended {n} rows ({rows['event_id'].nunique() if n else 0} games); usage {usage}")
            return
        if not (args.start and args.end):
            sys.exit("--historical needs --start and --end")
        for day in pd.date_range(args.start, args.end, freq="D"):
            for t in args.time:
                at = f"{day.date()}T{t}:00Z"
                rows, usage = fetch_historical(key, at, args.regions, args.bookmakers)
                n = append(out, rows)
                print(f"{at}: appended {n} rows; remaining credits {usage.get('x-requests-remaining')}")
                time.sleep(args.sleep)
    except OddsApiError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
