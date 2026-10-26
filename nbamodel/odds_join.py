"""Attach timestamped odds snapshots to games without look-ahead.

For each game, the quote used is the latest one observed at least
`lead_minutes` before tip-off: the price you could plausibly have bet when
deciding. Quotes from later (including live, in-game prices) are never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BEST = "best"


def attach_odds(games: pd.DataFrame, snapshots: pd.DataFrame, book: str = "draftkings",
                lead_minutes: float = 60.0) -> pd.DataFrame:
    """Return `games` with home_odds, away_odds, odds_timestamp, odds_book filled where available.

    book: a bookmaker key (e.g. "draftkings"), or "best" for the best price per
    side across all books in the chosen snapshot. "best" assumes you hold
    accounts at every book, so it is optimistic.
    Existing odds columns in `games` are replaced.
    """
    out = games.drop(columns=[c for c in ("home_odds", "away_odds", "home_moneyline", "away_moneyline",
                                          "odds_timestamp", "odds_book") if c in games.columns]).copy()
    for col in ("home_odds", "away_odds"):
        out[col] = np.nan
    out["odds_timestamp"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["odds_book"] = None
    if snapshots.empty:
        return out

    snap = snapshots.copy()
    snap["observed_at"] = pd.to_datetime(snap["observed_at"], utc=True)
    snap["commence_time"] = pd.to_datetime(snap["commence_time"], utc=True)
    snap["date"] = snap["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")
    if book != BEST:
        snap = snap[snap["book"] == book]

    # The provider's home/away can differ from the NBA's nominal home team at neutral
    # sites; index both orientations, swapping prices for the reversed one.
    flipped = snap.rename(columns={"home_team": "away_team", "away_team": "home_team",
                                   "home_price": "away_price", "away_price": "home_price"})
    both = pd.concat([snap, flipped], ignore_index=True)

    tip = (pd.to_datetime(out["tipoff_utc"], utc=True, errors="coerce")
           if "tipoff_utc" in out.columns else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"))
    lead = pd.Timedelta(minutes=lead_minutes)
    keys = out["date"].astype(str).str[:10]

    grouped = {k: g for k, g in both.groupby(["date", "home_team", "away_team"])}
    for i in out.index:
        cand = grouped.get((keys[i], out.at[i, "home_team"], out.at[i, "away_team"]))
        if cand is None:
            continue
        game_tip = tip[i] if pd.notna(tip[i]) else cand["commence_time"].iloc[0]
        cand = cand[cand["observed_at"] <= game_tip - lead]
        if cand.empty:
            continue
        latest = cand[cand["observed_at"] == cand["observed_at"].max()]
        if book == BEST:
            out.at[i, "home_odds"] = latest["home_price"].max()
            out.at[i, "away_odds"] = latest["away_price"].max()
            out.at[i, "odds_book"] = BEST
        else:
            row = latest.iloc[0]
            out.at[i, "home_odds"] = row["home_price"]
            out.at[i, "away_odds"] = row["away_price"]
            out.at[i, "odds_book"] = row["book"]
        out.at[i, "odds_timestamp"] = latest["observed_at"].iloc[0]
    return out


def attach_odds_with_close(games: pd.DataFrame, snapshots: pd.DataFrame, book: str = "draftkings",
                           lead_minutes: float = 60.0) -> pd.DataFrame:
    """attach_odds plus closing prices: the last quote observed before tip-off (lead 0).

    The decision price (home_odds/away_odds) is what a simulated bet may use; the close_* columns
    are only for measuring closing-line value after the fact.
    """
    out = attach_odds(games, snapshots, book, lead_minutes)
    close = attach_odds(games, snapshots, book, lead_minutes=0.0)
    out["close_home_odds"] = close["home_odds"].to_numpy()
    out["close_away_odds"] = close["away_odds"].to_numpy()
    out["close_odds_timestamp"] = close["odds_timestamp"].to_numpy()
    return out
