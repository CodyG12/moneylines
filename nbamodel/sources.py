"""Converters from provider formats into the canonical games CSV schema."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def team_logs_to_games(logs: pd.DataFrame) -> pd.DataFrame:
    """Convert nba_api LeagueGameFinder team rows (two per game) into one row per game.

    MATCHUP is "AAA vs. BBB" for the home team and "AAA @ BBB" for the away team.
    Neutral-site games (international games, NBA Cup knockouts in Las Vegas) come
    back with "@" on *both* rows, e.g. "MIA @ WAS" twice. Those are kept with
    neutral_site=True, using the listed (nominal) home team as home_team.
    """
    needed = {"SEASON_ID", "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS"}
    missing = needed - set(logs.columns)
    if missing:
        raise ValueError(f"team logs missing columns: {sorted(missing)}")
    logs = logs.copy()
    logs["_home_row"] = logs["MATCHUP"].str.contains(" vs. ", regex=False)
    logs["_away_row"] = logs["MATCHUP"].str.contains(" @ ", regex=False)
    logs["_nominal_home"] = logs["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[-1]
    per_game = logs.groupby("GAME_ID").agg(rows=("GAME_ID", "size"), homes=("_home_row", "sum"),
                                           aways=("_away_row", "sum"))

    regular = per_game[(per_game["rows"] == 2) & (per_game["homes"] == 1) & (per_game["aways"] == 1)].index
    neutral = per_game[(per_game["rows"] == 2) & (per_game["aways"] == 2)].index
    skipped = len(per_game) - len(regular) - len(neutral)
    if skipped:
        log.warning("%d games had an unexpected row layout and were skipped", skipped)

    # For a neutral game the row whose team equals the nominal home plays the "home" role.
    logs["_is_home"] = logs["_home_row"] | (
        logs["GAME_ID"].isin(neutral) & (logs["TEAM_ABBREVIATION"] == logs["_nominal_home"]))
    keep = logs[logs["GAME_ID"].isin(regular.union(neutral))]
    home, away = keep[keep["_is_home"]], keep[~keep["_is_home"]]
    merged = home.merge(away[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]], on="GAME_ID", suffixes=("_h", "_a"))
    if len(merged) != len(regular) + len(neutral):
        log.warning("%d games lost while pairing home/away rows",
                    len(regular) + len(neutral) - len(merged))

    games = pd.DataFrame({
        "date": pd.to_datetime(merged["GAME_DATE"]).dt.date.astype(str),
        "season": merged["SEASON_ID"].astype(str).str[1:].astype(int),
        "home_team": merged["TEAM_ABBREVIATION_h"],
        "away_team": merged["TEAM_ABBREVIATION_a"],
        "home_score": merged["PTS_h"],
        "away_score": merged["PTS_a"],
        "neutral_site": merged["GAME_ID"].isin(neutral).astype(int),
        "game_id_source": merged["GAME_ID"],
    })
    return games.sort_values(["date", "home_team"]).reset_index(drop=True)
