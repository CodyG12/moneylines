"""Converters from provider formats into the canonical games CSV schema."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def team_logs_to_games(logs: pd.DataFrame) -> pd.DataFrame:
    """Convert nba_api LeagueGameFinder team rows (two per game) into one row per game.

    MATCHUP is "AAA vs. BBB" for the home team and "AAA @ BBB" for the away team.
    Neutral-site games (international games, NBA Cup knockout rounds in Las Vegas)
    keep the league's nominal home/away designation, which overstates home edge.
    """
    needed = {"SEASON_ID", "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS"}
    missing = needed - set(logs.columns)
    if missing:
        raise ValueError(f"team logs missing columns: {sorted(missing)}")
    home = logs[logs["MATCHUP"].str.contains(" vs. ", regex=False)]
    away = logs[logs["MATCHUP"].str.contains(" @ ", regex=False)]
    merged = home.merge(away[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]], on="GAME_ID", suffixes=("_h", "_a"))
    unpaired = logs["GAME_ID"].nunique() - merged["GAME_ID"].nunique()
    if unpaired:
        log.warning("%d games could not be paired into home/away rows and were skipped", unpaired)
    games = pd.DataFrame({
        "date": pd.to_datetime(merged["GAME_DATE"]).dt.date.astype(str),
        "season": merged["SEASON_ID"].astype(str).str[1:].astype(int),
        "home_team": merged["TEAM_ABBREVIATION_h"],
        "away_team": merged["TEAM_ABBREVIATION_a"],
        "home_score": merged["PTS_h"],
        "away_score": merged["PTS_a"],
        "game_id_source": merged["GAME_ID"],
    })
    return games.sort_values(["date", "home_team"]).reset_index(drop=True)
