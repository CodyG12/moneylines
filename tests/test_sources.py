import pandas as pd

from nbamodel.data import prepare_games
from nbamodel.sources import team_logs_to_games


def test_team_logs_to_games():
    logs = pd.DataFrame({
        "SEASON_ID": ["22023"] * 4, "GAME_ID": ["001", "001", "002", "002"],
        "GAME_DATE": ["2023-10-24"] * 4, "TEAM_ABBREVIATION": ["DEN", "LAL", "GSW", "PHX"],
        "MATCHUP": ["DEN vs. LAL", "LAL @ DEN", "GSW vs. PHX", "PHX @ GSW"], "PTS": [119, 107, 104, 108]})
    games = team_logs_to_games(logs)
    assert games[["home_team", "away_team", "home_score", "away_score"]].values.tolist() == [
        ["DEN", "LAL", 119, 107], ["GSW", "PHX", 104, 108]]
    assert games["season"].tolist() == [2023, 2023]
    prepare_games(games)  # output satisfies the canonical schema


def test_neutral_site_games_are_kept_and_flagged():
    logs = pd.DataFrame({
        "SEASON_ID": ["22024"] * 2, "GAME_ID": ["147", "147"], "GAME_DATE": ["2024-11-02"] * 2,
        "TEAM_ABBREVIATION": ["MIA", "WAS"], "MATCHUP": ["MIA @ WAS", "MIA @ WAS"], "PTS": [118, 98]})
    games = team_logs_to_games(logs)
    assert games[["home_team", "away_team", "home_score", "away_score", "neutral_site"]].values.tolist() == [
        ["WAS", "MIA", 98, 118, 1]]
    assert bool(prepare_games(games)[0].loc[0, "neutral_site"])
