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


def test_schedule_to_games():
    from nbamodel.sources import schedule_to_games
    base = {"seasonYear": "2026-27", "isNeutral": False, "homeTeam_score": 0, "awayTeam_score": 0}
    rows = [
        {**base, "gameId": "0022600118", "gameDateEst": "2026-10-25T00:00:00Z", "gameDateTimeUTC": "2026-10-26T01:00:00Z",
         "gameStatus": 3, "homeTeam_teamTricode": "SAC", "awayTeam_teamTricode": "MEM",
         "homeTeam_score": 112, "awayTeam_score": 123},
        {**base, "gameId": "0022600119", "gameDateEst": "2026-10-26T00:00:00Z", "gameDateTimeUTC": "2026-10-26T23:00:00Z",
         "gameStatus": 1, "homeTeam_teamTricode": "CHA", "awayTeam_teamTricode": "WAS"},
        {**base, "gameId": "0022600189", "gameDateEst": "2026-11-07T00:00:00Z", "gameDateTimeUTC": "2026-11-07T22:00:00Z",
         "gameStatus": 1, "isNeutral": True, "homeTeam_teamTricode": "IND", "awayTeam_teamTricode": "DEN"},
        {**base, "gameId": "0012600009", "gameDateEst": "2026-10-03T00:00:00Z", "gameDateTimeUTC": "2026-10-03T23:00:00Z",
         "gameStatus": 3, "homeTeam_teamTricode": "TOR", "awayTeam_teamTricode": "MIA"},   # preseason
        {**base, "gameId": "0062600001", "gameDateEst": "2026-12-11T00:00:00Z", "gameDateTimeUTC": "2026-12-12T01:00:00Z",
         "gameStatus": 1, "homeTeam_teamTricode": None, "awayTeam_teamTricode": None},     # Cup final, TBD
    ]
    games = schedule_to_games(pd.DataFrame(rows))
    assert games["game_id_source"].tolist() == ["0022600118", "0022600119", "0022600189"]
    assert games["season"].tolist() == [2026] * 3
    assert games["home_score"].isna().tolist() == [False, True, True]
    assert games["neutral_site"].tolist() == [0, 0, 1]
    parsed, report = prepare_games(games)
    assert report.n_played == 1 and report.n_scheduled == 2
