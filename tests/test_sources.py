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


def test_box_scores_kept_and_schedule_merged():
    from nbamodel.sources import merge_box_scores, merge_schedule_info
    logs = pd.DataFrame({
        "SEASON_ID": ["22023"] * 2, "GAME_ID": ["0022300001"] * 2, "GAME_DATE": ["2023-10-24"] * 2,
        "TEAM_ABBREVIATION": ["DEN", "LAL"], "MATCHUP": ["DEN vs. LAL", "LAL @ DEN"], "PTS": [119, 107],
        "FGA": [91, 85], "FTA": [20, 17], "OREB": [11, 9], "TOV": [12, 15], "FGM": [48, 41], "FG3M": [8, 10],
        "DREB": [36, 33]})
    games = team_logs_to_games(logs)
    assert games.loc[0, ["home_fga", "away_fga", "home_tov", "away_tov"]].tolist() == [91, 85, 12, 15]
    games["game_id_source"] = games["game_id_source"].str.lstrip("0")   # as after a CSV round trip
    sched = pd.DataFrame({"game_id_source": ["0022300001"], "tipoff_utc": ["2023-10-24T23:30:00Z"],
                          "neutral_site": [1]})
    merged = merge_schedule_info(games, sched)
    assert merged.loc[0, "tipoff_utc"] == "2023-10-24T23:30:00Z" and merged.loc[0, "neutral_site"] == 1
    assert merged.loc[0, "game_id_source"] == "0022300001"
    bare = merged.drop(columns=[c for c in merged.columns if c.endswith("_fga")])
    assert merge_box_scores(bare, team_logs_to_games(logs)).loc[0, "home_fga"] == 91
    parsed = prepare_games(merged)[0]
    assert parsed.loc[0, "game_key"] == "0022300001" and parsed.loc[0, "home_fga"] == 91
