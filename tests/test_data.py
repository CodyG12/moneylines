import pandas as pd
import pytest

from nbamodel.data import DataError, load_games, prepare_games

BASE = {"date": ["2023-10-24", "2023-10-24"], "home_team": ["den", "GSW"], "away_team": ["LAL", "PHX"],
        "home_score": [119, 104], "away_score": [107, 108]}


def test_basic_load_normalises():
    games, report = prepare_games(pd.DataFrame(BASE))
    assert list(games["home_team"]) == ["DEN", "GSW"]
    assert list(games["home_win"]) == [1.0, 0.0]
    assert list(games["season"]) == [2023, 2023]
    assert report.season_inferred and not report.has_odds


def test_season_inference_spring_game():
    raw = pd.DataFrame({**BASE, "date": ["2024-03-01", "2024-03-01"]})
    assert list(prepare_games(raw)[0]["season"]) == [2023, 2023]


@pytest.mark.parametrize("change,msg", [
    ({"home_score": [119, None]}, "only one score"),
    ({"away_score": [119, 108]}, "tied"),
    ({"away_team": ["DEN", "PHX"]}, "home_team == away_team"),
    ({"away_team": ["LAL", "LAL"]}, "same date"),
    ({"date": ["2023-10-24", "not a date"]}, "unparseable"),
])
def test_invalid_rows_rejected(change, msg):
    with pytest.raises(DataError, match=msg):
        prepare_games(pd.DataFrame({**BASE, **change}))


def test_missing_columns_rejected():
    with pytest.raises(DataError, match="missing required"):
        prepare_games(pd.DataFrame({"date": ["2023-10-24"]}))


def test_missing_file():
    with pytest.raises(DataError, match="not found"):
        load_games("/nonexistent/games.csv")


def test_scheduled_games_allowed():
    raw = pd.DataFrame({**BASE, "home_score": [119, None], "away_score": [107, None]})
    games, report = prepare_games(raw)
    assert report.n_played == 1 and report.n_scheduled == 1


def test_american_odds_converted_and_untimed_odds_flagged():
    raw = pd.DataFrame({**BASE, "home_moneyline": [-150, 120], "away_moneyline": [130, None]})
    games, report = prepare_games(raw)
    assert games.loc[0, "home_odds"] == pytest.approx(1 + 100 / 150)
    assert pd.isna(games.loc[1, "home_odds"])  # incomplete market -> treated as missing
    assert report.n_odds_missing == 1
    assert not report.odds_timing_verified
    assert any("odds_timestamp" in w for w in report.warnings)


def test_odds_observed_after_game_are_dropped():
    raw = pd.DataFrame({**BASE, "home_odds": [1.5, 2.0], "away_odds": [2.6, 1.9],
                        "odds_timestamp": ["2023-10-24T12:00:00Z", "2023-10-25T02:00:00Z"]})
    games, report = prepare_games(raw)
    assert games["home_odds"].notna().tolist() == [True, False]
    assert report.n_odds_after_game_dropped == 1
