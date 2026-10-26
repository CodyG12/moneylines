import pandas as pd
import pytest

from nbamodel.data import prepare_games
from nbamodel.features import FEATURES, REST_CAP, build_features
from tests.conftest import tiny


def test_features_unchanged_when_current_and_future_results_change(synthetic_raw):
    """Core leakage test: scrambling every result on/after a cutoff date must
    not change any feature for games on the cutoff date."""
    games, _ = prepare_games(synthetic_raw)
    dates = games["date"].sort_values().unique()
    cutoff = dates[len(dates) // 2]
    base = build_features(games)

    tampered = games.copy()
    later = tampered["date"] >= cutoff
    tampered.loc[later, ["home_score", "away_score"]] = tampered.loc[later, ["away_score", "home_score"]].values
    tampered.loc[later, "home_win"] = 1 - tampered.loc[later, "home_win"]
    after = build_features(tampered)

    on_cutoff = games["date"] == cutoff
    pd.testing.assert_frame_equal(base.loc[on_cutoff, FEATURES], after.loc[on_cutoff, FEATURES])
    # sanity: the tampering does change later features
    assert not base.loc[games["date"] > cutoff, "elo_diff"].equals(after.loc[games["date"] > cutoff, "elo_diff"])


def test_rest_days_back_to_back_and_form():
    rows = [("2023-10-24", "A", "B", 110, 100), ("2023-10-25", "A", "C", 100, 120),
            ("2023-10-28", "B", "A", None, None)]
    f = build_features(tiny(rows), window=10)
    assert f.loc[0, "home_days_rest"] == REST_CAP           # first game of season
    assert f.loc[1, "home_days_rest"] == 1 and f.loc[1, "home_b2b"] == 1.0
    assert f.loc[1, "home_form_margin"] == 10 and f.loc[1, "home_form_winpct"] == 1.0
    # scheduled game: A's form covers both played games; B's covers one
    assert f.loc[2, "away_form_margin"] == pytest.approx((10 - 20) / 2)
    assert f.loc[2, "home_form_margin"] == -10
    assert f.loc[2, "away_days_rest"] == 3


def test_form_resets_each_season():
    rows = [("2023-03-01", "A", "B", 130, 90), ("2023-11-01", "A", "B", 100, 99)]
    f = build_features(tiny(rows))
    assert f.loc[1, "home_form_n"] == 0 and f.loc[1, "home_form_margin"] == 0
    assert f.loc[1, "home_history"] == 1  # career history count still accumulates
