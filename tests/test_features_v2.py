import numpy as np
import pandas as pd
import pytest

from nbamodel.features import FEATURES, FEATURES_V2, FORM_PRIOR, build_features, estimate_possessions
from nbamodel.leakage import (LeakageError, assert_no_leakage, find_leaking_features, perturb_outcomes,
                              suspicious_label_correlation)
from tests.conftest import tiny


def test_phase2_features_have_no_lookahead(multi_season_games):
    leaks = find_leaking_features(multi_season_games, build_features, FEATURES_V2 + FEATURES)
    assert leaks == {}


def test_deliberately_leaky_feature_is_caught(multi_season_games):
    """A feature built from a game's own final margin must be flagged by the validator."""
    def leaky(g):
        f = build_features(g)
        f["leak_margin"] = (g["home_score"] - g["away_score"]).fillna(0)
        return f
    leaks = find_leaking_features(multi_season_games, leaky, FEATURES_V2 + ["leak_margin"])
    assert set(leaks) == {"leak_margin"}
    with pytest.raises(LeakageError):
        assert_no_leakage(multi_season_games, leaky, ["leak_margin"])


def test_future_rolling_feature_is_caught(multi_season_games):
    """A 'rolling' feature that accidentally includes the current game (shift off by one) is caught."""
    def off_by_one(g):
        f = build_features(g)
        m = (g["home_score"] - g["away_score"]).fillna(0)
        f["bad_roll"] = m.groupby(g["home_team"]).transform(lambda s: s.rolling(5, min_periods=1).mean())
        return f
    assert "bad_roll" in find_leaking_features(multi_season_games, off_by_one, ["bad_roll"])


def test_label_copy_flagged_by_correlation_screen(multi_season_games):
    f = build_features(multi_season_games).assign(copy=lambda d: d["home_win"])
    assert "copy" in suspicious_label_correlation(f, FEATURES_V2 + ["copy"])
    assert suspicious_label_correlation(f, FEATURES_V2) == {}


def test_simultaneous_games_are_isolated():
    """Changing one game's result must not move features of other games on the same date."""
    rows = [("2023-10-24", "A", "B", 110, 100), ("2023-10-26", "A", "C", 100, 90), ("2023-10-26", "B", "D", 95, 99),
            ("2023-10-27", "C", "D", None, None)]
    g = tiny(rows)
    base = build_features(g)
    flipped = build_features(perturb_outcomes(g, g.index == 1))
    pd.testing.assert_series_equal(base.loc[2, FEATURES_V2], flipped.loc[2, FEATURES_V2], check_names=False)
    assert not np.isclose(base.loc[3, "home_form10_margin"], flipped.loc[3, "home_form10_margin"])


def test_rolling_form_excludes_current_game_and_is_shrunk():
    rows = [("2023-10-24", "A", "B", 110, 100), ("2023-10-26", "A", "C", 100, 120), ("2023-10-28", "A", "D", 99, 98)]
    f = build_features(tiny(rows))
    assert f.loc[0, "home_form10_margin"] == 0.0          # nothing before the first game
    assert f.loc[1, "home_form10_margin"] == pytest.approx(10 / (1 + FORM_PRIOR))
    assert f.loc[2, "home_form10_margin"] == pytest.approx((10 - 20) / (2 + FORM_PRIOR))
    assert f.loc[2, "home_form10_winpct"] == pytest.approx((1 + 0.5 * FORM_PRIOR) / (2 + FORM_PRIOR))


def test_form5_uses_only_last_five():
    rows = [(f"2023-11-{d:02d}", "A", t, 100 + m, 100) for d, t, m in
            [(1, "B", 30), (3, "C", 1), (5, "D", 1), (7, "E", 1), (9, "F", 1), (11, "G", 1), (13, "H", 0)]]
    rows[-1] = ("2023-11-13", "A", "H", None, None)
    f = build_features(tiny(rows))
    assert f.loc[6, "home_form5_margin"] == pytest.approx(5 / (5 + FORM_PRIOR))
    assert f.loc[6, "home_form10_margin"] == pytest.approx(35 / (6 + FORM_PRIOR))


def test_home_and_road_streaks_from_schedule():
    rows = [("2023-11-01", "A", "B", 100, 90), ("2023-11-03", "A", "C", 100, 90), ("2023-11-05", "A", "B", None, None),
            ("2023-11-07", "D", "B", None, None)]
    f = build_features(tiny(rows))
    assert f["home_streak"].tolist()[:3] == [1, 2, 3]
    assert f.loc[3, "away_streak"] == 3  # B: away at A, (no C game), away at A, away at D
    assert f.loc[1, "away_streak"] == 1


def test_head_to_head_is_regularized_and_home_perspective():
    rows = [("2023-11-01", "A", "B", 130, 90), ("2023-11-10", "B", "A", None, None)]
    f = build_features(tiny(rows))
    assert f.loc[0, "h2h_resid"] == 0
    resid = 40 - (100 / 28.0)                    # actual margin minus Elo expectation (equal ratings + HCA)
    assert f.loc[1, "h2h_resid"] == pytest.approx(-resid / (1 + 4))


def test_possession_estimate_and_efficiency_use_prior_games_only():
    assert estimate_possessions(90, 10, 14, 20, 88, 12, 12, 25) == pytest.approx(0.5 * ((90 - 10 + 14 + 8.8) +
                                                                                        (88 - 12 + 12 + 11)))
    raw = pd.DataFrame({"date": ["2023-11-01", "2023-11-03"], "home_team": ["A", "A"], "away_team": ["B", "C"],
                        "home_score": [120, 100], "away_score": [100, 101], "season": [2023, 2023]})
    for side in ("home", "away"):
        raw[f"{side}_fga"], raw[f"{side}_oreb"], raw[f"{side}_tov"], raw[f"{side}_fta"] = 90, 10, 15, 25
    from nbamodel.data import prepare_games
    f = build_features(prepare_games(raw)[0])
    assert f.loc[0, "home_eff_n"] == 0 and f.loc[1, "home_eff_n"] == 1
    assert f.loc[1, "home_ortg"] > f.loc[1, "home_drtg"]   # A outscored B in its only earlier game
    assert f.loc[0, "home_off_vs_away_def"] == pytest.approx(0)


def test_missing_box_scores_leave_neutral_efficiency(synthetic_games):
    games, _ = synthetic_games
    f = build_features(games.drop(columns=[c for c in games.columns if c.endswith("_fga")]))
    assert (f["home_eff_n"] == 0).all()
    assert np.allclose(f["home_off_vs_away_def"], 0)
