import pytest

from nbamodel.elo import EloConfig, compute_elo, expected_score
from tests.conftest import tiny


def test_expected_score_symmetry():
    assert expected_score(0) == pytest.approx(0.5)
    assert expected_score(100) + expected_score(-100) == pytest.approx(1.0)


def test_pre_game_ratings_ignore_own_and_same_day_results():
    rows = [("2023-10-24", "A", "B", 110, 100), ("2023-10-24", "C", "D", 90, 120),
            ("2023-10-25", "A", "C", 100, 95)]
    elo, _ = compute_elo(tiny(rows))
    assert (elo.loc[[0, 1], ["home_elo_pre", "away_elo_pre"]] == 1500).all().all()
    flipped = [rows[0], ("2023-10-24", "C", "D", 120, 90), rows[2]]
    elo2, _ = compute_elo(tiny(flipped))
    # Same-day result of another game cannot affect A-B, and A's pre-rating
    # for 10-25 depends only on A's own earlier game.
    assert elo2.loc[0, "home_elo_pre"] == elo.loc[0, "home_elo_pre"]
    assert elo2.loc[2, "home_elo_pre"] == elo.loc[2, "home_elo_pre"]
    assert elo.loc[2, "home_elo_pre"] > 1500  # A won its first game


def test_ratings_are_zero_sum_and_scheduled_games_do_not_update():
    rows = [("2023-10-24", "A", "B", 110, 100), ("2023-10-26", "B", "A", None, None)]
    _, ratings = compute_elo(tiny(rows))
    assert ratings["A"] + ratings["B"] == pytest.approx(3000)
    assert ratings["A"] > 1500


def test_season_regression_to_mean():
    rows = [("2023-03-01", "A", "B", 130, 90), ("2023-11-01", "A", "B", 100, 99)]
    cfg = EloConfig(season_carryover=0.5)
    elo, final = compute_elo(tiny(rows), cfg)
    _, after_first = compute_elo(tiny(rows[:1]), cfg)
    assert elo.loc[1, "home_elo_pre"] == pytest.approx(0.5 * after_first["A"] + 0.5 * 1500)


def test_upsets_move_ratings_more():
    fav_wins = [("2023-10-24", "A", "B", 101, 100)]
    _, r1 = compute_elo(tiny(fav_wins), EloConfig(use_margin=False))
    upset = [("2023-10-24", "A", "B", 100, 101)]
    _, r2 = compute_elo(tiny(upset), EloConfig(use_margin=False))
    # home team is favoured (home advantage), so losing costs more than winning gains
    assert 1500 - r2["A"] > r1["A"] - 1500
