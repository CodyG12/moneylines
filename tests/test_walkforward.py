import numpy as np
import pandas as pd
import pytest

from nbamodel.data import DataError
from nbamodel.leakage import perturb_outcomes
from nbamodel.report import champion_spec, choose_champion, evaluate_predictions
from nbamodel.walkforward import WalkForwardConfig, assign_periods, predict_frame, run_walk_forward

CFG = dict(calibration_dates=15, final_test_start="2022-08-01", models=("elo", "logistic"))


@pytest.fixture(scope="module")
def monthly(multi_season_games):
    return run_walk_forward(multi_season_games, WalkForwardConfig(retrain="monthly", **CFG))


def test_every_period_trains_strictly_before_it_predicts(monthly):
    per = monthly.periods
    assert (per["train_end"] < per["test_start"]).all()
    assert (per["cal_start"] <= per["train_end"]).all()
    assert per["test_start"].is_monotonic_increasing
    # expanding window: the training set only grows
    assert per["n_train"].is_monotonic_increasing
    preds = monthly.predictions.merge(per[["period", "train_end"]], on="period")
    assert (preds["date"] > preds["train_end"]).all()


def test_development_stops_before_final_test(monthly):
    p = monthly.predictions
    assert p["date"].max() < pd.Timestamp("2022-08-01")
    assert p["season"].min() == 2021            # first two seasons are training only
    assert p["game_key"].is_unique
    assert {"p_elo", "p_logistic", "p_logistic_raw", "p_home_base_rate"} <= set(p.columns)


def test_final_test_phase_covers_only_the_holdout(multi_season_games):
    r = run_walk_forward(multi_season_games, WalkForwardConfig(retrain="season", **CFG), phase="final_test")
    assert r.predictions["date"].min() >= pd.Timestamp("2022-08-01")
    assert (r.periods["train_end"] < pd.Timestamp("2022-08-01")).all()


def test_predictions_reproducible(multi_season_games, monthly):
    again = run_walk_forward(multi_season_games, WalkForwardConfig(retrain="monthly", **CFG))
    pd.testing.assert_frame_equal(monthly.predictions, again.predictions)


def test_future_outcomes_cannot_change_frozen_predictions(multi_season_games, monthly):
    """Flip every outcome in the last period: that period's first-day predictions are unchanged
    (later days legitimately see earlier days' results through daily feature updates)."""
    last = monthly.periods.iloc[-1]
    tampered = perturb_outcomes(multi_season_games, multi_season_games["date"] >= last["test_start"])
    other = run_walk_forward(tampered, WalkForwardConfig(retrain="monthly", **CFG))
    a = monthly.predictions[monthly.predictions["date"] == last["test_start"]]
    b = other.predictions[other.predictions["date"] == last["test_start"]]
    for c in ("p_elo", "p_logistic", "p_logistic_raw"):
        np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy())


def test_rolling_window_limits_training_seasons(multi_season_games):
    r = run_walk_forward(multi_season_games, WalkForwardConfig(retrain="season", window="rolling", rolling_seasons=1,
                                                               **CFG))
    p = r.periods.iloc[0]
    assert pd.Timestamp(p["train_start"]).year >= 2020


def test_weekly_and_season_periods():
    d = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-07", "2024-01-08", "2024-02-01"]))
    assert assign_periods(d, pd.Series([2023] * 4), "weekly").nunique() == 3
    assert assign_periods(d, pd.Series([2023] * 4), "season").nunique() == 1
    with pytest.raises(ValueError):
        assign_periods(d, pd.Series([2023] * 4), "daily")


def test_predict_frame_refuses_outcome_columns():
    with pytest.raises(DataError, match="outcome"):
        predict_frame({}, pd.DataFrame({"home_score": [1]}), 0.5)


def test_too_few_seasons_rejected(synthetic_games):
    games, _ = synthetic_games
    with pytest.raises(DataError, match="seasons"):
        run_walk_forward(games, WalkForwardConfig(**CFG))


def test_report_and_champion(monthly):
    rep = evaluate_predictions(monthly.predictions, n_boot=100)
    for c in ("p_elo", "p_logistic", "p_home_base_rate"):
        s = rep["models"][c]
        assert 0 < s["log_loss"] < 1 and 0 <= s["ece"] <= 1 and s["log_loss_ci"]["ci_low"] <= s["log_loss"]
    assert "p_logistic_minus_p_elo" in rep["comparisons"]
    assert sum(rep["confusion_matrix"]["p_elo"].values()) == len(monthly.predictions)
    ch = choose_champion(rep)
    assert ch["champion"] in rep["models"] and ch["champion"] != "p_home_base_rate"
    assert champion_spec("p_logistic_raw") == {"model": "logistic", "calibrated": False}
    assert champion_spec("p_elo") == {"model": "elo", "calibrated": False}


def test_ensemble_only_eligible_when_it_beats_both_components():
    rep = {"models": {"p_logistic": {"log_loss": 0.62}, "p_gbm": {"log_loss": 0.63},
                      "p_ensemble": {"log_loss": 0.625}}, "comparisons": {}}
    ch = choose_champion(rep)
    assert ch["champion"] == "p_logistic" and "note" in ch
    rep["models"]["p_ensemble"]["log_loss"] = 0.615
    assert choose_champion(rep)["champion"] == "p_ensemble"
