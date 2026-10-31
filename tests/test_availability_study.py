import numpy as np
import pandas as pd
import pytest

from nbamodel import availability_study as study
from nbamodel import player_impact as pi
from nbamodel.dataset import AVAILABILITY_FEATURES, attach_availability, build_training_dataset
from nbamodel.features import FEATURES_V2
from nbamodel.walkforward import WalkForwardConfig

from .test_player_impact import _changes, _league


def _features_for(ds, seed=0):
    """Availability rows for every game from 2020 on (so 2019 is uncovered, like the real data),
    with a signal: impact partly follows the game's actual margin. SYNTHETIC, for plumbing only."""
    rng = np.random.default_rng(seed)
    rows = ds[ds["played"] & (ds["season"] >= 2020)]
    margin = rows["home_score"] - rows["away_score"]
    f = pd.DataFrame({"game_key": rows["game_key"].to_numpy()})
    f["injury_impact_diff"] = 0.4 * margin.to_numpy() + rng.normal(0, 4, len(rows))
    f["injury_impact_sd"] = rng.uniform(0, 2, len(rows))
    f["rotation_out_diff"] = rng.integers(-2, 3, len(rows))
    f["rotation_uncertain_diff"] = rng.integers(-1, 2, len(rows))
    f["report_available"] = 1
    return f


def test_attach_marks_uncovered_games_as_having_no_report(multi_season_games):
    ds = build_training_dataset(multi_season_games)
    out = attach_availability(ds, _features_for(ds))
    assert len(out) == len(ds) and set(AVAILABILITY_FEATURES) <= set(out.columns)
    uncovered = out[out["season"] < 2020]
    assert (uncovered["report_available"] == 0).all() and uncovered["injury_impact_diff"].isna().all()
    assert (out[out["played"] & (out["season"] >= 2020)]["report_available"] == 1).all()
    assert not set(AVAILABILITY_FEATURES) & set(FEATURES_V2)      # production feature list is untouched


def test_attach_rejects_duplicate_feature_rows(multi_season_games):
    ds = build_training_dataset(multi_season_games)
    f = _features_for(ds)
    with pytest.raises(ValueError):
        attach_availability(ds, pd.concat([f, f.iloc[:1]]))


def _rows_and_changes():
    day = "2025-04-30"
    rows = pd.DataFrame({"game_key": ["g1"], "date": [day], "season": [2024], "home_team": ["AAA"],
                         "away_team": ["BBB"], "prediction_timestamp": ["2025-04-30T22:00:00+00:00"], "played": [True]})
    changes = _changes([[day, "BBB@AAA", "AAA", "aaastar", "out", "2025-04-30T17:00:00+00:00"]])
    return rows, changes, pd.Series({day: "2025-04-30T17:00:00+00:00"}), day


def test_dataset_features_ignore_box_scores_from_the_game_date_on():
    rows, changes, first, day = _rows_and_changes()
    lg = _league()
    cfg = pi.ImpactConfig(ridge_lambda=5)
    base = pi.dataset_features(rows, changes, first, lg, cfg)
    tampered = lg.copy()
    later = pd.to_datetime(tampered["game_date"]) >= pd.Timestamp(day)
    assert later.any()
    tampered.loc[later, "pts"] = 0.0
    tampered.loc[later, "minutes"] = 48.0
    pd.testing.assert_frame_equal(base, pi.dataset_features(rows, changes, first, tampered, cfg))


def test_dataset_features_skip_dates_with_no_reports_and_derive_differences():
    rows, changes, first, _ = _rows_and_changes()
    extra = rows.assign(game_key="g0", date="2025-04-29", prediction_timestamp="2025-04-29T22:00:00+00:00")
    f = pi.dataset_features(pd.concat([extra, rows], ignore_index=True), changes, first, _league(),
                            pi.ImpactConfig(ridge_lambda=5))
    assert f["game_key"].tolist() == ["g1"]
    r = f.iloc[0]
    assert r["report_available"] == 1
    assert r["rotation_out_diff"] == r["home_n_rotation_out"] - r["away_n_rotation_out"] == 1


def test_game_with_only_one_team_reported_is_not_covered():
    f = pd.DataFrame({"home_report_available": [1, 1, 0], "away_report_available": [1, 0, 0]})
    assert pi.both_reported(f).tolist() == [1, 0, 0]


def _report(best_diff, best_high, early, n=1000):
    return {"n": n, "overall": {"A_logistic": {"log_loss": 0.62},
                                "B_logistic": {"log_loss": 0.62 + best_diff,
                                               "log_loss_diff_vs_base": {"estimate": best_diff, "ci_low": best_diff - .002,
                                                                         "ci_high": best_high}},
                                "E_gbm": {"log_loss": 0.63,
                                          "log_loss_diff_vs_base": {"estimate": .01, "ci_low": .005, "ci_high": .015}}},
            "by_segment": {"early (<15 games)": {"B_logistic": {"log_loss_diff_vs_base": {"estimate": early}}}}}


@pytest.mark.parametrize("diff,high,early,adopt", [(-.003, -.001, -.001, True),      # clear win everywhere
                                                   (-.003, .001, -.001, False),      # CI includes zero
                                                   (-.003, -.001, .002, False),      # worse early in the season
                                                   (.001, .002, -.001, False)])      # no better at all
def test_adoption_rule(diff, high, early, adopt):
    d = study.decide(_report(diff, high, early), ["B_logistic", "E_gbm"])
    assert d["best"] == ("B_logistic" if diff < .01 else "E_gbm")
    assert d["adopt"] is adopt


@pytest.fixture(scope="module")
def result(multi_season_games):
    ds = build_training_dataset(multi_season_games)
    ds = attach_availability(ds, _features_for(ds))
    cfg = WalkForwardConfig(retrain="season", calibration_dates=5, final_test_start="2022-08-01", min_train_seasons=1)
    return study.run(multi_season_games, ds, cfg, n_boot=20, seed=0)


def test_study_scores_every_variant_on_the_same_covered_development_games(result):
    rep, preds = result
    assert set(study.VARIANTS) <= set(preds.columns)
    assert (preds["report_available"] == 1).all() and (pd.to_datetime(preds["date"]) < "2022-08-01").all()
    assert preds[list(study.VARIANTS)].notna().all().all()
    assert rep["n"] == len(preds)
    assert all(rep["overall"][c]["n"] == rep["n"] for c in study.VARIANTS)


def test_study_reports_required_comparisons(result):
    rep, _ = result
    assert rep["overall"]["B_logistic"]["log_loss_diff_vs_base"]["ci_high"] is not None
    assert "log_loss_diff_vs_twin" in rep["overall"]["E_gbm"]
    assert {"by_season", "by_segment", "injury_heavy", "coverage", "decision", "feature_diagnostics"} <= set(rep)
    assert rep["market"]["n_priced"] == int(result[1]["home_odds"].notna().sum()) if "home_odds" in result[1] else True
    assert 0 <= rep["overall"]["C_platt"]["ece"] <= 1
    assert rep["feature_diagnostics"]["logistic_coefficients"].keys() >= set(AVAILABILITY_FEATURES)


def test_study_finds_a_signal_planted_in_the_features(result):
    # impact follows the margin here, so the injury models must beat the baseline by a lot
    rep, _ = result
    assert rep["overall"]["B_logistic"]["log_loss_diff_vs_base"]["ci_high"] < 0
