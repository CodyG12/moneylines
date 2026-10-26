import json

import numpy as np
import pandas as pd
import pytest

import nba_model
from nbamodel.data import DataError, prepare_games
from nbamodel.pipeline import PipelineConfig, chronological_split, run_pipeline


def test_chronological_split_orders_and_never_splits_a_date():
    dates = pd.Series(pd.to_datetime(np.repeat(pd.date_range("2024-01-01", periods=10), 3)))
    split = chronological_split(dates, 0.6, 0.2)
    frame = pd.DataFrame({"date": dates, "split": split})
    assert frame.groupby("date")["split"].nunique().max() == 1
    assert frame.loc[frame.split == "train", "date"].max() < frame.loc[frame.split == "validation", "date"].min()
    assert frame.loc[frame.split == "validation", "date"].max() < frame.loc[frame.split == "holdout", "date"].min()


def test_pipeline_end_to_end_is_deterministic(synthetic_games):
    games, report = synthetic_games
    cfg = PipelineConfig(n_boot=200)
    r1 = run_pipeline(games, report, cfg)
    r2 = run_pipeline(games, report, cfg)
    pd.testing.assert_series_equal(r1.holdout["model_home_prob"], r2.holdout["model_home_prob"])
    m = r1.metrics
    assert set(m["holdout"]) == {"logistic", "elo", "home_base_rate"}
    assert m["holdout"]["logistic"]["n"] == len(r1.holdout)
    assert r1.holdout["model_home_prob"].between(0, 1).all()
    lo, hi = m["holdout"]["logistic"]["log_loss_ci"]["ci_low"], m["holdout"]["logistic"]["log_loss_ci"]["ci_high"]
    assert lo <= m["holdout"]["logistic"]["log_loss"] <= hi
    assert "market_comparison" in m and "paper_trading" in m
    assert r1.calibration["n"].sum() == len(r1.holdout)
    assert len(r1.upcoming) > 0 and r1.upcoming["home_score"].isna().all()


def test_holdout_scores_do_not_influence_model(synthetic_games):
    """Flipping holdout outcomes must not change holdout predictions."""
    games, report = synthetic_games
    cfg = PipelineConfig(n_boot=50)
    base = run_pipeline(games, report, cfg)
    tampered = games.copy()
    ho = tampered["game_id"].isin(base.holdout["game_id"])
    tampered.loc[ho, ["home_score", "away_score"]] = tampered.loc[ho, ["away_score", "home_score"]].values
    tampered.loc[ho, "home_win"] = 1 - tampered.loc[ho, "home_win"]
    # Features for the first holdout date are fixed; later holdout dates legitimately
    # use earlier holdout results via online Elo/form, but the fitted model must not.
    other = run_pipeline(tampered, report, cfg)
    first = base.holdout["date"].min()
    a = base.holdout.loc[base.holdout["date"] == first, "model_home_prob"]
    b = other.holdout.loc[other.holdout["date"] == first, "model_home_prob"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())
    assert base.metrics["model_selection"]["chosen_C"] == other.metrics["model_selection"]["chosen_C"]


def test_too_little_data_rejected():
    raw = pd.DataFrame({"date": ["2023-10-24"] * 2, "home_team": ["A", "C"], "away_team": ["B", "D"],
                        "home_score": [100, 100], "away_score": [90, 110]})
    games, report = prepare_games(raw)
    with pytest.raises(DataError, match="at least"):
        run_pipeline(games, report)


def test_cli_writes_outputs(tmp_path, synthetic_raw, capsys):
    src = tmp_path / "games.csv"
    synthetic_raw.to_csv(src, index=False)
    out = tmp_path / "out" / "predictions.csv"
    assert nba_model.main(["--games", str(src), "--output", str(out), "--n-boot", "100"]) == 0
    preds = pd.read_csv(out)
    assert {"model_home_prob", "fair_home_american", "elo_home_prob"} <= set(preds.columns)
    metrics = json.loads((tmp_path / "out" / "predictions_metrics.json").read_text())
    assert metrics["holdout"]["logistic"]["n"] == len(preds)
    assert (tmp_path / "out" / "predictions_upcoming.csv").exists()
    assert "Holdout" in capsys.readouterr().out


def test_cli_reports_bad_input(tmp_path, capsys):
    assert nba_model.main(["--games", str(tmp_path / "missing.csv")]) == 2
    assert "not found" in capsys.readouterr().err
