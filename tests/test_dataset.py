import pandas as pd
import pytest

from nbamodel.data import DataError
from nbamodel.dataset import build_training_dataset, validate_dataset


def test_dataset_rows_and_timestamps(multi_season_games):
    ds = build_training_dataset(multi_season_games)
    assert len(ds) == len(multi_season_games) and ds["game_key"].is_unique
    assert (ds["features_asof"] < ds["date"]).all()
    for side in ("home", "away"):
        last = ds[f"{side}_last_game_date"]
        assert (last.isna() | (last < ds["date"])).all()
    tip = pd.to_datetime(ds["tipoff_utc"], utc=True)
    assert (ds["prediction_timestamp"] == tip - pd.Timedelta(minutes=60)).all()
    assert (ds["prediction_timestamp_basis"] == "tipoff_minus_lead").all()
    assert not ds.loc[~ds["played"], "train_eligible"].any()
    assert ds.loc[ds["train_eligible"], ["home_history", "away_history"]].min().min() >= 5


def test_nominal_timestamp_without_tipoff(synthetic_games):
    games, _ = synthetic_games
    ds = build_training_dataset(games)
    assert (ds["prediction_timestamp_basis"] == "nominal_0900_et").all()
    first = ds.iloc[0]
    assert first["prediction_timestamp"] == pd.Timestamp(first["date"]).tz_localize("US/Eastern").tz_convert(
        "UTC") + pd.Timedelta(hours=9)


def test_validate_dataset_rejects_impossible_rows(multi_season_games):
    ds = build_training_dataset(multi_season_games)
    bad = ds.copy()
    bad.loc[bad.index[50], "home_last_game_date"] = bad.loc[bad.index[50], "date"]
    with pytest.raises(DataError, match="game date or later"):
        validate_dataset(bad)
    bad = ds.copy()
    bad.loc[bad.index[5], "prediction_timestamp"] = pd.Timestamp("2099-01-01", tz="UTC")
    with pytest.raises(DataError, match="tip-off"):
        validate_dataset(bad)
