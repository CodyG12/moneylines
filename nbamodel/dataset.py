"""Historical training dataset: one row per game, as known at a pregame prediction time.

Each row carries the game's identity and schedule, the label and final scores
(for scoring only, never as inputs), the Phase 2 feature vector, and the
timestamps that say what information the features could have used:

  features_asof              results through the end of this ET date are included
                             (always the day before the game; same-day results never are)
  home_/away_last_game_date  the team's previous game date (all results used are on or before it)
  prediction_timestamp       when a prediction from this row could be made:
                             tip-off minus `lead_minutes` where tip-off is known,
                             otherwise 09:00 US/Eastern on game day ("nominal")
  source_fetched_at          when the source rows were downloaded (if recorded)
  train_eligible             played, and both teams have >= min_history earlier games

Injury/lineup features are not included: no reliable timestamped historical
source is wired in. See nbamodel/availability.py for the ingestion adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DataError
from .elo import EloConfig
from .features import FEATURES_V2, build_features
from .leakage import outcome_columns

NOMINAL_PREDICTION_HOUR_ET = 9


@dataclass(frozen=True)
class DatasetConfig:
    elo: EloConfig = field(default_factory=EloConfig)
    window: int = 10
    lead_minutes: float = 60.0
    min_history: int = 5


METADATA = ["game_key", "game_id", "game_id_source", "season", "date", "tipoff_utc", "home_team", "away_team",
            "neutral_site", "played"]
DIAGNOSTICS = ["home_elo_pre", "away_elo_pre", "elo_home_prob", "home_season_games", "away_season_games",
               "home_history", "away_history", "home_eff_n", "away_eff_n", "h2h_n", "home_days_rest",
               "away_days_rest", "league_rtg"]
# Present only when odds were attached (data.prepare_games / odds_join); never model inputs.
ODDS_COLUMNS = ["home_odds", "away_odds", "odds_timestamp", "odds_book", "close_home_odds", "close_away_odds",
                "close_odds_timestamp"]


def prediction_timestamps(dates: pd.Series, tipoff: pd.Series, lead_minutes: float):
    tip = pd.to_datetime(tipoff, utc=True, errors="coerce")
    nominal = (pd.to_datetime(dates) + pd.Timedelta(hours=NOMINAL_PREDICTION_HOUR_ET)).dt.tz_localize(
        "US/Eastern").dt.tz_convert("UTC")
    ts = tip - pd.Timedelta(minutes=lead_minutes)
    basis = np.where(tip.notna(), "tipoff_minus_lead", "nominal_0900_et")
    return ts.where(tip.notna(), nominal), basis


def build_training_dataset(games: pd.DataFrame, config: DatasetConfig = DatasetConfig()) -> pd.DataFrame:
    feats = build_features(games, config.elo, config.window)
    if "tipoff_utc" not in feats.columns:
        feats["tipoff_utc"] = pd.NaT
    if "game_id_source" not in feats.columns:
        feats["game_id_source"] = None
    odds = [c for c in ODDS_COLUMNS if c in feats.columns]
    cols = METADATA + outcome_columns(feats) + FEATURES_V2 + DIAGNOSTICS + odds
    ds = feats[list(dict.fromkeys(cols))].copy()
    ds["features_asof"] = ds["date"] - pd.Timedelta(days=1)
    ds["home_last_game_date"] = feats["home_last_game_date"]
    ds["away_last_game_date"] = feats["away_last_game_date"]
    ds["prediction_timestamp"], ds["prediction_timestamp_basis"] = prediction_timestamps(
        ds["date"], ds["tipoff_utc"], config.lead_minutes)
    ds["source_fetched_at"] = feats["fetched_at"] if "fetched_at" in feats.columns else None
    ds["train_eligible"] = ds["played"] & (ds["home_history"] >= config.min_history) & (
        ds["away_history"] >= config.min_history)
    validate_dataset(ds)
    return ds


def validate_dataset(ds: pd.DataFrame) -> None:
    """Structural timing checks; raises DataError on any row that could not exist pregame."""
    for side in ("home", "away"):
        last = ds[f"{side}_last_game_date"]
        if (last.notna() & (last >= ds["date"])).any():
            raise DataError(f"{side} features use a result from the game date or later")
    if (ds["features_asof"] >= ds["date"]).any():
        raise DataError("features_asof must be before the game date")
    tip = pd.to_datetime(ds["tipoff_utc"], utc=True, errors="coerce")
    if (tip.notna() & (ds["prediction_timestamp"] >= tip)).any():
        raise DataError("prediction_timestamp at or after tip-off")
    if ds["game_key"].duplicated().any():
        raise DataError("duplicate game_key in dataset")
